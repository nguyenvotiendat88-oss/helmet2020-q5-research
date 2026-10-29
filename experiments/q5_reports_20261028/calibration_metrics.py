"""Read cached logits; audit calibration on unused-for-fit VAL clips. No GPU."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path("/workspace/HELMET2020_training_kit_v1")
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from helmetlab import kit_models as km
from helmetlab.common import digest,read_jsonl
from helmetlab.labels import LabelMap

DIAG=Path("/workspace/runs/q5_audit_20261028")
CAL=Path("/workspace/runs/gq5_calibration_pilot_20261028")

def in_calibration(clip):
    return int(hashlib.sha256(("calibration_v1:"+str(clip)).encode()).hexdigest(),16)%3==0

def reliability(conf,correct,bins=15):
    edges=np.linspace(0.,1.,bins+1)
    group=np.minimum((conf*bins).astype(int),bins-1)
    rows=[];ece=0.
    for b in range(bins):
        take=group==b
        n=int(take.sum())
        avg=float(conf[take].mean()) if n else None
        acc=float(correct[take].mean()) if n else None
        if n:ece+=n/len(conf)*abs(avg-acc)
        rows.append({"lower":float(edges[b]),"upper":float(edges[b+1]),"n":n,
                     "mean_confidence":avg,"accuracy":acc})
    return {"ece":ece,"bins":rows}

def metrics(logits,ids,labels):
    q=km.decode(torch.from_numpy(logits),"Q5",labels)
    pred=q.argmax(-1)
    correct=pred==ids
    target=labels.states[ids]
    present=target!=0
    helmet=target==1
    z=logits.astype(np.float64)
    native_nll=(np.logaddexp(0,z[:,:5])-present*z[:,:5]
                +present*(np.logaddexp(0,z[:,5:])-helmet*z[:,5:])).mean()
    brier=(np.square(q).sum(-1)-2*q[np.arange(len(ids)),ids]+1).mean()
    result={"n":len(ids),"accuracy":float(correct.mean()),"confidence":float(q.max(-1).mean()),
            "native_masked_NLL_div5":float(native_nll),
            "constrained_class_NLL":float(-np.log(np.maximum(q[np.arange(len(ids)),ids],1e-300)).mean()),
            "multiclass_Brier_sum_over_classes":float(brier),
            "top_label_reliability":reliability(q.max(-1),correct),"roles":[]}
    p=1/(1+np.exp(-np.clip(z,-80,80)))
    for r,name in enumerate(labels.data["positions"]):
        ground=present[:,r];prediction=z[:,r]>=0
        n=int(ground.sum())
        tp=int((prediction&ground).sum());fp=int((prediction&~ground).sum())
        row={"role":name,"GT_present":n,"presence_TP":tp,"presence_FP":fp,
             "presence_FN":int((~prediction&ground).sum()),"presence_recall":tp/n if n else None,
             "presence_precision":tp/(tp+fp) if tp+fp else None,
             "presence_Brier":float(((p[:,r]-ground)**2).mean()),
             "presence_probability_reliability":reliability(p[:,r],ground)}
        if n:
            ph=p[ground,r+5];yh=helmet[ground,r]
            row["conditional_helmet"]={"accuracy_at_0.5":float(((ph>=.5)==yh).mean()),
                "Brier":float(((ph-yh)**2).mean()),
                "probability_reliability":reliability(ph,yh)}
        result["roles"].append(row)
    return result

def main():
    labels=LabelMap(ROOT/"manifests/classes.json")
    frames=read_jsonl(ROOT/"manifests/val.jsonl")
    ds=km.CropDataset("/workspace/helmet_images",frames,training=False)
    selected=json.loads((DIAG/"samples.json").read_text())["val"]
    uniform=set(np.random.default_rng(20261028).choice(len(ds),4096,replace=False).tolist())
    keep=np.array([i in uniform and not in_calibration(frames[ds.items[i]["row_index"]]["clip_id"]) for i in selected])
    fit=np.array([i in uniform and in_calibration(frames[ds.items[i]["row_index"]]["clip_id"]) for i in selected])
    assert keep.sum()+fit.sum()==4096 and fit.sum()==1398
    assert not (keep&fit).any()
    calibration=json.loads((CAL/"calibration.json").read_text())
    records={"schema":"helmet2020.calibration_reliability_audit.v1","population":"uniform4096_crop_sample_intersect_noncalibration_VAL_clips",
             "n":int(keep.sum()),"fit_crop_count":int(fit.sum()),"test_accessed":False,
             "independent_confirmation":False,"prior_development_VAL_seen":True,
             "scope":"GT crops only, not detector/background calibration",
             "sampling":"uniform crops then clip-based exclusion; not class-enriched diagnostic population",
             "source_hashes":{"sample_indices":digest(DIAG/"samples.json"),"calibration":digest(CAL/"calibration.json")},
             "arms":{}}
    for arm in ("G","Q5"):
        path=DIAG/f"{arm}_val_gt_logits.npz"
        z=np.load(path)
        expected=np.array([ds.items[i]["class_id"] for i in selected])
        assert np.array_equal(expected,z["ids"])
        t=calibration[arm]["temperature"]
        before=metrics(z["logits"][keep],z["ids"][keep],labels)
        after=metrics(z["logits"][keep]/t,z["ids"][keep],labels)
        assert [(r["presence_TP"],r["presence_FP"],r["presence_FN"]) for r in before["roles"]]==[(r["presence_TP"],r["presence_FP"],r["presence_FN"]) for r in after["roles"]]
        records["arms"][arm]={"temperature":t,"logits_sha256":digest(path),"before":before,"after":after}
    print(json.dumps(records,indent=2,allow_nan=False))

if __name__=="__main__":main()
