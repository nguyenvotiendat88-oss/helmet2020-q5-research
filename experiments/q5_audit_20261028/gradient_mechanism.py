"""CPU-only TRAIN gradient audit; no optimizer, checkpoint mutation, VAL or TEST."""
import os
os.environ["OMP_NUM_THREADS"]="2"
import collections, hashlib, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path("/workspace/HELMET2020_training_kit_v1")
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from helmetlab import kit_models as km
from helmetlab.common import read_jsonl,digest
from helmetlab.labels import LabelMap
from helmetlab.kit_runner import atomic_json

OUT=Path("/workspace/runs/q5_gradient_audit_20261028")
CHECKPOINTS=[
 (4,Path("/workspace/runs/gq5_long12_20261027/Q5/epoch04.pt"),"ea3aada6cc2640830853570b842ad15ae89c677c5ea6a7da4ee82dab49221dd6"),
 (12,Path("/workspace/runs/gq5_long12_20261027/Q5/epoch12.pt"),"e4f67f02973b2e03246846105b80c6c9d33fa346fd4446eb4828ae04c5744bb0"),
 (24,Path("/workspace/runs/gq5_extend24_20261028/Q5/epoch24.pt"),"306a97ef63cf35342dc1665e265e7bbe458d9b4285b340cecac4d2b7c7a429ab")]

def cosine(a,b):
    na=float(a.norm());nb=float(b.norm())
    return float(torch.dot(a,b)/(na*nb)) if min(na,nb)>1e-12 else None

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"report.json").exists():raise RuntimeError("Existing audit retained")
    labels=LabelMap(ROOT/"manifests/classes.json")
    frames=read_jsonl(ROOT/"manifests/train.jsonl")
    receipt=json.loads(Path("/workspace/runs/research_v1_fp32/trainval_pixels.json").read_text())
    hashes={k:v["sha256"] for k,v in receipt["files"].items()}
    ds=km.CropDataset("/workspace/helmet_images",frames,training=False,image_hashes=hashes)
    by_class=collections.defaultdict(list)
    for i,item in enumerate(ds.items):by_class[item["class_id"]].append(i)
    rng=np.random.default_rng(20261028)
    classes=sorted(by_class)
    batches=[[int(rng.choice(by_class[c])) for c in classes] for _ in range(8)]
    tensors=[]
    for indices in batches:
        rows=[ds[i] for i in indices]
        tensors.append((torch.stack([x for x,y in rows]),torch.tensor([y for x,y in rows])))
    mass=torch.tensor([len(by_class[c])/len(ds) for c in classes],dtype=torch.float32)
    assert abs(float(mass.sum())-1.)<1e-6
    result={"schema":"helmet2020.query_gradient_diagnostic.v1","device":"cpu",
        "sample":"8 batches with one TRAIN crop per supported class per batch; replacement allowed",
        "sample_weights":"class TRAIN frequency; weighted estimator of original training objective",
        "sample_indices":batches,"checkpoint_epochs":[4,12,24],"roles":labels.data["positions"],
        "augmentation":False,"model_mode":"eval; frozen BatchNorm running statistics",
        "parameter_scope":"only each role query vector; not shared backbone/attention matrix gradients",
        "test_accessed":False,"val_accessed":False,"optimizer_updates":0,"observations":[],
        "limitations":"Small diagnostic snapshots, fixed eval mode, not causal proof of training failure"}
    atomic_json(OUT/"plan.json",{k:v for k,v in result.items() if k!="observations"})
    for epoch,path,expected in CHECKPOINTS:
        if digest(path)!=expected:raise RuntimeError("Checkpoint mismatch")
        ck=torch.load(path,map_location="cpu",weights_only=True)
        model=km.build_classifier("Q5",initialize=False)
        model.load_state_dict(ck["state_dict"],strict=True);del ck
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
        query=model.head.predictor.queries
        query.requires_grad_(True)
        for batch_index,(pixels,ids) in enumerate(tensors):
            with torch.no_grad():features=model.forward_features(pixels)
            logits=model.head(features)
            target=torch.tensor(labels.states)[ids]
            present=(target!=0).float();helmet=(target==1).float()
            occupancy=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,:5],present,reduction="none")
            conditional=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,5:],helmet,reduction="none")
            losses={"presence_positive":(mass[:,None]*present*occupancy).sum(0)/5.,
                    "presence_negative":(mass[:,None]*(1-present)*occupancy).sum(0)/5.,
                    "helmet_conditional":(mass[:,None]*present*conditional).sum(0)/5.}
            for role in range(5):
                gradients={name:torch.autograd.grad(value[role],query,retain_graph=True)[0][role].detach()
                           for name,value in losses.items()}
                gp=gradients["presence_positive"]+gradients["presence_negative"]
                gh=gradients["helmet_conditional"];gn=gradients["presence_negative"]
                if not all(torch.isfinite(g).all() for g in gradients.values()):raise RuntimeError("Nonfinite gradient")
                result["observations"].append({"epoch":epoch,"batch":batch_index,"role":labels.data["positions"][role],
                    "positive_samples":int(present[:,role].sum()),
                    "norms":{name:float(g.norm()) for name,g in gradients.items()},
                    "cos_presence_vs_helmet":cosine(gp,gh),
                    "cos_negative_presence_vs_helmet":cosine(gn,gh),
                    "negative_to_helmet_norm":float(gn.norm()/gh.norm()) if float(gh.norm())>1e-12 else None})
        del model
        print(json.dumps({"epoch":epoch,"observations":len(result["observations"])}),flush=True)
    summary=[]
    for epoch,path,h in CHECKPOINTS:
        for role in result["roles"]:
            rows=[x for x in result["observations"] if x["epoch"]==epoch and x["role"]==role]
            r={"epoch":epoch,"role":role,"batches":len(rows)}
            for key in ("cos_presence_vs_helmet","cos_negative_presence_vs_helmet"):
                vals=[x[key] for x in rows if x[key] is not None]
                r[key]={"mean":float(np.mean(vals)) if vals else None,
                        "negative_fraction":sum(x<0 for x in vals)/len(vals) if vals else None,
                        "valid_batches":len(vals)}
            vals=[x["negative_to_helmet_norm"] for x in rows if x["negative_to_helmet_norm"] is not None]
            r["negative_to_helmet_norm_median"]=float(np.median(vals)) if vals else None
            summary.append(r)
    result.update(status="completed",summary=summary,finished_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint_hashes={str(e):h for e,p,h in CHECKPOINTS})
    atomic_json(OUT/"report.json",result)
    print(json.dumps({"status":"completed","summary":summary}),flush=True)

if __name__=="__main__":main()
