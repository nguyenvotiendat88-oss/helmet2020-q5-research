"""Equal-weight three-checkpoint ensembles with bounded calibration-only selection."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ.setdefault("OMP_NUM_THREADS","2")
import collections,gc,json,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=Path("/workspace/HELMET2020_training_kit_v1")
sys.path[:0]=[str(ROOT),str(HERE)]
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from torchvision import transforms
from helmetlab import kit_models as km
from helmetlab.common import digest,read_jsonl,assert_aligned_frames
from helmetlab.kit_runner import atomic_json,canonical,source_bindings,seed_rng,runtime,execution_lock,score_predictions
from helmetlab.labels import LabelMap
import fast_eval
from posthoc_core import combine,train_prior,adjust,fit_temperature,select_tau,TAUS

OUT=Path("/workspace/runs/ensemble3_resume_20261029")
OLD=Path("/workspace/runs/research_v1_fp32")
REPLICATION=Path("/workspace/runs/gq5_confirmation_20261028")
DIAG=Path("/workspace/runs/q5_audit_20261028")
IMAGES=Path("/workspace/helmet_images")
DEVICE="cuda:0"

def now():return datetime.now(timezone.utc).isoformat()
def status(**values):
    row={"pid":os.getpid(),"time_utc":now(),**values}
    atomic_json(OUT/"status.json",row);print(json.dumps(row),flush=True)
def guard():
    if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError("8GiB reserve reached")
def clip_cal(clip):
    import hashlib
    return int(hashlib.sha256(("calibration_v1:"+str(clip)).encode()).hexdigest(),16)%3==0
def checkpoint_dir(seed):
    return Path("/workspace/runs/gq5_extend24_20261028") if seed==0 else REPLICATION/f"seed{seed}"/"phase24"

def setup():
    summary=json.loads((REPLICATION/"replication_summary.json").read_text())
    if summary["status"]!="completed" or summary["seeds_available"]!=[0,1,2]:
        raise RuntimeError("Need three completed seeds")
    parent=json.loads(Path("/workspace/runs/gq5_extend24_20261028/study.json").read_text())
    if source_bindings()!=parent["kit_sources"]:raise RuntimeError("Kit source drift")
    manifests={n:digest(ROOT/"manifests"/(n+(".json" if n=="classes" else ".jsonl"))) for n in ("train","val","classes")}
    if manifests!=parent["manifest_sha256"]:raise RuntimeError("Dataset identity differs")
    frames=read_jsonl(ROOT/"manifests/val.jsonl")
    if len(frames)!=14600:raise RuntimeError("Unexpected VAL")
    train=read_jsonl(ROOT/"manifests/train.jsonl")
    counts=np.bincount([o["class_id"] for r in train for o in r["objects"]],minlength=36)
    del train
    labels=LabelMap(ROOT/"manifests/classes.json")
    pixels=json.loads((OLD/"trainval_pixels.json").read_text())
    hashes={k:v["sha256"] for k,v in pixels["files"].items()}
    if canonical(hashes)!=parent["pixel_hash_map_sha256"]:raise RuntimeError("Pixel receipt changed")
    proposal_file=OLD/"seed0/retinanet/val_predictions.jsonl"
    if digest(proposal_file)!=parent["detector_proposals_sha256"]:raise RuntimeError("Proposal drift")
    proposals=read_jsonl(proposal_file)
    assert_aligned_frames(frames,proposals,len(labels))
    plans=[fast_eval._unique_boxes(f,p) for f,p in zip(frames,proposals)]
    offsets=np.r_[0,np.cumsum([len(p.boxes) for p in plans])].astype(np.int64)
    binding={"schema":"helmet2020.ensemble3.v1","source_hashes":source_bindings(),
             "implementation":{n:digest(HERE/n) for n in ("run_ensemble.py","posthoc_core.py","fast_eval.py","reuse_g.py")},
             "manifest_sha256":manifests,"proposal_sha256":digest(proposal_file),
             "pixel_hash_map_sha256":canonical(hashes),"runtime":runtime(DEVICE),
             "seeds":[0,1,2],"epoch":24,"equal_weight":True,"ensemble_space":"mean of decoded36 probabilities",
             "batch":64,"crop_size":224,"crop_margin":.05,"cache_dtype":"float64",
             "score_thresh":.001,"nms":.5,"cap":300,"train_counts":counts.tolist(),
             "prior_smoothing":1.,"tau_grid":list(TAUS),"T_bounds":[.5,10.],
             "calibration_clips":sorted({r["clip_id"] for r in frames if clip_cal(r["clip_id"])}),
             "evaluation_clips":sorted({r["clip_id"] for r in frames if not clip_cal(r["clip_id"])}),
             "calibration_selection":"max_macro_with_weighted_floor_minus0.5pp_vs_tau0_tie_lower_tau",
             "num_proposals":int(offsets[-1]),"num_frames":len(frames),
             "test_accessed":False,"training_updates":0,"checkpoints":{}}
    if set(binding["calibration_clips"])&set(binding["evaluation_clips"]):raise RuntimeError("Clip overlap")
    for row in summary["rows"]:
        seed=row["seed"]
        for arm,rec in row["endpoint"].items():
            p=checkpoint_dir(seed)/arm/"epoch24.pt"
            if digest(p)!=rec["checkpoint_sha256"]:raise RuntimeError("Checkpoint drift")
            binding["checkpoints"][f"{arm}_seed{seed}"]={"path":str(p),"sha256":rec["checkpoint_sha256"]}
    path=OUT/"study.json"
    if path.exists() and json.loads(path.read_text())!=binding:raise RuntimeError("Study binding differs")
    atomic_json(path,binding)
    return binding,frames,proposals,plans,offsets,hashes,labels,train_prior(counts)

def load_models(arm,binding):
    models=[]
    for seed in (0,1,2):
        row=binding["checkpoints"][f"{arm}_seed{seed}"]
        if digest(Path(row["path"]))!=row["sha256"]:raise RuntimeError("Checkpoint changed before load")
        model=km.build_classifier(arm,seed=seed,initialize=False).to(DEVICE).eval()
        state=torch.load(row["path"],map_location="cpu",weights_only=True)
        if state["epoch"]!=24 or state["seed"]!=seed or state["arm"]!=arm:raise RuntimeError("Wrong checkpoint identity")
        model.load_state_dict(state["state_dict"],strict=True);del state
        # Keep Q5 parameter flags identical to its original evaluation.
        # inference_mode still suppresses graph construction; no optimizer exists.
        if arm=="G":
            for param in model.parameters():param.requires_grad_(False)
        models.append(model)
    return models

def postprocess(frame,plan,q):
    objects=[]
    if len(plan.boxes):
        scores=q*np.asarray(plan.detector_scores)[:,None]
        boxids,classids=np.nonzero(scores>.001)
        if len(boxids):
            boxes=torch.tensor(np.asarray(plan.boxes)[boxids],dtype=torch.float32)
            values=torch.tensor(scores[boxids,classids],dtype=torch.float32)
            classes=torch.tensor(classids,dtype=torch.long)
            keep=km.batched_nms(boxes,values,classes,.5)[:300]
            for i in keep.tolist():
                b=int(boxids[i]);c=int(classids[i])
                objects.append({"bbox":list(plan.boxes[b]),"class_id":c,"score":float(values[i]),
                                "detector_score":float(plan.detector_scores[b])})
    return {**{k:v for k,v in frame.items() if k!="objects"},"objects":objects}

def verify_prefix(arm,models,frames,proposals,plans,hashes,labels):
    original=checkpoint_dir(0)/arm/"val_epoch24/predictions.jsonl"
    from itertools import islice
    with original.open() as f:expected=[json.loads(x) for x in islice(f,3)]
    actual=km.classifier_predictions(models[0],IMAGES,frames[:3],proposals[:3],labels,
        device=DEVICE,batch=64,margin=.05,size=224,score_thresh=.001,nms_thresh=.5,max_detections=300,image_hashes=hashes)
    if actual!=expected:raise RuntimeError("Original seed0 inference no longer matches")
    resize=transforms.Resize((224,224));mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    replay=[]
    with torch.inference_mode():
        for f,p in zip(frames[:3],plans[:3]):
            if p.boxes:
                crops=fast_eval._prepare_resized(IMAGES,p,.05,resize,hashes,km._rgb,km.crop_box)
                probs=[]
                for start in range(0,len(crops),64):
                    x=fast_eval._batched_tensorize(crops[start:start+64],None,mean,std).to(DEVICE)
                    probs.append(km.decode(models[0](x),arm,labels))
                q=np.concatenate(probs)
            else:q=np.empty((0,len(labels)))
            replay.append(postprocess(f,p,q))
    if replay!=expected:raise RuntimeError("Cached-replay scoring parity failed")
    return {"frames":3,"original_vs_saved_exact":True,"replay_vs_saved_exact":True}

def cal_probs(arm,models,binding,frames,hashes,labels):
    root=OUT/arm;ident=canonical(binding);done=root/"cal_probs_complete.json"
    if done.exists():
        r=json.loads(done.read_text())
        if r["study_sha256"]!=ident or digest(root/"cal_probs.npz")!=r["sha256"]:raise RuntimeError("Cal cache drift")
        z=np.load(root/"cal_probs.npz");return z["q"],z["ids"]
    ds=km.CropDataset(IMAGES,frames,training=False,image_hashes=hashes)
    indices=sorted(i for i in np.random.default_rng(20261028).choice(len(ds),4096,replace=False).tolist()
                   if clip_cal(frames[ds.items[i]["row_index"]]["clip_id"]))
    if len(indices)!=1398:raise RuntimeError("Calibration population differs")
    loader=DataLoader(Subset(ds,indices),batch_size=64,shuffle=False,num_workers=4,
                      multiprocessing_context="spawn",pin_memory=True)
    probs=[];ids=[]
    with torch.inference_mode():
        for x,y in loader:
            guard();x=x.to(DEVICE,non_blocking=True)
            probs.append(combine([km.decode(m(x),arm,labels) for m in models]));ids.append(y.numpy())
    q=np.concatenate(probs);y=np.concatenate(ids)
    np.savez_compressed(root/"cal_probs.npz",q=q,ids=y,indices=indices)
    atomic_json(done,{"study_sha256":ident,"sha256":digest(root/"cal_probs.npz"),"samples":len(y)})
    return q,y

def cache_probabilities(arm,models,binding,frames,plans,offsets,hashes,labels):
    root=OUT/arm;path=root/"proposal_probabilities.npy";receipt=root/"cache_complete.json"
    ident=canonical(binding);shape=(int(offsets[-1]),len(labels));progress=root/"cache_progress.json"
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r["study_sha256"]!=ident or r["sha256"]!=digest(path):raise RuntimeError("Cache hash mismatch")
        return np.load(path,mmap_mode="r")
    if progress.exists():
        rec=json.loads(progress.read_text())
        if rec["study_sha256"]!=ident:raise RuntimeError("Partial cache identity mismatch")
        start=rec["completed_frames"];mm=np.load(path,mmap_mode="r+")
        if mm.shape!=shape or mm.dtype!=np.float64:raise RuntimeError("Partial cache layout mismatch")
    else:
        if path.exists():raise RuntimeError("Unbound partial cache exists")
        mm=np.lib.format.open_memmap(path,mode="w+",dtype=np.float64,shape=shape)
        mm.flush();start=0
        atomic_json(progress,{"study_sha256":ident,"completed_frames":0})
    resize=transforms.Resize((224,224));mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    tick=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
        pending=deque();cursor=start
        def fill():
            nonlocal cursor
            while cursor<len(plans) and len(pending)<8:
                p=plans[cursor]
                future=pool.submit(fast_eval._prepare_resized,IMAGES,p,.05,resize,hashes,km._rgb,km.crop_box) if p.boxes else None
                pending.append((cursor,p,future));cursor+=1
        fill()
        while pending:
            i,p,future=pending.popleft();guard()
            resized=future.result() if future else []
            for begin in range(0,len(resized),64):
                x=fast_eval._batched_tensorize(resized[begin:begin+64],None,mean,std).to(DEVICE)
                q=combine([km.decode(m(x),arm,labels) for m in models])
                mm[offsets[i]+begin:offsets[i]+begin+len(q)]=q
            if (i+1)%100==0 or i+1==len(frames):
                mm.flush();atomic_json(progress,{"study_sha256":ident,"completed_frames":i+1})
            if (i+1)%250==0 or i+1==len(frames):
                status(state="running",arm=arm,phase="cache_inference",frames=i+1,total=len(frames),
                       resume_from_frame=start,elapsed_seconds=time.perf_counter()-tick)
            del resized,future
            fill()
    mm.flush();del mm
    atomic_json(receipt,{"study_sha256":ident,"sha256":digest(path),"frames":len(frames),"shape":list(shape),
                        "dtype":"float64","meaning":"equal arithmetic mean of three decoded36 probabilities"})
    return np.load(path,mmap_mode="r")

def score_group(frames,plans,offsets,cache,indices,T,tau,prior,arm,variant,group,write_path=None):
    pred=[];truth=[]
    status(state="running",arm=arm,phase="postprocess",variant=variant,group=group)
    for j,i in enumerate(indices):
        guard()
        q=np.asarray(cache[offsets[i]:offsets[i+1]])
        q=adjust(q,T,tau,prior) if len(q) else q
        pred.append(postprocess(frames[i],plans[i],q));truth.append(frames[i])
        if (j+1)%2500==0:status(state="running",arm=arm,phase="postprocess",variant=variant,group=group,frames=j+1,total=len(indices))
    status(state="running",arm=arm,phase="scoring",variant=variant,group=group,frames=len(indices))
    metrics=score_predictions(truth,pred,len(prior))
    if write_path:
        tmp=write_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in pred:f.write(json.dumps(row,allow_nan=False)+"\n")
            f.flush();os.fsync(f.fileno())
        tmp.replace(write_path)
    return metrics

def analyze(arm,binding,frames,plans,offsets,cache,labels,prior,qcal,ycal):
    root=OUT/arm;ident=canonical(binding)
    final=root/"result.json"
    if final.exists():
        r=json.loads(final.read_text())
        if r["study_sha256"]!=ident:raise RuntimeError("Completed result identity mismatch")
        for name,record in r["evaluation"].items():
            if digest(root/f"eval_{name}_predictions.jsonl")!=record["predictions_sha256"]:
                raise RuntimeError("Completed evaluation predictions changed")
        return r
    calids=[i for i,r in enumerate(frames) if clip_cal(r["clip_id"])]
    evids=[i for i,r in enumerate(frames) if not clip_cal(r["clip_id"])]
    if len(calids)!=5200 or len(evids)!=9400:raise RuntimeError("Clip split drift")
    selected_path=root/"selection.json"
    if selected_path.exists():
        selection=json.loads(selected_path.read_text())
        if selection["study_sha256"]!=ident:raise RuntimeError("Frozen selection identity mismatch")
        T=selection["temperature"];tau=selection["tau"]
    else:
        temp=fit_temperature(qcal,ycal);T=temp["temperature"]
        atomic_json(root/"temperature.json",{"study_sha256":ident,**temp})
        records=[]
        for tau in TAUS:
            path=root/f"cal_tau_{tau:g}.json"
            if path.exists():
                obj=json.loads(path.read_text())
                if obj["study_sha256"]!=ident or obj["temperature"]!=T:raise RuntimeError("Calibration metric drift")
                m=obj["metrics"]
            else:
                m=score_group(frames,plans,offsets,cache,calids,T,tau,prior,arm,f"T_tau{tau}","calibration")
                atomic_json(path,{"study_sha256":ident,"temperature":T,"tau":tau,"metrics":m})
            records.append({"tau":tau,"macro_AP50":m["macro_mAP"],"weighted_AP50":m["weighted_mAP"]})
        chosen=select_tau(records);tau=chosen["tau"]
        selection={"study_sha256":ident,"temperature":T,"tau":tau,"calibration_grid":records,
                   "rule":binding["calibration_selection"],"frozen_before_evaluation":True,"time_utc":now()}
        atomic_json(selected_path,selection)
    variants=[("raw",1.,0.),("temperature",T,0.),("temperature_logit_adjustment",T,tau)]
    evaluation={}
    for name,t,adjustment in variants:
        mpath=root/f"eval_{name}.json";ppath=root/f"eval_{name}_predictions.jsonl"
        if mpath.exists():
            rec=json.loads(mpath.read_text())
            if rec["study_sha256"]!=ident or digest(ppath)!=rec["predictions_sha256"]:raise RuntimeError("Eval artifact changed")
        else:
            m=score_group(frames,plans,offsets,cache,evids,t,adjustment,prior,arm,name,"noncalibration_clips",ppath)
            rec={"study_sha256":ident,"temperature":t,"tau":adjustment,"metrics":m,
                 "predictions_sha256":digest(ppath),"test_accessed":False}
            atomic_json(mpath,rec)
        evaluation[name]=rec
    result={"status":"completed","arm":arm,"study_sha256":ident,"selection":selection,"evaluation":evaluation,
            "scope":"VAL development;94 noncalibration clips; not independent TEST",
            "ensemble_members":3,"test_accessed":False,"training_updates":0}
    atomic_json(final,result)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with execution_lock(DEVICE):
        seed_rng(0,2);guard()
        status(state="running",phase="setup")
        binding,frames,proposals,plans,offsets,hashes,labels,prior=setup()
        results={}
        for arm in ("G","Q5"):
            if arm=="G":
                from reuse_g import verified_original_G
                results[arm]=verified_original_G(binding,OUT)
                status(state="running",arm="G",phase="verified_completed_G_reused")
                continue
            root=OUT/arm;root.mkdir(exist_ok=True)
            status(state="running",arm=arm,phase="loading_frozen_checkpoints")
            models=load_models(arm,binding)
            parity=verify_prefix(arm,models,frames,proposals,plans,hashes,labels)
            atomic_json(root/"parity.json",{"study_sha256":canonical(binding),**parity})
            qcal,ycal=cal_probs(arm,models,binding,frames,hashes,labels)
            cache=cache_probabilities(arm,models,binding,frames,plans,offsets,hashes,labels)
            del models;gc.collect();torch.cuda.empty_cache()
            results[arm]=analyze(arm,binding,frames,plans,offsets,cache,labels,prior,qcal,ycal)
            del cache;gc.collect()
        comparisons={}
        for variant in ("raw","temperature","temperature_logit_adjustment"):
            g=results["G"]["evaluation"][variant]["metrics"];q=results["Q5"]["evaluation"][variant]["metrics"]
            comparisons[variant]={"macro_Q5_minus_G_pp":100*(q["macro_mAP"]-g["macro_mAP"]),
                                  "weighted_Q5_minus_G_pp":100*(q["weighted_mAP"]-g["weighted_mAP"])}
        target=comparisons["temperature_logit_adjustment"]
        flag=target["macro_Q5_minus_G_pp"]<=0 or target["weighted_Q5_minus_G_pp"]<-.5
        if source_bindings()!=binding["source_hashes"]:raise RuntimeError("Kit source changed")
        for n,h in binding["implementation"].items():
            if digest(HERE/n)!=h:raise RuntimeError("Implementation changed")
        result={"status":"completed","study_sha256":canonical(binding),"arms":results,"comparisons":comparisons,
                "longtail_review_indicated":flag,"auto_train_longtail":False,
                "decision":"Review per-class/clip errors and freeze matched3seed long-tail recipe if indicated; no TEST tuning",
                "test_accessed":False,"training_updates":0}
        atomic_json(OUT/"comparison_result.json",result)
        status(state="completed",result="comparison_result.json",longtail_review_indicated=flag)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if OUT.exists():status(state="failed",error=str(exc),exception=type(exc).__name__)
        traceback.print_exc()
        raise
