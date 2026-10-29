"""G/Q5 seed0, fresh ImageNet, cosine12, full VAL at epochs4/8/12."""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("OMP_NUM_THREADS", "2")
import gc, hashlib, json, math, shutil, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path("/workspace/HELMET2020_training_kit_v1")
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
import torch
from torch.utils.data import DataLoader
from helmetlab import kit_models as km
from helmetlab.common import digest, read_jsonl, validate_frames
from helmetlab.checkpoints import atomic_save, cpu_tree, rng_state, restore_rng
from helmetlab.labels import LabelMap
from helmetlab.kit_runner import (atomic_json,canonical,seed_rng,runtime,source_bindings,
    stage_verified,binding_for,score_predictions,execution_lock,move_optimizer)
from fast_eval import classifier_predictions_prefetch

OLD=Path("/workspace/runs/research_v1_fp32")
OUT=Path("/workspace/runs/gq5_long12_20261027")
IMAGES=Path("/workspace/helmet_images")
DEVICE="cuda:0"
PLAN={
 "schema":"helmet2020.gq5_long12.v1","arms":["G","Q5"],"seed":0,
 "epochs_per_arm":12,"val_epochs":[4,8,12],"selection":"max_VAL_macro_AP50_tie_earliest",
 "train_batch":256,"loader_workers":8,"prefetch_factor":2,"persistent_workers":False,
 "worker_context":"spawn","cpu_threads":2,"fp32":True,"amp":False,"tf32":False,
 "optimizer":"AdamW","lr":0.0001,"weight_decay":0.0001,
 "scheduler":"CosineAnnealingLR","scheduler_T_max":12,"eta_min":0.0,
 "crop_size":224,"margin":0.05,"train_frames":58200,"train_crops":181176,
 "val_frames":14600,"val_batch":64,"val_prefetch_workers":4,"val_prefetch_frames":8,
 "score_thresh":0.001,"nms_thresh":0.5,"max_detections":300,"min_free_disk_gib":8,
 "initialization":"fresh_ImageNet_both_arms; no_screening_or_old_G_warm_start",
 "detector":"reuse_verified_seed0_RetinaNet; same_frozen_proposals",
 "test_accessed":False,"auto_expand":False,"hours_limit":None,"early_stopping":False,
 "checkpoint_policy":"save_each_completed_train_epoch_before_VAL; keep_milestones",
 "interpretation":"development_study_after_two_epoch_screen; one_seed; no_TEST_claim",
 "instance_auto_stop":False,
}

def now():
 return datetime.now(timezone.utc).isoformat()

def guard():
 if shutil.disk_usage(OUT).free<PLAN["min_free_disk_gib"]*1024**3:
  raise RuntimeError("Disk reserve reached; last completed training checkpoint retained")

def emit(kind,**data):
 row={"kind":kind,"time_utc":now(),**data}
 print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
 atomic_json(OUT/"progress.json",row)

def set_status(state,**data):
 atomic_json(OUT/"status.json",{"status":state,"pid":os.getpid(),"updated_utc":now(),**data})

def verify_sources(binding):
 if binding["kit_sources"]!=source_bindings():raise RuntimeError("Kit source changed")
 if any(digest(HERE/n)!=h for n,h in binding["implementation"].items()):
  raise RuntimeError("Run implementation changed")

def setup():
 old=json.loads((OLD/"study.json").read_text())
 if old["sources"]!=source_bindings():raise RuntimeError("Source kit differs from detector study")
 det=OLD/"seed0/retinanet"
 if not stage_verified(det,binding_for(old,0,"retinanet")):
  raise RuntimeError("Detector completion integrity failed")
 manifest={}
 for n in ("train","val","classes"):
  path=ROOT/"manifests"/(n+(".json" if n=="classes" else ".jsonl"))
  manifest[n]=digest(path)
  if manifest[n]!=old["dataset"]["files"][n]["sha256"]:raise RuntimeError("Manifest changed: "+n)
 frames={n:read_jsonl(ROOT/"manifests"/(n+".jsonl")) for n in ("train","val")}
 if len(frames["train"])!=PLAN["train_frames"] or len(frames["val"])!=PLAN["val_frames"]:
  raise RuntimeError("Wrong population")
 labels=LabelMap(ROOT/"manifests/classes.json")
 for records in frames.values():validate_frames(records,len(labels))
 receipt=json.loads((OLD/"trainval_pixels.json").read_text())
 hashes={p:v["sha256"] for p,v in receipt["files"].items()}
 if canonical(hashes)!=old["pixel_sha256"]:raise RuntimeError("Pixel receipt changed")
 props=read_jsonl(det/"val_predictions.jsonl")
 km.assert_aligned_frames(frames["val"],props,len(labels))
 weights=ROOT/"pretrained/resnet18-f37072fd.pth"
 if digest(weights)!=old["initialization"][weights.name]:raise RuntimeError("ImageNet asset changed")
 binding={
  "plan":PLAN,"kit_sources":source_bindings(),
  "implementation":{n:digest(HERE/n) for n in ("engine12.py","engine24.py","run_confirmation.py","fast_eval.py")},
  "detector_source_study_sha256":canonical(old),"detector_best_sha256":digest(det/"best.pt"),
  "detector_proposals_sha256":digest(det/"val_predictions.jsonl"),
  "manifest_sha256":manifest,"pixel_hash_map_sha256":canonical(hashes),
  "pretraining_sha256":digest(weights),"runtime":runtime(DEVICE),
 }
 path=OUT/"study.json"
 if path.exists():
  if json.loads(path.read_text())!=binding:raise RuntimeError("Study binding differs; refusing resume")
 else:atomic_json(path,binding)
 return binding,frames,props,hashes,labels

def tensor_digest(state):
 h=hashlib.sha256()
 for name,v in sorted(state.items()):
  a=v.detach().cpu().contiguous()
  h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
 return h.hexdigest()

def snapshot_epoch(d,epoch):
 target=d/f"epoch{epoch:02d}.pt"
 if target.exists():
  if digest(target)!=digest(d/"last.pt"):raise RuntimeError("Milestone snapshot differs")
 else:os.link(d/"last.pt",target)  # later last.pt saves use atomic replace, never in-place
 return target

def choose_best(records):
 if not records:raise ValueError("No evaluations")
 return min(records,key=lambda r:(-r["macro_AP50"],r["epoch"]))

def pending_val_epochs(completed):
 return [e for e in PLAN["val_epochs"] if e<=completed]

def verify_eval(d,epoch,ident):
 sub=d/f"val_epoch{epoch:02d}"
 receipt=sub/"complete.json"
 if not receipt.exists():return None
 rec=json.loads(receipt.read_text())
 if rec["study_sha256"]!=ident or rec["epoch"]!=epoch:
  raise RuntimeError("VAL identity mismatch")
 for rel,key in ((f"epoch{epoch:02d}.pt","checkpoint_sha256"),
                 (f"val_epoch{epoch:02d}/metrics.json","metrics_sha256"),
                 (f"val_epoch{epoch:02d}/predictions.jsonl","predictions_sha256")):
  if digest(d/rel)!=rec[key]:raise RuntimeError("VAL artifact hash mismatch: "+rel)
 return rec

def evaluate_milestone(arm,epoch,d,binding,frames,props,hashes,labels):
 ident=canonical(binding)
 rec=verify_eval(d,epoch,ident)
 if rec is not None:return rec
 verify_sources(binding);guard()
 p=d/f"epoch{epoch:02d}.pt"
 ck=torch.load(p,map_location="cpu",weights_only=True)
 if ck["study_sha256"]!=ident or ck["arm"]!=arm or ck["epoch"]!=epoch:
  raise RuntimeError("Evaluation checkpoint identity mismatch")
 model=km.build_classifier(arm,seed=PLAN["seed"],initialize=False).to(DEVICE)
 model.load_state_dict(ck["state_dict"],strict=True)
 del ck
 sub=d/f"val_epoch{epoch:02d}";sub.mkdir(exist_ok=True)
 set_status("running",method=arm,epoch=epoch,phase="val")
 emit("val_started",method=arm,epoch=epoch,total_frames=len(frames["val"]))
 tick=time.perf_counter()
 def progress(count,total):
  emit("val_progress",method=arm,epoch=epoch,completed_frames=count,total_frames=total,
       elapsed_seconds=time.perf_counter()-tick,phase="val")
 pred=classifier_predictions_prefetch(model,IMAGES,frames["val"],props,labels,DEVICE,
  batch=64,margin=.05,size=224,score_thresh=.001,nms_thresh=.5,max_detections=300,
  guard=guard,image_hashes=hashes,workers=4,prefetch_frames=8,
  kit_models_module=km,batched_cpu=True,progress_callback=progress)
 inference=time.perf_counter()-tick
 emit("val_scoring",method=arm,epoch=epoch,phase="AP")
 metrics=score_predictions(frames["val"],pred,len(labels))
 atomic_json(sub/"metrics.json",metrics)
 tmp=sub/"predictions.jsonl.tmp"
 with tmp.open("w") as f:
  for row in pred:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
  f.flush();os.fsync(f.fileno())
 tmp.replace(sub/"predictions.jsonl")
 rec={
  "status":"completed","arm":arm,"seed":PLAN["seed"],"epoch":epoch,"study_sha256":ident,
  "macro_AP50":metrics["macro_mAP"],"weighted_AP50":metrics["weighted_mAP"],
  "val_frames":len(frames["val"]),"supported_classes":metrics["supported_classes"],
  "inference_seconds":inference,"total_eval_seconds":time.perf_counter()-tick,
  "checkpoint_sha256":digest(p),"metrics_sha256":digest(sub/"metrics.json"),
  "predictions_sha256":digest(sub/"predictions.jsonl"),"test_accessed":False,
 }
 atomic_json(sub/"complete.json",rec);emit("val_completed",**rec)
 del model,pred;gc.collect();torch.cuda.empty_cache()
 return rec

def train_arm(arm,binding,frames,props,hashes,labels):
 d=OUT/arm;d.mkdir(exist_ok=True)
 ident=canonical(binding)
 done=d/"complete.json"
 if done.exists():
  rec=json.loads(done.read_text())
  if rec["study_sha256"]!=ident or digest(d/"last.pt")!=rec["last_checkpoint_sha256"]:
   raise RuntimeError("Completed arm identity mismatch")
  for e in PLAN["val_epochs"]:
   if verify_eval(d,e,ident) is None:raise RuntimeError("Missing scheduled VAL")
  return rec
 seed_rng(PLAN["seed"],PLAN["cpu_threads"])
 generator=torch.Generator().manual_seed(PLAN["seed"])
 last=d/"last.pt"
 previous=torch.load(last,map_location="cpu",weights_only=True) if last.exists() else None
 model=km.build_classifier(arm,seed=PLAN["seed"],
  backbone_path=ROOT/"pretrained/resnet18-f37072fd.pth" if previous is None else None,
  initialize=previous is None).to(DEVICE)
 optimizer=torch.optim.AdamW(model.parameters(),lr=PLAN["lr"],weight_decay=PLAN["weight_decay"],foreach=False,fused=False)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12,eta_min=0)
 ds=km.CropDataset(IMAGES,frames["train"],training=True,seed=PLAN["seed"],size=224,margin=.05,image_hashes=hashes)
 if len(ds)!=PLAN["train_crops"]:raise RuntimeError("TRAIN crop population differs")
 loader=DataLoader(ds,batch_size=256,shuffle=True,num_workers=8,pin_memory=True,
  prefetch_factor=2,persistent_workers=False,generator=generator,multiprocessing_context="spawn")
 states=torch.tensor(labels.states,dtype=torch.long,device=DEVICE)
 history=[];start=0
 if previous is not None:
  if previous["study_sha256"]!=ident or previous["arm"]!=arm or previous["seed"]!=PLAN["seed"]:
   raise RuntimeError("Checkpoint identity mismatch")
  start=previous["epoch"];history=previous["history"]
  if not 0<=start<=12 or len(history)!=start or [x["epoch"] for x in history]!=list(range(1,start+1)):
   raise RuntimeError("Checkpoint history mismatch")
  if previous["scheduler"]["T_max"]!=12 or previous["scheduler"]["last_epoch"]!=start:
   raise RuntimeError("Checkpoint cosine schedule mismatch")
  if any(float(s["step"])!=start*len(loader) for s in previous["optimizer"]["state"].values()):
   raise RuntimeError("Optimizer step exposure mismatch")
  model.load_state_dict(previous["state_dict"],strict=True)
  optimizer.load_state_dict(previous["optimizer"]);move_optimizer(optimizer,DEVICE)
  scheduler.load_state_dict(previous["scheduler"]);restore_rng(previous["rng"],generator)
  del previous
 else:
  init={"study_sha256":ident,"arm":arm,"seed":PLAN["seed"],"initial_backbone_sha256":tensor_digest(model.backbone.state_dict()),
        "initial_model_sha256":tensor_digest(model.state_dict()),"pretraining_sha256":binding["pretraining_sha256"]}
  initpath=d/"initialization.json"
  if initpath.exists() and json.loads(initpath.read_text())!=init:raise RuntimeError("Initialization drift")
  atomic_json(initpath,init)
  if arm=="Q5":
   base=json.loads((OUT/"G/initialization.json").read_text())
   if base["initial_backbone_sha256"]!=init["initial_backbone_sha256"]:
    raise RuntimeError("G/Q5 pretrained backbone mismatch")
 set_status("running",method=arm,completed_epochs=start,phase="train")
 emit("arm_started",method=arm,completed_epochs=start,planned_epochs=12)
 # If interrupted after training commit or during VAL, finish that VAL before new training.
 if start in PLAN["val_epochs"]:snapshot_epoch(d,start)
 evaluations=[]
 for e in pending_val_epochs(start):
  saved_rng=rng_state(generator)
  try:evaluations.append(evaluate_milestone(arm,e,d,binding,frames,props,hashes,labels))
  finally:restore_rng(saved_rng,generator)
 for epoch in range(start,12):
  guard();verify_sources(binding)
  ds.set_epoch(epoch);model.train()
  set_status("running",method=arm,epoch=epoch+1,phase="train")
  tick=time.perf_counter();wait_sec=compute_sec=0.;seen=0;loss_sum=0.
  torch.cuda.reset_peak_memory_stats()
  iterator=iter(loader);order_hash=hashlib.sha256()
  for idx in range(len(loader)):
   guard();t=time.perf_counter();x,y=next(iterator);wait_sec+=time.perf_counter()-t
   order_hash.update(y.numpy().tobytes());t=time.perf_counter()
   x=x.to(DEVICE,non_blocking=True);y=y.to(DEVICE,non_blocking=True)
   optimizer.zero_grad(set_to_none=True)
   loss=km.class_loss(model(x),y,states,arm)
   if not torch.isfinite(loss).item():raise RuntimeError("Nonfinite loss")
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"),error_if_nonfinite=True)
   optimizer.step();n=len(x);seen+=n;loss_sum+=loss.item()*n
   torch.cuda.synchronize();compute_sec+=time.perf_counter()-t
   if idx==0 or (idx+1)%50==0 or idx+1==len(loader):
    emit("train_progress",method=arm,seed=PLAN["seed"],epoch=epoch+1,batch=idx+1,total_batches=len(loader),
         loss=loss.item(),lr=optimizer.param_groups[0]["lr"],samples=seen,
         elapsed_seconds=time.perf_counter()-tick,data_wait_seconds=wait_sec,compute_seconds=compute_sec,phase="train")
  del iterator,x,y,loss
  if seen!=len(ds):raise RuntimeError("Incomplete training epoch")
  scheduler.step()
  row={"epoch":epoch+1,"samples":seen,"train_loss":loss_sum/seen,"train_seconds":time.perf_counter()-tick,
       "data_wait_seconds":wait_sec,"compute_seconds":compute_sec,"label_order_sha256":order_hash.hexdigest(),
       "next_lr":optimizer.param_groups[0]["lr"],"peak_cuda_bytes":torch.cuda.max_memory_allocated()}
  history.append(row)
  state={"schema":"helmet2020.gq5_long12.checkpoint.v1","study_sha256":ident,"arm":arm,"seed":PLAN["seed"],
         "epoch":epoch+1,"state_dict":cpu_tree(model.state_dict()),"optimizer":cpu_tree(optimizer.state_dict()),
         "scheduler":scheduler.state_dict(),"rng":rng_state(generator),"history":history}
  atomic_save(state,last);del state
  atomic_json(d/"history.json",history)
  emit("checkpoint_committed",method=arm,epoch=epoch+1,train_seconds=row["train_seconds"])
  if epoch+1 in PLAN["val_epochs"]:
   snapshot_epoch(d,epoch+1)
   saved_rng=rng_state(generator)
   try:evaluations.append(evaluate_milestone(arm,epoch+1,d,binding,frames,props,hashes,labels))
   finally:restore_rng(saved_rng,generator)
   atomic_json(d/"val_history.json",evaluations)
 verify_sources(binding)
 if [x["epoch"] for x in evaluations]!=PLAN["val_epochs"]:raise RuntimeError("VAL schedule incomplete")
 selected=choose_best(evaluations)
 rec={"status":"completed","arm":arm,"seed":PLAN["seed"],"epochs":12,"study_sha256":ident,
      "selection_rule":PLAN["selection"],"selected_epoch":selected["epoch"],
      "macro_AP50":selected["macro_AP50"],"weighted_AP50":selected["weighted_AP50"],
      "selected_checkpoint":f"{arm}/epoch{selected['epoch']:02d}.pt",
      "selected_checkpoint_sha256":selected["checkpoint_sha256"],
      "last_checkpoint_sha256":digest(last),
      "train_seconds":sum(x["train_seconds"] for x in history),
      "eval_seconds":sum(x["total_eval_seconds"] for x in evaluations),
      "evaluations":evaluations,"test_accessed":False}
 atomic_json(d/"val_history.json",evaluations);atomic_json(d/"complete.json",rec)
 emit("arm_completed",**rec)
 del model,optimizer,scheduler,loader,ds,states
 gc.collect();torch.cuda.empty_cache()
 return rec

def finalize(binding,summary):
 a,b=summary;ident=canonical(binding)
 pairs=[]
 for epoch in PLAN["val_epochs"]:
  ga=next(r for r in a["evaluations"] if r["epoch"]==epoch)
  qb=next(r for r in b["evaluations"] if r["epoch"]==epoch)
  pairs.append({"epoch":epoch,"G_macro_AP50":ga["macro_AP50"],"Q5_macro_AP50":qb["macro_AP50"],
      "G_weighted_AP50":ga["weighted_AP50"],"Q5_weighted_AP50":qb["weighted_AP50"],
      "macro_delta_pp":100*(qb["macro_AP50"]-ga["macro_AP50"]),
      "weighted_delta_pp":100*(qb["weighted_AP50"]-ga["weighted_AP50"])})
 selection={"schema":"helmet2020.gq5_long12.selection.v1","study_sha256":ident,"seed":PLAN["seed"],
            "candidates_epochs":PLAN["val_epochs"],"rule":PLAN["selection"],
            "entries":[{"method":r["arm"],"epoch":r["selected_epoch"],"checkpoint":r["selected_checkpoint"],
                        "checkpoint_sha256":r["selected_checkpoint_sha256"]} for r in summary],
            "TEST_accessed":False,"legacy_kit_test_release":False}
 atomic_json(OUT/"selection.json",selection)
 result={"status":"completed","study_sha256":ident,"arms":summary,"paired_epoch_comparisons":pairs,
         "selected_epoch_macro_delta_pp":100*(b["macro_AP50"]-a["macro_AP50"]),
         "selected_epoch_weighted_delta_pp":100*(b["weighted_AP50"]-a["weighted_AP50"]),
         "test_accessed":False,"other_seeds_started":False,
         "limitation":"Development VAL, one seed. Checkpoints selected among epochs4/8/12; no final TEST claim."}
 atomic_json(OUT/"comparison_result.json",result)
 import csv
 with (OUT/"comparison_val.csv").open("w",newline="") as f:
  fields=["method","selected_epoch","macro_AP50_percent","weighted_AP50_percent","seed"]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in summary:w.writerow({"method":r["arm"],"selected_epoch":r["selected_epoch"],
       "macro_AP50_percent":100*r["macro_AP50"],"weighted_AP50_percent":100*r["weighted_AP50"],"seed":PLAN["seed"]})
 set_status("completed",finished_utc=now(),result="comparison_result.json")
 emit("study_completed",macro_delta_pp=result["selected_epoch_macro_delta_pp"],
      weighted_delta_pp=result["selected_epoch_weighted_delta_pp"],test_accessed=False)

def main():
 OUT.mkdir(exist_ok=True,parents=True)
 with execution_lock(DEVICE):
  try:
   set_status("initializing",started_utc=now())
   guard();binding,frames,props,hashes,labels=setup()
   emit("study_verified",plan=PLAN,study_sha256=canonical(binding))
   summary=[train_arm(a,binding,frames,props,hashes,labels) for a in PLAN["arms"]]
   verify_sources(binding);finalize(binding,summary)
  except BaseException as exc:
   set_status("paused" if isinstance(exc,KeyboardInterrupt) else "failed",
              error=str(exc),exception=type(exc).__name__)
   traceback.print_exc()
   raise

if __name__=="__main__":raise SystemExit("Use run_confirmation.py; legacy study paths are read-only references")
