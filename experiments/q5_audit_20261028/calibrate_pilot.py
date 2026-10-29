"""One prespecified temperature calibration candidate per arm, clip-disjoint VAL audit."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import hashlib, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path("/workspace/HELMET2020_training_kit_v1")
sys.path[:0] = [str(ROOT), "/workspace/gq5_extend24_20261028"]
import numpy as np
import torch
from helmetlab import kit_models as km
from helmetlab.common import read_jsonl, digest
from helmetlab.labels import LabelMap
from helmetlab.kit_runner import atomic_json, seed_rng, execution_lock, score_predictions, source_bindings
from fast_eval import classifier_predictions_prefetch

BASE = Path("/workspace/runs/gq5_extend24_20261028")
DIAG = Path("/workspace/runs/q5_audit_20261028")
OUT = Path("/workspace/runs/gq5_calibration_pilot_20261028")

def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()

def calibration_clip(clip):
    return int(hashed("calibration_v1:"+str(clip)),16) % 3 == 0

def status(**data):
    atomic_json(OUT/"status.json", {"pid":os.getpid(), "time_utc":datetime.now(timezone.utc).isoformat(), **data})
    print(json.dumps(data), flush=True)

def fit_temperature(z, ids, states):
    # Optimize on calibration data only. No audit metrics enter selection.
    z = torch.tensor(z, dtype=torch.float64)
    target = torch.tensor(states[ids], dtype=torch.int64)
    present = (target != 0).to(z.dtype); helmet = (target == 1).to(z.dtype)
    def objective(log_t):
        value = z / math.exp(log_t)
        a = torch.nn.functional.binary_cross_entropy_with_logits(value[:,:5], present, reduction="none")
        b = torch.nn.functional.binary_cross_entropy_with_logits(value[:,5:], helmet, reduction="none")
        return ((a+present*b).mean()).item()
    left, right = math.log(.5), math.log(10.)
    ratio = (math.sqrt(5.)-1.)/2.
    x = right-ratio*(right-left); y = left+ratio*(right-left)
    fx, fy = objective(x), objective(y)
    for _ in range(64):
        if fx < fy:
            right,y,fy = y,x,fx
            x = right-ratio*(right-left); fx = objective(x)
        else:
            left,x,fx = x,y,fy
            y = left+ratio*(right-left); fy = objective(y)
    candidates = [(0.,objective(0.)), (left,objective(left)), (right,objective(right))]
    optimum,value = min(candidates,key=lambda x:x[1])
    return {"temperature":math.exp(optimum),"calibration_native_NLL_before":objective(0.),
        "calibration_native_NLL_after":value,"calibration_crops":len(ids),"bounds":[.5,10.]}

class Calibrated(torch.nn.Module):
    def __init__(self, base, temperature):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0: raise ValueError("Positive finite T required")
        self.base, self.temperature, self.kit_arm = base, temperature, base.kit_arm
    def forward(self,x):
        return self.base(x)/self.temperature

def subset_records(path, frame_ids):
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["image_id"] in frame_ids: rows.append(row)
    return rows

def infer(model,frames,proposals,labels,hashes,progress=None):
    return classifier_predictions_prefetch(model,"/workspace/helmet_images",frames,proposals,labels,"cuda:0",
        batch=64,margin=.05,size=224,score_thresh=.001,nms_thresh=.5,max_detections=300,
        image_hashes=hashes,workers=4,prefetch_frames=8,kit_models_module=km,batched_cpu=True,
        progress_callback=progress)

def main():
    OUT.mkdir(exist_ok=True)
    seed_rng(0,2)
    study = json.loads((BASE/"study.json").read_text())
    if source_bindings()!=study["kit_sources"]: raise RuntimeError("Completed kit source changed")
    if (OUT/"result.json").exists(): raise RuntimeError("Pilot already completed; preserve result")
    labels = LabelMap(ROOT/"manifests/classes.json")
    val = read_jsonl(ROOT/"manifests/val.jsonl")
    ds = km.CropDataset("/workspace/helmet_images",val,training=False)
    diag_indices = json.loads((DIAG/"samples.json").read_text())["val"]
    uniform = set(np.random.default_rng(20261028).choice(len(ds),4096,replace=False).tolist())
    mask = np.array([i in uniform and calibration_clip(val[ds.items[i]["row_index"]]["clip_id"]) for i in diag_indices])
    if mask.sum()<200: raise RuntimeError("Insufficient calibration samples")
    candidates = [r for r in val if not calibration_clip(r["clip_id"])]
    chosen = {r["image_id"] for r in sorted(candidates,key=lambda r:hashed("audit1000:"+str(r["image_id"])))[:1000]}
    frames = [r for r in val if r["image_id"] in chosen]
    assert len(frames)==1000 and not any(calibration_clip(r["clip_id"]) for r in frames)
    cal_clips = {val[ds.items[i]["row_index"]]["clip_id"] for i,m in zip(diag_indices,mask) if m}
    assert cal_clips.isdisjoint({r["clip_id"] for r in frames})
    plan = {"schema":"helmet2020.temperature_pilot.v1","arms":["G","Q5"],"checkpoint_epoch":24,
        "fit":"one_scalar_temperature_minimize_native_masked_binary_NLL",
        "calibration_clip_rule":"sha256(calibration_v1:clip_id)mod3==0",
        "calibration_population":"uniform4096_VAL_crop_sample_intersect_calibration_clips",
        "audit":"first1000_by_hash_audit1000:image_id_from_disjoint_VAL_clips",
        "audit_frame_ids":[r["image_id"] for r in frames],"calibration_clips":sorted(cal_clips),
        "audit_clips":sorted({r["clip_id"] for r in frames}),"test_accessed":False,"train_updates":0,
        "scope":"exploratory_VAL_development; prior_VAL_metrics_seen; not_independent_confirmation",
        "promotion_gate":"Q5 macro_AP50 improvement >=0.5pp and weighted loss <=0.2pp on this pilot",
        "script_sha256":digest(Path(__file__)),"parent_study_sha256":"8a2516d80f8c4507224b0d785e80c3b7853f88b034cf9757346b721ee28a6915"}
    atomic_json(OUT/"plan.json",plan)
    fitted = {}
    for arm in plan["arms"]:
        z = np.load(DIAG/f"{arm}_val_gt_logits.npz")
        if len(z["ids"])!=len(diag_indices): raise RuntimeError("Logit sample alignment differs")
        expected = np.array([ds.items[i]["class_id"] for i in diag_indices])
        if not np.array_equal(expected,z["ids"]): raise RuntimeError("GT identity mismatch")
        fitted[arm] = fit_temperature(z["logits"][mask],z["ids"][mask],labels.states)
    atomic_json(OUT/"calibration.json",fitted)
    status(state="running",phase="calibrated",temperatures=fitted)
    pixels = json.loads(Path("/workspace/runs/research_v1_fp32/trainval_pixels.json").read_text())
    hashes = {k:v["sha256"] for k,v in pixels["files"].items()}
    props = subset_records(Path("/workspace/runs/research_v1_fp32/seed0/retinanet/val_predictions.jsonl"),chosen)
    results = {"plan":plan,"calibration":fitted,"arms":{}}
    receipt = json.loads((BASE/"comparison_result.json").read_text())
    for arm in plan["arms"]:
        original = subset_records(BASE/arm/"val_epoch24/predictions.jsonl",chosen)
        baseline = score_predictions(frames,original,len(labels))
        checkpoint = BASE/arm/"epoch24.pt"
        expected = next(x["last_checkpoint_sha256"] for x in receipt["arms"] if x["arm"]==arm)
        if digest(checkpoint)!=expected: raise RuntimeError("Checkpoint changed")
        model = km.build_classifier(arm,initialize=False).cuda()
        ck = torch.load(checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(ck["state_dict"],strict=True);del ck
        model.eval()
        # Exact prediction parity for T=1 on same first two frames.
        parity = infer(Calibrated(model,1.),frames[:2],props[:2],labels,hashes)
        if parity != original[:2]: raise RuntimeError("T=1 pipeline parity failed")
        calibrated = Calibrated(model,fitted[arm]["temperature"])
        tick = time.perf_counter()
        predictions = infer(calibrated,frames,props,labels,hashes,
            lambda n,total:status(state="running",arm=arm,phase="audit",frames=n,total=total))
        measured = score_predictions(frames,predictions,len(labels))
        with (OUT/f"{arm}_predictions.jsonl").open("w") as f:
            for row in predictions: f.write(json.dumps(row)+"\n")
        record = {"baseline":baseline,"calibrated":measured,"t1_exact_parity_passed":True,
            "checkpoint_sha256":expected,"predictions_sha256":digest(OUT/f"{arm}_predictions.jsonl"),
            "macro_delta_pp":100*(measured["macro_mAP"]-baseline["macro_mAP"]),
            "weighted_delta_pp":100*(measured["weighted_mAP"]-baseline["weighted_mAP"]),
            "seconds":time.perf_counter()-tick}
        results["arms"][arm]=record
        atomic_json(OUT/f"{arm}_result.json",record)
        del model,calibrated,predictions,original
        torch.cuda.empty_cache()
    q = results["arms"]["Q5"]
    results["promotion_gate_passed"] = q["macro_delta_pp"]>=.5 and q["weighted_delta_pp"]>=-.2
    results["status"]="completed"
    atomic_json(OUT/"result.json",results)
    status(state="completed",promotion_gate_passed=results["promotion_gate_passed"])

if __name__=="__main__":
    with execution_lock("cuda:0"):
        main()
