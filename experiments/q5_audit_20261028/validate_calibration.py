"""Fixed-temperature full VAL follow-up; preserves original studies, never accesses TEST."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json, shutil, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import calibrate_pilot as p
import torch
from helmetlab.kit_runner import atomic_json, execution_lock, seed_rng, source_bindings
from helmetlab.common import digest, read_jsonl

OUT = Path("/workspace/runs/gq5_calibration_fullval_20261028")
PILOT = Path("/workspace/runs/gq5_calibration_pilot_20261028")

def status(**data):
    atomic_json(OUT/"status.json",{"pid":os.getpid(),"time_utc":p.datetime.now(p.timezone.utc).isoformat(),**data})
    print(json.dumps(data),flush=True)

def guard():
    if shutil.disk_usage(OUT).free < 8*1024**3:
        raise RuntimeError("Preserve 8GiB disk reserve")

def groups(frames):
    return {"all_val_development":frames,
            "val_disjoint_calibration_clips":[r for r in frames if not p.calibration_clip(r["clip_id"])]}

def measure(frames,predictions,labels):
    by_id={r["image_id"]:r for r in predictions}
    return {name:p.score_predictions(rows,[by_id[r["image_id"]] for r in rows],len(labels))
            for name,rows in groups(frames).items()}

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"comparison.json").exists():
        print("Already completed; preserve outputs",flush=True)
        return
    seed_rng(0,2);guard()
    binding=json.loads((p.BASE/"study.json").read_text())
    if source_bindings()!=binding["kit_sources"]:raise RuntimeError("Source drift")
    pilot=json.loads((PILOT/"result.json").read_text())
    if not pilot["promotion_gate_passed"]:raise RuntimeError("Pilot gate failed")
    calibration=json.loads((PILOT/"calibration.json").read_text())
    frames=read_jsonl(p.ROOT/"manifests/val.jsonl")
    labels=p.LabelMap(p.ROOT/"manifests/classes.json")
    if len(frames)!=14600:raise RuntimeError("Wrong VAL population")
    cal_clips=set(pilot["plan"]["calibration_clips"])
    disjoint={r["clip_id"] for r in groups(frames)["val_disjoint_calibration_clips"]}
    if cal_clips & disjoint:raise RuntimeError("Calibration clips leaked")
    plan={"schema":"helmet2020.temperature_fullval.v1","arms":["G","Q5"],"epoch":24,
          "temperatures":{arm:calibration[arm]["temperature"] for arm in ("G","Q5")},
          "calibration_sha256":digest(PILOT/"calibration.json"),"pilot_sha256":digest(PILOT/"result.json"),
          "manifest_sha256":digest(p.ROOT/"manifests/val.jsonl"),
          "script_sha256":digest(Path(__file__)),"pilot_script_sha256":digest(HERE/"calibrate_pilot.py"),
          "frozen_parent_study_sha256":"8a2516d80f8c4507224b0d785e80c3b7853f88b034cf9757346b721ee28a6915",
          "evaluation_groups":{n:len(r) for n,r in groups(frames).items()},
          "recipe":"fixed temperatures from prior 1398 calibration crops; no further tuning; same proposals/decoder/NMS/cap",
          "test_accessed":False,"training_updates":0,"auto_expand":False,
          "interpretation":"All-VAL includes calibration; disjoint subset avoids calibration clip reuse but is development data already seen in earlier VAL/pilot, not independent confirmation"}
    if (OUT/"plan.json").exists() and json.loads((OUT/"plan.json").read_text())!=plan:
        raise RuntimeError("Plan differs")
    atomic_json(OUT/"plan.json",plan)
    status(state="running",phase="preflight",groups=plan["evaluation_groups"])
    pixels=json.loads(Path("/workspace/runs/research_v1_fp32/trainval_pixels.json").read_text())
    hashes={k:v["sha256"] for k,v in pixels["files"].items()}
    props=read_jsonl(Path("/workspace/runs/research_v1_fp32/seed0/retinanet/val_predictions.jsonl"))
    receipt=json.loads((p.BASE/"comparison_result.json").read_text())
    results={}
    for arm in ("G","Q5"):
        guard()
        done=OUT/f"{arm}_result.json"
        if done.exists():
            cached=json.loads(done.read_text())
            if cached["temperature"]!=plan["temperatures"][arm] or digest(OUT/f"{arm}_predictions.jsonl")!=cached["predictions_sha256"]:
                raise RuntimeError("Cached result changed")
            results[arm]=cached
            continue
        ckfile=p.BASE/arm/"epoch24.pt"
        expected=next(x["last_checkpoint_sha256"] for x in receipt["arms"] if x["arm"]==arm)
        if digest(ckfile)!=expected:raise RuntimeError("Checkpoint differs")
        model=p.km.build_classifier(arm,initialize=False).cuda()
        ck=torch.load(ckfile,map_location="cpu",weights_only=True)
        model.load_state_dict(ck["state_dict"],strict=True);del ck
        net=p.Calibrated(model,plan["temperatures"][arm]);net.eval()
        tick=time.perf_counter()
        status(state="running",arm=arm,phase="inference",frames=0,total=len(frames))
        def progress(n,total):
            guard()
            status(state="running",arm=arm,phase="inference",frames=n,total=total,elapsed_seconds=time.perf_counter()-tick)
        predictions=p.classifier_predictions_prefetch(net,"/workspace/helmet_images",frames,props,labels,"cuda:0",
            batch=64,margin=.05,size=224,score_thresh=.001,nms_thresh=.5,max_detections=300,
            image_hashes=hashes,workers=4,prefetch_frames=8,kit_models_module=p.km,batched_cpu=True,
            progress_callback=progress,guard=guard)
        status(state="running",arm=arm,phase="scoring",frames=len(frames),total=len(frames))
        calibrated=measure(frames,predictions,labels)
        tmp=OUT/f"{arm}_predictions.jsonl.tmp"
        with tmp.open("w") as f:
            for row in predictions:f.write(json.dumps(row)+"\n")
            f.flush();os.fsync(f.fileno())
        tmp.replace(OUT/f"{arm}_predictions.jsonl")
        del predictions,net,model
        torch.cuda.empty_cache()
        old=read_jsonl(p.BASE/arm/"val_epoch24/predictions.jsonl")
        baseline=measure(frames,old,labels);del old
        record={"checkpoint_sha256":expected,"temperature":plan["temperatures"][arm],
                "baseline":baseline,"calibrated":calibrated,"seconds":time.perf_counter()-tick,
                "predictions_sha256":digest(OUT/f"{arm}_predictions.jsonl"),"test_accessed":False}
        atomic_json(done,record);results[arm]=record
    if source_bindings()!=binding["kit_sources"]:raise RuntimeError("Source changed during evaluation")
    if digest(PILOT/"calibration.json")!=plan["calibration_sha256"]:raise RuntimeError("Calibration changed")
    comparisons={}
    for group in plan["evaluation_groups"]:
        comparisons[group]={}
        for arm,r in results.items():
            b=r["baseline"][group];c=r["calibrated"][group]
            comparisons[group][arm]={"macro_before":b["macro_mAP"],"macro_after":c["macro_mAP"],
                "weighted_before":b["weighted_mAP"],"weighted_after":c["weighted_mAP"],
                "macro_delta_pp":100*(c["macro_mAP"]-b["macro_mAP"]),
                "weighted_delta_pp":100*(c["weighted_mAP"]-b["weighted_mAP"])}
        a=results["G"]["calibrated"][group];b=results["Q5"]["calibrated"][group]
        comparisons[group]["Q5_minus_G_after"]={"macro_delta_pp":100*(b["macro_mAP"]-a["macro_mAP"]),
                                                "weighted_delta_pp":100*(b["weighted_mAP"]-a["weighted_mAP"])}
    atomic_json(OUT/"comparison.json",{"status":"completed","plan":plan,"arms":results,"comparisons":comparisons})
    status(state="completed",comparison="comparison.json")

if __name__=="__main__":
    try:
        with execution_lock("cuda:0"):main()
    except Exception as exc:
        if OUT.exists():status(state="failed",error=str(exc))
        traceback.print_exc()
        raise
