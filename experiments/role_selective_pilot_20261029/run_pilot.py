"""Run one frozen CPU development pilot on cached logits; no model update."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
from selective_core import METHODS,probabilities,policies,matched_mask,fit_threshold,summarize,cluster_contributions,paired_cluster_interval
from test_selective import run as test

HERE=Path(__file__).resolve().parent
ROOT=Path("/local/helmet-research")
INPUTS=HERE/"inputs"
OUT=ROOT/"output/role_selective_pilot_20261029"
TARGETS=(.5,.7,.9)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atom(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)

def calibration_clip(clip):
    return int(hashlib.sha256(("calibration_v1:"+str(clip)).encode()).hexdigest(),16)%3==0

def main():
    test()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"result.json").exists():raise RuntimeError("Preserve completed results")
    meta=json.loads((INPUTS/"meta.json").read_text())
    for name,info in meta["files"].items():
        if sha(INPUTS/name)!=info["sha256"]:raise RuntimeError("Input changed: "+name)
    manifest=ROOT/"HELMET2020_training_kit_v1/manifests/val.jsonl"
    if sha(manifest)!=meta["val_manifest_sha256"]:raise RuntimeError("VAL identity changed")
    labels=json.loads((INPUTS/"classes.json").read_text())
    states=np.array([r["states"] for r in labels["classes"]])
    roles=labels["positions"]
    records=[]
    with manifest.open() as f:
        for line in f:
            frame=json.loads(line)
            for obj_index,obj in enumerate(frame["objects"]):
                records.append({"crop_index":len(records),"image_id":frame["image_id"],
                    "clip_id":frame["clip_id"],"site":frame.get("site_id",frame.get("site","unknown")),
                    "object_index":obj_index,"class_id":obj["class_id"]})
    assert len(records)==47672
    selected=json.loads((INPUTS/"samples.json").read_text())["val"]
    uniform=set(np.random.default_rng(20261028).choice(len(records),4096,replace=False).tolist())
    chosen=[records[i] for i in selected]
    cal_mask=np.array([i in uniform and calibration_clip(records[i]["clip_id"]) for i in selected])
    eval_mask=np.array([i in uniform and not calibration_clip(records[i]["clip_id"]) for i in selected])
    assert int(cal_mask.sum())==1398 and int(eval_mask.sum())==2698 and not (cal_mask&eval_mask).any()
    calibration_records=[x for x,m in zip(chosen,cal_mask) if m]
    evaluation_records=[x for x,m in zip(chosen,eval_mask) if m]
    cal_clips=sorted({x["clip_id"] for x in calibration_records})
    eval_clips=sorted({x["clip_id"] for x in evaluation_records})
    assert set(cal_clips).isdisjoint(eval_clips)
    tiebreak=np.array([int(hashlib.sha256(f"selective_tie_v1:{x['crop_index']}".encode()).hexdigest()[:16],16)
                      for x in evaluation_records],dtype=np.uint64)
    clip_lookup={c:i for i,c in enumerate(eval_clips)}
    clip_idx=np.array([clip_lookup[x["clip_id"]] for x in evaluation_records])
    rng=np.random.default_rng(20261029)
    draws=rng.integers(0,len(eval_clips),size=(1000,len(eval_clips)))
    bootstrap=np.array([np.bincount(row,minlength=len(eval_clips)) for row in draws])
    calibration=json.loads((INPUTS/"calibration.json").read_text())
    prereg={"schema":"helmet2020.selective_role_pilot.v1","seed":0,"epoch":24,
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "methods":list(METHODS),"primary_coverage":.7,"secondary_coverages":[.5,.9],
        "candidate":"set_consensus","bootstrap_repetitions":1000,"bootstrap_seed":20261029,
        "temperature_by_model":{a:calibration[a]["temperature"] for a in ("G","Q5")},
        "sample_counts":{"calibration":len(calibration_records),"evaluation":len(evaluation_records)},
        "clip_counts":{"calibration":len(cal_clips),"evaluation":len(eval_clips)},
        "input_meta":meta,"code_hashes":{n:sha(HERE/n) for n in ("selective_core.py","run_pilot.py","test_selective.py")},
        "plan_sha256":sha(HERE/"PLAN_VI.md"),"test_accessed":False,"GPU_used":False,
        "training_updates":0,"dataset_scope":"GT_crop_development_subset",
        "prior_VAL_seen":True,"conformal_guarantee_claimed":False,
        "numeric_policy":"float64 analysis sigmoid, class distributions keep original1e-12 state-probability floor; same for all policies",
        "gate":{"risk_improvement_pp_vs_every_baseline":1.,"max_GT_present_risk_increase_pp":1.,
                "paired_clip_CI95_upper_below_zero":True},
        "followup":"report_all_results_no_auto_refinement"}
    atom(OUT/"plan.json",prereg)
    atom(OUT/"sample_manifest.json",{"calibration":calibration_records,"evaluation":evaluation_records,
                                   "calibration_clips":cal_clips,"evaluation_clips":eval_clips})
    y=np.array([x["class_id"] for x in chosen])
    truth=states[y[eval_mask]]
    report={"schema":prereg["schema"],"plan":prereg,"models":{},
            "evaluation_role_state_support":{roles[r]:np.bincount(truth[:,r],minlength=3).tolist() for r in range(len(roles))}}
    arrays={"truth":truth,"class_ids":y[eval_mask],"clip_index":clip_idx,"tiebreak":tiebreak,
            "crop_index":np.array([x["crop_index"] for x in evaluation_records])}
    for arm in ("G","Q5"):
        cache=np.load(INPUTS/f"{arm}_val_gt_logits.npz")
        assert np.array_equal(cache["ids"],y) and len(cache["ids"])==len(selected)
        native,q=probabilities(cache["logits"],states,calibration[arm]["temperature"])
        options=policies(native,q,states)
        entries={}
        contributions={}
        for name,(pred,confidence) in options.items():
            evaluated=pred[eval_mask];scores=confidence[eval_mask]
            arrays[f"{arm}_{name}_pred"]=evaluated
            arrays[f"{arm}_{name}_score"]=scores
            entry={"matched":{},"calibration_thresholds":{},"risk_coverage_curve":[]}
            for target in TARGETS:
                key=str(target)
                accepted=matched_mask(scores,target,tiebreak)
                assert np.all(accepted.sum(0)==int(np.floor(target*len(truth)+1e-9)))
                if name=="crop_confidence":assert np.all(accepted==accepted[:,[0]])
                measured=summarize(evaluated,accepted,truth,states,roles)
                if name in ("crop_confidence","topclass_role_marginal","set_consensus"):
                    assert measured["inconsistent_partial_crops"]==0
                entry["matched"][key]=measured
                thresholds=fit_threshold(confidence[cal_mask],target)
                actual=scores>=thresholds
                if name=="crop_confidence":assert np.all(actual==actual[:,[0]])
                entry["calibration_thresholds"][key]={
                    "thresholds":thresholds.tolist(),"target":target,
                    "calibration_coverage_per_role":(confidence[cal_mask]>=thresholds).mean(0).tolist(),
                    "evaluation":summarize(evaluated,actual,truth,states,roles)}
                if target==.7:
                    arrays[f"{arm}_{name}_primary_mask"]=accepted
                    contributions[name]=cluster_contributions(evaluated,accepted,truth,clip_idx,len(eval_clips))
            for target in np.linspace(.1,1.,10):
                keep=matched_mask(scores,float(target),tiebreak)
                e=summarize(evaluated,keep,truth,states,roles)
                entry["risk_coverage_curve"].append({"target":float(target),"coverage":e["coverage"],
                                                    "risk":e["risk"],"GT_present_risk":e["GT_present_risk"]})
            entry["mean_grid_risk_10_to100"]=float(np.mean([x["risk"] for x in entry["risk_coverage_curve"]]))
            entries[name]=entry
        baselines=[x for x in METHODS if x!="set_consensus"]
        comparison={name:paired_cluster_interval(contributions["set_consensus"],contributions[name],bootstrap)
                    for name in baselines}
        strongest=min(baselines,key=lambda name:entries[name]["matched"]["0.7"]["risk"])
        a=entries["set_consensus"]["matched"]["0.7"]
        b=entries[strongest]["matched"]["0.7"]
        candidate_gain_pp=100*(b["risk"]-a["risk"])
        present_delta_pp=100*(a["GT_present_risk"]-b["GT_present_risk"])
        passed=candidate_gain_pp>=1 and present_delta_pp<=1 and comparison[strongest]["percentile95_pp"][1]<0
        report["models"][arm]={"methods":entries,"primary_comparisons":comparison,
            "strongest_simple_baseline_on_primary_metric":strongest,
            "candidate_gain_vs_strongest_pp":candidate_gain_pp,"candidate_GT_present_risk_delta_pp":present_delta_pp,
            "candidate_gate_passed":passed}
        print(arm,"best_baseline",strongest,"candidate_risk_gain_pp",round(candidate_gain_pp,4),"gate",passed,flush=True)
    np.savez_compressed(OUT/"decisions.npz",**arrays)
    report.update(status="completed",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  decisions_sha256=sha(OUT/"decisions.npz"),
                  expand_candidate=all(r["candidate_gate_passed"] for r in report["models"].values()))
    atom(OUT/"result.json",report)
    atom(OUT/"verification.json",{"status":"passed","input_hashes":True,"same_GT_order":True,
        "clip_disjoint_fit_eval":True,"exact_per_role_matched_coverage":True,"unknown_separate_from_absent":True,
        "candidate_partial_predictions_extendable":True,"no_TEST":True,"no_training_updates":True,
        "result_sha256":sha(OUT/"result.json"),"decisions_sha256":sha(OUT/"decisions.npz")})
    print("Completed fixed selective pilot; expansion gate:",report["expand_candidate"])

if __name__=="__main__":main()
