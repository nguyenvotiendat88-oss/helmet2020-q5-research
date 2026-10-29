"""Bounded matched G/Q5 x seeds0/1/2, fixed12 epochs, one long-tail loss."""
import copy,gc,json,statistics,sys,traceback
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import engine as e
import presence_asl
import torch
from helmetlab import kit_models as native
from helmetlab.kit_runner import atomic_json,canonical,execution_lock
from helmetlab.common import digest

OUT=Path("/workspace/runs/longtail3_20261029")
ARMS=("G","Q5")
SEEDS=(0,1,2)
BASE_PLAN=copy.deepcopy(e.PLAN)
REFERENCE_SEED0_SHA="9691a80f17a000127ffac5244b704eb17260ea4826b7020e5344dc304bc9cb7f"
PREDECLARED={
 "schema":"helmet2020.longtail3.v1","arms":list(ARMS),"seeds":list(SEEDS),
 "epochs_per_arm":12,"val_epochs":[12],"selection":"fixed_epoch12",
 "objective":"presence simplified asymmetric focusing gamma+=0 gamma-=1; conditional helmet BCE unchanged; average batch*5",
 "train_from":"ImageNet fresh, each paired seed","sampling":"original shuffle all181176 crops; no resampling",
 "optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"scheduler":"cosine12,no_restart",
 "batch":256,"workers":8,"FP32":True,"AMP":False,"TF32":False,
 "held_constant":"ResNet18, crop224,margin0.05,ColorJitter,detector_seed0,proposals,decoder,threshold0.001,NMS0.5,cap300,scorer",
 "historical_comparison":"BCE baseline same model and seed at epoch12, never epoch24",
 "prior_development_VAL_seen":True,"test_accessed":False,"auto_expand":False,
 "train_epochs_total":72,"new_full_val_evaluations_total":6,
 "evaluation_schedule_change":"Only VAL12; previous VAL4/8 passes did not update training parameters and preserved RNG; no early selection",
 "limitation":"Fixed development validation across3classifier seeds; detector fixed; no independent confirmation",
 "prior_art":"Query2Label section3.2 simplified asymmetric loss; adaptation,not newly invented objective"}

def baseline_root(seed):
    if seed not in SEEDS:raise ValueError("Unsupported seed")
    if seed==0:return Path("/workspace/runs/gq5_long12_20261027")
    return Path("/workspace/runs/gq5_confirmation_20261028")/f"seed{seed}"/"phase12"

def configure(seed):
    if seed not in SEEDS:raise ValueError("Unsupported seed")
    e.PLAN=copy.deepcopy(BASE_PLAN)
    e.PLAN.update(schema="helmet2020.longtail3.seed.v1",seed=seed,arms=list(ARMS),
        epochs_per_arm=12,val_epochs=[12],selection="fixed_epoch12",
        primary_comparison="paired_G_vs_Q5_and_each_vs_original_BCE_at_epoch12",
        objective=PREDECLARED["objective"],gamma_positive=0.,gamma_negative=1.,
        probability_shift=0.,focus_detached=False,conditional_helmet_term_changed=False,
        balanced_sampling=False,scheduler_T_max=12,auto_expand=False,early_stopping=False,
        initialization="fresh_ImageNet_same_paired_seed_as_BCE_control",
        interpretation="matched_fixed_longtail_recipe_chosen_after_previous_development;3seeds;no_TEST")
    e.OUT=OUT/f"seed{seed}"
    e.km=SimpleNamespace(**{n:getattr(native,n) for n in dir(native) if not n.startswith("__")})
    e.km.class_loss=presence_asl.loss

def prior_baseline(seed,binding):
    root=baseline_root(seed)
    original=json.loads((root/"study.json").read_text())
    if original["plan"]["seed"]!=seed or original["plan"]["epochs_per_arm"]!=12:
        raise RuntimeError("Baseline seed or budget differs")
    if seed==0 and canonical(original)!=REFERENCE_SEED0_SHA:
        raise RuntimeError("Seed0 baseline identity mismatch")
    for key in ("kit_sources","manifest_sha256","pixel_hash_map_sha256","pretraining_sha256",
                "detector_best_sha256","detector_proposals_sha256","runtime"):
        if binding[key]!=original[key]:raise RuntimeError("Baseline environment/data differ: "+key)
    match=("train_batch","loader_workers","prefetch_factor","persistent_workers","worker_context",
           "cpu_threads","fp32","amp","tf32","optimizer","lr","weight_decay",
           "scheduler","scheduler_T_max","eta_min","crop_size","margin","train_frames","train_crops",
           "val_frames","val_batch","val_prefetch_workers","val_prefetch_frames",
           "score_thresh","nms_thresh","max_detections")
    for key in match:
        if e.PLAN[key]!=original["plan"][key]:raise RuntimeError("Baseline recipe differs: "+key)
    refs={}
    for arm in ARMS:
        d=root/arm;done=json.loads((d/"complete.json").read_text())
        rec=next(x for x in done["evaluations"] if x["epoch"]==12)
        if rec["seed"]!=seed or rec["arm"]!=arm or rec["study_sha256"]!=canonical(original):
            raise RuntimeError("Baseline completion identity differs")
        for relative,key in (("epoch12.pt","checkpoint_sha256"),("val_epoch12/metrics.json","metrics_sha256"),
                             ("val_epoch12/predictions.jsonl","predictions_sha256")):
            if digest(d/relative)!=rec[key]:raise RuntimeError("Baseline artifact drift")
        refs[arm]=rec
    return {"root":str(root),"study_sha256":canonical(original),"endpoints":refs}

def paired_init(seed,arm,reference):
    root=Path(reference["root"])/arm
    expected=json.loads((root/"initialization.json").read_text())
    model=native.build_classifier(arm,seed=seed,backbone_path=e.ROOT/"pretrained/resnet18-f37072fd.pth")
    actual=e.tensor_digest(model.state_dict())
    if actual!=expected["initial_model_sha256"]:raise RuntimeError("Paired initialization mismatch")
    del model
    return expected

def delta(a,b):
    return {key+"_delta_pp":100*(a[key]-b[key]) for key in ("macro_AP50","weighted_AP50")}

def run_seed(seed):
    configure(seed);e.OUT.mkdir(parents=True,exist_ok=True)
    e.set_status("initializing",seed=seed)
    try:
        e.guard();binding,frames,proposals,hashes,labels=e.setup()
        reference=prior_baseline(seed,binding)
        atomic_json(e.OUT/"baseline_binding.json",reference)
        results={}
        for arm in ARMS:
            atomic_json(OUT/"status.json",{"state":"running","pid":__import__("os").getpid(),
                        "seed":seed,"arm":arm,"seed_progress_path":str(e.OUT/"progress.json"),"time_utc":e.now()})
            expected=paired_init(seed,arm,reference)
            rec=e.train_arm(arm,binding,frames,proposals,hashes,labels)
            initialized=json.loads((e.OUT/arm/"initialization.json").read_text())
            if initialized["initial_model_sha256"]!=expected["initial_model_sha256"]:
                raise RuntimeError("Saved initialization mismatch")
            history=json.loads((e.OUT/arm/"history.json").read_text())
            baseline=json.loads((baseline_root(seed)/arm/"history.json").read_text())
            if len(history)!=12 or [x["label_order_sha256"] for x in history]!=[x["label_order_sha256"] for x in baseline]:
                raise RuntimeError("Training sample order differs")
            measured=rec["evaluations"][0]
            results[arm]={"completion":rec,"baseline_epoch12":reference["endpoints"][arm],
                         "vs_same_arm_BCE":delta(measured,reference["endpoints"][arm]),
                         "paired_init_and_sample_order":True}
            atomic_json(e.OUT/f"{arm}_comparison.json",results[arm])
        e.verify_sources(binding)
        g=results["G"]["completion"]["evaluations"][0]
        q=results["Q5"]["completion"]["evaluations"][0]
        result={"status":"completed","seed":seed,"epoch":12,"study_sha256":canonical(binding),
                "arms":results,"Q5_vs_G_same_loss":delta(q,g),
                "Q5_new_vs_G_BCE":delta(q,reference["endpoints"]["G"]),
                "test_accessed":False,"not_an_ensemble":True}
        atomic_json(e.OUT/"comparison_result.json",result)
        e.set_status("completed",seed=seed,finished_utc=e.now())
        return result
    except BaseException as exc:
        e.set_status("failed",seed=seed,error=str(exc),exception=type(exc).__name__)
        raise

def aggregate():
    rows=[]
    for seed in SEEDS:
        p=OUT/f"seed{seed}/comparison_result.json"
        if p.exists():rows.append(json.loads(p.read_text()))
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for kind in ("new_loss","baseline_BCE"):
            stats[arm][kind]={}
            for key in ("macro_AP50","weighted_AP50"):
                vals=[100*(r["arms"][arm]["completion"]["evaluations"][0] if kind=="new_loss"
                            else r["arms"][arm]["baseline_epoch12"])[key] for r in rows]
                stats[arm][kind][key]={"mean_percent":statistics.mean(vals),
                    "sample_sd_pp":statistics.stdev(vals) if len(vals)>1 else None,"values_percent":vals}
    completed=len(rows)==3
    gate=None
    if completed:
        deltas=[r["Q5_vs_G_same_loss"] for r in rows]
        gate=all(r["macro_AP50_delta_pp"]>0 and r["weighted_AP50_delta_pp"]>=-.5 for r in deltas) and statistics.mean(r["macro_AP50_delta_pp"] for r in deltas)>=1.
    result={"status":"completed" if completed else "partial","seeds_available":[r["seed"] for r in rows],
            "epoch":12,"rows":rows,"statistics":stats,"descriptive_seed_gate_passed":gate,
            "gate_is_not_statistical_proof":True,"detector_fixed_seed":0,
            "test_accessed":False,"auto_expand":False,
            "limitation":"Development VAL has been repeatedly inspected; need clip uncertainty and locked TEST before final claims"}
    atomic_json(OUT/"summary.json",result)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with execution_lock("cuda:0"):
        plan={**PREDECLARED,"implementation_sha256":{n:digest(HERE/n) for n in ("engine.py","run_longtail.py","presence_asl.py","fast_eval.py")}}
        if (OUT/"plan.json").exists() and json.loads((OUT/"plan.json").read_text())!=plan:
            raise RuntimeError("Global plan drift")
        atomic_json(OUT/"plan.json",plan)
        for seed in SEEDS:
            run_seed(seed)
            aggregate()
            gc.collect();torch.cuda.empty_cache()
        atomic_json(OUT/"status.json",{"state":"completed","pid":__import__("os").getpid(),
                    "time_utc":e.now(),"summary":"summary.json","test_accessed":False,"instance_auto_stop":False})
        print(json.dumps({"state":"completed","seeds":list(SEEDS),"arms":list(ARMS),"epoch":12,"test_accessed":False}),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if OUT.exists():atomic_json(OUT/"status.json",{"state":"failed","time_utc":e.now(),"error":str(exc)})
        traceback.print_exc()
        raise
