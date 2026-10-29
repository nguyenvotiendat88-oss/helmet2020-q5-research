"""One bounded residual-architecture pilot, paired with a capacity control."""
import copy,gc,json,sys,traceback
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import residual_engine as e
import residual_candidate as candidate
import torch
from helmetlab import kit_models as native
from helmetlab.kit_runner import atomic_json,canonical,execution_lock
from helmetlab.common import digest

OUT=Path("/workspace/runs/gq5_residual_pilot_20261029")
CONTROL=Path("/workspace/runs/gq5_long12_20261027")
CONTROL_SHA="9691a80f17a000127ffac5244b704eb17260ea4826b7020e5344dc304bc9cb7f"
ARMS=("GQ5_residual","G_capacity392")
BASE_PLAN=copy.deepcopy(e.PLAN)

def build_classifier(arm,seed=0,backbone_path=None,smoke=False,*,initialize=True,num_classes=36):
    if arm not in ARMS or smoke or num_classes!=36:raise ValueError("Unexpected model request")
    return candidate.build(arm,seed=seed,backbone_path=backbone_path,initialize=initialize)

def class_loss(logits,ids,states,arm,weighting=None):
    if arm not in ARMS or weighting is not None:raise ValueError("Use original matched BCE")
    return native.class_loss(logits,ids,states,"G")

def configure():
    e.PLAN=copy.deepcopy(BASE_PLAN)
    e.PLAN.update(schema="helmet2020.residual_pilot.v1",arms=list(ARMS),seed=0,
        epochs_per_arm=4,val_epochs=[4],selection="fixed_epoch4",
        primary_comparison="residual_vs_original_G_and_capacity_control_at_epoch4",
        initialization="fresh_ImageNet_seed0; residual_global_path_exact_G; zero_role_gates",
        objective="unchanged_native10_presence_BCE_plus_masked_helmet_BCE_div5",
        scheduler_T_max=12,early_stopping=False,auto_expand=False,
        historical_baselines={"G":"original_seed0_epoch4","Q5":"original_seed0_epoch4"},
        residual_head_parameters=205209,capacity_control_head_parameters=205026,
        residual_gates="5 signed bounded tanh coefficients; initialize0; shared across each role pair",
        warmstart_from_trained_G=False,one_backbone_per_inference=True,calibration=False,
        primary_gate="macro_gain>=1pp_and_weighted_delta>=-0.5pp_vs_both_G_and_capacity_control",
        support_sensitivity="report every class support and contribution; no class exclusion for headline",
        measured_expansion=False,maximum_new_fits=2,
        interpretation="posthoc_development_pilot; adapted_global_plus_spatial_residual; no_novelty_or_TEST_claim")
    e.OUT=OUT
    e.km=SimpleNamespace(**{n:getattr(native,n) for n in dir(native) if not n.startswith("__")})
    e.km.build_classifier=build_classifier
    e.km.class_loss=class_loss

def verify_baseline(binding):
    original=json.loads((CONTROL/"study.json").read_text())
    if canonical(original)!=CONTROL_SHA:raise RuntimeError("Baseline identity changed")
    for key in ("kit_sources","manifest_sha256","pixel_hash_map_sha256",
                "pretraining_sha256","detector_best_sha256","detector_proposals_sha256","runtime"):
        if original[key]!=binding[key]:raise RuntimeError("Paired setup mismatch: "+key)
    receipts={}
    for arm in ("G","Q5"):
        d=CONTROL/arm
        record=json.loads((d/"complete.json").read_text())
        rec=next(v for v in record["evaluations"] if v["epoch"]==4)
        for rel,key in (("epoch04.pt","checkpoint_sha256"),("val_epoch04/metrics.json","metrics_sha256"),
                        ("val_epoch04/predictions.jsonl","predictions_sha256")):
            if digest(d/rel)!=rec[key]:raise RuntimeError("Baseline artifact changed")
        receipts[arm]=rec
    return receipts

def sensitivity(rec,control):
    cm=json.loads((CONTROL/control/"val_epoch04/metrics.json").read_text())["classes"]
    mm=json.loads((OUT/rec["arm"]/"val_epoch04/metrics.json").read_text())["classes"]
    supported=sum(c["support"]>0 for c in cm)
    rows=[]
    for a,b in zip(cm,mm):
        if a["class_id"]!=b["class_id"] or a["support"]!=b["support"]:raise RuntimeError("Population mismatch")
        if a["support"]:
            delta=100*(b["ap"]-a["ap"])
            rows.append({"class_id":a["class_id"],"support":a["support"],
                         "baseline_AP50":a["ap"],"candidate_AP50":b["ap"],
                         "delta_pp":delta,"contribution_to_macro_delta_pp":delta/supported})
    return {"control":control,"classes":rows,
            "support_le5_macro_delta_contribution_pp":sum(x["contribution_to_macro_delta_pp"] for x in rows if x["support"]<=5),
            "macro_metric_unchanged":True}

def compare(a,b):
    return {"macro_delta_pp":100*(a["macro_AP50"]-b["macro_AP50"]),
            "weighted_delta_pp":100*(a["weighted_AP50"]-b["weighted_AP50"])}

def main():
    configure()
    OUT.mkdir(parents=True,exist_ok=True)
    with execution_lock("cuda:0"):
        try:
            e.set_status("initializing",started_utc=e.now())
            e.guard();binding,frames,props,hashes,labels=e.setup()
            baselines=verify_baseline(binding)
            summaries=[]
            expected_backbone=json.loads((CONTROL/"G/initialization.json").read_text())["initial_backbone_sha256"]
            expected_history=json.loads((CONTROL/"G/history.json").read_text())[:4]
            for arm in ARMS:
                # Establish head/backbone initialization evidence before starting this arm.
                check=build_classifier(arm,seed=0,backbone_path=e.ROOT/"pretrained/resnet18-f37072fd.pth")
                if e.tensor_digest(check.backbone.state_dict())!=expected_backbone:
                    raise RuntimeError("Backbone not paired with G")
                spec={"architecture":check.architecture_metadata(),
                      "initial_model_sha256":e.tensor_digest(check.state_dict()),
                      "initial_backbone_sha256":e.tensor_digest(check.backbone.state_dict())}
                spec_path=OUT/f"{arm}_spec.json"
                if spec_path.exists() and json.loads(spec_path.read_text())!=spec:raise RuntimeError("Spec drift")
                atomic_json(spec_path,spec)
                del check
                rec=e.train_arm(arm,binding,frames,props,hashes,labels)
                init=json.loads((OUT/arm/"initialization.json").read_text())
                if init["initial_model_sha256"]!=spec["initial_model_sha256"]:raise RuntimeError("Initial model changed")
                hist=json.loads((OUT/arm/"history.json").read_text())
                if [x["label_order_sha256"] for x in hist]!=[x["label_order_sha256"] for x in expected_history]:
                    raise RuntimeError("Training order mismatch")
                summaries.append(rec)
                gc.collect();torch.cuda.empty_cache()
            e.verify_sources(binding)
            endpoints={r["arm"]:r["evaluations"][0] for r in summaries}
            residual=endpoints["GQ5_residual"]
            comparisons={name:compare(residual,rec) for name,rec in
                         {**baselines,"G_capacity392":endpoints["G_capacity392"]}.items()}
            passed=all(comparisons[a]["macro_delta_pp"]>=1. and comparisons[a]["weighted_delta_pp"]>=-.5
                       for a in ("G","G_capacity392"))
            result={"status":"completed","study_sha256":canonical(binding),"seed":0,"epoch":4,
                    "arms":summaries,"historical_baselines":baselines,"residual_comparisons":comparisons,
                    "screen_gate_passed":passed,
                    "class_sensitivity_vs_G":sensitivity(summaries[0],"G"),
                    "test_accessed":False,"auto_expand":False,
                    "limitation":"Single-seed four-epoch development; pass is not stable superiority or novelty"}
            atomic_json(OUT/"comparison_result.json",result)
            atomic_json(OUT/"selection.json",{"rule":"fixed_epoch4_no_alternative_checkpoint_selection",
                         "entries":[{"arm":a,"checkpoint":f"{a}/epoch04.pt","sha256":r["checkpoint_sha256"]} for a,r in endpoints.items()],
                         "test_accessed":False})
            e.set_status("completed",finished_utc=e.now(),screen_gate_passed=passed,result="comparison_result.json")
            e.emit("study_completed",comparisons=comparisons,screen_gate_passed=passed,test_accessed=False)
        except BaseException as exc:
            e.set_status("failed",error=str(exc),exception=type(exc).__name__)
            traceback.print_exc()
            raise

if __name__=="__main__":main()
