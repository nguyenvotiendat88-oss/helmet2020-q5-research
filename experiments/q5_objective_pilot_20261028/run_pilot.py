"""Finite mechanism pilot: G/Q5 x two objective variants, four epochs each."""
import copy,gc,json,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot_engine as e
import torch
import valid_state_objective as candidate
from helmetlab.kit_runner import atomic_json,canonical,execution_lock
from helmetlab.common import digest
from helmetlab import kit_models as native

ROOT_OUT=Path("/workspace/runs/q5_objective_pilot_20261028")
ORIGINAL=Path("/workspace/runs/gq5_long12_20261027")
ORIGINAL_SHA="9691a80f17a000127ffac5244b704eb17260ea4826b7020e5344dc304bc9cb7f"
VARIANTS=("omit_constant_presence","valid_state_conditional")
BASE_PLAN=copy.deepcopy(e.PLAN)

def objective(logits,ids,states,arm,weighting=None):
    if arm not in ("G","Q5") or weighting is not None:
        raise ValueError("Only matched unweighted G/Q5 pilot")
    mode=e.PLAN["objective"]
    if mode=="omit_constant_presence":
        return candidate.binary_loss_without_constant_presence(logits,ids,states)
    if mode=="valid_state_conditional":
        return candidate.valid_state_nll(logits,ids,states)
    raise ValueError("Unknown objective")

def configure(mode):
    if mode not in VARIANTS:raise ValueError("Unregistered variant")
    e.PLAN=copy.deepcopy(BASE_PLAN)
    e.PLAN.update(schema="helmet2020.objective_pilot.v1",arms=["G","Q5"],seed=0,
        epochs_per_arm=4,val_epochs=[4],objective=mode,scheduler_T_max=12,
        selection="fixed_epoch4",primary_comparison="same_architecture_vs_frozen_native_BCE_epoch4",
        interpretation="posthoc_development_pilot; conditional_likelihood_is_known; no_novelty_or_TEST_claim",
        early_stopping=False,auto_expand=False,
        historical_controls="original_G_Q5_epoch4; same_initialization_data_order_cosine12_prefix",
        decoder="unchanged_legacy_floor1e-12_for_all_arms",
        candidate_loss="logspace_conditional_NLL_div5" if mode=="valid_state_conditional" else "remove_constant_presence_BCE_keep_div5",
        promotion_rule="Q5_macro_delta>=1pp_and_weighted_delta>=-0.5pp; inspect_role_metrics_before_expansion",
        next_stage="manual_scientific_review_no_auto_candidate_expansion")
    e.OUT=ROOT_OUT/mode
    # Isolated process-local adapter. No kit files or other processes are changed.
    e.km=SimpleNamespace(**{name:getattr(native,name) for name in dir(native) if not name.startswith("__")})
    e.km.class_loss=objective

def verify_controls(binding):
    original=json.loads((ORIGINAL/"study.json").read_text())
    if canonical(original)!=ORIGINAL_SHA:raise RuntimeError("Wrong baseline study")
    for key in ("kit_sources","manifest_sha256","pretraining_sha256","detector_best_sha256",
                "detector_proposals_sha256","pixel_hash_map_sha256"):
        if binding[key]!=original[key]:raise RuntimeError("Historical control differs: "+key)
    controls={}
    for arm in ("G","Q5"):
        d=ORIGINAL/arm
        done=json.loads((d/"complete.json").read_text())
        rec=next(v for v in done["evaluations"] if v["epoch"]==4)
        for rel,key in (("epoch04.pt","checkpoint_sha256"),
                        ("val_epoch04/metrics.json","metrics_sha256"),
                        ("val_epoch04/predictions.jsonl","predictions_sha256")):
            if digest(d/rel)!=rec[key]:raise RuntimeError("Baseline artifact differs")
        controls[arm]=rec
    return controls

def run_variant(mode):
    configure(mode)
    e.OUT.mkdir(parents=True,exist_ok=True)
    e.set_status("initializing",objective=mode)
    try:
        e.guard()
        binding,frames,proposals,hashes,labels=e.setup()
        controls=verify_controls(binding)
        summary=[]
        for arm in ("G","Q5"):
            rec=e.train_arm(arm,binding,frames,proposals,hashes,labels)
            init=json.loads((e.OUT/arm/"initialization.json").read_text())
            expected=json.loads((ORIGINAL/arm/"initialization.json").read_text())
            if init["initial_model_sha256"]!=expected["initial_model_sha256"]:
                raise RuntimeError("Pilot initialization differs from control")
            history=json.loads((e.OUT/arm/"history.json").read_text())
            old_history=json.loads((ORIGINAL/arm/"history.json").read_text())[:4]
            if [h["label_order_sha256"] for h in history]!=[h["label_order_sha256"] for h in old_history]:
                raise RuntimeError("Training sample exposure differs")
            measured=rec["evaluations"][0]
            summary.append({"arm":arm,"control":controls[arm],"candidate":measured,
                "macro_delta_pp":100*(measured["macro_AP50"]-controls[arm]["macro_AP50"]),
                "weighted_delta_pp":100*(measured["weighted_AP50"]-controls[arm]["weighted_AP50"]),
                "paired_initialization_and_order":True})
        e.verify_sources(binding)
        q=next(x for x in summary if x["arm"]=="Q5")
        result={"status":"completed","objective":mode,"arms":summary,"seed":0,"epochs":4,
                "study_sha256":canonical(binding),"screen_gate_passed":q["macro_delta_pp"]>=1 and q["weighted_delta_pp"]>=-.5,
                "test_accessed":False,"not_a_final_method_comparison":True}
        atomic_json(e.OUT/"comparison_result.json",result)
        e.set_status("completed",objective=mode)
        return result
    except BaseException as exc:
        e.set_status("failed",objective=mode,error=str(exc))
        raise

def main():
    ROOT_OUT.mkdir(parents=True,exist_ok=True)
    with execution_lock("cuda:0"):
        results=[]
        for mode in VARIANTS:
            atomic_json(ROOT_OUT/"status.json",{"state":"running","objective":mode,"time_utc":e.now()})
            results.append(run_variant(mode))
            gc.collect();torch.cuda.empty_cache()
        atomic_json(ROOT_OUT/"comparison_result.json",{"status":"completed","variants":results,
            "test_accessed":False,"scope":"four_epoch_mechanism_pilot_seed0","auto_expand":False})
        atomic_json(ROOT_OUT/"status.json",{"state":"completed","time_utc":e.now()})

if __name__=="__main__":main()
