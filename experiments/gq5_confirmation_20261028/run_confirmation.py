"""Bounded G/Q5 seed replication with the original 12+12 training operations."""
import argparse, copy, gc, json, statistics, sys, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import engine12 as first
import engine24 as second
import torch
from helmetlab.kit_runner import atomic_json, canonical, execution_lock
from helmetlab.common import digest

ROOT_OUT=Path("/workspace/runs/gq5_confirmation_20261028")
FIRST_PLAN=copy.deepcopy(first.PLAN)
SECOND_PLAN=copy.deepcopy(second.PLAN)
ARM_ORDER=("G","Q5")

def configure_first(seed,arms=ARM_ORDER,root=ROOT_OUT):
    if seed not in (0,1,2):raise ValueError("Only prespecified seeds0,1,2")
    if not arms or len(set(arms))!=len(arms) or any(a not in ("G","Q5","G+","Q5shift") for a in arms):
        raise ValueError("Unsupported arm list")
    first.PLAN=copy.deepcopy(FIRST_PLAN)
    first.PLAN.update(seed=seed,arms=list(arms),schema="helmet2020.replication.phase12.v1",
        primary_comparison="fixed_global_epoch24_after_two_stage_schedule",
        interpretation="seed_replication_conditional_on_fixed_seed0_detector; recipe_frozen_after_seed0_development",
        evaluation_selection_used_to_stop=False)
    first.OUT=root/f"seed{seed}"/"phase12"
    return first

def configure_second(seed,arms=ARM_ORDER,root=ROOT_OUT):
    parent=root/f"seed{seed}"/"phase12"
    binding=json.loads((parent/"study.json").read_text())
    if binding["plan"]["seed"]!=seed or binding["plan"]["arms"]!=list(arms):
        raise RuntimeError("Parent seed or arms mismatch")
    second.PLAN=copy.deepcopy(SECOND_PLAN)
    second.PLAN.update(seed=seed,arms=list(arms),schema="helmet2020.replication.phase24.v1",
        interpretation="seed_replication_after_seed0_development; primary_epoch24; no_TEST")
    second.OUT=root/f"seed{seed}"/"phase24"
    second.PARENT=parent
    second.PARENT_CODE=HERE
    second.PARENT_SHA=canonical(binding)
    second.PARENT_CHECKPOINTS={}
    for arm in arms:
        rec=json.loads((parent/arm/"complete.json").read_text())
        e=next(r for r in rec["evaluations"] if r["epoch"]==12)
        if rec["seed"]!=seed or rec["arm"]!=arm or e["study_sha256"]!=second.PARENT_SHA:
            raise RuntimeError("Parent completion identity mismatch")
        second.PARENT_CHECKPOINTS[arm]=e["checkpoint_sha256"]
    return second

def finalize(module,binding,summaries):
    for rec in summaries:
        if rec["seed"]!=module.PLAN["seed"] or rec["study_sha256"]!=canonical(binding):
            raise RuntimeError("Mixed seed/study summaries")
    histories=[json.loads((module.OUT/a/"history.json").read_text()) for a in module.PLAN["arms"]]
    if any([x["label_order_sha256"] for x in h]!=[x["label_order_sha256"] for x in histories[0]] for h in histories[1:]):
        raise RuntimeError("Paired training label order differs")
    epochs=module.PLAN["val_epochs"]
    endpoint=epochs[-1]
    primary={r["arm"]:next(v for v in r["evaluations"] if v["epoch"]==endpoint) for r in summaries}
    result={"status":"completed","study_sha256":canonical(binding),"seed":module.PLAN["seed"],
        "arms":summaries,"endpoint_epoch":endpoint,"endpoint":primary,
        "paired_label_orders_match":True,"test_accessed":False,
        "interpretation":"Classifier seed replication; detector seed0 fixed; no independent TEST conclusion"}
    if "G" in primary and "Q5" in primary:
        result["Q5_minus_G"]={key+"_delta_pp":100*(primary["Q5"][key]-primary["G"][key])
                              for key in ("macro_AP50","weighted_AP50")}
    atomic_json(module.OUT/"comparison_result.json",result)
    atomic_json(module.OUT/"selection.json",{"rule":"fixed_last_epoch_for_primary_comparison",
        "epoch":endpoint,"seed":module.PLAN["seed"],"test_accessed":False,
        "entries":[{"arm":a,"checkpoint":f"{a}/epoch{endpoint:02d}.pt",
                    "sha256":r["checkpoint_sha256"]} for a,r in primary.items()]})
    module.set_status("completed",finished_utc=module.now(),result="comparison_result.json")
    module.emit("study_completed",seed=module.PLAN["seed"],epoch=endpoint,
        deltas=result.get("Q5_minus_G"),test_accessed=False)

def run_phase(module):
    module.OUT.mkdir(parents=True,exist_ok=True)
    try:
        module.set_status("initializing",started_utc=module.now(),seed=module.PLAN["seed"])
        module.guard()
        binding,frames,props,hashes,labels=module.setup()
        module.emit("study_verified",plan=module.PLAN,study_sha256=canonical(binding))
        summaries=[module.train_arm(a,binding,frames,props,hashes,labels) for a in module.PLAN["arms"]]
        module.verify_sources(binding)
        if module is second and module.parent_evidence()!=binding["parent"]:
            raise RuntimeError("Parent changed")
        finalize(module,binding,summaries)
    except BaseException as exc:
        module.set_status("failed",error=str(exc),exception=type(exc).__name__)
        raise

def aggregate_available():
    prior=Path("/workspace/runs/gq5_extend24_20261028/comparison_result.json")
    raw=json.loads(prior.read_text())
    if raw["study_sha256"]!="8a2516d80f8c4507224b0d785e80c3b7853f88b034cf9757346b721ee28a6915":
        raise RuntimeError("Wrong seed0 study")
    seed0={r["arm"]:next(v for v in r["evaluations"] if v["epoch"]==24) for r in raw["arms"]}
    rows=[{"seed":0,"endpoint":seed0,"origin":"exploratory_seed0_preceded_protocol_freeze"}]
    for seed in (1,2):
        p=ROOT_OUT/f"seed{seed}"/"phase24/comparison_result.json"
        if p.exists():
            rec=json.loads(p.read_text())
            if rec["status"]!="completed" or rec["endpoint_epoch"]!=24:raise RuntimeError("Incomplete seed")
            rows.append({"seed":seed,"endpoint":rec["endpoint"],"origin":"subsequent_seed_replication"})
    stats={}
    for arm in ARM_ORDER:
        stats[arm]={}
        for key in ("macro_AP50","weighted_AP50"):
            values=[r["endpoint"][arm][key]*100 for r in rows]
            stats[arm][key]={"mean_percent":statistics.mean(values),
                "sample_sd_pp":statistics.stdev(values) if len(values)>1 else None,"values_percent":values}
    atomic_json(ROOT_OUT/"replication_summary.json",{"status":"completed" if len(rows)==3 else "partial",
        "seeds_available":[r["seed"] for r in rows],"rows":rows,"statistics":stats,
        "detector_fixed_seed":0,"calibration_for_new_seeds":"not_run_yet",
        "test_accessed":False,"warning":"Seed0 guided recipe; disclose development history; no TEST or paper superiority claim"})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=(1,2),required=True)
    args=parser.parse_args()
    ROOT_OUT.mkdir(parents=True,exist_ok=True)
    with execution_lock("cuda:0"):
        run_phase(configure_first(args.seed))
        gc.collect();torch.cuda.empty_cache()
        run_phase(configure_second(args.seed))
        aggregate_available()

if __name__=="__main__":main()
