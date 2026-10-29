"""CPU checks of seed propagation, recipe parity, restart and finite queue."""
import ast, copy, gc, json, tempfile
from pathlib import Path
import torch
torch.set_num_threads(2)
import engine12 as first
import engine24 as second
import run_confirmation as driver
import queue_confirmation as queue
import check_phase12
import check_phase24

class NormalizeSeed(ast.NodeTransformer):
    def visit_Subscript(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=="PLAN" and isinstance(node.slice,ast.Constant) and node.slice.value=="seed":
            return ast.copy_location(ast.Constant(value=0),node)
        return self.generic_visit(node)

def fn(path,name,normalize=False):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    if normalize:node=NormalizeSeed().visit(node)
    return ast.dump(node,include_attributes=False)

def run():
    here=Path(__file__).resolve().parent
    old12=Path("/workspace/gq5_long12_20261027/run_gq5_12.py")
    old24=Path("/workspace/gq5_extend24_20261028/run_gq5_extend24.py")
    for name in ("train_arm","evaluate_milestone"):
        assert fn(old12,name)==fn(here/"engine12.py",name,True)
        assert fn(old24,name)==fn(here/"engine24.py",name,True)
    assert fn(old24,"validate_parent_checkpoint")==fn(here/"engine24.py","validate_parent_checkpoint",True)
    check_phase12.run()
    check_phase24.main()
    hashes={}
    for seed in (0,1,2):
        for arm in ("G","Q5"):
            rng=torch.get_rng_state().clone()
            m=first.km.build_classifier(arm,seed=seed,backbone_path=first.ROOT/"pretrained/resnet18-f37072fd.pth")
            assert torch.equal(rng,torch.get_rng_state())
            hashes[(seed,arm)]={"backbone":first.tensor_digest(m.backbone.state_dict()),
                               "model":first.tensor_digest(m.state_dict())}
            if seed==0:
                original=json.loads((Path("/workspace/runs/gq5_long12_20261027")/arm/"initialization.json").read_text())
                assert original["initial_model_sha256"]==hashes[(seed,arm)]["model"]
            del m;gc.collect()
    assert len({x["backbone"] for x in hashes.values()})==1
    for arm in ("G","Q5"):
        assert len({hashes[(s,arm)]["model"] for s in (0,1,2)})==3
    row=json.loads((first.ROOT/"manifests/train.jsonl").open().readline())
    tensors=[]
    for seed in (1,1,2):
        ds=first.km.CropDataset(first.IMAGES,[row],training=True,seed=seed)
        ds.set_epoch(3)
        tensors.append(ds[0][0])
    assert torch.equal(tensors[0],tensors[1]) and not torch.equal(tensors[0],tensors[2])
    for seed in (1,2):
        m=driver.configure_first(seed)
        assert m.PLAN["seed"]==seed and m.PLAN["arms"]==["G","Q5"]
        assert m.OUT.name=="phase12" and m.OUT.parent.name==f"seed{seed}"
    # Seed mismatch on parent continuation is explicitly rejected.
    oldplan=copy.deepcopy(second.PLAN);oldsha=second.PARENT_SHA
    second.PLAN["seed"]=1;second.PARENT_SHA="test-fixture"
    checkpoint={"schema":"helmet2020.gq5_long12.checkpoint.v1","study_sha256":"test-fixture",
        "arm":"G","seed":0,"epoch":12,"history":[{"epoch":e} for e in range(1,13)],
        "scheduler":{"T_max":12,"last_epoch":12},
        "optimizer":{"state":{0:{"step":12}},"param_groups":[{"lr":0.}]}}
    try:second.validate_parent_checkpoint(checkpoint,"G",1)
    except RuntimeError:pass
    else:raise AssertionError("Wrong seed parent accepted")
    checkpoint["seed"]=1
    second.validate_parent_checkpoint(checkpoint,"G",1)
    second.PLAN=oldplan;second.PARENT_SHA=oldsha
    assert queue.dependency_state({"state":"running"},False,True)=="waiting"
    assert queue.dependency_state({"state":"completed"},True,True)=="waiting"
    assert queue.dependency_state({"state":"completed"},True,False)=="ready"
    for state,exists,live in [("failed",False,False),("completed",False,False),("running",False,False)]:
        try:queue.dependency_state({"state":state},exists,live)
        except RuntimeError:pass
        else:raise AssertionError("Invalid dependency allowed")
    result={"status":"passed","training_started":False,"gpu_model_updates":0,
        "checks":["train_and_VAL_operations_AST_equal_to_original_except_seed",
                  "phase12_resume_and_artifact_tamper_regressions",
                  "phase24_optimizer_moments_and_LR_restart_regressions",
                  "seed0_initial_model_hashes_equal_original_run",
                  "seed1_2_heads_differ_same_ImageNet_backbone",
                  "paired_augmentation_identical_and_cross_seed_different",
                  "wrong_seed_parent_rejected",
                  "queue_waits_for_completed_dependency_process_exit",
                  "failed_dead_missing_dependency_stops_queue"]}
    (here/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":run()
