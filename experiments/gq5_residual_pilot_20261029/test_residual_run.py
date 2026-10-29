"""CPU contract/resume tests for a new architecture without dataset fitting."""
import ast,copy,json,tempfile
from pathlib import Path
import run_residual_pilot as run
import test_candidate
import torch
torch.set_num_threads(2)

class RemoveDiagnostic(ast.NodeTransformer):
    def visit_If(self,node):
        if isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Name) and node.test.func.id=="hasattr":
            if len(node.test.args)==2 and isinstance(node.test.args[1],ast.Constant) and node.test.args[1].value=="role_scale":
                return None
        return self.generic_visit(node)

def function(path,name,normalize=False):
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    if normalize:fn=RemoveDiagnostic().visit(fn)
    return ast.dump(fn,include_attributes=False)

def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def update(head,opt,features,labels,states,arm):
    opt.zero_grad(set_to_none=True)
    loss=run.class_loss(head(features),labels,states,arm)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    opt.step()

def main():
    here=Path(__file__).resolve().parent
    old=Path("/workspace/q5_objective_pilot_20261028/pilot_engine.py")
    for name in ("train_arm","evaluate_milestone"):
        assert function(old,name)==function(here/"residual_engine.py",name,True)
    test_candidate.main()
    native_builder=run.native.build_classifier;native_loss=run.native.class_loss
    run.configure()
    assert run.e.PLAN["arms"]==["GQ5_residual","G_capacity392"]
    assert run.e.PLAN["epochs_per_arm"]==4 and run.e.PLAN["val_epochs"]==[4]
    assert run.e.PLAN["scheduler_T_max"]==12 and run.e.PLAN["auto_expand"] is False
    assert run.native.build_classifier is native_builder and run.native.class_loss is native_loss
    original=json.loads((run.CONTROL/"study.json").read_text())
    assert run.canonical(original)==run.CONTROL_SHA
    refs=run.verify_baseline(original)
    assert set(refs)=={"G","Q5"}
    from helmetlab.labels import LabelMap
    states=torch.tensor(LabelMap(run.e.ROOT/"manifests/classes.json").states)
    torch.manual_seed(33)
    f=torch.randn(8,512,7,7);ids=torch.arange(8)
    weights=run.e.ROOT/"pretrained/resnet18-f37072fd.pth"
    for arm in run.ARMS:
        model=run.build_classifier(arm,seed=0,backbone_path=weights)
        opt=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12)
        for _ in range(2):update(model.head,opt,f,ids,states,arm);sched.step()
        assert all(float(x["step"])==2 for x in opt.state.values())
        if arm=="GQ5_residual":
            assert float(model.head.role_scale.abs().sum())>0
            assert sum(float(p.grad.abs().sum()) for p in model.head.query_head.parameters())>0
        restored=run.build_classifier(arm,seed=0,initialize=False)
        restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
        opt2=torch.optim.AdamW(restored.head.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
        sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=12)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()));sched2.load_state_dict(copy.deepcopy(sched.state_dict()))
        update(model.head,opt,f,ids,states,arm);sched.step()
        update(restored.head,opt2,f,ids,states,arm);sched2.step()
        assert same(model.head.state_dict(),restored.head.state_dict()) and same(opt.state_dict(),opt2.state_dict())
        z=model.head(f)
        assert torch.equal(run.class_loss(z,ids,states,arm),native_loss(z,ids,states,"G"))
        del model,restored,opt,opt2,sched,sched2
    # Synthetic immutable-milestone and tamper rejection fixture.
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        run.e.atomic_save({"epoch":4,"value":torch.arange(2)},d/"last.pt")
        snap=run.e.snapshot_epoch(d,4);h=run.e.digest(snap)
        run.e.atomic_save({"epoch":5,"value":torch.ones(2)},d/"last.pt")
        assert run.e.digest(snap)==h
        v=d/"val_epoch04";v.mkdir()
        (v/"metrics.json").write_text("{}");(v/"predictions.jsonl").write_text("{}\n")
        binding={"fixture":True};ident=run.canonical(binding)
        receipt={"study_sha256":ident,"epoch":4,"checkpoint_sha256":h,
                 "metrics_sha256":run.e.digest(v/"metrics.json"),"predictions_sha256":run.e.digest(v/"predictions.jsonl")}
        run.atomic_json(v/"complete.json",receipt)
        assert run.e.verify_eval(d,4,ident)==receipt
        (v/"metrics.json").write_text('{"changed":true}')
        try:run.e.verify_eval(d,4,ident)
        except RuntimeError:pass
        else:raise AssertionError("Changed metric accepted")
    result={"status":"passed","dataset_training_started":False,"test_accessed":False,
       "checks":["candidate_gradient_and_single_backbone_tests",
                 "training_loop_unchanged_except_gate_diagnostic","native_module_not_modified",
                 "original_BCE_loss_exact","paired_baseline_hashes","bounded4epoch_plan",
                 "AdamW_zero_gate_wakes_queries","AdamW_resume_same_next_update",
                 "milestone_immutable","metrics_tamper_rejected"]}
    run.atomic_json(here/"run_verification.json",result)
    print(json.dumps(result),flush=True)

if __name__=="__main__":main()
