"""CPU checks: masked asymmetric loss, exact paired initial states and resume."""
import ast,copy,json,tempfile
from pathlib import Path
import run_longtail as run
import presence_asl
import torch
torch.set_num_threads(2)
from helmetlab.labels import LabelMap
from helmetlab.kit_runner import source_bindings

def fn(path,name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def step(head,opt,sched,x,y,states,arm):
    opt.zero_grad(set_to_none=True)
    value=presence_asl.loss(head(x),y,states,arm)
    value.backward()
    assert torch.isfinite(value)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    opt.step();sched.step()

def main():
    here=Path(__file__).resolve().parent
    source=Path("/workspace/gq5_confirmation_20261028/engine12.py")
    for name in ("train_arm","evaluate_milestone"):
        assert fn(source,name)==fn(here/"engine.py",name)
    kit_before=source_bindings()
    native_loss=run.native.class_loss
    labels=LabelMap(run.e.ROOT/"manifests/classes.json")
    states=torch.tensor(labels.states)
    torch.manual_seed(20261029)
    z=torch.randn(7,10,dtype=torch.float64,requires_grad=True);ids=torch.arange(7)
    original=native_loss(z,ids,states,"G")
    gamma0=presence_asl.loss(z,ids,states,"G",gamma_negative=0)
    assert torch.equal(original,gamma0)
    g0=torch.autograd.grad(original,z,retain_graph=True)[0]
    g1=torch.autograd.grad(presence_asl.loss(z,ids,states,"Q5"),z,retain_graph=True)[0]
    target=states[ids];present=target!=0
    # All helmet derivatives and positive presence derivatives are unchanged.
    torch.testing.assert_close(g0[:,5:],g1[:,5:],rtol=0,atol=0)
    torch.testing.assert_close(g0[:,:5][present],g1[:,:5][present],rtol=0,atol=0)
    assert torch.count_nonzero(g1[:,5:][~present])==0
    p=z[:,:5].sigmoid()
    manual=(-(present*torch.nn.functional.logsigmoid(z[:,:5]))
            -(~present)*p*torch.nn.functional.logsigmoid(-z[:,:5])
            +present*torch.nn.functional.binary_cross_entropy_with_logits(z[:,5:],(target==1).double(),reduction="none")).mean(-1).mean()
    torch.testing.assert_close(presence_asl.loss(z,ids,states,"G"),manual)
    assert torch.autograd.gradcheck(lambda x:presence_asl.loss(x,ids,states,"G"),(z,),eps=1e-6,atol=1e-4)
    # Easy negative derivatives are suppressed; hard negatives still get gradient.
    easy=torch.full((1,10),-8.,dtype=torch.float64,requires_grad=True)
    all_absent=torch.zeros((1,5),dtype=torch.long)
    y=torch.zeros(1,dtype=torch.long)
    ga=torch.autograd.grad(presence_asl.loss(easy,y,all_absent,"G"),easy,retain_graph=True)[0]
    gb=torch.autograd.grad(native_loss(easy,y,all_absent,"G"),easy)[0]
    assert (ga[:,:5].abs()<gb[:,:5].abs()).all()
    extremes=torch.tensor([[100.,-100.]*5,[-100.,100.]*5],requires_grad=True)
    loss=presence_asl.loss(extremes,torch.tensor([0,1]),states,"Q5")
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(extremes.grad).all()
    seed_model_hashes={}
    for seed in run.SEEDS:
        run.configure(seed)
        assert run.e.PLAN["seed"]==seed and run.e.PLAN["val_epochs"]==[12]
        assert run.e.PLAN["epochs_per_arm"]==12 and run.e.PLAN["scheduler_T_max"]==12
        original_binding=json.loads((run.baseline_root(seed)/"study.json").read_text())
        ref=run.prior_baseline(seed,original_binding)
        assert set(ref["endpoints"])=={"G","Q5"}
        for arm in run.ARMS:
            expected=run.paired_init(seed,arm,ref)
            seed_model_hashes[f"{arm}_seed{seed}"]=expected["initial_model_sha256"]
    assert run.native.class_loss is native_loss
    assert kit_before==source_bindings()
    # Small real head AdamW update and interruption-resume equivalence.
    run.configure(0)
    weights=run.e.ROOT/"pretrained/resnet18-f37072fd.pth"
    for arm in run.ARMS:
        model=run.native.build_classifier(arm,seed=0,backbone_path=weights)
        x=torch.randn(8,512,7,7);y=torch.arange(8)
        opt=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12)
        step(model.head,opt,sched,x,y,states,arm)
        restored=run.native.build_classifier(arm,seed=0,initialize=False)
        restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
        opt2=torch.optim.AdamW(restored.head.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
        sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=12)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()));sched2.load_state_dict(copy.deepcopy(sched.state_dict()))
        step(model.head,opt,sched,x,y,states,arm);step(restored.head,opt2,sched2,x,y,states,arm)
        assert equal(model.head.state_dict(),restored.head.state_dict()) and equal(opt.state_dict(),opt2.state_dict())
        del model,restored,opt,opt2,sched,sched2
    # Data augmentation stays paired by seed and epoch, independent of model.
    row=json.loads((run.e.ROOT/"manifests/train.jsonl").open().readline())
    data=run.native.CropDataset(run.e.IMAGES,[row],training=True,seed=1);data.set_epoch(4)
    assert torch.equal(data[0][0],data[0][0])
    data2=run.native.CropDataset(run.e.IMAGES,[row],training=True,seed=2);data2.set_epoch(4)
    assert not torch.equal(data[0][0],data2[0][0])
    result={"status":"passed","dataset_training_started":False,"GPU_used":False,"test_accessed":False,
        "checks":["train_and_eval_loops_AST_identical","gamma0_exact_BCE",
                  "positive_presence_and_conditional_helmet_gradients_unchanged",
                  "absent_helmet_gradient_zero","asymmetric_formula_and_finite_difference",
                  "easy_negative_suppression_extreme_logits_finite",
                  "all3seed_baseline_artifacts_verified","all6_initial_model_hashes_paired",
                  "original_kit_and_native_loss_not_modified","AdamW_resume_same_next_update",
                  "augmentation_paired_and_seed_sensitive"],
        "paired_initialization_hashes":seed_model_hashes}
    run.atomic_json(here/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
