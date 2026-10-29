"""CPU synthetic checks only: identity, live gradients, checkpoint and capacity."""
import io,json,sys
from pathlib import Path
ROOT=Path("/workspace/HELMET2020_training_kit_v1")
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
from helmetlab import kit_models as km
from helmetlab.labels import LabelMap
from residual_candidate import build,count

def main():
    weights=ROOT/"pretrained/resnet18-f37072fd.pth"
    labels=LabelMap(ROOT/"manifests/classes.json")
    states=torch.tensor(labels.states)
    torch.manual_seed(20261028)
    rng=torch.get_rng_state().clone()
    model=build("GQ5_residual",seed=0,backbone_path=weights)
    assert torch.equal(rng,torch.get_rng_state())
    control=build("G_capacity392",seed=0,backbone_path=weights)
    baseline=km.build_classifier("G",seed=0,backbone_path=weights)
    assert count(model.head)==205209 and count(control.head)==205026
    assert abs(count(control.head)-count(model.head))/count(model.head)<.005
    assert all(torch.equal(v,baseline.backbone.state_dict()[k]) for k,v in model.backbone.state_dict().items())
    for k,v in model.head.global_head.state_dict().items():
        assert torch.equal(v,baseline.head.state_dict()[k])
    features=torch.randn(8,512,7,7,requires_grad=True)
    ids=torch.arange(8)
    initial=model.head(features)
    expected=baseline.head(features)
    assert torch.equal(initial,expected)
    for a,b in zip(km.decode(initial,model.kit_arm,labels),km.decode(expected,"G",labels)):
        assert (a==b).all()
    loss=km.class_loss(initial,ids,states,"Q5");loss.backward()
    baseline.zero_grad(set_to_none=True)
    km.class_loss(expected,ids,states,"G").backward()
    for a,b in zip(model.head.global_head.parameters(),baseline.head.parameters()):
        assert torch.equal(a.grad,b.grad)
    assert model.head.role_scale.grad is not None
    assert torch.isfinite(model.head.role_scale.grad).all() and float(model.head.role_scale.grad.abs().sum())>0
    assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in model.head.query_head.parameters())
    # A gate initialized at zero must wake the query branch after an update.
    opt=torch.optim.SGD(model.head.parameters(),lr=.01)
    opt.step();opt.zero_grad(set_to_none=True)
    out=model.head(features.detach())
    km.class_loss(out,ids,states,"Q5").backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.head.query_head.parameters())>0
    # Full model must execute the backbone once, not duplicate two classifiers.
    calls=[]
    hook=model.backbone.register_forward_hook(lambda *args:calls.append(1))
    model.eval()
    with torch.no_grad():full=model(torch.randn(2,3,64,64))
    hook.remove()
    assert len(calls)==1 and full.shape==(2,10) and torch.isfinite(full).all()
    # Strict checkpoint round trip, including gates.
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    restored=build("GQ5_residual",seed=0,initialize=False)
    restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model.head(features.detach()),restored.head(features.detach()))
    # Both outputs of a role use that role's coefficient.
    with torch.no_grad():
        model.head.role_scale.zero_();model.head.role_scale[2]=.4
        f=torch.randn(2,512,7,7)
        delta=model.head(f)-model.head.global_head(f)
        assert torch.count_nonzero(delta[:,[0,1,3,4,5,6,8,9]])==0
        assert torch.count_nonzero(delta[:,[2,7]])>0
    result={"status":"passed","dataset_training_started":False,"synthetic_optimizer_steps":1,"test_accessed":False,
            "model":model.architecture_metadata(),"control":control.architecture_metadata(),
            "checks":["paired_G_initial_logits_and_decoder_exact",
                      "initial_global_gradients_match_G","zero_gate_wakes_query_gradients",
                      "all_gradients_finite","one_backbone_per_forward",
                      "strict_checkpoint_roundtrip","native_role_gate_order","capacity_match_within0.5percent"]}
    (Path(__file__).parent/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
