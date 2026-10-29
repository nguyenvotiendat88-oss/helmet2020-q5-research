"""Check the mathematical mechanism; no training/model quality claims."""
import json,sys
from pathlib import Path
ROOT=Path("/workspace/HELMET2020_training_kit_v1")
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from helmetlab.labels import LabelMap
from helmetlab import kit_models as km
from valid_state_objective import joint_log_scores,valid_state_log_probs,valid_state_nll,binary_loss_without_constant_presence

def run():
    labels=LabelMap(ROOT/"manifests/classes.json")
    states=torch.tensor(labels.states)
    assert (states[:,0]!=0).all()
    generator=torch.Generator().manual_seed(20261028)
    z=torch.randn(12,10,dtype=torch.float64,generator=generator,requires_grad=True)
    ids=torch.arange(12)
    logq=valid_state_log_probs(z,states)
    legacy=km.decode(z,"Q5",labels)
    np.testing.assert_allclose(logq.exp().detach().numpy(),legacy,rtol=1e-10,atol=1e-12)
    changed=z.detach().clone()
    changed[:,0]=torch.linspace(-80,80,len(z),dtype=z.dtype)
    torch.testing.assert_close(valid_state_log_probs(changed,states),logq.detach(),rtol=0,atol=0)
    # Every class score has zero derivative wrt the driver-presence logit.
    gradient=torch.autograd.grad(valid_state_nll(z,ids,states),z,retain_graph=True)[0]
    assert torch.count_nonzero(gradient[:,0])==0
    original=km.class_loss(z,ids,states,"Q5")
    old_gradient=torch.autograd.grad(original,z,retain_graph=True)[0]
    assert torch.count_nonzero(old_gradient[:,0])==len(z)
    scores=joint_log_scores(z,states)
    manual=(-scores[torch.arange(len(z)),ids]+scores.logsumexp(-1)).mean()/5.
    torch.testing.assert_close(valid_state_nll(z,ids,states),manual)
    torch.testing.assert_close(original,-scores[torch.arange(len(z)),ids].mean()/5.)
    assert torch.autograd.gradcheck(lambda value:valid_state_nll(value,ids,states),(z,),eps=1e-6,atol=1e-4)
    reduced=binary_loss_without_constant_presence(z,ids,states)
    missing=torch.nn.functional.binary_cross_entropy_with_logits(z[:,0],torch.ones(len(z),dtype=z.dtype))/5.
    torch.testing.assert_close(original-reduced,missing)
    result={"status":"passed","model_training":False,"test_accessed":False,
       "constant_presence_roles":[labels.data["positions"][i] for i in range(5) if len(set((states[:,i]!=0).tolist()))==1],
       "checks":["probability_matches_legacy_for_moderate_logits",
                 "driver_presence_invariance_logspace_minus80_to80",
                 "driver_presence_gradient_zero_in_conditional_objective",
                 "original_BCE_has_nonzero_redundant_gradient",
                 "conditional_likelihood_identity","autograd_finite_difference",
                 "constant_task_ablation_removes_exactly_its_loss"],
       "limits":"Legacy probability floor can break exact invariance at extreme logits; test moderate-logit parity only. Objective is conditional likelihood, an established principle, not new mathematics."}
    (Path(__file__).resolve().parent/"valid_state_objective_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=="__main__":run()
