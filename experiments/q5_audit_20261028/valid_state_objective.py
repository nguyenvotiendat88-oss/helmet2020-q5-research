"""Research candidate: known conditional likelihood applied to role-query outputs.

Not a novelty claim. Frozen kit and active runs remain unchanged. No training
entry point is provided. Stable log-space probabilities differ from the legacy
decoder's 1e-12 floor at extremes; isolate that numerical change in ablations.
"""
import torch
from torch import nn

def joint_log_scores(logits, states, omit_constant_presence=False):
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError("Expected batch x ten native logits")
    states=torch.as_tensor(states,device=logits.device,dtype=torch.long)
    if states.ndim != 2 or states.shape[1] != 5 or bool(((states<0)|(states>2)).any()):
        raise ValueError("Expected valid class x five role states")
    if len(torch.unique(states,dim=0))!=len(states):
        raise ValueError("Duplicate valid states")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Finite logits required")
    present=states!=0
    presence=torch.where(present[None],nn.functional.logsigmoid(logits[:,:5])[:,None],
                         nn.functional.logsigmoid(-logits[:,:5])[:,None])
    if omit_constant_presence:
        variable=present.any(0)&(~present).any(0)
        presence=presence*variable[None,None]
    helmet=torch.where((states==1)[None],nn.functional.logsigmoid(logits[:,5:])[:,None],
                       nn.functional.logsigmoid(-logits[:,5:])[:,None])
    return (presence+present[None]*helmet).sum(-1)

def valid_state_log_probs(logits,states):
    scores=joint_log_scores(logits,states,omit_constant_presence=True)
    return scores-scores.logsumexp(-1,keepdim=True)

def valid_state_nll(logits,ids,states):
    """Mean conditional class NLL / five, matching the declared native scale."""
    return nn.functional.nll_loss(valid_state_log_probs(logits,states),ids)/5.

def binary_loss_without_constant_presence(logits,ids,states):
    """Diagnostic ablation: remove only structurally constant presence tasks."""
    states=torch.as_tensor(states,device=logits.device,dtype=torch.long)
    present=(states[ids]!=0).to(logits.dtype)
    helmet=(states[ids]==1).to(logits.dtype)
    variable=(states!=0).any(0)&(states==0).any(0)
    occupancy=nn.functional.binary_cross_entropy_with_logits(logits[:,:5],present,reduction="none")
    conditional=nn.functional.binary_cross_entropy_with_logits(logits[:,5:],helmet,reduction="none")
    return (occupancy*variable[None]+present*conditional).mean()
