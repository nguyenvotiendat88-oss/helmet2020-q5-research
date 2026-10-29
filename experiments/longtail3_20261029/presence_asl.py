"""One fixed loss adaptation: asymmetric focusing on presence only.

Inherited idea: simplified asymmetric focal loss (Query2Label, section3.2).
gamma_positive=0, gamma_negative=1, no probability shift or detached weights.
Helmet conditional BCE and global /5 normalization remain unchanged.
"""
import torch
from torch import nn

def loss(logits,ids,states,arm,weighting=None,*,gamma_negative=1.):
    if arm not in ("G","Q5") or weighting is not None:
        raise ValueError("Matched G/Q5 only, without extra weighting")
    if not isinstance(gamma_negative,(int,float)) or not 0<=gamma_negative<=1:
        raise ValueError("Only diagnostic gamma0 or fixed gamma1 permitted")
    if logits.ndim!=2 or logits.shape[1]!=10 or len(logits)==0 or not torch.isfinite(logits).all():
        raise ValueError("Finite nonempty batch x10 logits required")
    if ids.dtype!=torch.long or ids.ndim!=1 or len(ids)!=len(logits):
        raise ValueError("Aligned int64 class labels required")
    states=torch.as_tensor(states,device=logits.device,dtype=torch.long)
    if states.ndim!=2 or states.shape[1]!=5 or ((states<0)|(states>2)).any():
        raise ValueError("Expected class x5 states in {0,1,2}")
    if ((ids<0)|(ids>=len(states))).any():raise ValueError("Invalid class ID")
    targets=states[ids]
    present=(targets!=0).to(logits.dtype)
    helmet=(targets==1).to(logits.dtype)
    occupancy=nn.functional.binary_cross_entropy_with_logits(logits[:,:5],present,reduction="none")
    focus=present+(1-present)*torch.sigmoid(logits[:,:5]).pow(gamma_negative)
    helmet_loss=nn.functional.binary_cross_entropy_with_logits(logits[:,5:],helmet,reduction="none")
    return (focus*occupancy+present*helmet_loss).mean(-1).mean()
