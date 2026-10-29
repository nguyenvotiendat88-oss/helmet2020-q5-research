"""Prepared fallback, not an executed experiment or a novelty claim.

Retains G's pooled path and adds a five-role query residual on the same backbone.
Global + spatial residual attention has established prior art (CSRA, ICCV2021).
This adaptation must be named GQ5_residual, never reported as unchanged Q5.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from helmetlab import kit_models as km

VARIANTS=("GQ5_residual","G_capacity392")
ROLES=("D","P1","P2","P3","P0")

def count(module):
    return sum(p.numel() for p in module.parameters())

def matched_width(target,channels=512):
    ideal=(target-10)/(channels+11)
    choices={max(1,math.floor(ideal)),max(1,math.ceil(ideal))}
    return min(choices,key=lambda width:(abs((channels+11)*width+10-target),width))

class ResidualRoleHead(nn.Module):
    def __init__(self,global_head,query_head):
        super().__init__()
        self.global_head=global_head
        self.query_head=query_head
        # tanh(0)=0: the first forward is exactly the paired G forward.
        self.role_scale=nn.Parameter(torch.zeros(5))

    def forward(self,features):
        global_logits=self.global_head(features)
        residual_logits=self.query_head(features)
        scale=self.role_scale.tanh()
        native_scale=torch.cat((scale,scale))
        return global_logits+native_scale[None]*residual_logits

class CapacityGlobalHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.predictor=nn.Sequential(nn.Linear(512,width),nn.GELU(),nn.Linear(width,10))
    def forward(self,features):
        return self.predictor(features.mean((2,3)))

class Candidate(nn.Module):
    def __init__(self,variant,backbone,head,metadata):
        super().__init__()
        self.variant=variant
        self.backbone=backbone
        self.head=head
        # Decoder adapter only: both use the same ten native binary logits.
        self.kit_arm="Q5" if variant=="GQ5_residual" else "G+"
        self.details=metadata
    def forward_features(self,pixels):
        return self.backbone(pixels)
    def forward(self,pixels):
        return self.head(self.forward_features(pixels))
    def architecture_metadata(self):
        return {**self.details,"variant":self.variant,"backbone_count":1,
                "head_parameters":count(self.head),"total_parameters":count(self),
                "native_output_order":[f"presence_{x}" for x in ROLES]+[f"helmet_{x}" for x in ROLES],
                "not_original_Q5":True,"new_method_validated":False}

def build(variant,seed,backbone_path=None,initialize=True):
    if variant not in VARIANTS:raise ValueError("Unregistered variant")
    global_model=km.build_classifier("G",seed=seed,backbone_path=backbone_path,initialize=initialize)
    query_model=km.build_classifier("Q5",seed=seed,backbone_path=backbone_path,initialize=initialize)
    if not all(torch.equal(v,query_model.backbone.state_dict()[k]) for k,v in global_model.backbone.state_dict().items()):
        raise RuntimeError("G/Q5 backbone initialization differs")
    target=count(global_model.head)+count(query_model.head)+5
    if variant=="GQ5_residual":
        head=ResidualRoleHead(global_model.head,query_model.head)
        details={"formula":"G_logits + repeat_native(tanh(role_scale))*Q5_logits",
                 "gate_initialization":0.,"trainable_role_gates":5,
                 "global_path_initialization":"exact paired G",
                 "query_path_initialization":"exact paired Q5",
                 "prior_art":"Global/spatial residual attention; CSRA ICCV2021",
                 "does_not_guarantee_test_noninferiority":True}
    else:
        width=matched_width(target)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head=CapacityGlobalHead(width)
        details={"global_hidden_width":width,"target_head_parameters":target,
                 "relative_head_parameter_mismatch":abs(count(head)-target)/target,
                 "purpose":"capacity control for hybrid, separate from original G+"}
    return Candidate(variant,global_model.backbone,head,details)
