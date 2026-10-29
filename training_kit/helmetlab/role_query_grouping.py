"""Equal-five-query role grouping control; no fitting or novelty claim.

Q5shift changes which supervised tasks share a query, not model capacity.
Identical raw parameter bytes intentionally give permuted native helmet
initial logits. One fixed cyclic grouping is an exploratory control.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from . import role_queries as rq

ARMS = ("Q5", "Q5shift")
SCHEMA = "helmetlab.helmet2020.role_query_grouping_prototype.v1"
HELMET_TARGET_FOR_SHIFTED_QUERY = (1, 2, 3, 4, 0)
QUERY_FOR_NATIVE_SHIFTED_HELMET = (4, 0, 1, 2, 3)
classification_loss = rq.classification_loss


def _arm(arm: str) -> str:
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Expected Q5 or Q5shift")
    return arm


def to_native_logits(query_outputs: Tensor, arm: str) -> Tensor:
    """Restore helmet target-role order before the common loss and decoder."""
    arm = _arm(arm)
    if (not isinstance(query_outputs, Tensor) or query_outputs.ndim != 2 or
            query_outputs.shape[0] < 1 or query_outputs.shape[1] != 10 or
            not query_outputs.is_floating_point()):
        raise ValueError("Expected nonempty floating native-ten query outputs")
    if arm == "Q5":
        return query_outputs
    return torch.cat((query_outputs[:, :5], query_outputs[:, 5:].roll(1, dims=1)), dim=1)


class RoleGroupingHead(nn.Module):
    """Use exactly the frozen Q5 predictor construction and parameter keys."""

    def __init__(self, arm: str, in_channels: int = rq.FEATURE_CHANNELS):
        super().__init__()
        self.arm = _arm(arm)
        self.in_channels = in_channels
        self.predictor = rq.SpatialQueryHead(in_channels, 5, 2)

    def forward(self, features: Tensor) -> Tensor:
        return to_native_logits(self.predictor(features), self.arm)

    def architecture_metadata(self) -> dict:
        target = HELMET_TARGET_FOR_SHIFTED_QUERY if self.arm == "Q5shift" else tuple(range(5))
        inverse = QUERY_FOR_NATIVE_SHIFTED_HELMET if self.arm == "Q5shift" else tuple(range(5))
        return {"schema": SCHEMA, "source_architecture_schema": rq.ARCHITECTURE_SCHEMA,
            "arm": self.arm, "input_feature_channels": self.in_channels,
            "native_output_order": list(rq.NATIVE_OUTPUT_ORDER),
            "head_parameters": rq.parameter_count(self),
            "trainable_head_parameters": rq.parameter_count(self, trainable_only=True),
            "query_count": 5, "outputs_per_query": 2, "query_dimension": rq.QUERY_DIM,
            "attention_heads": rq.ATTENTION_HEADS, "FFN_dimension": rq.FFN_DIM,
            "cross_attention_blocks": 1, "query_self_attention": False,
            "attention_dropout": 0., "role_boxes_or_zoom": False, "KD_or_rebalancing": False,
            "helmet_target_role_for_query": [rq.ROLES[index] for index in target],
            "query_index_for_native_helmet": list(inverse),
            "query_tasks": [{"query_index": index, "presence_target": role,
                "helmet_target": rq.ROLES[target[index]]} for index, role in enumerate(rq.ROLES)],
            "helmet_mask_policy": "native target-role GT presence after output reordering",
            "initialization_contract": "pair raw parameter bytes and RNG state; shifted native helmet logits are permuted",
            "contrast_limit": "one prespecified cyclic task grouping; does not isolate localization, task correlation or optimization effects"}


class RoleGroupingClassifier(rq.RoleQueryClassifier):
    """Construct one backbone then one head, matching frozen Q5 RNG consumption."""

    def __init__(self, arm: str, backbone_state: Mapping[str, Tensor] | None = None):
        nn.Module.__init__(self)
        self.arm = _arm(arm)
        self.backbone = rq.ResNet18SpatialBackbone()
        if backbone_state is not None:
            self.load_backbone_state_dict(backbone_state)
        self.head = RoleGroupingHead(arm)


def build_role_grouping_model(arm: str, backbone_state: Mapping[str, Tensor] | None = None) -> RoleGroupingClassifier:
    return RoleGroupingClassifier(arm, backbone_state)
