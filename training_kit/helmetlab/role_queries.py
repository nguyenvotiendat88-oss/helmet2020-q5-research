"""Matched engineering prototypes for global and spatial-query native-ten heads.

This is a standard grouped-query adaptation, not a novelty claim. Q5 versus
Q10 compares query count and representation sharing together; it does not
isolate sharing as the cause of a performance change.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torchvision import models

from .positional import classification_loss as _native_classification_loss

ARMS = ("G", "G+", "Q10", "Q5")
ROLES = ("D", "P1", "P2", "P3", "P0")
NATIVE_OUTPUT_ORDER = tuple("presence_" + role for role in ROLES) + tuple("helmet_" + role for role in ROLES)
ARCHITECTURE_SCHEMA = "helmetlab.helmet2020.role_query_prototype.v1"
FEATURE_CHANNELS, QUERY_DIM, ATTENTION_HEADS, FFN_DIM = 512, 128, 4, 256
HEAD_MATCH_TOLERANCE = .005
BACKBONE_PARTS = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if not trainable_only or parameter.requires_grad)


def fixed_2d_position_encoding(height: int, width: int, *, device=None, dtype=torch.float32) -> Tensor:
    """Row-major y-sin/y-cos/x-sin/x-cos encoding, independent of arm and batch."""
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("Spatial dimensions must be positive integers")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Position encoding needs a floating dtype")
    quarter = QUERY_DIM // 4
    frequency = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.) / quarter))
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequency[None, :]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequency[None, :]
    y = torch.cat((y.sin(), y.cos()), dim=1)[:, None, :].expand(height, width, -1)
    x = torch.cat((x.sin(), x.cos()), dim=1)[None, :, :].expand(height, width, -1)
    return torch.cat((y, x), dim=2).reshape(height * width, QUERY_DIM).to(dtype=dtype)


class SpatialQueryHead(nn.Module):
    """One cross-attention block; Q5 shares each role token across its two outputs."""

    def __init__(self, in_channels: int, query_count: int, outputs_per_query: int):
        super().__init__()
        if type(in_channels) is not int or in_channels < 1 or (query_count, outputs_per_query) not in ((10, 1), (5, 2)):
            raise ValueError("Use a positive channel count and either ten label or five role queries")
        self.in_channels, self.query_count, self.outputs_per_query = in_channels, query_count, outputs_per_query
        self.projection = nn.Conv2d(in_channels, QUERY_DIM, kernel_size=1)
        self.queries = nn.Parameter(torch.empty(query_count, QUERY_DIM))
        self.attention = nn.MultiheadAttention(QUERY_DIM, ATTENTION_HEADS, dropout=0., batch_first=True)
        self.norm1 = nn.LayerNorm(QUERY_DIM)
        self.ffn = nn.Sequential(nn.Linear(QUERY_DIM, FFN_DIM), nn.GELU(), nn.Linear(FFN_DIM, QUERY_DIM))
        self.norm2 = nn.LayerNorm(QUERY_DIM)
        self.readout_weight = nn.Parameter(torch.empty(query_count, outputs_per_query, QUERY_DIM))
        self.readout_bias = nn.Parameter(torch.empty(query_count, outputs_per_query))
        nn.init.normal_(self.queries, std=.02)
        nn.init.uniform_(self.readout_weight, -1. / math.sqrt(QUERY_DIM), 1. / math.sqrt(QUERY_DIM))
        nn.init.uniform_(self.readout_bias, -1. / math.sqrt(QUERY_DIM), 1. / math.sqrt(QUERY_DIM))

    def memory_tokens(self, features: Tensor) -> Tensor:
        _validate_features(features, self.in_channels)
        projected = self.projection(features)
        _, _, height, width = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)
        position = fixed_2d_position_encoding(height, width, device=tokens.device, dtype=tokens.dtype)
        return tokens + position.unsqueeze(0)

    def query_representations(self, features: Tensor) -> Tensor:
        memory = self.memory_tokens(features)
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, _ = self.attention(queries, memory, memory, need_weights=False)
        queries = self.norm1(queries + attended)
        return self.norm2(queries + self.ffn(queries))

    def forward(self, features: Tensor) -> Tensor:
        representations = self.query_representations(features)
        outputs = torch.einsum("bqd,qod->bqo", representations, self.readout_weight) + self.readout_bias
        if self.outputs_per_query == 1:
            return outputs.squeeze(-1)
        return torch.cat((outputs[:, :, 0], outputs[:, :, 1]), dim=1)


def _validate_features(features: Tensor, in_channels: int) -> None:
    if (not isinstance(features, Tensor) or features.ndim != 4 or features.shape[1] != in_channels or
            min(features.shape[0], features.shape[2], features.shape[3]) < 1 or not features.is_floating_point()):
        raise ValueError("Expected nonempty floating NCHW features with the configured channel count")


def q5_head_parameter_count(in_channels: int = FEATURE_CHANNELS) -> int:
    if type(in_channels) is not int or in_channels < 1:
        raise ValueError("Channel count must be positive")
    projection = in_channels * QUERY_DIM + QUERY_DIM
    queries = 5 * QUERY_DIM
    attention = 4 * QUERY_DIM * QUERY_DIM + 4 * QUERY_DIM
    norms = 4 * QUERY_DIM
    ffn = 2 * QUERY_DIM * FFN_DIM + FFN_DIM + QUERY_DIM
    readout = 10 * QUERY_DIM + 10
    return projection + queries + attention + norms + ffn + readout


def matched_global_width(in_channels: int = FEATURE_CHANNELS) -> int:
    target = q5_head_parameter_count(in_channels)
    denominator = in_channels + 11
    ideal = (target - 10) / denominator
    candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
    return min(candidates, key=lambda width: (abs(denominator * width + 10 - target), width))


class RoleQueryHead(nn.Module):
    """Feature-map API for all four controls, without allocating a backbone."""

    def __init__(self, arm: str, in_channels: int = FEATURE_CHANNELS):
        super().__init__()
        if arm not in ARMS or type(in_channels) is not int or in_channels < 1:
            raise ValueError("Expected G, G+, Q10 or Q5 and a positive channel count")
        self.arm, self.in_channels = arm, in_channels
        self.hidden_width = matched_global_width(in_channels) if arm == "G+" else None
        if arm == "G":
            self.predictor = nn.Linear(in_channels, 10)
        elif arm == "G+":
            self.predictor = nn.Sequential(nn.Linear(in_channels, self.hidden_width), nn.GELU(), nn.Linear(self.hidden_width, 10))
        else:
            self.predictor = SpatialQueryHead(in_channels, 10 if arm == "Q10" else 5, 1 if arm == "Q10" else 2)

    def forward(self, features: Tensor) -> Tensor:
        _validate_features(features, self.in_channels)
        return self.predictor(features.mean(dim=(2, 3))) if self.arm in ("G", "G+") else self.predictor(features)

    def architecture_metadata(self) -> dict:
        count = parameter_count(self, trainable_only=True)
        target = q5_head_parameter_count(self.in_channels)
        return {"schema": ARCHITECTURE_SCHEMA, "arm": self.arm, "input_feature_channels": self.in_channels,
            "native_output_order": list(NATIVE_OUTPUT_ORDER), "head_parameters": parameter_count(self),
            "trainable_head_parameters": count, "global_hidden_width": self.hidden_width,
            "G_plus_target_Q5_head_parameters": target if self.arm == "G+" else None,
            "G_plus_relative_parameter_error": abs(count - target) / target if self.arm == "G+" else None,
            "G_plus_match_tolerance": HEAD_MATCH_TOLERANCE if self.arm == "G+" else None,
            "G_plus_match_within_tolerance": abs(count - target) / target <= HEAD_MATCH_TOLERANCE if self.arm == "G+" else None,
            "query_count": 10 if self.arm == "Q10" else 5 if self.arm == "Q5" else 0,
            "outputs_per_query": 1 if self.arm == "Q10" else 2 if self.arm == "Q5" else None,
            "query_dimension": QUERY_DIM if self.arm in ("Q10", "Q5") else None,
            "attention_heads": ATTENTION_HEADS if self.arm in ("Q10", "Q5") else None,
            "cross_attention_blocks": 1 if self.arm in ("Q10", "Q5") else 0,
            "FFN_dimension": FFN_DIM if self.arm in ("Q10", "Q5") else None,
            "query_self_attention": False, "attention_dropout": 0.,
            "token_policy": "all spatial cells, row-major, 1x1 learned projection; no cap or mask" if self.arm in ("Q10", "Q5") else "global spatial mean",
            "position_encoding": "fixed absolute y-sin/y-cos/x-sin/x-cos, base10000, d128; added to keys and values" if self.arm in ("Q10", "Q5") else None,
            "block_policy": "cross_attention residual then LayerNorm; GELU FFN residual then LayerNorm" if self.arm in ("Q10", "Q5") else None,
            "role_boxes_or_zoom": False, "KD_or_rebalancing": False,
            "contrast_limit": "Q5 vs Q10 changes query count and representation sharing together; no causal isolation"}


class ResNet18SpatialBackbone(nn.Module):
    """Keep native ResNet18 keys so accepted backbone bytes load strictly."""

    def __init__(self):
        super().__init__()
        source = models.resnet18(weights=None)
        for name in BACKBONE_PARTS:
            self.add_module(name, getattr(source, name))

    def forward(self, pixels: Tensor) -> Tensor:
        if (not isinstance(pixels, Tensor) or pixels.ndim != 4 or pixels.shape[1] != 3 or
                min(pixels.shape[0], pixels.shape[2], pixels.shape[3]) < 1 or not pixels.is_floating_point()):
            raise ValueError("Expected nonempty floating NCHW RGB pixels")
        for name in BACKBONE_PARTS:
            pixels = getattr(self, name)(pixels)
        return pixels


class RoleQueryClassifier(nn.Module):
    def __init__(self, arm: str, backbone_state: Mapping[str, Tensor] | None = None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Expected G, G+, Q10 or Q5")
        self.arm = arm
        self.backbone = ResNet18SpatialBackbone()
        if backbone_state is not None:
            self.load_backbone_state_dict(backbone_state)
        self.head = RoleQueryHead(arm)

    def forward_features(self, pixels: Tensor) -> Tensor:
        return self.backbone(pixels)

    def forward(self, pixels: Tensor) -> Tensor:
        return self.head(self.forward_features(pixels))

    def backbone_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict((name, value.detach().cpu().clone()) for name, value in self.backbone.state_dict().items())

    def load_backbone_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if not isinstance(state, Mapping) or any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
            raise ValueError("Expected a named tensor backbone state mapping")
        expected = self.backbone.state_dict()
        if any(value.dtype != expected[name].dtype for name, value in state.items() if name in expected):
            raise ValueError("Backbone state dtype differs; implicit byte-changing casts are forbidden")
        self.backbone.load_state_dict(state, strict=True)

    def architecture_metadata(self) -> dict:
        return {**self.head.architecture_metadata(), "backbone": "torchvision.resnet18 spatial conv1 through layer4",
            "backbone_weights_argument": None, "backbone_pretrained_download": False,
            "backbone_parameters": parameter_count(self.backbone),
            "trainable_backbone_parameters": parameter_count(self.backbone, trainable_only=True),
            "total_parameters": parameter_count(self), "trainable_total_parameters": parameter_count(self, trainable_only=True),
            "loss": "unchanged positional.classification_loss binary10; presence BCE plus GT-present-masked helmet BCE, mean over batch*5"}


def build_role_query_model(arm: str, backbone_state: Mapping[str, Tensor] | None = None) -> RoleQueryClassifier:
    return RoleQueryClassifier(arm, backbone_state)


def classification_loss(logits: Tensor, ids: Tensor, states: Tensor) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[1] != 10 or logits.shape[0] < 1:
        raise ValueError("Expected nonempty native-ten logits")
    if not isinstance(ids, Tensor) or ids.ndim != 1 or len(ids) != len(logits) or ids.dtype != torch.int64:
        raise ValueError("Expected one int64 class ID per logit row")
    if (not isinstance(states, Tensor) or states.ndim != 2 or states.shape[1] != 5 or
            states.shape[0] < 1 or states.dtype not in (torch.int32, torch.int64) or
            bool(((states < 0) | (states > 2)).any()) or bool(((ids < 0) | (ids >= len(states))).any())):
        raise ValueError("Expected valid absent/helmet/no-helmet native states and class IDs")
    return _native_classification_loss(logits, ids, states, "binary10")
