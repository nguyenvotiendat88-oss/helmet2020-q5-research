"""Lazy CPU construction of the existing five matched role-query arms.

This factory supplies no weights, recipe, checkpoint loading or execution admission.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import importlib
import random
import threading

from .role_query_study_contract import (
    ARMS, MODEL_SCHEMAS, NATIVE_ORDER, ROLES, canonical_sha, validate_binding,
)

_CONSTRUCTION_LOCK = threading.RLock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _arguments(arm, initialization_seed, backbone_state, binding):
    _require(type(arm) is str and arm in ARMS, "Expected exactly G, G+, Q10, Q5 or Q5shift")
    _require(type(initialization_seed) is int and 0 <= initialization_seed < 2**63,
             "Explicit integer initialization_seed in [0, 2**63) required")
    _require(isinstance(backbone_state, Mapping) and backbone_state
             and all(type(name) is str and name for name in backbone_state),
             "Supply a nonempty named CPU backbone tensor mapping, not a path")
    if binding is not None:
        validate_binding(binding)


@contextmanager
def _cpu_initialization(seed):
    # Serialize this factory; unrelated concurrent RNG consumers remain caller-owned.
    with _CONSTRUCTION_LOCK:
        python_state = random.getstate()
        torch = np = torch_state = numpy_state = default_dtype = None
        try:
            torch = importlib.import_module("torch")
            torch_state = torch.random.get_rng_state().clone()
            default_dtype = torch.get_default_dtype()
            np = importlib.import_module("numpy")
            numpy_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            torch.set_default_dtype(torch.float32)
            with torch.device("cpu"):
                yield torch
        finally:
            if torch_state is not None:
                torch.random.set_rng_state(torch_state)
            if default_dtype is not None:
                torch.set_default_dtype(default_dtype)
            if numpy_state is not None:
                np.random.set_state(numpy_state)
            random.setstate(python_state)


def _input_identity(state, torch, semantic_sha):
    _require(all(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                 and tensor.layout == torch.strided and not tensor.is_meta
                 for tensor in state.values()), "Backbone state must contain materialized dense CPU tensors")
    signature = tuple((name, id(tensor), tuple(tensor.shape), tensor.dtype,
                       tuple(tensor.stride()), tensor.storage_offset(), tensor._version)
                      for name, tensor in state.items())
    return signature, semantic_sha(state)


def _validate_model(model, arm, torch, queries):
    metadata = model.architecture_metadata()
    canonical_sha(metadata)
    _require(type(model.arm) is str and model.arm == arm and metadata.get("arm") == arm
             and metadata.get("schema") == MODEL_SCHEMAS[1 if arm in ("Q5", "Q5shift") else 0]
             and metadata.get("native_output_order") == list(NATIVE_ORDER),
             "Constructed arm/schema/native output order differs")
    for component, total_key, trainable_key in (
            (model, "total_parameters", "trainable_total_parameters"),
            (model.backbone, "backbone_parameters", "trainable_backbone_parameters"),
            (model.head, "head_parameters", "trainable_head_parameters")):
        parameters = list(component.parameters())
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        _require(total > 0 and total == trainable
                 and type(metadata.get(total_key)) is int and metadata[total_key] == total
                 and type(metadata.get(trainable_key)) is int and metadata[trainable_key] == trainable,
                 "Actual trainable parameter counts differ from metadata")
    _require(all(tensor.device.type == "cpu" and tensor.layout == torch.strided and not tensor.is_meta
                 and (not tensor.is_floating_point() or tensor.dtype == torch.float32)
                 for tensor in model.state_dict().values()), "Factory must return a dense CPU FP32 model")
    expected_queries, expected_outputs = (0, None) if arm in ("G", "G+") else (10, 1) if arm == "Q10" else (5, 2)
    _require(type(metadata.get("query_count")) is int and metadata["query_count"] == expected_queries
             and "outputs_per_query" in metadata
             and (metadata["outputs_per_query"] is None if expected_outputs is None
                  else type(metadata["outputs_per_query"]) is int and metadata["outputs_per_query"] == expected_outputs),
             "Constructed query/output controls differ")
    _require(metadata.get("backbone_weights_argument") is None
             and metadata.get("backbone_pretrained_download") is False,
             "Implicit backbone download is forbidden")
    if arm == "G+":
        target = queries.q5_head_parameter_count(queries.FEATURE_CHANNELS)
        error = abs(metadata["head_parameters"] - target) / target
        _require(metadata.get("G_plus_target_Q5_head_parameters") == target
                 and metadata.get("G_plus_relative_parameter_error") == error
                 and metadata.get("G_plus_match_tolerance") == .005
                 and metadata.get("G_plus_match_within_tolerance") is True and error <= .005,
                 "Actual G+ capacity must match Q5 within 0.5 percent")
    if arm in ("Q5", "Q5shift"):
        targets = ROLES if arm == "Q5" else ("P1", "P2", "P3", "P0", "D")
        inverse = [0, 1, 2, 3, 4] if arm == "Q5" else [4, 0, 1, 2, 3]
        _require(metadata.get("helmet_target_role_for_query") == list(targets)
                 and canonical_sha(metadata.get("query_index_for_native_helmet")) == canonical_sha(inverse)
                 and metadata.get("helmet_mask_policy") == "native target-role GT presence after output reordering"
                 and canonical_sha(metadata.get("query_tasks")) == canonical_sha([
                     {"query_index": index, "presence_target": role, "helmet_target": targets[index]}
                     for index, role in enumerate(ROLES)]), "Constructed native role grouping differs")
    return metadata


def build_model(arm, *, initialization_seed, backbone_state, binding=None):
    """Return one existing CPU FP32 model with explicitly seeded fresh head.

    All inputs are required except ``binding``. A supplied full study binding
    additionally pins this arm's exact metadata and semantic initialization.
    Callers supply already admitted CPU tensor weights and serialize unrelated
    RNG consumers. No checkpoint is loaded, CUDA API initialized or fit launched.
    """
    _arguments(arm, initialization_seed, backbone_state, binding)
    binding_sha = None if binding is None else canonical_sha(binding)
    with _cpu_initialization(initialization_seed) as torch:
        queries = importlib.import_module("helmetlab.role_queries")
        semantic_sha = importlib.import_module("helmetlab.role_query_scientific_training").semantic_state_sha256
        original = _input_identity(backbone_state, torch, semantic_sha)
        if binding is not None:
            _require(original[1] == binding["initial_backbone_sha256"],
                     "Supplied backbone differs from the known study binding")
        try:
            if arm in ("G", "G+", "Q10"):
                model = queries.build_role_query_model(arm, backbone_state)
            else:
                grouping = importlib.import_module("helmetlab.role_query_grouping")
                model = grouping.build_role_grouping_model(arm, backbone_state)
            metadata = _validate_model(model, arm, torch, queries)
            _require(semantic_sha(model.backbone.state_dict()) == original[1],
                     "Constructed backbone semantic bytes differ from supplied weights")
            model_sha = semantic_sha(model.state_dict())
            if binding is not None:
                _require(canonical_sha(metadata) == canonical_sha(binding["arm_metadata"][arm])
                         and model_sha == binding["initial_model_sha256"][arm],
                         "Constructed metadata or semantic initialization differs from known binding")
            return model
        finally:
            _require(_input_identity(backbone_state, torch, semantic_sha) == original,
                     "Supplied backbone tensor mapping changed during construction")
            if binding is not None:
                _require(canonical_sha(binding) == binding_sha, "Supplied study binding changed during construction")
