"""Explicit matched-study contracts; validation does not authorize execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import re

ARMS = ("G", "G+", "Q10", "Q5", "Q5shift")
ROLES = ("D", "P1", "P2", "P3", "P0")
NATIVE_ORDER = tuple(kind + "_" + role for kind in ("presence", "helmet") for role in ROLES)
SCHEMA = "helmetlab.helmet2020.role_query_matched_recipe.v1"
BINDING_SCHEMA = "helmetlab.helmet2020.role_query_matched_binding.v1"
RECIPE_FIELDS = {
    "schema", "arms", "seed", "epochs", "batch_size", "lr", "weight_decay", "stride",
    "cpu_threads", "workers", "optimizer", "scheduler", "precision", "input_size",
    "crop_margin", "sample_stream", "drop_last", "augmentation", "selection", "fit_scope",
}
BINDING_HASH_FIELDS = {
    "class_map_sha256", "train_manifest_sha256", "cache_manifest_sha256", "cache_pixels_sha256",
    "items_sha256", "initial_backbone_sha256", "class_states_sha256",
}
BINDING_FIELDS = BINDING_HASH_FIELDS | {
    "schema", "num_classes", "crop_count", "initialization_scope", "source_sha256", "versions",
    "arm_metadata", "initial_model_sha256", "Q5_Q5shift_raw_head_equal",
}
INITIALIZATION_SCOPES = (
    "accepted_exposed_HELMET_teacher_backbone_exploratory",
    "ImageNet_pretrained_backbone_exploratory",
)
MODEL_SCHEMAS = (
    "helmetlab.helmet2020.role_query_prototype.v1",
    "helmetlab.helmet2020.role_query_grouping_prototype.v1",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        require(math.isfinite(value), "Finite JSON values required")
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("Strict finite JSON types required")


def canonical_sha(value):
    _json_value(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def valid_sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
        and (value > 0 if positive else value >= 0))


def validate_recipe(recipe):
    canonical_sha(recipe)
    require(type(recipe) is dict and set(recipe) == RECIPE_FIELDS, "Recipe exact fields differ")
    require(recipe["schema"] == SCHEMA and recipe["arms"] == list(ARMS), "Matched five-arm recipe required")
    require(type(recipe["seed"]) is int and 0 <= recipe["seed"] < 2**63, "Invalid matched seed")
    for name in ("epochs", "stride", "cpu_threads"):
        require(type(recipe[name]) is int and recipe[name] >= 1, "Explicit positive integer required: " + name)
    require(type(recipe["batch_size"]) is int and 2 <= recipe["batch_size"] <= 64, "Explicit batch 2..64 required")
    require(type(recipe["workers"]) is int and recipe["workers"] == 0, "Initial scientific core requires workers=0")
    require(_number(recipe["lr"], positive=True) and _number(recipe["weight_decay"]), "Invalid explicit optimizer rates")
    require(canonical_sha(recipe["optimizer"]) == canonical_sha({"name": "AdamW", "betas": [.9, .999], "eps": 1e-8,
        "foreach": False, "fused": False}), "Declared deterministic AdamW contract differs")
    require(canonical_sha(recipe["scheduler"]) == canonical_sha({"name": "CosineAnnealingLR", "unit": "completed_epoch", "eta_min": 0.}),
        "Declared epoch cosine scheduler differs")
    require(recipe["precision"] == "FP32_TF32_disabled_deterministic"
        and canonical_sha(recipe["input_size"]) == canonical_sha([224, 224]) and type(recipe["crop_margin"]) is float
        and recipe["crop_margin"] == .05, "Fixed crop/precision contract differs")
    require(recipe["sample_stream"] == "torch_randperm_all_crops_once_per_epoch_v1"
        and recipe["drop_last"] is False
        and recipe["augmentation"] == "CachedCrops_item_epoch_ColorJitter_v1", "Common exposure/augmentation differs")
    require(recipe["selection"] == "final_completed_planned_epoch_only"
        and recipe["fit_scope"] == "TRAIN_only_exploratory_development", "Fixed final selection and development scope required")
    return recipe


def _source_name(name):
    require(type(name) is str and name and "\\" not in name and ":" not in name,
        "Portable source name required")
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and ".." not in path.parts and str(path) == name,
        "Canonical relative source name required")


def validate_binding(binding):
    canonical_sha(binding)
    require(type(binding) is dict and set(binding) == BINDING_FIELDS, "Binding exact fields differ")
    require(binding["schema"] == BINDING_SCHEMA, "Study binding schema differs")
    require(type(binding["num_classes"]) is int and binding["num_classes"] >= 1
        and type(binding["crop_count"]) is int and binding["crop_count"] >= 2, "Invalid class/crop population")
    require(all(valid_sha(binding[name]) for name in BINDING_HASH_FIELDS), "Missing exact data/backbone identity")
    require(binding["initialization_scope"] in INITIALIZATION_SCOPES, "Explicit exploratory initialization exposure required")
    sources, versions = binding["source_sha256"], binding["versions"]
    require(type(sources) is dict and sources and all(valid_sha(value) for value in sources.values()),
        "Explicit source identities required")
    for name in sources:
        _source_name(name)
    require(type(versions) is dict and versions
        and all(type(key) is str and key and type(value) is str and value for key, value in versions.items()),
        "Explicit runtime versions required")
    metadata, initial = binding["arm_metadata"], binding["initial_model_sha256"]
    require(type(metadata) is dict and set(metadata) == set(ARMS)
        and type(initial) is dict and set(initial) == set(ARMS)
        and all(valid_sha(value) for value in initial.values()), "All five initial model identities required")
    require(binding["Q5_Q5shift_raw_head_equal"] is True and initial["Q5"] == initial["Q5shift"],
        "Q5/Q5shift complete raw initialization hashes must match")
    for arm in ARMS:
        value = metadata[arm]
        require(type(value) is dict and value.get("arm") == arm and value.get("schema") in MODEL_SCHEMAS
            and value.get("native_output_order") == list(NATIVE_ORDER), "Arm/native metadata differs: " + arm)
    for arm in ("G", "G+", "Q10"):
        value = metadata[arm]
        queries, outputs = (10, 1) if arm == "Q10" else (0, None)
        require(value["schema"] == MODEL_SCHEMAS[0]
            and type(value.get("query_count")) is int and value["query_count"] == queries
            and "outputs_per_query" in value
            and (value.get("outputs_per_query") is None if outputs is None
                else type(value.get("outputs_per_query")) is int and value["outputs_per_query"] == outputs),
            "Global/independent-query representation differs: " + arm)
    plus, q5 = metadata["G+"], metadata["Q5"]
    count, target = plus.get("trainable_head_parameters"), q5.get("trainable_head_parameters")
    require(type(count) is int and count > 0 and type(target) is int and target > 0,
        "Capacity control requires explicit head parameter counts")
    error = abs(count - target) / target
    require(type(plus.get("head_parameters")) is int and plus["head_parameters"] == count
        and type(plus.get("G_plus_target_Q5_head_parameters")) is int
        and plus["G_plus_target_Q5_head_parameters"] == target
        and type(plus.get("G_plus_relative_parameter_error")) in (int, float)
        and plus["G_plus_relative_parameter_error"] == error
        and type(plus.get("G_plus_match_tolerance")) is float and plus["G_plus_match_tolerance"] == .005
        and error <= .005 and plus.get("G_plus_match_within_tolerance") is True,
        "G+ capacity must match Q5 within the frozen 0.5 percent tolerance")
    for arm, targets, inverse in (("Q5", ROLES, (0, 1, 2, 3, 4)),
                                  ("Q5shift", ("P1", "P2", "P3", "P0", "D"), (4, 0, 1, 2, 3))):
        value = metadata[arm]
        require(value["schema"] == MODEL_SCHEMAS[1]
            and type(value.get("query_count")) is int and value["query_count"] == 5
            and type(value.get("outputs_per_query")) is int and value["outputs_per_query"] == 2
            and value.get("helmet_target_role_for_query") == list(targets)
            and canonical_sha(value.get("query_index_for_native_helmet")) == canonical_sha(list(inverse))
            and value.get("helmet_mask_policy") == "native target-role GT presence after output reordering"
            and canonical_sha(value.get("query_tasks")) == canonical_sha([{"query_index": index, "presence_target": role,
                "helmet_target": targets[index]} for index, role in enumerate(ROLES)]),
            "Aligned/shifted grouping semantics differ: " + arm)
    return binding


def exposure(recipe, crop_count, epochs):
    validate_recipe(recipe)
    require(type(crop_count) is int and crop_count >= 2
        and type(epochs) is int and 0 <= epochs <= recipe["epochs"], "Invalid committed epoch exposure")
    return {"completed_epochs": epochs,
        "completed_updates": ((crop_count + recipe["batch_size"] - 1) // recipe["batch_size"]) * epochs,
        "seen_crops": crop_count * epochs}
