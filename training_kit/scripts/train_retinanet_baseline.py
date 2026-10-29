"""Train a declared torchvision RetinaNet-R50-FPN adapted HELMET baseline.

Dry-run by default. Smoke mode cannot verify 800-pixel GPU memory or speed.
Use --stop-after 2 with the full configuration for a resumable operational pilot.
This worker neither downloads weights nor evaluates or selects on the test set.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import FrozenBatchNorm2d
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.transforms.functional import pil_to_tensor
import yaml

from helmetlab.checkpoints import atomic_save, cpu_tree, restore_rng, rng_state
from helmetlab.common import digest, read_json, read_jsonl, safe_child, validate_frames
from helmetlab.data import verify_yolo_export
from helmetlab.labels import LabelMap
from helmetlab.metrics import frame_metrics
from helmetlab.training import environment, source_hashes

SCRIPT = Path(__file__).resolve()
DEFAULT_CONFIG = PROJECT / "configs/RETINANET_ADAPTED.yaml"
SCHEMA = "helmetlab.retinanet_adapted.v1"
OBJECTIVES = ("weighted_AP50", "macro_AP50")
IMAGENET_SHA_PREFIX = "0676ba61"
COMPLETE_ARTIFACTS = {
    "weights/last.pt", "weights/best.pt", "weights/best_macro.pt", "weighted_checkpoint.json",
    "macro_checkpoint.json", "history.json", "epoch_timing.json", "selected_val.jsonl",
    "selected_val_AP50.json", "selected_val.jsonl.meta.json", "preflight.json", "run.json",
}


def binding_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def record_bytes(record):
    return (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def records_digest(records):
    hasher = hashlib.sha256()
    for record in records:
        hasher.update(record_bytes(record))
    return hasher.hexdigest()


def atomic_predictions(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        for record in records:
            handle.write(record_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest(path)


def load_config(path):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("RetinaNet configuration must be a mapping")
    required = {"model", "torchvision_version", "backbone_initialization", "optimizer", "lr", "batch",
                "epochs", "patience", "min_size", "max_size", "trainable_backbone_layers", "workers",
                "cpu_threads", "amp", "score_thresh", "nms_thresh", "detections_per_img", "topk_candidates"}
    if set(settings) != required:
        raise ValueError(f"RetinaNet config keys differ: {sorted(set(settings) ^ required)}")
    expected = {"model": "torchvision_retinanet_resnet50_fpn_v1", "torchvision_version": "0.23.0",
                "backbone_initialization": "torchvision_resnet50_imagenet1k_v1", "optimizer": "Adam",
                "lr": 1e-5, "batch": 4, "patience": 2, "min_size": 800, "max_size": 1333,
                "score_thresh": .001, "nms_thresh": .5, "detections_per_img": 300, "topk_candidates": 1000}
    if any(settings.get(key) != value or isinstance(settings.get(key), bool) for key, value in expected.items()):
        raise ValueError("Use the declared adapted reference training and inference settings")
    for name in ("epochs", "batch", "patience", "min_size", "max_size", "cpu_threads", "detections_per_img", "topk_candidates"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["epochs"] > 20:
        raise ValueError("This adapted protocol declares a maximum of 20 epochs")
    if type(settings["workers"]) is not int or settings["workers"] < 0:
        raise ValueError("workers must be a nonnegative integer")
    if type(settings["trainable_backbone_layers"]) is not int or settings["trainable_backbone_layers"] != 3:
        raise ValueError("The adapted reference uses exactly three trainable backbone layers")
    if type(settings["amp"]) is not bool:
        raise ValueError("amp must be a boolean")
    return dict(settings)


def effective_settings(config, smoke_test):
    settings = dict(config)
    if smoke_test:
        settings.update(batch=2, epochs=2, min_size=64, max_size=64, workers=0)
    return settings


def smoke_records(records, limit=2):
    chosen = []
    for empty in (False, True):
        match = next((index for index, row in enumerate(records) if (not row["objects"]) == empty), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    chosen.extend(index for index in range(len(records)) if index not in chosen)
    return [records[index] for index in sorted(chosen[:limit])]


def preflight(dataset, smoke_test=False):
    root = Path(dataset).resolve()
    lock = read_json(root / "lock.json")
    mandatory = {"train.jsonl", "val.jsonl", "test.jsonl", "classes.json", "audit.json"}
    if not isinstance(lock, dict) or not mandatory <= set(lock):
        raise ValueError("Dataset lock must bind train/val/test/classes/audit")
    checked = verify_yolo_export(root, image_label_splits=("train", "val"))
    if not smoke_test and read_json(root / "audit.json").get("subset_only") is not False:
        raise ValueError("The full adapted baseline requires a full dataset audit, not a subset-only export")
    labels = LabelMap(root / "classes.json")
    if not smoke_test and len(labels) != 36:
        raise ValueError("The full adapted HELMET baseline requires the 36-class map")
    records = {split: read_jsonl(root / f"{split}.jsonl") for split in ("train", "val")}
    if any(not rows for rows in records.values()) or not any(row["objects"] for row in records["val"]):
        raise ValueError("Need nonempty train/val frame sets and positive validation GT support")
    if smoke_test:
        records = {split: smoke_records(rows) for split, rows in records.items()}
    image_hashes = {}
    for split, rows in records.items():
        validate_frames(rows, len(labels))
        for index, row in enumerate(rows, 1):
            width, height = row["width"], row["height"]
            if type(width) is not int or type(height) is not int or min(width, height) < 1:
                raise ValueError("Image dimensions must be positive integers")
            for obj in row["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                    raise ValueError("Use the prepared clipped GT geometry; do not silently revise the locked manifests")
            path = safe_child(root, row["image"])
            contents = path.read_bytes()
            with Image.open(io.BytesIO(contents)) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError(f"Image pixels/dimensions differ from manifest: {row['image_id']}")
            image_hashes[row["image"]] = hashlib.sha256(contents).hexdigest()
            if index % 1000 == 0:
                print(f"Pixel preflight {split}: {index}/{len(rows)}", flush=True)
    files = [root / name for name in (*sorted(mandatory), "lock.json", "dataset.yaml")]
    identity = {"dataset": str(root), "dataset_lock_sha256": checked["dataset_lock_sha256"],
                "input_hashes": {str(path): digest(path) for path in files}, "image_sha256": image_hashes,
                "evaluated_records_sha256": {split: records_digest(rows) for split, rows in records.items()},
                "frame_ids": {split: [row["image_id"] for row in rows] for split, rows in records.items()},
                "smoke_test": smoke_test}
    report = {"data_identity": identity, "export_check": checked,
              "selected_frames": {split: len(rows) for split, rows in records.items()},
              "consumed_train_val_pixels_decoded_and_sha256_bound": True,
              "test_manifest_annotations_integrity_checked": True,
              "test_pixels_or_yolo_labels_checked": False,
              "test_predictions_generated": False, "test_metrics_or_selection": False,
              "smoke_subset_is_full_training_validation": False if smoke_test else True}
    return report, records, labels


class ManifestDataset(Dataset):
    def __init__(self, root, records, image_hashes=None):
        self.root, self.records = Path(root).resolve(), records
        self.image_hashes = image_hashes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        contents = safe_child(self.root, row["image"]).read_bytes()
        if self.image_hashes is not None and hashlib.sha256(contents).hexdigest() != self.image_hashes[row["image"]]:
            raise ValueError("Image bytes changed after preflight")
        with Image.open(io.BytesIO(contents)) as source:
            if source.size != (row["width"], row["height"]):
                raise ValueError("Image dimensions differ from the locked manifest")
            image = pil_to_tensor(source.convert("RGB")).to(torch.float32).div(255.)
        boxes = torch.tensor([obj["bbox"] for obj in row["objects"]], dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() and (not torch.isfinite(boxes).all() or bool((boxes[:, :2] < 0).any())
                             or bool((boxes[:, 2] > row["width"]).any()) or bool((boxes[:, 3] > row["height"]).any())
                             or bool((boxes[:, 2:] <= boxes[:, :2]).any())):
            raise ValueError("Targets must use valid prepared clipped geometry")
        targets = {"boxes": boxes,
                   "labels": torch.tensor([obj["class_id"] + 1 for obj in row["objects"]], dtype=torch.int64)}
        return image, targets


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class ForegroundRetinaNet(RetinaNet):
    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        diagnostics = {"anchors_decoded": 0, "degenerate_clipped_anchors_removed_before_topk": 0,
                       "foreground_hypotheses_above_threshold": 0, "foreground_hypotheses_after_geometry": 0,
                       "candidate_limit_hypotheses_removed": 0, "NMS_hypotheses_removed": 0,
                       "final_cap_hypotheses_removed": 0, "frames_hitting_final_cap": 0}
        detections = []
        for index, image_shape in enumerate(image_shapes):
            image_boxes, image_scores, image_labels = [], [], []
            levels = zip(head_outputs["bbox_regression"], head_outputs["cls_logits"], anchors[index])
            for regression, logits, level_anchors in levels:
                original = logits[index]
                if not bool(torch.isfinite(original).all()):
                    raise ValueError("RetinaNet classification logits became nonfinite")
                masked = original.clone()
                masked[..., 0] = -torch.inf
                num_classes = masked.shape[-1]
                # Decode all anchors before ranking so clipped empty boxes cannot consume a candidate slot.
                decoded = self.box_coder.decode_single(regression[index], level_anchors)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError("RetinaNet decoded boxes became nonfinite")
                clipped = box_ops.clip_boxes_to_image(decoded, image_shape)
                valid = (clipped[:, 2:] > clipped[:, :2]).all(dim=1)
                diagnostics["anchors_decoded"] += len(clipped)
                diagnostics["degenerate_clipped_anchors_removed_before_topk"] += int((~valid).sum())
                scores = torch.sigmoid(masked)
                diagnostics["foreground_hypotheses_above_threshold"] += int((scores[:, 1:] > self.score_thresh).sum())
                keep = (scores > self.score_thresh) & valid[:, None]
                candidate_ids = torch.where(keep.flatten())[0]
                diagnostics["foreground_hypotheses_after_geometry"] += len(candidate_ids)
                diagnostics["candidate_limit_hypotheses_removed"] += max(0, len(candidate_ids) - self.topk_candidates)
                candidate_scores = scores.flatten()[candidate_ids]
                candidate_scores, order = candidate_scores.topk(min(len(candidate_ids), self.topk_candidates))
                candidate_ids = candidate_ids[order]
                anchor_ids = torch.div(candidate_ids, num_classes, rounding_mode="floor")
                image_boxes.append(clipped[anchor_ids])
                image_scores.append(candidate_scores)
                image_labels.append(candidate_ids % num_classes)
            boxes = torch.cat(image_boxes, dim=0)
            scores = torch.cat(image_scores, dim=0)
            labels = torch.cat(image_labels, dim=0)
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            diagnostics["NMS_hypotheses_removed"] += len(scores) - len(keep)
            diagnostics["final_cap_hypotheses_removed"] += max(0, len(keep) - self.detections_per_img)
            diagnostics["frames_hitting_final_cap"] += int(len(keep) >= self.detections_per_img)
            keep = keep[:self.detections_per_img]
            detections.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        self.last_postprocess_diagnostics = diagnostics
        return detections


def backbone_state(path):
    path = Path(path).resolve()
    if not path.is_file() or not digest(path).startswith(IMAGENET_SHA_PREFIX):
        raise ValueError("Use the explicit local torchvision ImageNet V1 resnet50-0676ba61.pth asset; SHA prefix differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("ImageNet backbone file must contain the original tensor state_dict")
    if "fc.weight" not in state or "fc.bias" not in state or tuple(state["fc.weight"].shape) != (1000, 2048) or tuple(state["fc.bias"].shape) != (1000,):
        raise ValueError("ImageNet backbone needs the original 1000-way ResNet50 classifier shapes")
    return state


def build_model(settings, num_classes, backbone_weights=None, random_smoke=False):
    if torchvision.__version__.split("+", 1)[0] != settings["torchvision_version"]:
        raise ValueError("Use the pinned torchvision 0.23.0 implementation")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be positive")
    if backbone_weights is None and not random_smoke:
        raise FileNotFoundError("Model initialization requires explicit local ImageNet backbone weights")
    if random_smoke and (settings["min_size"] != 64 or settings["max_size"] != 64 or settings["epochs"] > 2):
        raise ValueError("Random initialization is allowed only in the bounded 64-pixel smoke configuration")
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d,
                                  trainable_layers=settings["trainable_backbone_layers"], returned_layers=[2, 3, 4],
                                  extra_blocks=LastLevelP6P7(256, 256))
    if backbone_weights is not None:
        state = backbone_state(backbone_weights)
        filtered = {key: value for key, value in state.items() if key not in {"fc.weight", "fc.bias"}
                    and not key.endswith(".num_batches_tracked")}
        if set(filtered) != set(backbone.body.state_dict()):
            raise ValueError("ImageNet backbone state keys differ from the declared ResNet50 architecture")
        backbone.body.load_state_dict(filtered, strict=True)
    return ForegroundRetinaNet(backbone, num_classes=num_classes + 1,
                              min_size=settings["min_size"], max_size=settings["max_size"],
                              score_thresh=settings["score_thresh"], nms_thresh=settings["nms_thresh"],
                              detections_per_img=settings["detections_per_img"], topk_candidates=settings["topk_candidates"])


def export_predictions(outputs, frames, num_classes, diagnostics=None):
    if len(outputs) != len(frames):
        raise ValueError("Prediction output frame count differs")
    if diagnostics is not None:
        for name in ("decoded_predictions", "exported_predictions", "dropped_degenerate_predictions"):
            diagnostics.setdefault(name, 0)
    result = []
    for output, frame in zip(outputs, frames):
        boxes, labels, scores = [output[key].detach().cpu() for key in ("boxes", "labels", "scores")]
        if boxes.shape != (len(labels), 4) or scores.shape != (len(labels),):
            raise ValueError("Malformed torchvision prediction tensor shapes")
        objects = []
        if diagnostics is not None:
            diagnostics["decoded_predictions"] += len(labels)
        for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            if type(label) is not int or not 0 <= label <= num_classes:
                raise ValueError("Prediction label outside the declared foreground class map")
            if label == 0:
                continue
            if not all(math.isfinite(value) for value in box) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prediction has nonfinite geometry or confidence")
            clipped = [max(0., min(float(frame["width"]), box[0])), max(0., min(float(frame["height"]), box[1])),
                       max(0., min(float(frame["width"]), box[2])), max(0., min(float(frame["height"]), box[3]))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                if diagnostics is not None:
                    diagnostics["dropped_degenerate_predictions"] += 1
                continue
            objects.append({"bbox": clipped, "class_id": label - 1, "score": score})
        if diagnostics is not None:
            diagnostics["exported_predictions"] += len(objects)
        result.append({**{key: value for key, value in frame.items() if key != "objects"}, "objects": objects})
    validate_frames(result, num_classes, prediction=True)
    return result


def scores(report):
    values = {"weighted_AP50": report.get("weighted_mAP", report.get("weighted_AP50")),
              "macro_AP50": report.get("macro_mAP", report.get("macro_AP50"))}
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1
           for value in values.values()):
        raise ValueError("Validation AP must be finite with positive GT support")
    return values


def advance_stop_state(state, report, epoch, patience):
    values = scores(report)
    previous = state.get("best", {name: {"value": None, "epoch": 0} for name in OBJECTIVES})
    best = {name: dict(previous[name]) for name in OBJECTIVES}
    for name in OBJECTIVES:
        if best[name]["value"] is None or values[name] > best[name]["value"]:
            best[name] = {"value": float(values[name]), "epoch": epoch}
    bad = 0 if best["weighted_AP50"]["epoch"] == epoch else state.get("weighted_bad_epochs", 0) + 1
    return {"best": best, "weighted_bad_epochs": bad, "stop": bad >= patience}


def verify_static(binding):
    if source_hashes() != binding["source_sha256"]:
        raise ValueError("Package source changed")
    if any(not Path(path).is_file() or digest(path) != expected for path, expected in binding["static_hashes"].items()):
        raise ValueError("Worker/config/dataset/backbone binding changed")


def make_binding(args, settings, checked, seed):
    asset = None
    if args.backbone_weights is not None:
        path = args.backbone_weights.resolve()
        if not path.is_file() or not digest(path).startswith(IMAGENET_SHA_PREFIX):
            raise ValueError("Explicit backbone weight asset must match SHA prefix 0676ba61")
        asset = {"path": str(path), "sha256": digest(path), "kind": "torchvision_ResNet50_IMAGENET1K_V1"}
    elif not args.smoke_random_init:
        raise FileNotFoundError("--execute requires existing --backbone-weights; no model files are downloaded")
    static = {**checked["data_identity"]["input_hashes"], str(SCRIPT): digest(SCRIPT),
              str(args.config.resolve()): digest(args.config)}
    if asset:
        static[asset["path"]] = asset["sha256"]
    env = environment()
    return {"schema": SCHEMA, "seed": seed, "device": str(torch.device(args.device)), "settings": settings,
            "source_sha256": source_hashes(), "static_hashes": static, "environment": env,
            "data_identity_sha256": binding_digest(checked["data_identity"]),
            "dataset_lock_sha256": checked["data_identity"]["dataset_lock_sha256"],
            "class_map_sha256": checked["data_identity"]["input_hashes"][str(args.dataset.resolve() / "classes.json")],
            "initialization": asset or {"kind": "explicit_random_untrained_smoke_only"},
            "smoke_test": args.smoke_test, "class_label_offset": 1,
            "background_suppression": "logit0_masked_before_threshold_topk_class_aware_NMS_and_final_cap",
            "geometry_guard": "clip_and_remove_degenerate_anchors_before_threshold_topk_NMS_and_cap; reject_nonfinite_predictions",
            "validation_inference_precision": "float32; no_autocast",
            "selection": "strict_weighted_validation_all_point_AP50; first_exact_equal_peak_retained",
            "macro_snapshot": "diagnostic_only; does_not_reset_weighted_patience",
            "augmentation": "none; aspect_resize_and_RGB_ImageNet_mean_std_inside_torchvision",
            "resume": "completed_epoch_state; rerun_partial_epoch; CUDA_future_numerical_identity_not_verified",
            "test_metadata_integrity_checked": True, "test_metrics_or_selection": False,
            "historical_exact_reproduction": False, "novelty_claim": False, "benchmark_claim": False}


def load_checkpoint(path, binding):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("binding") != binding:
        raise ValueError("RetinaNet checkpoint config/source/dataset/weights/environment binding differs")
    epoch = checkpoint.get("epoch")
    history = checkpoint.get("history")
    if type(epoch) is not int or not 0 <= epoch <= binding["settings"]["epochs"] or not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Checkpoint lacks a complete epoch history")
    stopper = {}
    for index, row in enumerate(history, 1):
        if row.get("epoch") != index:
            raise ValueError("Checkpoint history epochs differ")
        stopper = advance_stop_state(stopper, row, index, binding["settings"]["patience"])
    if checkpoint.get("stopper") != stopper:
        raise ValueError("Checkpoint lacks its exact weighted stopper and both objective peaks")
    if any(key not in checkpoint for key in ("state_dict", "optimizer", "scaler", "rng", "validation_generator", "best")):
        raise ValueError("Checkpoint lacks optimizer/AMP/RNG/sampler state")
    for name, reference in checkpoint["best"].items():
        if name not in OBJECTIVES or reference["epoch"] != stopper["best"][name]["epoch"]:
            raise ValueError("Best snapshot reference differs from recorded objective peak")
        for key, expected in (("weights", "weights_sha256"), ("predictions", "predictions_sha256"), ("report", "report_sha256")):
            actual = safe_child(Path(path).parent.parent, reference[key])
            if not actual.is_file() or digest(actual) != reference[expected]:
                raise ValueError("Committed RetinaNet best snapshot changed or is missing")
    if epoch and set(checkpoint["best"]) != set(OBJECTIVES):
        raise ValueError("Checkpoint must retain both objective peaks")
    return checkpoint


def inspect_existing(output, binding, resume):
    output = Path(output)
    if not output.exists():
        return None
    if not output.is_dir():
        raise FileExistsError("Run output is not a directory")
    if not any(output.iterdir()):
        return None
    if not resume:
        raise FileExistsError("Choose a fresh run output or --resume with identical bindings")
    latest = output / "weights/last.pt"
    if not latest.is_file():
        run_path = output / "run.json"
        if run_path.is_file():
            run = read_json(run_path)
            uncommitted = ((run.get("status") == "initializing" and run.get("completed_epochs") == 0)
                           or (run.get("status") == "interrupted_or_failed"
                               and "last_committed_epoch" in run and run["last_committed_epoch"] is None))
            allowed_files = {"run.json", "preflight.json", "weights/last.pt.tmp"}
            owned_initialization = run.get("schema") == SCHEMA and run.get("binding") == binding and uncommitted
            for item in output.rglob("*"):
                name = item.relative_to(output).as_posix()
                if (item.is_symlink() or (item.is_file() and name not in allowed_files)
                        or (item.is_dir() and name != "weights")):
                    owned_initialization = False
            if owned_initialization:
                return None
        raise FileNotFoundError("Resume needs this worker's completed-epoch weights/last.pt")
    checkpoint = load_checkpoint(latest, binding)
    complete = output / "complete.json"
    if complete.exists():
        final = read_json(complete)
        artifacts = final.get("artifacts")
        if (final.get("schema") != SCHEMA or final.get("binding") != binding
                or not isinstance(artifacts, dict) or set(artifacts) != COMPLETE_ARTIFACTS
                or not (checkpoint["stopper"].get("stop") or checkpoint["epoch"] == binding["settings"]["epochs"])
                or any(not safe_child(output, path).is_file() or digest(safe_child(output, path)) != expected
                       for path, expected in artifacts.items())):
            raise ValueError("Completed RetinaNet artifacts changed")
    return checkpoint


def copy_atomic(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def publish_best(output, checkpoint):
    for name, filename in (("weighted_AP50", "best.pt"), ("macro_AP50", "best_macro.pt")):
        if name in checkpoint["best"]:
            reference = checkpoint["best"][name]
            source = safe_child(output, reference["weights"])
            if digest(source) != reference["weights_sha256"]:
                raise ValueError("Best immutable checkpoint changed before publication")
            copy_atomic(source, Path(output) / "weights" / filename)
            atomic_json(Path(output) / ("weighted_checkpoint.json" if name == "weighted_AP50" else "macro_checkpoint.json"),
                        {"objective": name, "epoch": reference["epoch"], "value": checkpoint["stopper"]["best"][name]["value"],
                         "weights_sha256": reference["weights_sha256"], "metric": "frame_AP_all_points_IoU0.5"})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_model(model, loader, frames, device, num_classes, diagnostics=None):
    model.eval()
    predictions = []
    count = 0
    with torch.inference_mode():
        for images, _ in loader:
            outputs = model([image.to(device) for image in images])
            if diagnostics is not None:
                for name, value in getattr(model, "last_postprocess_diagnostics", {}).items():
                    diagnostics[name] = diagnostics.get(name, 0) + value
            predictions.extend(export_predictions(outputs, frames[count:count + len(images)], num_classes, diagnostics))
            count += len(images)
    if count != len(frames):
        raise ValueError("Validation loader lost frames")
    return predictions


def timing_estimate(rows, settings, smoke_test):
    if smoke_test or len(rows) < 2:
        return None
    usable = [row for row in rows if row["epoch"] > 1 and row.get("epoch_total_wall_seconds") is not None]
    if not usable:
        return None
    seconds = sum(row["epoch_total_wall_seconds"] for row in usable) / len(usable)
    remaining = max(0, settings["epochs"] - rows[-1]["epoch"])
    return {"reference_epochs_excluding_first": [row["epoch"] for row in usable],
            "mean_epoch_seconds": seconds, "remaining_epochs_upper_plan": remaining,
            "linear_remaining_seconds_upper_plan": seconds * remaining,
            "caveat": "Linear operational projection; patience may stop earlier and later throughput may differ."}


def artifact_bytes(output):
    return sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file())


def finalize(output, checkpoint, binding, checked, status, timings):
    publish_best(output, checkpoint)
    atomic_json(Path(output) / "history.json", checkpoint["history"])
    atomic_json(Path(output) / "epoch_timing.json", timings)
    run = {"schema": SCHEMA, "binding": binding, "status": status, "completed_epochs": checkpoint["epoch"],
           "stopper": checkpoint["stopper"], "artifact_bytes": artifact_bytes(output),
           "timing_estimate": timing_estimate(timings, binding["settings"], binding["smoke_test"]),
           "smoke_does_not_verify_800_pixel_GPU_memory_or_speed": True,
           "full_configuration_two_epoch_pilot": "--stop-after 2, then same command with --resume and no stop-after"}
    atomic_json(Path(output) / "run.json", run)
    if status != "completed":
        return run
    best = checkpoint["best"]["weighted_AP50"]
    prediction = Path(output) / "selected_val.jsonl"
    copy_atomic(safe_child(output, best["predictions"]), prediction)
    copy_atomic(safe_child(output, best["report"]), Path(output) / "selected_val_AP50.json")
    atomic_json(Path(output) / "selected_val.jsonl.meta.json", {
        "split": "val", "backend": "torchvision_retinanet_resnet50_fpn_v1", "frames": checked["selected_frames"]["val"],
        "dataset_lock_sha256": binding["dataset_lock_sha256"], "class_map_sha256": binding["class_map_sha256"],
        "source_sha256": binding["source_sha256"], "environment_versions": binding["environment"]["versions"],
        "weights_sha256": best["weights_sha256"], "classifier_sha256": None,
        "tracking": False, "dense": False, "dense_mode": None, "tile": 0, "nms": True,
        "confidence": binding["settings"]["score_thresh"], "max_hypotheses_per_frame": binding["settings"]["detections_per_img"],
        "output_prediction_sha256": digest(prediction), "smoke_test": binding["smoke_test"],
        "online_policy_compatible": False, "scope": "offline_torchvision_detector_frame_AP_only"})
    verify_static(binding)
    atomic_json(Path(output) / "complete.json", {"schema": SCHEMA, "binding": binding,
                "artifacts": {path: digest(Path(output) / path) for path in sorted(COMPLETE_ARTIFACTS)},
                "stop_reason": "weighted_patience" if checkpoint["stopper"]["stop"] else "declared_max_epochs",
                "historical_exact_reproduction": False, "novelty_claim": False, "benchmark_claim": False})
    return run


def train_one(args, settings, checked, records, labels, seed, output, binding, previous):
    output = Path(output)
    if (output / "complete.json").is_file():
        return read_json(output / "run.json")
    try:
        restarted_initialization = args.resume and previous is None and (output / "run.json").is_file()
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "run.json", {"schema": SCHEMA, "binding": binding, "status": "initializing",
                    "completed_epochs": previous["epoch"] if previous is not None else 0,
                    "restarted_initialization_without_committed_checkpoint": restarted_initialization})
        return _train_one(args, settings, checked, records, labels, seed, output, binding, previous)
    except BaseException:
        try:
            latest = output / "weights/last.pt"
            failure = {"schema": SCHEMA, "binding": binding, "status": "interrupted_or_failed",
                       "last_committed_epoch": None}
            if latest.is_file():
                try:
                    failure["last_committed_epoch"] = load_checkpoint(latest, binding)["epoch"]
                except Exception as checkpoint_error:
                    failure["checkpoint_verification_error"] = str(checkpoint_error)
            atomic_json(output / "run.json", failure)
        except Exception as status_error:
            print(f"RetinaNet failure status could not be saved: {status_error}", file=sys.stderr, flush=True)
        raise


def _train_one(args, settings, checked, records, labels, seed, output, binding, previous):
    output = Path(output)
    if (output / "complete.json").is_file():
        return read_json(output / "run.json")
    if previous is not None and (previous["stopper"].get("stop") or previous["epoch"] == settings["epochs"]):
        atomic_json(output / "preflight.json", checked)
        timing_path = output / "epoch_timing.json"
        timings = [row for row in read_json(timing_path) if row["epoch"] <= previous["epoch"]] if timing_path.is_file() else []
        return finalize(output, previous, binding, checked, "completed", timings)
    device = torch.device(args.device)
    torch.set_num_threads(settings["cpu_threads"])
    seed_everything(seed)
    model = build_model(settings, len(labels), args.backbone_weights, args.smoke_random_init).to(device)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=settings["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["amp"])
    generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    images = checked["data_identity"]["image_sha256"]
    options = {"batch_size": settings["batch"], "num_workers": settings["workers"], "collate_fn": collate,
               "pin_memory": device.type == "cuda", "persistent_workers": False}
    train_loader = DataLoader(ManifestDataset(args.dataset, records["train"], images), shuffle=True, generator=generator, **options)
    val_loader = DataLoader(ManifestDataset(args.dataset, records["val"], images), shuffle=False, generator=val_generator, **options)
    history, stopper, best, epoch = [], {}, {}, 0
    timings = []
    if previous is not None:
        model.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        history, stopper, best, epoch = previous["history"], previous["stopper"], previous["best"], previous["epoch"]
        restore_rng(previous["rng"], generator)
        val_generator.set_state(previous["validation_generator"])
        timing_path = output / "epoch_timing.json"
        if timing_path.exists():
            timings = [row for row in read_json(timing_path) if row["epoch"] <= epoch]
        publish_best(output, previous)
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights").mkdir(exist_ok=True)
    atomic_json(output / "preflight.json", checked)

    def checkpoint():
        return {"schema": SCHEMA, "binding": binding, "state_dict": cpu_tree(model.state_dict()),
                "optimizer": cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
                "epoch": epoch, "history": history, "stopper": stopper, "best": best,
                "rng": rng_state(generator), "validation_generator": val_generator.get_state()}

    current = previous
    if current is None:
        current = checkpoint()
        atomic_save(current, output / "weights/last.pt")
    if stopper.get("stop") or epoch == settings["epochs"]:
        return finalize(output, current, binding, checked, "completed", timings)
    if args.stop_after is not None and args.stop_after < epoch:
        raise ValueError("stop-after precedes the committed checkpoint epoch")
    atomic_json(output / "run.json", {"schema": SCHEMA, "binding": binding, "status": "training",
                "completed_epochs": epoch, "smoke_does_not_verify_800_pixel_GPU_memory_or_speed": True})
    try:
        while epoch < settings["epochs"]:
            if args.stop_after is not None and epoch >= args.stop_after:
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            model.train()
            loss_total, examples = 0., 0
            for step, (batch_images, targets) in enumerate(train_loader, 1):
                batch_images = [image.to(device) for image in batch_images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and settings["amp"]):
                    losses = model(batch_images, targets)
                    loss = sum(losses.values())
                if not bool(torch.isfinite(loss)):
                    raise ValueError("RetinaNet training loss became nonfinite")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_total += float(loss.detach()) * len(batch_images)
                examples += len(batch_images)
                if step == len(train_loader) or step % max(1, len(train_loader) // 10) == 0:
                    print(f"Seed {seed} epoch {epoch + 1}: train {step}/{len(train_loader)}", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            trained = time.perf_counter()
            prediction_diagnostics = {}
            predictions = validate_model(model, val_loader, records["val"], device, len(labels), prediction_diagnostics)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validated = time.perf_counter()
            report = frame_metrics(records["val"], predictions, len(labels))
            values = scores(report)
            scored = time.perf_counter()
            verify_static(binding)
            epoch += 1
            stopper = advance_stop_state(stopper, report, epoch, settings["patience"])
            name = f"epoch_{epoch:04d}"
            report_path = output / "epochs" / (name + ".json")
            pred_sha = records_digest(predictions)
            report.update(epoch=epoch, split="val", smoke_test=args.smoke_test,
                          prediction_geometry_diagnostics=prediction_diagnostics,
                          pred_sha256=pred_sha, gt_source_manifest_sha256=binding["static_hashes"][str(args.dataset.resolve() / "val.jsonl")],
                          evaluated_gt_records_sha256=checked["data_identity"]["evaluated_records_sha256"]["val"],
                          classes_sha256=binding["class_map_sha256"], worker_binding_sha256=binding_digest(binding))
            atomic_json(report_path, report)
            improved = [objective for objective in OBJECTIVES if stopper["best"][objective]["epoch"] == epoch]
            if improved:
                weights = output / "weights" / (name + ".pt")
                atomic_save({"schema": SCHEMA, "binding": binding, "state_dict": cpu_tree(model.state_dict()),
                             "epoch": epoch, "validation": values}, weights)
                prediction = output / "validation" / (name + ".jsonl")
                if atomic_predictions(prediction, predictions) != pred_sha:
                    raise ValueError("Saved validation predictions differ from scored records")
                reference = {"epoch": epoch, "weights": str(weights.relative_to(output)), "weights_sha256": digest(weights),
                             "predictions": str(prediction.relative_to(output)), "predictions_sha256": pred_sha,
                             "report": str(report_path.relative_to(output)), "report_sha256": digest(report_path)}
                for objective in improved:
                    best[objective] = dict(reference)
            history.append({"epoch": epoch, "loss": loss_total / examples, **values,
                            "report": str(report_path.relative_to(output)), "report_sha256": digest(report_path)})
            current = checkpoint()
            atomic_save(current, output / "weights/last.pt")
            publish_best(output, current)
            atomic_json(output / "history.json", history)
            committed = time.perf_counter()
            timings.append({"epoch": epoch, "train_seconds": trained - started,
                            "full_validation_inference_seconds": validated - trained,
                            "canonical_AP_seconds": scored - validated, "output_commit_seconds": committed - scored,
                            "epoch_total_wall_seconds": committed - started,
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                            "train_frames": len(records["train"]), "validation_frames": len(records["val"]),
                            "smoke_test": args.smoke_test})
            atomic_json(output / "epoch_timing.json", timings)
            atomic_json(output / "run.json", {"schema": SCHEMA, "binding": binding, "status": "training",
                        "completed_epochs": epoch, "artifact_bytes": artifact_bytes(output),
                        "timing_estimate": timing_estimate(timings, settings, args.smoke_test),
                        "smoke_does_not_verify_800_pixel_GPU_memory_or_speed": True})
            print(json.dumps({"seed": seed, "epoch": epoch, **values, "weighted_bad_epochs": stopper["weighted_bad_epochs"]}), flush=True)
            del predictions
            if stopper["stop"]:
                break
        status = "completed" if stopper.get("stop") or epoch == settings["epochs"] else "paused_at_requested_epoch"
        return finalize(output, current, binding, checked, status, timings)
    except BaseException:
        raise


def train(args):
    config = load_config(args.config)
    settings = effective_settings(config, args.smoke_test)
    if len(set(args.seeds)) != len(args.seeds) or any(type(seed) is not int or not 0 <= seed <= 2**32 - 2 for seed in args.seeds):
        raise ValueError("Seeds must be distinct integers in 0..2**32-2")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.smoke_random_init and (not args.smoke_test or args.backbone_weights is not None):
        raise ValueError("--smoke-random-init requires --smoke-test and no pretrained asset")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use CPU or one CUDA device")
    if args.stop_after is not None and not 1 <= args.stop_after <= settings["epochs"]:
        raise ValueError("stop-after must lie within the declared effective epoch budget")
    plan = {"schema": SCHEMA, "execute": args.execute, "settings": settings, "device": str(device),
            "seeds_serial": args.seeds, "dataset": str(args.dataset.resolve()), "output": str(args.output.resolve()),
            "initial_backbone": str(args.backbone_weights.resolve()) if args.backbone_weights else None,
            "random_untrained_smoke_only": args.smoke_random_init, "smoke_test": args.smoke_test,
            "selection": "weighted all-point validation AP50, patience2; macro snapshot diagnostic only",
            "historical_exact_reproduction": False, "novelty_claim": False,
            "test_metrics_or_selection": False, "smoke_does_not_verify_800_pixel_GPU_memory_or_speed": True,
            "full_configuration_pilot": "--stop-after 2 followed by identical --resume without stop-after",
            "preflight_performed": False}
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
    if not args.execute:
        return plan
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; full-resolution GPU execution remains unverified")
    output_root = args.output.resolve()
    if output_root.exists() and (not output_root.is_dir() or not args.resume and any(output_root.iterdir())):
        raise FileExistsError("Choose a fresh output root or --resume with this worker's exact bindings")
    if not args.smoke_random_init and (args.backbone_weights is None or not args.backbone_weights.is_file()):
        raise FileNotFoundError("--execute needs an existing explicit local ImageNet backbone asset")
    checked, records, labels = preflight(args.dataset, args.smoke_test)
    ancestor = next(path for path in (output_root, *output_root.parents) if path.exists())
    disk = {"free_disk_bytes_before_training": shutil.disk_usage(ancestor).free,
            "storage_note": "Optimizer last.pt, both best aliases and immutable improving-epoch models/predictions can be large; no fixed disk requirement assumed."}
    outputs = [output_root / f"seed{seed}" for seed in args.seeds]
    bindings = [make_binding(args, settings, checked, seed) for seed in args.seeds]
    root_binding = {"schema": SCHEMA, "seeds_serial": args.seeds,
                    "seed_bindings": {str(seed): binding for seed, binding in zip(args.seeds, bindings)}}
    if args.resume:
        previous_plan = output_root / "plan.json"
        if not previous_plan.is_file() or read_json(previous_plan).get("root_binding") != root_binding:
            raise ValueError("Resume root binding differs; need the original recipe and declared serial seeds")
    states = [inspect_existing(output, binding, args.resume) for output, binding in zip(outputs, bindings)]
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "plan.json", {**plan, "preflight_performed": True, **disk, "root_binding": root_binding,
                "data_identity_sha256": binding_digest(checked["data_identity"])})
    results = []
    for seed, output, binding, state in zip(args.seeds, outputs, bindings, states):
        verify_static(binding)
        results.append(train_one(args, settings, checked, records, labels, seed, output, binding, state))
    return {"schema": SCHEMA, "runs": results, "seeds_serial": args.seeds, "test_metrics_or_selection": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--dataset", type=Path, default=Path("data/helmet"))
    result.add_argument("--output", type=Path, default=Path("runs/retinanet_adapted_v1"))
    result.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--backbone-weights", type=Path, help="Existing local ImageNet V1 resnet50-0676ba61.pth; SHA-bound, never downloaded")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--smoke-test", action="store_true", help="Bounded 2-train/2-val, 64-pixel, max2epoch test; does not verify 800-pixel GPU speed/VRAM")
    result.add_argument("--smoke-random-init", action="store_true", help="Explicit untrained random weights only with --smoke-test; no baseline result claim")
    result.add_argument("--stop-after", type=int, help="Operational completed-epoch pause; use2 with full configuration for timing/VRAM pilot, then --resume without this flag")
    return result


def main(argv=None):
    train(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
