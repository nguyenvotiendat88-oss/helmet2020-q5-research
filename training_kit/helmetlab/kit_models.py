"""Model/data adapters for the new split training kit.

These adapters reuse the existing detector and role heads without changing any
frozen study. They never download weights or select a checkpoint. ``initialize``
is false only when a caller immediately loads its already verified kit state.
Random training initialization is permitted explicitly for smoke fixtures.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision import models, transforms
from torchvision.ops import batched_nms

from .common import assert_aligned_frames, digest, safe_child, validate_frames
from .positional import crop_box, decode_logits
from .role_query_model_factory import build_model as build_role_model
from scripts import train_retinanet_baseline as retina

ARMS = ("flat", "position", "G", "G_sqrt", "G+", "Q10", "Q5", "Q5shift")
RESNET18_SHA_PREFIX = "f37072fd"


def _guard(callback):
    if callback is not None and callback() is False:
        raise RuntimeError("Execution guard denied work")


def _arm(value):
    if value not in ARMS:
        raise ValueError("Unknown training kit classifier arm")
    return value


def build_detector(settings, backbone_path=None, smoke=False, *, initialize=True,
                   num_classes=36):
    """Build the existing foreground RetinaNet; never fetch pretrained assets.

    Full new fits use the original ImageNet ResNet50 asset. ``initialize=False``
    creates only architecture, intended for strict loading of a kit checkpoint.
    """
    config = dict(settings)
    if smoke:
        config.update(min_size=64, max_size=64, epochs=min(config.get("epochs", 2), 2))
    if initialize:
        return retina.build_model(config, num_classes, backbone_path,
                                  random_smoke=bool(smoke and backbone_path is None))
    if torchvision.__version__.split("+", 1)[0] != config["torchvision_version"]:
        raise ValueError("Use the pinned torchvision implementation")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be positive")
    backbone = retina.resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=retina.FrozenBatchNorm2d,
        trainable_layers=config["trainable_backbone_layers"], returned_layers=[2, 3, 4],
        extra_blocks=retina.LastLevelP6P7(256, 256))
    return retina.ForegroundRetinaNet(
        backbone, num_classes=num_classes + 1, min_size=config["min_size"],
        max_size=config["max_size"], score_thresh=config["score_thresh"],
        nms_thresh=config["nms_thresh"], detections_per_img=config["detections_per_img"],
        topk_candidates=config["topk_candidates"])


class DetectionDataset(retina.ManifestDataset):
    """Original manifest loader, including negatives and SHA-bound image reads."""


collate_detection = retina.collate


def detector_predictions(model, loader, frames, device, num_classes=36, guard=None):
    """Predict an ordered, complete frame list, including every empty frame."""
    model.eval()
    result, cursor = [], 0
    with torch.inference_mode():
        for images, _targets in loader:
            _guard(guard)
            outputs = model([image.to(device) for image in images])
            result.extend(retina.export_predictions(outputs, frames[cursor:cursor + len(images)], num_classes))
            cursor += len(images)
            _guard(guard)
    if cursor != len(frames):
        raise ValueError("Prediction loader did not cover every frame exactly once")
    assert_aligned_frames(frames, result, num_classes)
    return result


def _resnet18_state(path):
    path = Path(path)
    before = digest(path)
    if not before.startswith(RESNET18_SHA_PREFIX):
        raise ValueError("Use the original resnet18-f37072fd.pth ImageNet asset, not a HELMET checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("ImageNet ResNet18 must be its original tensor state_dict")
    if digest(path) != before:
        raise ValueError("ImageNet asset changed while loading")
    return state


def build_classifier(arm, seed=0, backbone_path=None, smoke=False, *, initialize=True,
                     num_classes=36):
    """Return a fresh, paired ResNet18 crop arm without implicit downloads.

    G_sqrt uses exactly G architecture. Its separate training-only weighting is
    applied by ``class_loss``. No old HELMET teacher is admissible here.
    """
    _arm(arm)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be positive")
    if initialize and backbone_path is None and not smoke:
        raise FileNotFoundError("Classifier fit requires the local original ImageNet ResNet18 asset")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        source = models.resnet18(weights=None)
        if initialize and backbone_path is not None:
            state = _resnet18_state(backbone_path)
            expected = source.state_dict()
            missing = set(expected) - set(state)
            extra = set(state) - set(expected)
            # The official legacy resnet18-f37072fd checkpoint predates the
            # BatchNorm num_batches_tracked buffers. These buffers are runtime
            # counters, not learned weights; torchvision initializes them to
            # zero when constructing the model. Fill only this known, safe
            # compatibility gap and reject every other key mismatch.
            if extra or any(not key.endswith(".num_batches_tracked") for key in missing):
                raise ValueError("ImageNet ResNet18 tensor keys, shapes, dtypes or values differ")
            state = {**state, **{key: expected[key].detach().clone() for key in missing}}
            if any(
                    tuple(state[k].shape) != tuple(expected[k].shape) or state[k].dtype != expected[k].dtype
                    or not bool(torch.isfinite(state[k]).all()) for k in expected):
                raise ValueError("ImageNet ResNet18 tensor keys, shapes, dtypes or values differ")
            source.load_state_dict(state, strict=True)
        if arm in ("flat", "position"):
            source.fc = nn.Linear(source.fc.in_features, num_classes if arm == "flat" else 15)
            model = source
        else:
            backbone = {name: value.detach().clone() for name, value in source.state_dict().items()
                        if name not in ("fc.weight", "fc.bias")}
            model = build_role_model("G" if arm == "G_sqrt" else arm,
                                     initialization_seed=seed, backbone_state=backbone)
    model.kit_arm = arm
    return model


def _rgb(root, row, image_hashes=None):
    contents = safe_child(root, row["image"]).read_bytes()
    if image_hashes is not None and hashlib.sha256(contents).hexdigest() != image_hashes[row["image"]]:
        raise ValueError("Image bytes changed after preflight")
    with Image.open(io.BytesIO(contents)) as source:
        source.load()
        if source.size != (row["width"], row["height"]):
            raise ValueError("Image dimensions differ from manifest")
        return source.convert("RGB")


def _transform(size, training):
    if type(size) is not int or size < 32:
        raise ValueError("Crop size must be an integer >=32")
    ops = [transforms.Resize((size, size))]
    if training:
        ops.append(transforms.ColorJitter(.2, .2, .1, .02))
    return transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class CropDataset(Dataset):
    """All TRAIN GT objects with item/epoch deterministic crop augmentation.

    Empty images correctly supply zero GT crops; the detector still consumes
    those negative frames. Original image path provenance is preserved.
    """
    def __init__(self, image_root, records, training=False, margin=.05, size=224,
                 seed=0, image_hashes=None):
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("Nonnegative crop margin required")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Invalid crop augmentation seed")
        self.root, self.records = Path(image_root), list(records)
        self.training, self.margin, self.seed = training, margin, seed
        self.image_hashes, self.epoch = image_hashes, 0
        self.transform = _transform(size, training)
        self.items = []
        for row_index, row in enumerate(self.records):
            for obj in row["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                if not (all(math.isfinite(v) for v in (x1, y1, x2, y2))
                        and 0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
                    raise ValueError("Crop targets must use valid prepared clipped geometry")
                self.items.append({"row_index": row_index, "image_id": row["image_id"],
                                   "image": row["image"], "bbox": list(obj["bbox"]),
                                   "class_id": obj["class_id"]})

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, index):
        item = self.items[index]
        image = _rgb(self.root, self.records[item["row_index"]], self.image_hashes)
        crop = crop_box(image, item["bbox"], self.margin)
        if self.training:
            key = f"crop_augmentation_v1:{self.seed}:{self.epoch}:{index}"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                tensor = self.transform(crop)
        else:
            tensor = self.transform(crop)
        return tensor, item["class_id"]


def presence_weights(train_class_ids, states, balance="none"):
    """Five TRAIN-only positive occupancy weights adapted from prior work.

    Only presence BCE is changed. Conditional helmet BCE remains unchanged.
    This reproduces the formula in train_presence_crops.py, not its old run.
    ``train_class_ids`` may be IDs, crop-item dictionaries, or frame records.
    """
    if balance not in ("none", "sqrt"):
        raise ValueError("balance must be none or sqrt")
    states = np.asarray(states.cpu() if isinstance(states, torch.Tensor) else states)
    if (states.ndim != 2 or states.shape[1] != 5 or len(states) < 1
            or not np.issubdtype(states.dtype, np.integer) or np.any((states < 0) | (states > 2))):
        raise ValueError("Expected class x five integer role states in {0,1,2}")
    ids = []
    for item in train_class_ids:
        if isinstance(item, dict) and "objects" in item:
            ids.extend(obj["class_id"] for obj in item["objects"])
        else:
            ids.append(item["class_id"] if isinstance(item, dict) else item)
    if not ids or any(not isinstance(cid, (int, np.integer)) or isinstance(cid, (bool, np.bool_))
                      or not 0 <= cid < len(states) for cid in ids):
        raise ValueError("Need nonempty valid TRAIN class IDs")
    present = states[np.asarray(ids, dtype=np.int64)] != 0
    positives = present.sum(axis=0).astype(np.int64)
    negatives = len(ids) - positives
    weights = np.ones(5, dtype=np.float64)
    if balance == "sqrt":
        supported = (positives > 0) & (negatives > 0)
        weights[supported] = np.clip(np.sqrt(negatives[supported] / positives[supported]), 1., 10.)
    normalizers = (negatives + weights * positives) / len(ids)
    return {"balance": balance, "scope": "TRAIN_crops_only", "sampled_crops": len(ids),
            "positions": ["D", "P1", "P2", "P3", "P0"],
            "positive_counts": positives.tolist(), "negative_counts": negatives.tolist(),
            "positive_weights": weights.tolist(), "occupancy_normalizers": normalizers.tolist(),
            "conditional_helmet_term_changed": False,
            "formula": "alpha=clip(sqrt(neg/pos),1,10), alpha=1 if pos*neg=0; Z=(neg+alpha*pos)/N; occupancy_BCE(pos_weight=alpha)/Z"}


def class_loss(logits, ids, states, arm, weighting=None):
    _arm(arm)
    if ids.ndim != 1 or ids.dtype != torch.int64 or len(ids) != len(logits) or len(ids) == 0:
        raise ValueError("Expected one int64 target per nonempty logit row")
    states = torch.as_tensor(states, device=logits.device, dtype=torch.int64)
    if states.ndim != 2 or states.shape[1] != 5 or bool(((states < 0) | (states > 2)).any()):
        raise ValueError("Invalid role states")
    if bool(((ids < 0) | (ids >= len(states))).any()):
        raise ValueError("Target outside class map")
    expected = len(states) if arm == "flat" else 15 if arm == "position" else 10
    if logits.ndim != 2 or logits.shape[1] != expected or not bool(torch.isfinite(logits).all()):
        raise ValueError("Invalid classifier logits")
    if arm == "flat":
        losses = nn.functional.cross_entropy(logits, ids, reduction="none")
    elif arm == "position":
        losses = nn.functional.cross_entropy(logits.reshape(-1, 3), states[ids].reshape(-1),
                                             reduction="none").reshape(-1, 5).mean(-1)
    else:
        target = states[ids]
        present = (target != 0).to(logits.dtype)
        helmet = (target == 1).to(logits.dtype)
        weights = torch.ones(5, device=logits.device, dtype=logits.dtype)
        normalizers = torch.ones_like(weights)
        if arm == "G_sqrt":
            if not isinstance(weighting, dict) or weighting.get("balance") != "sqrt":
                raise ValueError("G_sqrt needs occupancy weights computed exclusively from TRAIN")
            weights = torch.as_tensor(weighting["positive_weights"], device=logits.device, dtype=logits.dtype)
            normalizers = torch.as_tensor(weighting["occupancy_normalizers"], device=logits.device, dtype=logits.dtype)
            if (weights.shape != (5,) or normalizers.shape != (5,) or not bool(torch.isfinite(weights).all())
                    or not bool(torch.isfinite(normalizers).all()) or bool(((weights < 1) | (weights > 10)).any())
                    or bool((normalizers <= 0).any())):
                raise ValueError("Invalid five-role occupancy weights/normalizers")
        occupancy = nn.functional.binary_cross_entropy_with_logits(logits[:, :5], present,
                                                                  pos_weight=weights, reduction="none") / normalizers
        helmet_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 5:], helmet, reduction="none")
        losses = (occupancy + present * helmet_loss).mean(-1)
    if arm != "G_sqrt" and weighting is not None and weighting.get("balance", "none") != "none":
        raise ValueError("Matched role arms use unweighted loss; G_sqrt is separate")
    return losses.mean()


def decode(logits, arm, labels):
    _arm(arm)
    mode = arm if arm in ("flat", "position") else "binary10"
    return decode_logits(logits, mode, labels)[0]


def classifier_predictions(model, image_root, frames, detector_predictions, labels,
                           device, batch=64, margin=.05, size=224, score_thresh=.001,
                           nms_thresh=.5, max_detections=300, guard=None, image_hashes=None):
    """Reclassify frozen detector boxes using multi-class product scores.

    Identical geometry emitted for multiple detector classes is deduplicated;
    its proposal confidence is the maximum detector score for that exact box.
    This score is not claimed to be calibrated class-agnostic objectness.
    """
    if type(batch) is not int or batch < 1 or type(max_detections) is not int or max_detections < 1:
        raise ValueError("Positive inference batch and detection cap required")
    if not 0 <= score_thresh <= 1 or not 0 < nms_thresh <= 1 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid inference thresholds or margin")
    assert_aligned_frames(frames, detector_predictions, len(labels))
    predictions_by_id = {row["image_id"]: row for row in detector_predictions}
    arm = _arm(getattr(model, "kit_arm", getattr(model, "arm", None)))
    transform = _transform(size, False)
    model.eval()
    result = []
    with torch.inference_mode():
        for row in frames:
            _guard(guard)
            source = predictions_by_id[row["image_id"]]
            if source["image"] != row["image"]:
                raise ValueError("Proposal image provenance differs")
            unique = {}
            for obj in source["objects"]:
                box = tuple(float(v) for v in obj["bbox"])
                if not (0 <= box[0] < box[2] <= row["width"] and 0 <= box[1] < box[3] <= row["height"]):
                    raise ValueError("Proposal geometry must be clipped and nondegenerate")
                unique[box] = max(unique.get(box, 0.), obj["score"])
            boxes = list(unique)
            objects = []
            if boxes:
                image = _rgb(image_root, row, image_hashes)
                distributions = []
                for start in range(0, len(boxes), batch):
                    _guard(guard)
                    chunk = boxes[start:start + batch]
                    pixels = torch.stack([transform(crop_box(image, b, margin)) for b in chunk]).to(device)
                    distributions.append(decode(model(pixels), arm, labels))
                q = np.concatenate(distributions)
                if q.shape != (len(boxes), len(labels)) or not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1), 1., atol=1e-5):
                    raise ValueError("Classifier decoder returned invalid distributions")
                scores = q * np.asarray(list(unique.values()))[:, None]
                box_ids, class_ids = np.nonzero(scores > score_thresh)
                if len(box_ids):
                    candidate_boxes = torch.tensor(np.asarray(boxes)[box_ids], dtype=torch.float32)
                    candidate_scores = torch.tensor(scores[box_ids, class_ids], dtype=torch.float32)
                    candidate_classes = torch.tensor(class_ids, dtype=torch.int64)
                    keep = batched_nms(candidate_boxes, candidate_scores, candidate_classes, nms_thresh)[:max_detections]
                    for i in keep.tolist():
                        box_index, cid = int(box_ids[i]), int(class_ids[i])
                        objects.append({"bbox": list(boxes[box_index]), "class_id": cid,
                                        "score": float(candidate_scores[i]),
                                        "detector_score": float(unique[boxes[box_index]])})
            result.append({**{key: value for key, value in row.items() if key != "objects"}, "objects": objects})
            _guard(guard)
    validate_frames(result, len(labels), prediction=True)
    return result
