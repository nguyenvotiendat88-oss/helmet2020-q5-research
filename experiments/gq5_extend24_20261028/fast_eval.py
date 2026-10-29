"""Read-only performance candidate for frozen classifier proposal rescoring.

This module intentionally lives outside HELMET2020_training_kit_v1. It does not
change the study recipe or monkeypatch the active runner. The default path keeps
the original PIL crop/resize and per-crop ToTensor/Normalize operations, while
prefetching up to a bounded number of frames in CPU threads. Forward chunk
boundaries, frame order, decoder, score multiplication, NMS, and output order are
kept equal to kit_models.classifier_predictions.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torchvision import transforms


@dataclass(frozen=True)
class _FramePlan:
    row: Mapping
    boxes: tuple[tuple[float, float, float, float], ...]
    detector_scores: tuple[float, ...]


def _unique_boxes(row: Mapping, source: Mapping) -> _FramePlan:
    """Match the original per-frame geometry/deduplication exactly."""
    if source["image"] != row["image"]:
        raise ValueError("Proposal image provenance differs")
    unique: dict[tuple[float, float, float, float], float] = {}
    for obj in source["objects"]:
        box = tuple(float(v) for v in obj["bbox"])
        if not (0 <= box[0] < box[2] <= row["width"] and
                0 <= box[1] < box[3] <= row["height"]):
            raise ValueError("Proposal geometry must be clipped and nondegenerate")
        unique[box] = max(unique.get(box, 0.0), obj["score"])
    return _FramePlan(row=row, boxes=tuple(unique), detector_scores=tuple(unique.values()))


def _prepare_resized(
    image_root,
    plan: _FramePlan,
    margin: float,
    resize,
    image_hashes,
    rgb_loader: Callable,
    cropper: Callable,
) -> list:
    """Worker-only operation: source read, hash check, crop and PIL resize."""
    image = rgb_loader(image_root, plan.row, image_hashes)
    return [resize(cropper(image, box, margin)) for box in plan.boxes]


def _exact_main_tensorize(resized: Sequence, to_tensor, normalize) -> torch.Tensor:
    """Preserve original transform order: ToTensor then Normalize per crop."""
    # The output is stacked once for the unchanged model call. Each crop still
    # follows the original ToTensor -> Normalize order, giving a strict parity
    # mode while JPEG/crop/resize work runs ahead in worker threads.
    return torch.stack([normalize(to_tensor(crop)) for crop in resized])


def _batched_tensorize(resized: Sequence, to_tensor, mean, std) -> torch.Tensor:
    """Optional faster CPU path; use only after exact parity validation."""
    # All crops are RGB PIL images, so their pixels are uint8. This preserves
    # ToTensor's elementwise cast/divide followed by Normalize's subtract/divide.
    array = np.stack([np.array(crop, dtype=np.uint8, copy=True) for crop in resized])
    pixels = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    pixels = pixels.to(dtype=torch.get_default_dtype()).div(255)
    return pixels.sub(mean).div(std)


def classifier_predictions_prefetch(
    model,
    image_root,
    frames,
    detector_predictions,
    labels,
    device,
    *,
    batch=64,
    margin=.05,
    size=224,
    score_thresh=.001,
    nms_thresh=.5,
    max_detections=300,
    guard=None,
    image_hashes=None,
    workers=4,
    prefetch_frames=8,
    kit_models_module=None,
    batched_cpu=False,
    progress_callback=None,
):
    """Proposal rescoring candidate with bounded CPU prefetch.

    ``kit_models_module`` is passed explicitly so the candidate does not import
    or patch the active study. It must provide ``_rgb``, ``crop_box``, ``_arm``,
    ``decode`` and ``_guard`` plus ``validate_frames``/``assert_aligned_frames``
    and ``batched_nms``. In normal use pass the imported kit_models module.

    The default ``batched_cpu=False`` is strict-parity mode. Set
    ``batched_cpu=True`` only after ``compare_preprocessing_exact`` and a small
    end-to-end equivalence test pass.
    """
    if kit_models_module is None:
        raise ValueError("Pass the study kit_models module explicitly")
    km = kit_models_module
    if type(batch) is not int or batch < 1 or type(max_detections) is not int or max_detections < 1:
        raise ValueError("Positive inference batch and detection cap required")
    if type(workers) is not int or not 1 <= workers <= 8 or type(prefetch_frames) is not int or not 1 <= prefetch_frames <= 8:
        raise ValueError("workers and prefetch_frames must be integers from 1 to 8")
    if not 0 <= score_thresh <= 1 or not 0 < nms_thresh <= 1 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid inference thresholds")
    if type(size) is not int or size < 32:
        raise ValueError("Crop size must be an integer >=32")
    km.assert_aligned_frames(frames, detector_predictions, len(labels))
    predictions_by_id = {row["image_id"]: row for row in detector_predictions}
    arm = km._arm(getattr(model, "kit_arm", getattr(model, "arm", None)))
    resize = transforms.Resize((size, size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    mean = torch.tensor([.485, .456, .406], dtype=torch.get_default_dtype()).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=torch.get_default_dtype()).view(1, 3, 1, 1)
    model.eval()
    result = []

    def plans() -> Iterable[_FramePlan]:
        for row in frames:
            km._guard(guard)
            source = predictions_by_id[row["image_id"]]
            yield _unique_boxes(row, source)

    plan_iter = iter(plans())
    pending: deque[tuple[_FramePlan, Future | None]] = deque()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="helmet-prefetch") as pool, torch.inference_mode():
        def fill_queue() -> None:
            while len(pending) < prefetch_frames:
                try:
                    plan = next(plan_iter)
                except StopIteration:
                    return
                future = None
                if plan.boxes:
                    future = pool.submit(
                        _prepare_resized, image_root, plan, margin, resize,
                        image_hashes, km._rgb, km.crop_box,
                    )
                pending.append((plan, future))

        fill_queue()
        while pending:
            km._guard(guard)
            plan, future = pending.popleft()
            resized = future.result() if future is not None else []
            objects = []
            if plan.boxes:
                distributions = []
                # Preserve the original per-frame forward chunk boundaries.
                for start in range(0, len(plan.boxes), batch):
                    km._guard(guard)
                    chunk = resized[start:start + batch]
                    if batched_cpu:
                        pixels_cpu = _batched_tensorize(chunk, to_tensor, mean, std)
                    else:
                        pixels_cpu = _exact_main_tensorize(chunk, to_tensor, normalize)
                    pixels = pixels_cpu.to(device)
                    distributions.append(km.decode(model(pixels), arm, labels))
                q = np.concatenate(distributions)
                if (q.shape != (len(plan.boxes), len(labels)) or
                        not np.isfinite(q).all() or (q < 0).any() or
                        not np.allclose(q.sum(1), 1., atol=1e-5)):
                    raise ValueError("Classifier decoder returned invalid distributions")
                scores = q * np.asarray(plan.detector_scores)[:, None]
                box_ids, class_ids = np.nonzero(scores > score_thresh)
                if len(box_ids):
                    candidate_boxes = torch.tensor(np.asarray(plan.boxes)[box_ids], dtype=torch.float32)
                    candidate_scores = torch.tensor(scores[box_ids, class_ids], dtype=torch.float32)
                    candidate_classes = torch.tensor(class_ids, dtype=torch.int64)
                    keep = km.batched_nms(candidate_boxes, candidate_scores, candidate_classes, nms_thresh)[:max_detections]
                    for i in keep.tolist():
                        box_index, cid = int(box_ids[i]), int(class_ids[i])
                        objects.append({"bbox": list(plan.boxes[box_index]), "class_id": cid,
                                        "score": float(candidate_scores[i]),
                                        "detector_score": float(plan.detector_scores[box_index])})
            result.append({**{key: value for key, value in plan.row.items() if key != "objects"}, "objects": objects})
            if progress_callback is not None and (len(result) % 250 == 0 or len(result) == len(frames)):
                progress_callback(len(result), len(frames))
            km._guard(guard)
            # Retain at most prefetch_frames frames including the active one.
            if plan.boxes:
                del chunk, pixels, pixels_cpu, distributions, q
            del resized
            future = None
            fill_queue()
    km.validate_frames(result, len(labels), prediction=True)
    return result


def compare_preprocessing_exact(crops: Sequence, *, size=224, batch=64) -> dict:
    """Compare both CPU candidates with original transforms and Torch RNG."""
    resize = transforms.Resize((size, size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    exact = []
    originals, resized = [], []
    before_rng = torch.random.get_rng_state().clone()
    for crop in crops:
        scaled = resize(crop)
        original = normalize(to_tensor(scaled))
        candidate = _exact_main_tensorize([scaled], to_tensor, normalize)[0]
        exact.append(bool(torch.equal(original, candidate)))
        originals.append(original)
        resized.append(scaled)
    batched_equal = []
    mean = torch.tensor([.485, .456, .406], dtype=torch.get_default_dtype()).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=torch.get_default_dtype()).view(1, 3, 1, 1)
    for start in range(0, len(resized), batch):
        expected = torch.stack(originals[start:start+batch])
        candidate = _batched_tensorize(resized[start:start+batch], to_tensor, mean, std)
        batched_equal.append(bool(torch.equal(expected, candidate)))
    return {"count": len(exact), "all_bitwise_equal": all(exact), "equal": exact,
            "batched_all_bitwise_equal": all(batched_equal),
            "rng_unchanged": bool(torch.equal(before_rng, torch.random.get_rng_state()))}


def compare_outputs(baseline_fn: Callable, fast_fn: Callable, *, kwargs: Mapping) -> dict:
    """Run old/new functions once and require exact serialized output equality."""
    baseline = baseline_fn(**kwargs)
    fast = fast_fn(**kwargs)
    equal = baseline == fast
    return {"equal": equal, "baseline_frames": len(baseline), "fast_frames": len(fast),
            "baseline": baseline if not equal else None, "fast": fast if not equal else None}
