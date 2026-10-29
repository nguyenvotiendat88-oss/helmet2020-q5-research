from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def safe_child(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {name}")
    return path


def validate_frames(records, num_classes, prediction=False):
    seen = set()
    for r in records:
        key = r["image_id"]
        if key in seen:
            raise ValueError(f"Duplicate frame: {key}")
        seen.add(key)
        tracks = set()
        for obj in r["objects"]:
            cid = obj["class_id"]
            if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < num_classes:
                raise ValueError(f"Invalid class id: {cid}")
            b = obj["bbox"]
            if len(b) != 4 or not all(math.isfinite(x) for x in b) or b[2] <= b[0] or b[3] <= b[1]:
                raise ValueError(f"Invalid xyxy box in {key}: {b}")
            if prediction and not (math.isfinite(obj["score"]) and 0 <= obj["score"] <= 1):
                raise ValueError(f"Invalid confidence in {key}")
            tid = obj.get("track_id")
            if tid is not None:
                if tid in tracks:
                    raise ValueError(f"Duplicate track in frame: {key}/{tid}")
                tracks.add(tid)
    return seen


def assert_aligned_frames(gt, pred, num_classes):
    """A shared image ID alone does not establish shared coordinates or clip identity."""
    keys = validate_frames(gt, num_classes)
    if validate_frames(pred, num_classes, True) != keys:
        raise ValueError("Prediction frames must exactly match ground truth, including empty frames")
    lookup = {r["image_id"]: r for r in gt}
    for row in pred:
        reference = lookup[row["image_id"]]
        for field in ("clip_id", "frame_index", "width", "height", "site_id"):
            if reference.get(field) != row.get(field):
                raise ValueError(f"Frame metadata mismatch: {row['image_id']}/{field}")
    return keys
