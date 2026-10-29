"""Portable, hash-bound HELMET manifests and streaming JPEG preflight.

No downloads or training occur here.  ``prepare_manifests`` is deliberately
specific to the approved split; the loader also supports small synthetic kits.
Training loads only TRAIN/VAL.  TEST annotations are opened only when explicitly
requested, including the one-time preparation and integrity audit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


SCHEMA = "helmet2020.portable_dataset.v1"
CLASSES_SHA256 = "d0560a66226362c6cd3ad2fa03853db54a7b9b6d4c552f5ab1fea1114e4e9ca4"
TEST_SHA256 = "aecb6d78f4da0b26d24ca0a09b335c8ad2f6f47c38f204d09ba9d6d85dc3daf0"
RESPLIT_SHA256 = {
    "train.jsonl": "8cdd05f75faff5aeff71fe753b632265cc3dd1f9e75140c9eda9ef9dd84c6de3",
    "val.jsonl": "40f98a44aa944e57277f8b60bc397d0db35164dedae31d8f8697e34c8f524c8b",
    "classes.json": CLASSES_SHA256,
}
ORIGINAL_SOURCE_SHA256 = {
    "train.jsonl": "0a44cd8013100489f4cf60e24f05cd5709c8f3b8cb950f6181448eff8b2cb788",
    "val.jsonl": "d196d2b198efd4ecd4de3f710e137e8dc23176b4edb8c5006190268a63c332eb",
    "classes.json": CLASSES_SHA256,
}
SPLITS = ("train", "val", "test")
IMAGE_RESERVE_BYTES = 1024 ** 3
_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class DataValidationError(ValueError):
    """Invalid or unbound data; callers must not continue with partial data."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise DataValidationError(f"Expected JSON object: {path}")
    return obj


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _verify_binding(path: Path, binding: dict[str, Any]) -> None:
    if not path.is_file():
        raise DataValidationError(f"Missing file: {path}")
    actual = _binding(path)
    if actual != {key: binding.get(key) for key in ("bytes", "sha256")}:
        raise DataValidationError(f"Size/SHA256 mismatch: {path}")


def _safe_relative(value: Any, *, image: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise DataValidationError(f"Unsafe relative path: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or any("_v2" in p.lower() for p in parts):
        raise DataValidationError(f"Unsafe or excluded relative path: {value!r}")
    result = PurePosixPath(value)
    if result.is_absolute():
        raise DataValidationError(f"Absolute paths are not portable: {value!r}")
    if image and (len(parts) != 3 or parts[0] != "images" or parts[1] not in SPLITS
                  or result.suffix.lower() not in (".jpg", ".jpeg")):
        raise DataValidationError(f"Expected images/{{train,val,test}}/name.jpg: {value!r}")
    return result


def _under(root: Path, relative: str, *, image: bool = False) -> Path:
    rel = _safe_relative(relative, image=image)
    candidate = root.joinpath(*rel.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except (ValueError, OSError) as exc:
        raise DataValidationError(f"Path escapes data root: {relative}") from exc
    return candidate


def _classes(value: dict[str, Any]) -> list[dict[str, Any]]:
    classes = value.get("classes")
    if not isinstance(classes, list) or not classes:
        raise DataValidationError("classes.json requires a nonempty classes list")
    if any(not isinstance(c, dict) or type(c.get("id")) is not int for c in classes):
        raise DataValidationError("Class IDs must be integers")
    if [c.get("id") if isinstance(c, dict) else None for c in classes] != list(range(len(classes))):
        raise DataValidationError("Class IDs must be unique, contiguous, zero-based and sorted")
    for c in classes:
        if not isinstance(c.get("name"), str) or not c["name"]:
            raise DataValidationError("Each class needs a name")
        states = c.get("states")
        if not isinstance(states, list) or len(states) != 5 or any(type(s) is not int or s not in (0, 1, 2) for s in states):
            raise DataValidationError("Each class needs five absent/helmet/no_helmet states")
    if len({tuple(c["states"]) for c in classes}) != len(classes):
        raise DataValidationError("Duplicate state vectors in class map")
    return classes


def _rows(path: Path, split: str, num_classes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise DataValidationError("Expected JSON object")
                for key in ("image_id", "clip_id", "site_id"):
                    if not isinstance(row.get(key), str) or not _ID.fullmatch(row[key]):
                        raise DataValidationError(f"Invalid {key}")
                frame = row.get("frame_index")
                if type(frame) is not int or not 1 <= frame <= 100:
                    raise DataValidationError("frame_index must be an integer in 1..100")
                if row["image_id"] != f"{row['clip_id']}__{frame:03d}":
                    raise DataValidationError("image_id does not match clip_id/frame_index")
                image = _safe_relative(row.get("image"), image=True)
                if image.stem != row["image_id"]:
                    raise DataValidationError("Image filename does not match image_id")
                if (split == "test") != (image.parts[1] == "test"):
                    raise DataValidationError("TEST image path mixed with development split")
                width, height = row.get("width"), row.get("height")
                if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                    raise DataValidationError("Dimensions must be positive integers")
                if row["image_id"] in ids or row["image"] in paths:
                    raise DataValidationError("Duplicate image ID or path")
                objects = row.get("objects")
                if not isinstance(objects, list):
                    raise DataValidationError("objects must be a list, including empty negative frames")
                tracks = set()
                for obj in objects:
                    if not isinstance(obj, dict):
                        raise DataValidationError("Invalid object")
                    cls, track, bbox = obj.get("class_id"), obj.get("track_id"), obj.get("bbox")
                    if type(cls) is not int or not 0 <= cls < num_classes:
                        raise DataValidationError("Out-of-range class_id")
                    if not isinstance(track, str) or not track or track in tracks:
                        raise DataValidationError("Missing or duplicate track_id in frame")
                    tracks.add(track)
                    if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
                        raise DataValidationError("bbox must contain four finite numbers")
                    x1, y1, x2, y2 = bbox
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise DataValidationError("bbox outside image or degenerate geometry")
                ids.add(row["image_id"])
                paths.add(row["image"])
                rows.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise DataValidationError(f"{path.name}:{number}: {exc}") from exc
    if not rows:
        raise DataValidationError(f"Empty manifest: {path}")
    return rows


def _counts(rows: list[dict[str, Any]], num_classes: int) -> dict[str, Any]:
    support = Counter(obj["class_id"] for row in rows for obj in row["objects"])
    return {
        "frames": len(rows), "clips": len({r["clip_id"] for r in rows}),
        "sites": len({r["site_id"] for r in rows}), "objects": sum(support.values()),
        "empty_frames": sum(not r["objects"] for r in rows),
        "clip_tracks": len({(r["clip_id"], o["track_id"]) for r in rows for o in r["objects"]}),
        "class_supports": [support[c] for c in range(num_classes)],
    }


def _disjoint(groups: dict[str, list[dict[str, Any]]]) -> None:
    seen = {key: {} for key in ("image_id", "image", "clip_id", "clip_track")}
    for split, rows in groups.items():
        for row in rows:
            values = {key: [row[key]] for key in ("image_id", "image", "clip_id")}
            values["clip_track"] = [(row["clip_id"], o["track_id"]) for o in row["objects"]]
            for key, entries in values.items():
                for value in entries:
                    previous = seen[key].setdefault(value, split)
                    if previous != split:
                        raise DataValidationError(f"Cross-split {key} overlap: {previous}/{split}: {value}")


def prepare_manifests(resplit_dir: str | Path, test_manifest: str | Path,
                      output: str | Path) -> dict[str, Any]:
    """Copy the pinned approved 58,200/14,600/18,200 split to a fresh kit.

    Returns the exact dataset.json object.  All source rows and source paths are
    copied byte for byte.  This verifies annotations, not the presence of JPEGs.
    """
    source, test, target = Path(resplit_dir), Path(test_manifest), Path(output)
    if target.exists():
        raise FileExistsError(f"Use a fresh manifest directory: {target}")
    receipt_binding = _binding(source / "completion.json")
    receipt = _read_json(source / "completion.json")
    if (receipt.get("status") != "metadata_split_verified" or receipt.get("seed") != 20261023
            or receipt.get("sources_unchanged") is not True
            or receipt.get("exact_source_rows_verified") != 72800
            or receipt.get("total_objects") != 228848
            or receipt.get("input_sha256") != ORIGINAL_SOURCE_SHA256):
        raise DataValidationError("Resplit completion receipt does not bind the approved source")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or not set(RESPLIT_SHA256).issubset(outputs):
        raise DataValidationError("Incomplete resplit output receipt")
    for name, expected in outputs.items():
        _verify_binding(_under(source, name), expected)
    for name, digest in RESPLIT_SHA256.items():
        if outputs[name]["sha256"] != digest:
            raise DataValidationError(f"Unapproved resplit source: {name}")
    if sha256_file(test) != TEST_SHA256:
        raise DataValidationError("Original TEST manifest SHA256 differs from the pinned source")
    classes = _classes(_read_json(source / "classes.json"))
    if len(classes) != 36:
        raise DataValidationError("Expected original 36-class map")
    sources = {"train": source / "train.jsonl", "val": source / "val.jsonl", "test": test}
    groups = {split: _rows(path, split, len(classes)) for split, path in sources.items()}
    _disjoint(groups)
    counts = {split: _counts(rows, len(classes)) for split, rows in groups.items()}
    for split, expected in {"train": (58200, 582), "val": (14600, 146), "test": (18200, 182)}.items():
        if (counts[split]["frames"], counts[split]["clips"]) != expected:
            raise DataValidationError(f"Incorrect approved population: {split}")
        frame_counts = Counter(r["clip_id"] for r in groups[split])
        if set(frame_counts.values()) != {100}:
            raise DataValidationError(f"Incomplete clip in {split}")
    if counts["train"]["objects"] + counts["val"]["objects"] != 228848 or counts["test"]["objects"] != 54529:
        raise DataValidationError("Annotation population changed")
    policy = _read_json(source / "split_policy.json")
    dataset: dict[str, Any] = {
        "schema": SCHEMA, "created_utc": _now(), "status": "metadata_verified_pixels_pending",
        "num_classes": 36, "files": {}, "splits": counts,
        "source_receipt": {"name": "source_completion.json", **receipt_binding},
        "protocol": {
            "name": "HELMET2020_trainval_resplit_20261023_v1_plus_original_test",
            "trainval_policy": policy, "test_role": "untouched model-selection holdout; evaluate only after frozen selection",
            "test_used_for_split_or_training": False, "test_annotations_audited_during_preparation": True,
            "split_overlap_checked": ["image_id", "image", "clip_id", "clip_track"],
            "shared_sites_allowed": True, "no_unseen_camera_or_person_claim": True,
            "image_paths_preserved": True, "all_negative_frames_preserved": True,
            "initialization": "Generic external pretraining; no checkpoint previously trained on any new VAL or TEST image",
            "comparison_scope": "Run every comparator on the same new split; not interchangeable with published/original-split AP",
            "pixels_verified": False,
        },
    }
    target.mkdir(parents=True, exist_ok=False)
    for split, path in sources.items():
        name = f"{split}.jsonl"
        shutil.copyfile(path, target / name)
        dataset["files"][split] = {"path": name, **_binding(target / name)}
        if dataset["files"][split]["sha256"] != (TEST_SHA256 if split == "test" else RESPLIT_SHA256[name]):
            raise DataValidationError(f"Source changed while copying {split}")
    shutil.copyfile(source / "classes.json", target / "classes.json")
    dataset["files"]["classes"] = {"path": "classes.json", **_binding(target / "classes.json")}
    if dataset["files"]["classes"]["sha256"] != CLASSES_SHA256:
        raise DataValidationError("Class map changed while copying")
    shutil.copyfile(source / "completion.json", target / "source_completion.json")
    _verify_binding(target / "source_completion.json", dataset["source_receipt"])
    # This also catches edits to supporting policy/provenance while preparing.
    for name, expected in outputs.items():
        _verify_binding(_under(source, name), expected)
    _verify_binding(source / "completion.json", receipt_binding)
    _write_json(target / "dataset.json", dataset)
    return dataset


def dataset_identity(manifest_dir: str | Path) -> dict[str, Any]:
    """Read only dataset.json. Does NOT read TEST labels or pixels.

    The returned hash binds the entire lock. Call load_manifest/check_pixels for
    verification of actual selected files; identity alone is not verification.
    """
    path = Path(manifest_dir) / "dataset.json"
    value = _read_json(path)
    if value.get("schema") != SCHEMA:
        raise DataValidationError("Unsupported dataset.json schema")
    if type(value.get("num_classes")) is not int or value["num_classes"] < 1:
        raise DataValidationError("Invalid num_classes")
    if not isinstance(value.get("files"), dict) or not {"classes", *SPLITS}.issubset(value["files"]):
        raise DataValidationError("dataset.json lacks required files")
    for name, info in value["files"].items():
        if not isinstance(info, dict):
            raise DataValidationError(f"Invalid file binding: {name}")
        _safe_relative(info.get("path"))
        if not isinstance(info.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"]):
            raise DataValidationError(f"Invalid SHA256 binding: {name}")
        if type(info.get("bytes")) is not int or info["bytes"] < 0:
            raise DataValidationError(f"Invalid size binding: {name}")
    result = copy.deepcopy(value)
    result["dataset_json_sha256"] = sha256_file(path)
    return result


def load_manifest(manifest_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Verify class-map + requested manifest and return validated raw row objects."""
    if split not in SPLITS:
        raise DataValidationError(f"Unknown split: {split!r}")
    root = Path(manifest_dir)
    identity = dataset_identity(root)
    info = identity["files"]
    for key in ("classes", split):
        _verify_binding(_under(root, info[key]["path"]), info[key])
    classes = _classes(_read_json(_under(root, info["classes"]["path"])))
    if len(classes) != identity["num_classes"]:
        raise DataValidationError("Class-map size differs from dataset.json")
    rows = _rows(_under(root, info[split]["path"]), split, len(classes))
    if _counts(rows, len(classes)) != identity.get("splits", {}).get(split):
        raise DataValidationError(f"{split} population differs from dataset.json")
    return rows


def check_pixels(manifest_dir: str | Path, image_root: str | Path,
                 splits: Iterable[str] = ("train", "val"),
                 output: str | Path | None = None,
                 guard: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Stream each selected JPEG, verify decode + dimensions, bind its full SHA.

    ``image_root`` contains the preserved ``images/train``, ``images/val`` and
    optional ``images/test`` paths. Output is a fresh JSON receipt *file*. No
    JPEG bytes are retained between images. Cross-split identical JPEGs fail.
    A saved receipt identifies the scan and cannot establish future immutability.
    """
    from PIL import Image

    if guard is not None:
        guard()
    requested = tuple(splits)
    if not requested or len(requested) != len(set(requested)) or any(s not in SPLITS for s in requested):
        raise DataValidationError("Choose one or more distinct train/val/test splits")
    dest = Path(output) if output is not None else None
    if dest is not None and dest.exists():
        raise FileExistsError(f"Use a fresh pixel receipt: {dest}")
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise DataValidationError(f"Image root does not exist: {root}")
    identity = dataset_identity(manifest_dir)
    groups = {split: load_manifest(manifest_dir, split) for split in requested}
    _disjoint(groups)
    if guard is not None:
        guard()
    result: dict[str, Any] = {
        "schema": "helmet2020.pixel_preflight.v1", "created_utc": _now(),
        "dataset_json_sha256": identity["dataset_json_sha256"],
        "image_root": str(root), "splits": {}, "files": {},
        "classes_sha256": identity["files"]["classes"]["sha256"],
        "manifest_sha256": {split: identity["files"][split]["sha256"] for split in requested},
        "test_pixels_read": "test" in requested, "status": "checking",
    }
    seen: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    invalid: list[str] = []
    missing_count = invalid_count = processed = 0
    total = sum(len(rows) for rows in groups.values())
    for split, rows in groups.items():
        totals = {"frames": len(rows), "verified_frames": 0, "bytes": 0, "same_split_identical_jpegs": 0}
        for row in rows:
            # Run outside image-error handling: deadline/cancellation must abort,
            # never be swallowed and counted as a bad JPEG.
            if guard is not None:
                guard()
            processed += 1
            if processed % 1000 == 0:
                print(f"pixel preflight {processed}/{total} frames", flush=True)
            path = _under(root, row["image"], image=True)
            if not path.is_file():
                missing_count += 1
                if len(missing) < 8:
                    missing.append(row["image"])
                continue
            try:
                before = path.stat()
                binding = _binding(path)
                with Image.open(path) as img:
                    if img.format != "JPEG" or img.size != (row["width"], row["height"]):
                        raise DataValidationError(f"Not JPEG or wrong dimensions: {row['image']}")
                    img.verify()
                # Pillow.verify does not decode JPEG payload. load catches truncated
                # files while holding at most one frame's pixels in memory.
                with Image.open(path) as img:
                    img.load()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise DataValidationError(f"Image changed during validation: {row['image']}")
                previous = seen.get(binding["sha256"])
                if previous and previous[0] != split:
                    raise DataValidationError(f"Cross-split identical JPEG: {previous[1]} and {row['image']}")
                if previous:
                    totals["same_split_identical_jpegs"] += 1
                else:
                    seen[binding["sha256"]] = (split, row["image"])
                result["files"][row["image"]] = {"split": split, **binding, "width": row["width"], "height": row["height"]}
                totals["verified_frames"] += 1
                totals["bytes"] += binding["bytes"]
            except (OSError, ValueError) as exc:
                invalid_count += 1
                if len(invalid) < 8:
                    invalid.append(f"{row['image']}: {exc}")
        result["splits"][split] = totals
    if guard is not None:
        guard()
    if missing_count or invalid_count:
        raise DataValidationError(f"Pixel preflight failed: {missing_count} missing, {invalid_count} invalid. "
                                  f"First missing: {missing}; first invalid: {invalid}")
    # Ensure metadata did not change during a potentially long scan.
    if dataset_identity(manifest_dir)["dataset_json_sha256"] != identity["dataset_json_sha256"]:
        raise DataValidationError("dataset.json changed during pixel preflight")
    for name in ("classes", *requested):
        _verify_binding(_under(Path(manifest_dir), identity["files"][name]["path"]), identity["files"][name])
    if guard is not None:
        guard()
    result["status"] = "verified"
    result["total_frames"] = sum(v["verified_frames"] for v in result["splits"].values())
    result["total_bytes"] = sum(v["bytes"] for v in result["splits"].values())
    result["completed_utc"] = _now()
    result["reusable_only_if_pixel_bytes_unchanged"] = True
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_json(dest, result)
    return result


def _reject_link(path: Path) -> None:
    """Reject Unix links and Windows reparse points without following either."""
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise DataValidationError(f"Symlink/reparse source or destination is not allowed: {path}")


def _reject_link_ancestors(path: Path) -> None:
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            _reject_link(item)


def _validate_jpeg(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    _reject_link(path)
    before = path.stat()
    result = _binding(path)
    with Image.open(path) as image:
        if image.format != "JPEG" or image.size != (row["width"], row["height"]):
            raise DataValidationError(f"Not JPEG or wrong dimensions: {path.name}")
        image.verify()
    with Image.open(path) as image:
        image.load()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise DataValidationError(f"Image changed during validation: {path}")
    return result


def stage_images(manifest_dir: str | Path, source_root: str | Path,
                 image_root: str | Path,
                 splits: Iterable[str] = ("train", "val", "test"),
                 mode: str = "copy", guard: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Import existing local JPEGs into a fresh portable image root.

    Match exact basenames recursively under the explicitly supplied source root.
    Ambiguous/missing matches fail before copying.  Copy is the only supported
    mode: no originals are removed, linked or changed.  Every staged file is
    decoded, dimension checked and SHA256 bound before one final directory rename.
    A later failure retains its private staging directory and a failed receipt;
    the requested final image_root does not appear as a completed dataset.
    """
    if guard is not None:
        guard()
    if mode != "copy":
        raise DataValidationError("Only mode='copy' is supported; originals must remain independent")
    requested = tuple(splits)
    if not requested or len(requested) != len(set(requested)) or any(s not in SPLITS for s in requested):
        raise DataValidationError("Choose one or more distinct train/val/test splits")
    source, destination = Path(source_root).absolute(), Path(image_root).absolute()
    _reject_link_ancestors(source)
    _reject_link_ancestors(destination)
    if not source.is_dir():
        raise DataValidationError(f"Source image directory does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Use a fresh image_root: {destination}")
    source = source.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise DataValidationError("image_root must be outside source_root")
    identity = dataset_identity(manifest_dir)
    groups = {split: load_manifest(manifest_dir, split) for split in requested}
    _disjoint(groups)
    expected = {PurePosixPath(row["image"]).name: (split, row)
                for split, rows in groups.items() for row in rows}
    if len(expected) != sum(len(rows) for rows in groups.values()):
        raise DataValidationError("Ambiguous image basenames in manifests")
    matches: dict[str, list[Path]] = {name: [] for name in expected}

    def walk_error(error: OSError) -> None:
        raise DataValidationError(f"Cannot enumerate source tree: {error}") from error

    for directory, dirs, names in os.walk(source, followlinks=False, onerror=walk_error):
        if guard is not None:
            guard()
        directory_path = Path(directory)
        _reject_link(directory_path)
        for child in sorted(dirs + names):
            _reject_link(directory_path / child)
        for name in sorted(names):
            if name in matches:
                candidate = directory_path / name
                if not candidate.is_file():
                    raise DataValidationError(f"Nonregular matching image: {candidate}")
                matches[name].append(candidate)
    missing = [name for name, paths in matches.items() if not paths]
    ambiguous = [name for name, paths in matches.items() if len(paths) > 1]
    if missing or ambiguous:
        raise DataValidationError(f"Image import inventory failed: {len(missing)} missing, {len(ambiguous)} ambiguous. "
                                  f"First missing: {missing[:8]}; first ambiguous: {ambiguous[:8]}")
    required = sum(paths[0].stat().st_size for paths in matches.values())
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_ancestors(destination.parent)
    free = shutil.disk_usage(destination.parent).free
    if free < required + IMAGE_RESERVE_BYTES:
        raise DataValidationError(f"Insufficient space: need {required} JPEG bytes + {IMAGE_RESERVE_BYTES} reserve, "
                                  f"available {free}, deficit {required + IMAGE_RESERVE_BYTES - free}")
    staging = destination.parent / (".kit-images-staging-" + uuid.uuid4().hex)
    staging.mkdir(exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "helmet2020.image_import.v1", "status": "staging", "created_utc": _now(),
        "source_root": str(source), "image_root": str(destination), "mode": mode,
        "dataset_json_sha256": identity["dataset_json_sha256"],
        "classes_sha256": identity["files"]["classes"]["sha256"],
        "manifest_sha256": {split: identity["files"][split]["sha256"] for split in requested},
        "expected_frames": len(expected), "staged_frames": 0, "total_bytes": 0,
        "reserve_bytes": IMAGE_RESERVE_BYTES, "free_bytes_before": free,
        "files": {}, "originals_modified": False,
    }
    sha_owners: dict[str, tuple[str, str]] = {}
    try:
        for number, (name, (split, row)) in enumerate(expected.items(), 1):
            if guard is not None:
                guard()
            source_file = matches[name][0]
            _reject_link_ancestors(source_file)
            source_binding = _validate_jpeg(source_file, row)
            old = sha_owners.get(source_binding["sha256"])
            if old is not None and old[0] != split:
                raise DataValidationError(f"Cross-split identical JPEG: {old[1]} and {row['image']}")
            sha_owners.setdefault(source_binding["sha256"], (split, row["image"]))
            target = _under(staging, row["image"], image=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents replacing anything, including partials.
            with source_file.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            _verify_binding(target, source_binding)
            _reject_link_ancestors(source_file)
            if _binding(source_file) != source_binding:
                raise DataValidationError(f"Source image changed during copy: {source_file}")
            receipt["files"][row["image"]] = {
                "source_relative": source_file.relative_to(source).as_posix(),
                "split": split, **source_binding, "width": row["width"], "height": row["height"],
            }
            receipt["staged_frames"] += 1
            receipt["total_bytes"] += source_binding["bytes"]
            if number % 1000 == 0:
                print(f"image import {number}/{len(expected)} frames", flush=True)
        if guard is not None:
            guard()
        if dataset_identity(manifest_dir)["dataset_json_sha256"] != identity["dataset_json_sha256"]:
            raise DataValidationError("dataset.json changed during image import")
        for name in ("classes", *requested):
            _verify_binding(_under(Path(manifest_dir), identity["files"][name]["path"]), identity["files"][name])
        if shutil.disk_usage(destination.parent).free < IMAGE_RESERVE_BYTES:
            raise DataValidationError("Disk reserve was consumed during image import")
        if guard is not None:
            guard()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Destination appeared during import: {destination}")
        receipt["status"] = "verified"
        receipt["completed_utc"] = _now()
        _write_json(staging / "image_import_receipt.json", receipt)
        # Same filesystem, fresh final name. Both source and destination stay local.
        staging.rename(destination)
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["failed_utc"] = _now()
        receipt["partial_directory"] = str(staging)
        failed_path = staging / "image_import_failed.json"
        try:
            if staging.is_dir():
                # A failed final rename must not leave a success receipt.
                pending_receipt = staging / "image_import_receipt.json"
                if pending_receipt.exists():
                    pending_receipt.unlink()
                if not failed_path.exists():
                    _write_json(failed_path, receipt)
        except OSError:
            # Preserve the original failure (including out-of-space/cancel).
            pass
        if hasattr(exc, "add_note"):
            exc.add_note(f"Image import incomplete; partial data retained at {staging}")
        raise
    return receipt
