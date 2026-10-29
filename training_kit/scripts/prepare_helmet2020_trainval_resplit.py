"""Prepare one fixed, clip-grouped development split; metadata only, no model scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEED = 20261023
ASSEMBLY = Path("runs/clore_backups_order2302991_assembly_20261017T022100Z/assembly_metadata_v1")
PINNED = {
    "train.jsonl": "0a44cd8013100489f4cf60e24f05cd5709c8f3b8cb950f6181448eff8b2cb788",
    "val.jsonl": "d196d2b198efd4ecd4de3f710e137e8dc23176b4edb8c5006190268a63c332eb",
    "classes.json": "d0560a66226362c6cd3ad2fa03853db54a7b9b6d4c552f5ab1fea1114e4e9ca4",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(clip):
    return hashlib.sha256(f"{SEED}:{clip}".encode("utf-8")).hexdigest(), clip


def assign_clips(clips):
    """Hamilton site quotas, rare-class coverage, then fixed hashed fill.

    Every supported class must retain a TRAIN clip. Singleton-class clips stay
    in TRAIN. A failure means this greedy procedure failed, not infeasibility.
    """
    by_site = defaultdict(list)
    by_class = defaultdict(set)
    for clip, info in sorted(clips.items()):
        by_site[info["site"]].append(clip)
        for cid in info["counts"]:
            by_class[cid].add(clip)
    n = len(clips)
    require(n > 1, "At least two clips required")
    target = (n + 2) // 5  # nearest integer to 20%; integer arithmetic
    require(0 < target < n, "Invalid validation target")
    quotas = {site: target * len(items) // n for site, items in by_site.items()}
    remainder_order = sorted(by_site, key=lambda s: (-(target * len(by_site[s]) % n), s))
    for site in remainder_order[:target - sum(quotas.values())]:
        quotas[site] += 1
    selected = set()
    site_count, class_count = Counter(), Counter()

    def allowed(clip):
        info = clips[clip]
        return (clip not in selected and site_count[info["site"]] < quotas[info["site"]]
                and all(class_count[cid] + 1 < len(by_class[cid]) for cid in info["counts"]))

    def add(clip):
        selected.add(clip)
        site_count[clips[clip]["site"]] += 1
        class_count.update(clips[clip]["counts"].keys())

    for cid in sorted(by_class, key=lambda c: (len(by_class[c]), c)):
        if len(by_class[cid]) == 1 or class_count[cid]:
            continue
        candidates = [clip for clip in sorted(by_class[cid], key=rank) if allowed(clip)]
        require(candidates, f"Fixed procedure did not achieve VAL coverage for class {cid}")
        add(candidates[0])
    for site in sorted(by_site):
        for clip in sorted(by_site[site], key=rank):
            if allowed(clip):
                add(clip)
        require(site_count[site] == quotas[site], f"Fixed procedure did not fill site quota: {site}")
    require(len(selected) == target, "VAL target not achieved")
    assignment = {clip: "val" if clip in selected else "train" for clip in sorted(clips)}
    return assignment, quotas


def load_inputs(source):
    for name, expected in PINNED.items():
        require(sha(source / name) == expected, f"Pinned input changed: {name}")
    classes = json.loads((source / "classes.json").read_text(encoding="utf-8"))
    require([r["id"] for r in classes["classes"]] == list(range(36)), "Unexpected class map")
    clips, row_hashes = {}, {}
    old_counts, old_objects = Counter(), Counter()
    for old_split in ("train", "val"):
        with (source / f"{old_split}.jsonl").open("rb") as f:
            for raw in f:
                r = json.loads(raw)
                iid, clip, site = r["image_id"], r["clip_id"], r["site_id"]
                require(iid not in row_hashes, f"Duplicate image_id: {iid}")
                require(r["image"] == f"images/{old_split}/{iid}.jpg" and "_v2" not in iid.lower(), "Disallowed image path")
                row_hashes[iid] = hashlib.sha256(raw).hexdigest()
                if clip not in clips:
                    clips[clip] = dict(site=site, old_split=old_split, frames=set(), counts=Counter(), empty=0, tracks=set())
                c = clips[clip]
                require(c["site"] == site and c["old_split"] == old_split, "Inconsistent clip provenance")
                require(r["frame_index"] not in c["frames"], "Duplicate clip/frame")
                c["frames"].add(r["frame_index"])
                c["empty"] += not r["objects"]
                frame_tracks = set()
                for o in r["objects"]:
                    require(type(o["class_id"]) is int and 0 <= o["class_id"] < 36, "Invalid class")
                    tid = str(o["track_id"])
                    require(tid not in frame_tracks, "Duplicate track within frame")
                    frame_tracks.add(tid)
                    c["tracks"].add(tid)
                    c["counts"][o["class_id"]] += 1
                old_counts[old_split] += 1
                old_objects[old_split] += len(r["objects"])
    require(old_counts == {"train": 63600, "val": 9200}, "Original frame population mismatch")
    require(old_objects == {"train": 199603, "val": 29245}, "Original annotation population mismatch")
    require(len(clips) == 728 and all(c["frames"] == set(range(1, 101)) for c in clips.values()), "Expected 728 complete 100-frame clips")
    return classes, clips, row_hashes


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def prepare(root, output):
    source = root / ASSEMBLY
    require(not output.exists(), "Output already exists; never overwrite a split")
    classes, clips, row_hashes = load_inputs(source)
    assignments, quotas = assign_clips(clips)
    output.mkdir(parents=True, exist_ok=False)
    (output / "classes.json").write_bytes((source / "classes.json").read_bytes())
    with (output / "train.jsonl").open("xb") as train, (output / "val.jsonl").open("xb") as val, (output / "image_assignments.csv").open("x", encoding="utf-8-sig", newline="") as mapping:
        writer = csv.writer(mapping)
        writer.writerow(["image_id", "image", "clip_id", "site_id", "original_split", "new_split"])
        handles = {"train": train, "val": val}
        for old_split in ("train", "val"):
            with (source / f"{old_split}.jsonl").open("rb") as f:
                for raw in f:
                    r = json.loads(raw)
                    new = assignments[r["clip_id"]]
                    handles[new].write(raw)
                    writer.writerow([r["image_id"], r["image"], r["clip_id"], r["site_id"], old_split, new])

    # Reopen outputs and independently audit byte identity, populations and overlap.
    seen, sets, summary = set(), {}, {}
    counts = {s: Counter() for s in ("train", "val")}
    for split in ("train", "val"):
        ids, clip_set, track_set = set(), set(), set()
        empty = objects = 0
        with (output / f"{split}.jsonl").open("rb") as f:
            for raw in f:
                r = json.loads(raw)
                iid, clip = r["image_id"], r["clip_id"]
                require(iid not in seen, "Duplicate output frame")
                require(row_hashes.get(iid) == hashlib.sha256(raw).hexdigest(), "Changed source row bytes")
                require(assignments[clip] == split, "Clip crossed split boundary")
                ids.add(iid); seen.add(iid); clip_set.add(clip)
                empty += not r["objects"]
                objects += len(r["objects"])
                for o in r["objects"]:
                    track_set.add((clip, str(o["track_id"])))
                    counts[split][o["class_id"]] += 1
        sets[split] = dict(ids=ids, clips=clip_set, tracks=track_set)
        summary[split] = dict(frames=len(ids), clips=len(clip_set), objects=objects, empty_frames=empty, tracks=len(track_set), supported_classes=sorted(counts[split]))
    require(seen == row_hashes.keys(), "Union differs from original TRAIN+VAL")
    for kind in ("ids", "clips", "tracks"):
        require(not sets["train"][kind] & sets["val"][kind], f"Cross-split overlap: {kind}")
    require(summary["train"]["frames"] == 58200 and summary["val"]["frames"] == 14600, "Unexpected new population")
    require(sum(s["objects"] for s in summary.values()) == 228848, "Annotation count changed")
    require(sum(s["empty_frames"] for s in summary.values()) == 7915, "Negative-frame count changed")
    require(set(counts["train"]) == set(range(32)), "TRAIN lost a supported class")
    expected_val = set(range(32)) - {28, 31}
    require(set(counts["val"]) == expected_val, "VAL class coverage differs from locked rule")
    class_rows = []
    for c in classes["classes"]:
        row = {"class_id": c["id"], "class_name": c["name"]}
        for split in ("train", "val"):
            row[f"{split}_objects"] = counts[split][c["id"]]
            row[f"{split}_clips"] = sum(assignments[k] == split and c["id"] in v["counts"] for k, v in clips.items())
        class_rows.append(row)
    write_csv(output / "class_supports.csv", class_rows)
    write_csv(output / "clip_assignments.csv", [dict(clip_id=k, site_id=v["site"], original_split=v["old_split"], new_split=assignments[k], frames=len(v["frames"]), objects=sum(v["counts"].values()), empty_frames=v["empty"]) for k, v in sorted(clips.items())])
    site_rows = []
    for site in sorted(quotas):
        row = dict(site_id=site, val_clip_quota=quotas[site])
        for split in ("train", "val"):
            group = [v for k, v in clips.items() if v["site"] == site and assignments[k] == split]
            row.update({f"{split}_clips": len(group), f"{split}_frames": sum(len(v["frames"]) for v in group), f"{split}_objects": sum(sum(v["counts"].values()) for v in group)})
        site_rows.append(row)
    write_csv(output / "site_supports.csv", site_rows)
    original_to_new = {old: {new: sum(len(c["frames"]) for k, c in clips.items() if c["old_split"] == old and assignments[k] == new) for new in ("train", "val")} for old in ("train", "val")}
    write_json(output / "summary.json", dict(splits=summary, original_to_new_frames=original_to_new, shared_sites=[r["site_id"] for r in site_rows if r["train_clips"] and r["val_clips"]], overlap_images=0, overlap_clips=0, overlap_clip_tracks=0, source_row_bytes_unchanged=True))
    policy = dict(schema="helmet2020.trainval_resplit.v1", seed=SEED, requested_ratio="80/20 working assumption", target_val_clips=146, method="Hamilton site quotas; rarest-class-first coverage; SHA256(seed:clip_id) ranking and fill; leave at least one TRAIN clip per supported class", singleton_train_only=[28, 31], absent_from_union=[32, 33, 34, 35], no_model_scores_used=True, split_seed_search=False, test_accessed=False, pixels_checked=False, training_launched=False, initial_weights="Generic external pretraining only; refit every HELMET-trained component excluding new VAL", evaluation_scope="New development split, not untouched confirmation or original official split", macro_ap_classes=sorted(expected_val), macro_ap_denominator=30, weighted_ap="Weight the same 30 supported classes by this VAL object's GT support", missing_class_ap="null, not zero or perfect", no_unseen_site_or_session_claim=True, source_image_paths_preserved=True, trainer_integration_verified=False)
    write_json(output / "split_policy.json", policy)
    readme = f"""# HELMET 2020 — gộp TRAIN + VAL và chia lại

Đây là bản chia metadata mới theo yêu cầu, chưa chạy training và chưa kiểm tra đủ JPEG trên máy.

| Tập mới | Clip | Ảnh | Đối tượng | Ảnh rỗng |
|---|---:|---:|---:|---:|
| TRAIN | 582 | 58.200 | {summary['train']['objects']:,} | {summary['train']['empty_frames']:,} |
| VAL | 146 | 14.600 | {summary['val']['objects']:,} | {summary['val']['empty_frames']:,} |

## Cách chia đã cố định

- Gộp đúng 63.600 ảnh TRAIN và 9.200 ảnh VAL gốc; không dùng TEST hoặc `_v2`.
- 80/20 là tỷ lệ làm việc chọn cho yêu cầu này; làm tròn theo 728 clip. Mỗi clip 100 ảnh nằm hoàn toàn trong một tập.
- Seed {SEED}. Phân bổ quota VAL theo site bằng phần dư lớn nhất; ưu tiên phủ lớp hiếm, rồi chọn theo SHA256(seed:clip_id). Không chọn seed theo điểm model.
- Giữ ít nhất một clip TRAIN cho mọi lớp hiện có. Hai lớp 28 và 31 chỉ có một clip nên giữ ở TRAIN; VAL có 30 lớp. Lớp 32–35 không có trong dữ liệu gộp.
- Còn chung 12 site; không chứng minh độc lập theo camera, phiên quay, người hoặc mọi pixel trùng lặp. Chưa kiểm tra trùng ảnh bằng nội dung JPEG.

## Dùng cho bước train tiếp theo

1. Đọc `train.jsonl`, `val.jsonl`, `classes.json` trong thư mục này. Image path vẫn là `images/train/...` hoặc `images/val/...` theo vị trí ảnh gốc; dataset root phải chứa cả hai thư mục. Không suy ra tập mới từ tên thư mục ảnh.
2. Xác minh đủ 72.800 JPEG và môi trường thực tế trước train. Chưa tích hợp/chạy thử trainer với manifest mới; các script frozen cũ có kiểm tra population gốc nên không thay manifest trực tiếp vào chúng.
3. Khởi tạo lại mọi thành phần đã học HELMET, kể cả detector/classifier/teacher; có thể dùng pretraining chung ngoài HELMET đã khai báo. Checkpoint HELMET cũ đã thấy một phần VAL mới nên không dùng làm phép đánh giá sạch cho bản chia này.
4. Mọi baseline và cải tiến phải train trên cùng bản chia, cùng evaluator và ngân sách so sánh. VAL dùng phát triển/chọn model; việc chia lại không làm dữ liệu đã xem trở thành confirmation chưa từng dùng.
5. Báo AP50 theo lớp và support; macro lấy trung bình 30 lớp có support VAL, weighted lấy trọng số support VAL. Lớp thiếu ghi null. Nếu chọn checkpoint/early stopping, khóa endpoint trước train.

Bản chia mới không bảo đảm điểm tăng. Điểm của nó thuộc protocol mới, không thay thế điểm VAL cũ hoặc chứng minh vượt bài 2020. Tổng TRAIN mới ít hơn TRAIN gốc (58.200 so với 63.600 ảnh).

`image_assignments.csv` và `clip_assignments.csv` giữ nguồn tập cũ/tập mới. `class_supports.csv`, `site_supports.csv` và `summary.json` ghi thống kê. `completion.json` ghi SHA256; giữ nguyên toàn bộ kết quả và dữ liệu gốc.
"""
    with (output / "README_VI.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write(readme)
    for name, expected in PINNED.items():
        require(sha(source / name) == expected, f"Source changed during preparation: {name}")
    require(sha(output / "classes.json") == PINNED["classes.json"], "Class map copy changed")
    receipt = dict(status="metadata_split_verified", created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)), seed=SEED, source_directory=str(ASSEMBLY).replace("\\", "/"), input_sha256=PINNED, outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()}, sources_unchanged=True, exact_source_rows_verified=72800, total_objects=228848, pixels_verified=False, training_started=False)
    write_json(output / "completion.json", receipt)
    print(json.dumps(dict(output=str(output), status=receipt["status"], splits=summary), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
