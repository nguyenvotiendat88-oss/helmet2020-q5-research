"""Auditable tables for measured training-kit results, without training or AP replay.

The caller supplies already scored records and a fresh output directory. This
module requires identical protocol bindings and truth supports before comparing
methods. It does not establish evaluator parity with any external paper.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METHODS = ("RetinaNet", "flat", "position", "G", "G_sqrt", "G+", "Q10", "Q5", "Q5shift")
SPLITS = ("val", "test")
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label, *, maximum=None):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             f"{label} must be a finite nonnegative number")
    if maximum is not None:
        _require(value <= maximum, f"{label} must be <= {maximum}")
    return float(value)


def _sha(value, label):
    _require(isinstance(value, str) and _SHA.fullmatch(value), f"{label} must be a SHA256 hex digest")
    return value.lower()


def _expected(values, label, allowed=None):
    if values is None:
        return None
    values = list(values)
    _require(values, f"{label} must not be empty")
    for item in values:
        if allowed is None:
            _integer(item, label)
        else:
            _require(type(item) is str and item in allowed, f"Unknown {label}: {item}")
    _require(len(values) == len(set(values)), f"Duplicate {label}")
    return set(values)


def _validate(records, expected_seeds, expected_methods):
    _require(isinstance(records, (list, tuple)) and len(records) > 0,
             "Supply a nonempty list of measured result records")
    seeds = _expected(expected_seeds, "expected_seeds")
    methods = _expected(expected_methods, "expected_methods", METHODS)
    # Snapshot the JSON evidence before output; rejects NaN and non-JSON objects.
    try:
        rows = json.loads(json.dumps(records, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("Result records must be finite JSON data") from exc
    keys, truth, bindings = set(), {}, set()
    for row in rows:
        _require(isinstance(row, dict), "Each result must be an object")
        for key in ("seed", "method", "split", "metrics", "selected_epoch", "train_seconds",
                    "inference_seconds", "checkpoint_sha256", "protocol_sha256"):
            _require(key in row, f"Missing result field: {key}")
        _integer(row["seed"], "seed")
        _integer(row["selected_epoch"], "selected_epoch")
        _require(row["method"] in METHODS, f"Unknown method: {row['method']}")
        _require(row["split"] in SPLITS, "Only val/test results can be reported")
        key = (row["split"], row["seed"], row["method"])
        _require(key not in keys, f"Duplicate split/seed/method: {key}")
        keys.add(key)
        for field in ("train_seconds", "inference_seconds"):
            _number(row[field], field)
        _sha(row["checkpoint_sha256"], "checkpoint_sha256")
        bindings.add(_sha(row["protocol_sha256"], "protocol_sha256"))
        metrics = row["metrics"]
        _require(isinstance(metrics, dict), "metrics must be an object")
        _require(metrics.get("metric") == "frame_AP_all_points", "Expected frame_AP_all_points metrics")
        _require(metrics.get("iou", .5) == .5, "Report expects canonical IoU=0.5")
        _require(metrics.get("weight_basis", "evaluation") == "evaluation",
                 "Report expects weighted AP using evaluation truth supports")
        _require(metrics.get("small_min_side_lt") is None, "Report expects full-object metrics, not a small-object slice")
        _require(metrics.get("absent_class_policy", "exclude_from_mean_and_renormalize_weights")
                 == "exclude_from_mean_and_renormalize_weights", "Unsupported absent-class policy")
        classes = metrics.get("classes")
        _require(isinstance(classes, list) and classes, "Metrics need per-class rows")
        ids, support_map, aps, weighted_terms = set(), {}, [], []
        for item in classes:
            _require(isinstance(item, dict) and all(name in item for name in
                     ("class_id", "support", "ap", "predictions_scored")), "Malformed per-class row")
            cid = _integer(item["class_id"], "class_id")
            _require(cid not in ids, "Duplicate class_id")
            ids.add(cid)
            support = _integer(item["support"], "support")
            _integer(item["predictions_scored"], "predictions_scored")
            support_map[cid] = support
            if support == 0:
                _require(item["ap"] is None, "Absent classes must have ap=null, not zero")
            else:
                ap = _number(item["ap"], "ap", maximum=1)
                aps.append(ap)
                weighted_terms.append(ap * support)
        _require(ids == set(range(len(ids))), "Class IDs must cover contiguous IDs starting at zero")
        _require(sum(support_map.values()) > 0, "Cannot compare a split without any truth objects")
        for field, actual in (("macro_mAP", statistics.fmean(aps)),
                              ("weighted_mAP", math.fsum(weighted_terms) / sum(support_map.values()))):
            _require(field in metrics, f"Missing metric: {field}")
            value = _number(metrics[field], field, maximum=1)
            _require(math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-12),
                     f"{field} disagrees with per-class AP/supports")
        previous = truth.setdefault(row["split"], support_map)
        _require(previous == support_map, f"Truth supports differ within {row['split']} split")
        if seeds is not None:
            _require(row["seed"] in seeds, "Observed seed is outside expected_seeds")
        if methods is not None:
            _require(row["method"] in methods, "Observed method is outside expected_methods")
    _require(len(bindings) == 1, "Cannot compare different protocol_sha256 bindings")
    # Expectations apply independently to every represented evaluation split.
    for split in truth:
        present = {(seed, method) for current, seed, method in keys if current == split}
        target_seeds = seeds if seeds is not None else {seed for seed, _ in present}
        target_methods = methods if methods is not None else {method for _, method in present}
        if seeds is not None or methods is not None:
            missing = sorted({(seed, method) for seed in target_seeds for method in target_methods} - present)
            _require(not missing, f"Missing expected seed/method cells in {split}: {missing}")
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["seed"], METHODS.index(row["method"])))
    return rows, truth, next(iter(bindings)), seeds, methods


def _write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _std(values):
    return statistics.stdev(values) if len(values) >= 2 else None


def _plots(output, aggregates):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return {"status": "not_created", "reason": "matplotlib is not installed", "files": []}
    files = []
    for split in SPLITS:
        rows = [row for row in aggregates if row["split"] == split]
        if not rows:
            continue
        figure = Figure(figsize=(max(7, len(rows) * 1.2), 4.5), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        positions = list(range(len(rows)))
        for field, offset, label, color in (("macro_mAP", -.18, "Macro AP50", "#2563eb"),
                                           ("weighted_mAP", .18, "Weighted AP50", "#059669")):
            means = [100 * row[f"{field}_mean"] for row in rows]
            axis.bar([x + offset for x in positions], means, width=.36, label=label, color=color)
            # A single seed has no sample SD; never draw a fabricated zero-CI.
            for x, mean, row in zip(positions, means, rows):
                spread = row[f"{field}_std"]
                if spread is not None:
                    axis.errorbar(x + offset, mean, yerr=100 * spread, fmt="none", color="#111827", capsize=3)
        axis.set_xticks(positions, [f"{row['method']}\nn={row['n_seeds']}" for row in rows])
        axis.set_ylim(0, 100)
        axis.set_ylabel("AP50 (%)")
        axis.set_title(f"{split.upper()} | measured means; bars: sample SD when n >= 2")
        axis.legend(loc="upper left")
        name = f"comparison_{split}.png"
        figure.savefig(output / name, dpi=160)
        files.append(name)
    return {"status": "created", "files": files}


def write_report(records, output, *, expected_seeds=None, expected_methods=None):
    """Write measured per-seed, aggregate, paired and per-class tables.

    ``output`` must not exist. Expected seeds/methods require their full product
    in every represented split; absence of TEST is explicitly disclosed rather
    than inferred as completion. Returns a JSON-compatible manifest. Standard
    deviations use ddof=1; one-seed SD and unavailable pairs remain undefined.
    """
    rows, truth, binding, seeds, methods = _validate(records, expected_seeds, expected_methods)
    output = Path(output)
    _require(not output.exists(), "Report output must be fresh; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=False)
    lookup = {(row["split"], row["seed"], row["method"]): row for row in rows}
    comparison, per_class, aggregates, pairs, paired_summary = [], [], [], [], []
    for row in rows:
        metrics = row["metrics"]
        flat = {key: row[key] for key in ("split", "seed", "method", "selected_epoch", "train_seconds",
                 "inference_seconds", "checkpoint_sha256", "protocol_sha256")}
        flat.update({key: metrics[key] for key in ("macro_mAP", "weighted_mAP")})
        flat.update(macro_AP50_percent=100 * metrics["macro_mAP"], weighted_AP50_percent=100 * metrics["weighted_mAP"],
                    truth_objects=sum(item["support"] for item in metrics["classes"]),
                    supported_classes=sum(item["support"] > 0 for item in metrics["classes"]),
                    predictions_scored=sum(item["predictions_scored"] for item in metrics["classes"]))
        comparison.append(flat)
        baseline = lookup.get((row["split"], row["seed"], "G"))
        base_classes = {} if baseline is None else {item["class_id"]: item for item in baseline["metrics"]["classes"]}
        for item in sorted(metrics["classes"], key=lambda entry: entry["class_id"]):
            before = base_classes.get(item["class_id"], {}).get("ap")
            delta = item["ap"] - before if item["ap"] is not None and before is not None else None
            per_class.append({"split": row["split"], "seed": row["seed"], "method": row["method"], **item,
                              "AP50_percent": None if item["ap"] is None else 100 * item["ap"],
                              "ap_delta_vs_G": delta, "ap_delta_vs_G_pp": None if delta is None else 100 * delta})
    for split in SPLITS:
        for method in METHODS:
            group = [row for row in comparison if row["split"] == split and row["method"] == method]
            if not group:
                continue
            aggregate = {"split": split, "method": method, "n_seeds": len(group),
                         "seeds": ";".join(str(row["seed"]) for row in group)}
            for field in ("macro_mAP", "weighted_mAP", "train_seconds", "inference_seconds"):
                values = [row[field] for row in group]
                aggregate[f"{field}_mean"], aggregate[f"{field}_std"] = statistics.fmean(values), _std(values)
            aggregates.append(aggregate)
    wanted_pairs = [(method, "G") for method in METHODS if method != "G"] + [("Q5", "Q10"), ("Q5", "Q5shift")]
    for split in SPLITS:
        split_seeds = sorted({row["seed"] for row in rows if row["split"] == split})
        for method, baseline in wanted_pairs:
            group = []
            for seed in split_seeds:
                after, before = lookup.get((split, seed, method)), lookup.get((split, seed, baseline))
                if after is None or before is None:
                    continue
                pair = {"split": split, "seed": seed, "method": method, "baseline": baseline}
                for field in ("macro_mAP", "weighted_mAP"):
                    difference = after["metrics"][field] - before["metrics"][field]
                    pair[f"{field}_delta"], pair[f"{field}_delta_pp"] = difference, 100 * difference
                pairs.append(pair)
                group.append(pair)
            if group:
                summary = {"split": split, "method": method, "baseline": baseline, "n_pairs": len(group),
                           "seeds": ";".join(str(row["seed"]) for row in group)}
                for field in ("macro_mAP", "weighted_mAP"):
                    values = [row[f"{field}_delta"] for row in group]
                    mean, std = statistics.fmean(values), _std(values)
                    summary.update({f"{field}_delta_mean": mean, f"{field}_delta_std": std,
                                    f"{field}_delta_mean_pp": 100 * mean,
                                    f"{field}_delta_std_pp": None if std is None else 100 * std})
                paired_summary.append(summary)
    _write_csv(output / "comparison.csv", comparison, list(comparison[0]))
    _write_csv(output / "aggregate.csv", aggregates, list(aggregates[0]))
    pair_fields = ["split", "seed", "method", "baseline", "macro_mAP_delta", "macro_mAP_delta_pp", "weighted_mAP_delta", "weighted_mAP_delta_pp"]
    _write_csv(output / "paired_deltas.csv", pairs, pair_fields)
    summary_fields = ["split", "method", "baseline", "n_pairs", "seeds"] + [f"{field}_delta_{suffix}" for field in ("macro_mAP", "weighted_mAP") for suffix in ("mean", "std", "mean_pp", "std_pp")]
    _write_csv(output / "paired_summary.csv", paired_summary, summary_fields)
    class_fields = ["split", "seed", "method", "class_id", "support", "ap", "predictions_scored", "AP50_percent", "ap_delta_vs_G", "ap_delta_vs_G_pp"]
    # Ignore optional labels in source class rows; the immutable input JSON keeps them.
    _write_csv(output / "per_class.csv", [{key: row[key] for key in class_fields} for row in per_class], class_fields)
    with (output / "records.json").open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    plot_status = _plots(output, aggregates)
    represented = ", ".join(split.upper() for split in SPLITS if split in truth)
    missing = ", ".join(split.upper() for split in SPLITS if split not in truth) or "không"
    smoke_note = "DỮ LIỆU GIẢ LẬP: chỉ kiểm tra luồng chạy; không phải kết quả HELMET.\n\n" if any(row.get("smoke") for row in rows) else ""
    note = f"""# Bảng kết quả bộ chạy HELMET\n\n{smoke_note}
Đây là tổng hợp {len(rows)} bản ghi kết quả đã đo, gồm: {represented}. Tập chưa có kết quả trong báo cáo: {missing}.
Mỗi bản ghi gắn SHA256 checkpoint và cùng protocol `{binding}`. Kiểm tra support không tự chứng minh ảnh/nhãn đúng; runner phải xác minh manifest và nguồn ảnh theo protocol.

- `comparison.csv`: từng seed, epoch được chọn, AP, thời gian và hash.
- `aggregate.csv`: trung bình và độ lệch chuẩn mẫu giữa seed (ddof=1).
- `paired_deltas.csv`: hiệu **method trừ baseline** trên cùng seed, cùng split; đơn vị gốc [0,1] và điểm phần trăm (pp).
- `paired_summary.csv`: trung bình/độ lệch chuẩn của chính các hiệu ghép cặp, không phải trừ hai trung bình từ seed khác nhau.
- `per_class.csv`: AP/support/số dự đoán từng lớp và hiệu so với G; hiệu âm là hồi quy cần báo cáo.
- `records.json`: bản ghi đầu vào đầy đủ. `manifest.json`: hash các tệp xuất.

AP50 là frame AP all-points tại IoU 0,5. Macro lấy trung bình các lớp có ground truth; weighted dùng số đối tượng ground truth của tập đang đánh giá. Lớp support=0 có AP trống, không được coi là AP=0. Thời gian là số giây do runner ghi, không tự suy ra FPS hay tốc độ trên phần cứng khác.

Thời gian inference của các đầu crop chỉ gồm đọc ảnh/crop và phân loại lại box có sẵn; phải cộng thời gian tạo proposal của RetinaNet nếu tính độ trễ cả pipeline. Không so sánh trực tiếp con số đó với thời gian detector để tuyên bố nhanh hơn.

Một seed không có độ lệch chuẩn mẫu: ô trống có nghĩa chưa xác định, không phải 0. Không tạo khoảng tin cậy hay tuyên bố ý nghĩa thống kê từ số seed thiếu. Cặp thiếu một nhánh không được điền bằng kết quả seed khác. Biểu đồ (nếu có) dùng thanh sai số SD giữa seed, không phải khoảng tin cậy. Không có biểu đồ khi môi trường chưa cài matplotlib.

Q5 chia sẻ đặc trưng truy vấn cho presence và helmet cùng vai trò. Cần đối chiếu G, G+ (kiểm soát dung lượng), Q10 và Q5shift trên cùng recipe. Q5 so với Q10 thay đổi cả số truy vấn/cách ghép tác vụ; Q5shift kiểm tra cách ghép vai trò. Chênh lệch AP riêng lẻ chưa chứng minh nhân quả, cơ chế tốt hơn, novelty hay khả năng được nhận bài. Chỉ gọi là cải thiện khi kết quả đo cho thấy mức tăng và báo cáo cả các hồi quy.

VAL dùng phát triển/chọn checkpoint; TEST chỉ đánh giá sau khi khóa recipe và checkpoint, không chọn mô hình bằng điểm TEST. Báo cáo không chứng nhận việc khóa TEST đã được thực thi. Bản chia TRAIN+VAL mới và split cũ/bài 2020 không phải phép so sánh cùng điều kiện: không trộn điểm cũ vào bảng ghép cặp và không suy ra vượt bài báo. Chưa xác nhận tương đương evaluator chính thức của tác giả; không hứa điểm số hay chấp nhận hội nghị.
"""
    (output / "README_VI.md").write_text(note, encoding="utf-8")
    manifest = {"schema": "helmetlab_training_kit_report_v1", "record_count": len(rows),
                "protocol_sha256": binding, "splits": [split for split in SPLITS if split in truth],
                "expected_seeds": sorted(seeds) if seeds is not None else None,
                "expected_methods": [method for method in METHODS if method in methods] if methods is not None else None,
                "truth_supports": {split: {str(cid): count for cid, count in sorted(counts.items())} for split, counts in truth.items()},
                "standard_deviation_ddof": 1, "confidence_intervals_computed": False,
                "official_evaluator_parity_verified": False, "plots": plot_status, "files": []}
    for path in sorted(output.iterdir()):
        payload = path.read_bytes()
        manifest["files"].append({"path": path.name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest
