"""Frame AP and track-event F1 are separate metrics; no fabricated benchmark scores."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .common import validate_frames, assert_aligned_frames


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.


def integral_ap(tp, fp, support):
    if support == 0:
        return None
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = np.r_[0., tp / support, 1.]
    precision = np.r_[0., tp / np.maximum(tp+fp, 1), 0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx+1] - recall[idx]) * precision[idx+1]))


def frame_metrics(gt, pred, num_classes, threshold=.5, small_side=None, weight_basis="evaluation", global_support=None):
    assert_aligned_frames(gt, pred, num_classes)
    if not 0 < threshold <= 1:
        raise ValueError("IoU must be in (0,1]")
    classes = []
    for cid in range(num_classes):
        targets = {r["image_id"]: [o for o in r["objects"] if o["class_id"] == cid] for r in gt}
        # Small-object diagnostic: min(width,height)<40 in original pixels, NOT head size.
        included = lambda o: small_side is None or min(o["bbox"][2]-o["bbox"][0], o["bbox"][3]-o["bbox"][1]) < small_side
        support = sum(included(o) for objs in targets.values() for o in objs)
        dets = sorted([(o["score"], r["image_id"], j, o) for r in pred for j,o in enumerate(r["objects"]) if o["class_id"] == cid], key=lambda x:(-x[0], x[1], x[2]))
        matched = defaultdict(set)
        tp, fp = [], []
        for score, image_id, _, obj in dets:
            candidates = [(iou(obj["bbox"], t["bbox"]), j) for j,t in enumerate(targets[image_id]) if j not in matched[image_id] and included(t)]
            overlap, j = max(candidates, default=(0., -1))
            if j >= 0 and overlap >= threshold:
                matched[image_id].add(j); tp.append(1); fp.append(0)
            elif small_side is not None and (not included(obj) or any(not included(t) and iou(obj["bbox"],t["bbox"]) >= threshold for t in targets[image_id])):
                continue
            else:
                tp.append(0); fp.append(1)
        classes.append({"class_id":cid, "support":support, "predictions_scored":len(tp), "ap":integral_ap(tp,fp,support)})
    counts = np.array([c["support"] for c in classes], float)
    if weight_basis == "evaluation":
        weights = counts
    elif weight_basis == "global" and global_support is not None:
        weights = np.asarray(global_support, float)
        if weights.shape != counts.shape or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("Invalid global support")
        weights = weights * (counts > 0)
    else:
        raise ValueError("weight_basis must be evaluation, or global with support")
    valid = [c["ap"] for c in classes if c["ap"] is not None]
    return {"metric":"frame_AP_all_points", "iou":threshold, "small_min_side_lt":small_side,
            "weight_basis":weight_basis, "absent_class_policy":"exclude_from_mean_and_renormalize_weights",
            "weighted_mAP":float(np.dot(weights,[c["ap"] or 0 for c in classes])/weights.sum()) if weights.sum() else None,
            "macro_mAP":float(np.mean(valid)) if valid else None, "classes":classes,
            "official_evaluator_parity_verified":False}


def _events(records, num_classes, prediction):
    validate_frames(records, num_classes, prediction)
    events = {}
    for r in records:
        for obj in r["objects"]:
            if obj.get("track_id") is None:
                raise ValueError("CHUE evaluation requires track_id for every object")
            key = (r["clip_id"], str(obj["track_id"]))
            e = events.setdefault(key,{"class_id":obj["class_id"], "boxes":{}})
            if e["class_id"] != obj["class_id"]:
                raise ValueError("One event must have a single class. Apply offline voting first.")
            if r["frame_index"] in e["boxes"]:
                raise ValueError("Duplicate frame index within track")
            e["boxes"][r["frame_index"]] = obj["bbox"]
    return events


def event_f1(gt, pred, num_classes):
    assert_aligned_frames(gt, pred, num_classes)
    g = _events(gt,num_classes,False); p = _events(pred,num_classes,True)
    return score_events(g,p,num_classes)


def score_events(g,p,num_classes):
    # Faithful port of OSF evaluate_tracks_sim.m and metric.m:
    # intersection frames with spatial IoU >= .5 / UNION frames >= .5.
    # Greedy first valid prediction in input order, then remove it (one-to-one).
    # This differs from the prose in the paper. Do not substitute GT coverage.
    owners = {}; used=set();candidates=defaultdict(list)
    for pk,pe in p.items():candidates[(pk[0],pe["class_id"])].append((pk,pe))
    for gk,ge in g.items():
        for pk,pe in candidates[(gk[0],ge["class_id"])]:
            if pk in used: continue
            hits=sum(f in pe["boxes"] and iou(b,pe["boxes"][f]) >= .5 for f,b in ge["boxes"].items())
            union=len(set(ge["boxes"]) | set(pe["boxes"]))
            if hits/union >= .5:
                owners[gk]=pk;used.add(pk);break
    classes = []
    for cid in range(num_classes):
        support = sum(e["class_id"] == cid for e in g.values())
        count = sum(e["class_id"] == cid for e in p.values())
        tp = sum(g[k]["class_id"] == cid for k in owners)
        classes.append({"class_id":cid,"support":support,"predicted_events":count,"tp":tp,
                        "precision":tp/count if count else 0.,"recall":tp/support if support else 0.,
                        "f1":2*tp/(support+count) if support+count else None})
    total = len(g)
    present = [c['f1'] for c in classes if c['support'] > 0]
    tp = len(owners)
    return {"metric":"OSF_CHUE_weighted_F1_python_port", "spatial_iou_gte":.5,"temporal_iou_gte":.5,
            "temporal_denominator":"union_of_frame_ids", "matching":"greedy_first_one_to_one_input_order", "official_matlab_runtime_parity_verified":False,
            "weighted_f1":sum(c["support"]*(c["f1"] or 0) for c in classes)/total if total else None,
            "macro_f1":float(np.mean(present)) if present else None,
            "micro_f1":2*tp/(len(g)+len(p)) if len(g)+len(p) else None,
            "macro_absent_class_policy":"exclude_classes_without_ground_truth",
            "classes":classes}
