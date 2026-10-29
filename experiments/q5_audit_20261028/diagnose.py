"""Matched read-only GPU diagnostics, fixed samples, no fitting or TEST."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import collections, copy, hashlib, json, math, sys, time
from pathlib import Path
ROOT = Path("/workspace/HELMET2020_training_kit_v1")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from helmetlab import kit_models as km
from helmetlab.common import read_jsonl, digest
from helmetlab.labels import LabelMap
from helmetlab.kit_runner import seed_rng, atomic_json, execution_lock
from helmetlab.metrics import iou
from helmetlab.positional import decode_logits

BASE = Path("/workspace/runs/gq5_extend24_20261028")
OUT = Path("/workspace/runs/q5_audit_20261028")

def selected_indices(ds):
    # Fixed before inference: uniform sample plus rare-class coverage.
    rng = np.random.default_rng(20261028)
    selected = set(rng.choice(len(ds), min(4096, len(ds)), replace=False).tolist())
    groups = collections.defaultdict(list)
    for i, item in enumerate(ds.items):
        groups[item["class_id"]].append(i)
    for ids in groups.values():
        selected.update(rng.choice(ids, min(64, len(ids)), replace=False).tolist())
    return sorted(selected)

def metrics(logits, ids, labels):
    z = torch.from_numpy(logits)
    target = labels.states[ids]
    q, native = decode_logits(z, "binary10", labels)
    predicted = q.argmax(-1)
    per_class = []
    for cid in range(len(labels)):
        mask = ids == cid
        per_class.append({"class_id": cid, "support": int(mask.sum()),
            "correct": int((predicted[mask] == cid).sum()),
            "recall": float((predicted[mask] == cid).mean()) if mask.any() else None})
    roles = []
    for j, role in enumerate(("D", "P1", "P2", "P3", "P0")):
        present = target[:, j] != 0
        ph = logits[:, j] >= 0
        tp = int((ph & present).sum()); fp = int((ph & ~present).sum())
        roles.append({"role": role, "present": int(present.sum()),
            "presence_recall": tp / max(1, int(present.sum())),
            "presence_precision": tp / max(1, tp+fp),
            "helmet_accuracy_given_GT_presence": float(((logits[present, j+5] >= 0) == (target[present, j] == 1)).mean()) if present.any() else None,
            "native_state_accuracy": float((native[:, j] == target[:, j]).mean())})
    confusion = collections.Counter(zip(ids.tolist(), predicted.tolist()))
    return {"n": len(ids), "loss": km.class_loss(z, torch.tensor(ids), torch.tensor(labels.states), "Q5").item(),
        "crop_accuracy": float((predicted == ids).mean()),
        "macro_crop_recall": float(np.mean([c["recall"] for c in per_class if c["recall"] is not None])),
        "mean_confidence": float(q.max(1).mean()), "per_class": per_class, "roles": roles,
        "top_confusions": [{"truth": int(a), "pred": int(b), "count": n} for (a,b),n in confusion.most_common() if a != b][:20]}

def main():
    OUT.mkdir(exist_ok=True)
    seed_rng(0, 2)
    labels = LabelMap(ROOT / "manifests/classes.json")
    frames = {s: read_jsonl(ROOT / "manifests" / (s+".jsonl")) for s in ("train", "val")}
    receipts = json.loads(Path("/workspace/runs/research_v1_fp32/trainval_pixels.json").read_text())
    hashes = {k: v["sha256"] for k,v in receipts["files"].items()}
    datasets, samples = {}, {}
    for split in ("train", "val"):
        ds = km.CropDataset("/workspace/helmet_images", frames[split], training=False, image_hashes=hashes)
        ids = selected_indices(ds)
        samples[split] = ids
        datasets[split+"_gt"] = Subset(ds, ids)
    valds = datasets["val_gt"].dataset
    by_frame = collections.defaultdict(list)
    for i in samples["val"]:
        by_frame[valds.items[i]["image_id"]].append(i)
    matched, overlaps = [], []
    proposal_ds = copy.copy(valds)
    proposal_ds.items = [dict(x) for x in valds.items]
    props_file = Path("/workspace/runs/research_v1_fp32/seed0/retinanet/val_predictions.jsonl")
    with props_file.open() as f:
        for line in f:
            row = json.loads(line)
            for i in by_frame.get(row["image_id"], []):
                target = valds.items[i]["bbox"]
                candidates = [(o["score"], iou(target, o["bbox"]), o["bbox"]) for o in row["objects"]]
                candidates = [c for c in candidates if c[1] >= .5]
                if candidates:
                    _, overlap, box = max(candidates, key=lambda x: (x[0], x[1]))
                    proposal_ds.items[i]["bbox"] = box
                    matched.append(i); overlaps.append(overlap)
    matched.sort()
    datasets["val_matched_gt"] = Subset(valds, matched)
    datasets["val_matched_proposal"] = Subset(proposal_ds, matched)
    plan = {"schema": "q5_readonly_diagnostic.v1", "train_samples": len(samples["train"]),
        "val_samples": len(samples["val"]), "matched": len(matched), "mean_match_iou": float(np.mean(overlaps)),
        "sample_policy": "fixed_seed20261028_uniform4096_plus_up_to64_per_class",
        "scope": "stratified_diagnostic_crops_not_population_AP", "test_accessed": False,
        "sample_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()}
    atomic_json(OUT/"plan.json", plan)
    atomic_json(OUT/"samples.json", samples)
    report = {"plan": plan, "models": {}}
    for arm in ("G", "Q5"):
        ck = torch.load(BASE/arm/"epoch24.pt", map_location="cpu", weights_only=True)
        model = km.build_classifier(arm, initialize=False).cuda()
        model.load_state_dict(ck["state_dict"], strict=True)
        assert ck["arm"] == arm and ck["epoch"] == 24
        del ck
        model.eval()
        entry = {"checkpoint_sha256": digest(BASE/arm/"epoch24.pt"), "datasets": {}}
        for name, ds in datasets.items():
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4,
                multiprocessing_context="spawn", prefetch_factor=2, pin_memory=True)
            all_z, all_y = [], []
            tick = time.perf_counter()
            with torch.inference_mode():
                for batch, (x,y) in enumerate(loader):
                    x = x.cuda(non_blocking=True)
                    z = model(x)
                    assert torch.isfinite(z).all()
                    all_z.append(z.cpu().numpy()); all_y.append(y.numpy())
                    if arm == "Q5" and name == "val_gt" and batch == 0:
                        feat = model.forward_features(x)
                        head = model.head.predictor
                        memory = head.memory_tokens(feat)
                        query = head.queries[None].expand(len(x), -1, -1)
                        _, w = head.attention(query, memory, memory, need_weights=True, average_attn_weights=False)
                        avg = w.mean(1)
                        ent = -(avg * avg.clamp_min(1e-12).log()).sum(-1) / math.log(avg.shape[-1])
                        unit = torch.nn.functional.normalize(avg, dim=-1)
                        cos = unit @ unit.transpose(1,2)
                        mask = ~torch.eye(5, dtype=torch.bool, device=x.device)
                        entry["attention_diagnostic"] = {"n":len(x),"normalized_entropy_by_role":ent.mean(0).cpu().tolist(),
                            "mean_inter_role_attention_cosine":cos[:,mask].mean().item(),
                            "feature_shape":list(feat.shape), "query_norms":head.queries.norm(dim=-1).cpu().tolist()}
            z = np.concatenate(all_z); y = np.concatenate(all_y)
            entry["datasets"][name] = metrics(z,y,labels)
            np.savez_compressed(OUT/f"{arm}_{name}_logits.npz", logits=z, ids=y)
            print(json.dumps({"arm":arm,"dataset":name,"n":len(y),"seconds":time.perf_counter()-tick,
                "loss":entry["datasets"][name]["loss"],"accuracy":entry["datasets"][name]["crop_accuracy"]}),flush=True)
        # Verify gradients reach all branches without updating model/checkpoints.
        x,y = next(iter(DataLoader(datasets["train_gt"], batch_size=64, shuffle=False)))
        model.train(); model.zero_grad(set_to_none=True)
        loss = km.class_loss(model(x.cuda()),y.cuda(),torch.tensor(labels.states,device="cuda"),arm)
        loss.backward()
        entry["gradient_checks"] = {n:{"finite":bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
            "norm":float(p.grad.norm()) if p.grad is not None else None} for n,p in model.named_parameters() if n.startswith("head.")}
        report["models"][arm] = entry
        atomic_json(OUT/"diagnostic.json",report)
        del model
        torch.cuda.empty_cache()
    report["status"] = "completed"
    atomic_json(OUT/"diagnostic.json",report)
    print("DIAGNOSTIC_COMPLETED",flush=True)

if __name__ == "__main__":
    with execution_lock("cuda:0"):
        main()
