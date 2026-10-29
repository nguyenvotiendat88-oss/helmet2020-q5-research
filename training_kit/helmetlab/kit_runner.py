"""Portable resplit study: fresh RetinaNet, paired role heads, VAL freeze, final TEST.

This is a new protocol. Frozen rental controllers and their results are not edited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

from .common import digest, read_json, read_jsonl, write_jsonl

SCHEMA = "helmet2020.training_kit.v1"
ARMS = ("flat", "position", "G", "G_sqrt", "G+", "Q10", "Q5", "Q5shift")
PROJECT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def load_config(path):
    c = read_json(path)
    require(set(c) == {"schema", "seeds", "arms", "cpu_threads", "workers", "min_free_disk_gib", "detector", "heads"}, "Unknown/missing config field")
    require(c["schema"] == SCHEMA, "Config schema differs")
    require(c["seeds"] and len(set(c["seeds"])) == len(c["seeds"]) and all(type(s) is int and 0 <= s < 2**32 for s in c["seeds"]), "Invalid paired seeds")
    require(c["arms"] and len(set(c["arms"])) == len(c["arms"]) and set(c["arms"]) <= set(ARMS), "Invalid arms")
    require("G" in c["arms"], "Keep G as the paired control")
    require(type(c["cpu_threads"]) is int and c["cpu_threads"] > 0, "cpu_threads must be positive")
    require(c["workers"] == 0, "This reproducible v1 uses workers=0; data/augmentation RNG is epoch-resumable")
    require(type(c["min_free_disk_gib"]) in (int, float) and math.isfinite(c["min_free_disk_gib"]) and c["min_free_disk_gib"] > 0, "Invalid disk reserve")
    d, h = c["detector"], c["heads"]
    require(set(d) == {"epochs", "patience", "batch", "lr", "min_size", "max_size", "trainable_backbone_layers", "score_thresh", "nms_thresh", "detections_per_img", "topk_candidates", "amp", "selection"}, "Invalid detector config")
    require(set(h) == {"epochs", "batch", "lr", "weight_decay", "selection", "crop_size", "crop_margin", "score_thresh", "nms_thresh", "detections_per_img"}, "Invalid head config")
    for settings in (d, h):
        for key in ("epochs", "batch", "detections_per_img"):
            require(type(settings[key]) is int and settings[key] > 0, f"Invalid {key}")
        for key in ("lr", "score_thresh", "nms_thresh"):
            require(type(settings[key]) in (int, float) and math.isfinite(settings[key]) and 0 < settings[key] < 1, f"Invalid {key}")
    require(d["selection"] == "weighted_mAP" and h["selection"] == "macro_mAP", "Endpoint fixed before fit: detector weighted AP50, heads macro AP50")
    require(type(d["amp"]) is bool and type(d["patience"]) is int and d["patience"] > 0, "Invalid precision/early stopping")
    require(d["trainable_backbone_layers"] == 3 and type(d["min_size"]) is int and type(d["max_size"]) is int and 64 <= d["min_size"] <= d["max_size"], "Invalid RetinaNet geometry")
    require(type(d["topk_candidates"]) is int and d["topk_candidates"] > 0, "Invalid proposal topk")
    require(h["crop_size"] == 224 and h["crop_margin"] == .05 and 0 <= h["weight_decay"] < 1, "Keep matched 224px/.05 crops and valid weight decay")
    return c


def source_bindings():
    # Portable relative names; adding a module changes identity before resume.
    paths = list((PROJECT / "helmetlab").glob("*.py"))
    paths += [PROJECT / "scripts/train_retinanet_baseline.py"]
    return {p.relative_to(PROJECT).as_posix(): digest(p) for p in sorted(paths)}


def runtime(device):
    import torch
    import torchvision
    require(torch.__version__.split("+")[0] == "2.8.0" and torchvision.__version__.split("+")[0] == "0.23.0", "Install pinned torch2.8.0 / torchvision0.23.0")
    versions = {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "Pillow", "PyYAML")}
    require(device == "cpu" or device.startswith("cuda:"), "Use cpu or cuda:N")
    if device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA unavailable; run doctor on the rented GPU")
        torch.cuda.set_device(torch.device(device))
    return dict(versions=versions, python=sys.version.split()[0], device=device, gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None, cuda=torch.version.cuda)


def seed_rng(seed, threads):
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


@contextmanager
def execution_lock(device):
    path = Path(tempfile.gettempdir()) / ("helmetlab_training_kit_" + device.replace(":", "_") + ".lock")
    f = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                f.write(b"0"); f.flush()
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        f.close()  # OS releases locks even after a killed worker; no stale-PID override.


class BudgetExceeded(RuntimeError):
    pass


class Guard:
    def __init__(self, hours, root, reserve_gib):
        require(hours > 0 and math.isfinite(hours), "max-hours must be finite and positive")
        self.end = time.monotonic() + hours * 3600
        self.root, self.reserve = Path(root), int(reserve_gib * 1024**3)

    def __call__(self):
        if time.monotonic() >= self.end:
            raise BudgetExceeded("Time limit reached between operations; resume from last completed epoch")
        if shutil.disk_usage(self.root).free < self.reserve:
            raise BudgetExceeded("Disk reserve reached; preserve outputs and free storage before resuming")


def save_predictions(path, records):
    tmp = Path(path).with_suffix(".jsonl.tmp")
    write_jsonl(tmp, records)
    os.replace(tmp, path)


def score_predictions(truth, pred, classes):
    from .metrics import frame_metrics
    report = frame_metrics(truth, pred, classes, threshold=.5)
    require(report["macro_mAP"] is not None, "Cannot score a split with no GT objects")
    return {**report, "frames": len(truth), "supported_classes": sum(c["support"] > 0 for c in report["classes"])}


def binding_for(study, seed, method, extra=None):
    return dict(study_sha256=canonical(study), seed=seed, method=method, extra=extra or {})


def load_stage(path, binding, resume):
    import torch
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        return None
    require(resume, f"Existing output requires --resume: {path}")
    if not (path / "last.pt").exists():
        require(not (path / "complete.json").exists(), "Completion without last checkpoint")
        require((path / "run.json").exists() and read_json(path / "run.json")["binding"] == binding, "No matching resumable stage identity")
        return None  # Interrupted before the first epoch: restart deterministically.
    ck = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
    require(ck["binding"] == binding and ck["schema"] == SCHEMA, "Checkpoint data/config/code/runtime identity changed")
    return ck


def history_progress(history, settings, samples=None):
    require(isinstance(history, list) and 0 < len(history) <= settings["epochs"], "Invalid checkpoint history length")
    field = "weighted_AP50" if settings["selection"] == "weighted_mAP" else "macro_AP50"
    best, epoch = -1., 0
    for i, row in enumerate(history, 1):
        require(type(row["epoch"]) is int and row["epoch"] == i, "History epochs must be contiguous")
        require(type(row["samples"]) is int and row["samples"] > 0 and (samples is None or row["samples"] == samples), "History sample exposure differs")
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k] >= 0 for k in ("train_loss","train_val_seconds","macro_AP50","weighted_AP50")), "Invalid checkpoint history values")
        require(row["macro_AP50"] <= 1 and row["weighted_AP50"] <= 1, "Invalid AP in history")
        if row[field] > best:
            best, epoch = row[field], i
    return best, epoch, len(history)-epoch


def validate_checkpoint_progress(ck, settings, samples, batches, detector):
    best, epoch, stale = history_progress(ck["history"], settings, samples)
    require(type(ck["epoch"]) is int and ck["epoch"] == len(ck["history"]), "Checkpoint epoch/history progress mismatch")
    require(ck["best_metric"] == settings["selection"] and ck["best_score"] == best and ck["best_epoch"] == epoch and ck["stale"] == stale, "Checkpoint best/stopping state differs from history")
    require(ck["optimizer"].get("state"), "Missing optimizer progress")
    for state in ck["optimizer"]["state"].values():
        require(float(state["step"]) == ck["epoch"] * batches, "Optimizer step count differs from committed exposure")
    if detector:
        require(ck["scheduler"] is None, "Detector scheduler differs")
    else:
        require(ck["scheduler"]["last_epoch"] == ck["epoch"] and ck["scheduler"]["T_max"] == settings["epochs"], "Scheduler progress differs")


def publish_best(path, ck):
    from .checkpoints import atomic_save
    atomic_save(dict(schema=SCHEMA, binding=ck["binding"], state_dict=ck["best_model"], selected_epoch=ck["best_epoch"], metric=ck["best_metric"], score=ck["best_score"]), Path(path) / "best.pt")


def move_optimizer(optimizer, device):
    import torch
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def fit_stage(*, model, dataset, evaluate, output, binding, config, stage_config, device, seed, guard, resume, detector, stop_after=None, weights=None):
    """Atomic last checkpoint contains best state: publication can be repaired after a crash."""
    import torch
    from torch.utils.data import DataLoader
    from .checkpoints import atomic_save, cpu_tree, rng_state, restore_rng
    from . import kit_models as km
    output = Path(output)
    previous = load_stage(output, binding, resume)
    seed_rng(seed, config["cpu_threads"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=stage_config["batch"], shuffle=True, num_workers=0, generator=generator, drop_last=False, collate_fn=km.collate_detection if detector else None)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=stage_config["lr"], foreach=False, fused=False) if detector else torch.optim.AdamW(model.parameters(), lr=stage_config["lr"], weight_decay=stage_config["weight_decay"], foreach=False, fused=False)
    scheduler = None if detector else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_config["epochs"])
    amp = detector and stage_config["amp"] and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    metric = stage_config["selection"]
    history, best_score, best_epoch, stale, start, best_model = [], -1., 0, 0, 0, None
    if previous is not None:
        validate_checkpoint_progress(previous, stage_config, len(dataset), len(loader), detector)
        model.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer"]); move_optimizer(optimizer, device)
        if scheduler:
            scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        restore_rng(previous["rng"], generator)
        history, best_score, best_epoch, stale, start, best_model = [previous[k] for k in ("history", "best_score", "best_epoch", "stale", "epoch", "best_model")]
        publish_best(output, previous)
    run = dict(schema=SCHEMA, binding=binding, status="training", selected_by=metric, resume_granularity="completed_epoch", training_samples=len(dataset), started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / "run.json", run)
    try:
        for epoch in range(start, stage_config["epochs"]):
            if detector and stale >= stage_config["patience"]:
                break
            if stop_after is not None and epoch >= stop_after:
                atomic_json(output / "run.json", {**run, "status": "paused", "completed_epochs": epoch})
                return False
            guard()
            tick = time.perf_counter()
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            model.train(); loss_sum = samples = 0
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            for index, (x, y) in enumerate(loader):
                guard(); optimizer.zero_grad(set_to_none=True)
                if detector:
                    x = [v.to(device) for v in x]
                    y = [{k: v.to(device) for k, v in t.items()} for t in y]
                    with torch.autocast(device_type="cuda", enabled=amp):
                        loss = sum(model(x, y).values())
                else:
                    loss = km.class_loss(model(x.to(device)), y.to(device), weights["states"], binding["method"], weights.get("presence"))
                require(torch.isfinite(loss).item(), "Nonfinite training loss")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
                n = len(x); samples += n; loss_sum += float(loss.detach()) * n
                if index % 100 == 0:
                    print(json.dumps(dict(method=binding["method"], seed=seed, epoch=epoch+1, batch=index+1, total_batches=len(loader), loss=float(loss.detach()))), flush=True)
            if scheduler:
                scheduler.step()
            guard(); metrics, _ = evaluate(model)
            score = metrics[metric]
            require(type(score) in (float, int) and math.isfinite(score), "VAL metric must be finite")
            improved = score > best_score
            if improved:
                best_score, best_epoch, stale = score, epoch + 1, 0
                best_model = cpu_tree(model.state_dict())
            else:
                stale += 1
            history.append(dict(epoch=epoch+1, train_loss=loss_sum/samples, samples=samples, macro_AP50=metrics["macro_mAP"], weighted_AP50=metrics["weighted_mAP"], train_val_seconds=time.perf_counter()-tick, peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None))
            checkpoint = dict(schema=SCHEMA, binding=binding, state_dict=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict() if scheduler else None, scaler=scaler.state_dict(), rng=rng_state(generator), history=history, epoch=epoch+1, best_model=best_model, best_epoch=best_epoch, best_score=best_score, best_metric=metric, stale=stale)
            # last is the commit point; best and history are reproducible projections.
            atomic_save(checkpoint, output / "last.pt")
            if improved or not (output / "best.pt").exists():
                publish_best(output, checkpoint)
            atomic_json(output / "history.json", history)
            print(json.dumps(dict(method=binding["method"], seed=seed, epoch=epoch+1, checkpoint_committed=True, seconds=time.perf_counter()-tick)), flush=True)
            previous = None
        require(best_model is not None, "No completed epoch")
        require(len(history) == stage_config["epochs"] or (detector and stale >= stage_config["patience"]), "Incomplete planned training cannot be finalized")
        model.load_state_dict(best_model, strict=True)
        guard(); metrics, pred = evaluate(model)
        save_predictions(output / "val_predictions.jsonl", pred)
        atomic_json(output / "val_metrics.json", metrics)
        atomic_json(output / "history.json", history)
        completion = dict(schema=SCHEMA, binding=binding, status="completed", epochs=len(history), selected_epoch=best_epoch, best_metric=metric, best_score=best_score, best_sha256=digest(output / "best.pt"), val_predictions_sha256=digest(output / "val_predictions.jsonl"), val_metrics_sha256=digest(output / "val_metrics.json"), history_sha256=digest(output/"history.json"), total_train_val_seconds=sum(h["train_val_seconds"] for h in history))
        atomic_json(output / "complete.json", completion)
        atomic_json(output / "run.json", {**run, "status": "completed", "completed_epochs": len(history)})
        return True
    except BaseException as exc:
        atomic_json(output / "run.json", {**run, "status": "paused" if isinstance(exc, (BudgetExceeded, KeyboardInterrupt)) else "failed", "error": str(exc), "checkpoint": "last.pt" if (output/"last.pt").exists() else None})
        raise


def stage_verified(path, binding):
    path = Path(path)
    if not (path / "complete.json").exists():
        return False
    c = read_json(path / "complete.json")
    require(c["schema"] == SCHEMA and c["binding"] == binding and c["status"] == "completed", "Completed stage identity differs")
    for name, key in (("best.pt", "best_sha256"), ("val_metrics.json", "val_metrics_sha256"), ("val_predictions.jsonl", "val_predictions_sha256"), ("history.json", "history_sha256")):
        require(digest(path / name) == c[key], f"Completed stage artifact changed: {name}")
    return True


def expected_stage_binding(study, output, seed, method):
    require(seed in study["config"]["seeds"] and method in ("retinanet", *study["config"]["arms"]), "Unknown stage owner")
    if method == "retinanet":
        return binding_for(study, seed, method)
    detector = Path(output)/f"seed{seed}"/"retinanet"
    require(stage_verified(detector, binding_for(study,seed,"retinanet")), "Detector stage is not complete")
    return binding_for(study,seed,method,{"detector_best_sha256":digest(detector/"best.pt"),"val_proposals_sha256":digest(detector/"val_predictions.jsonl")})


def test_completion_verified(path, binding):
    path=Path(path)
    done=read_json(path/"complete.json")
    require(done.get("schema") == SCHEMA and done.get("status") == "completed" and done.get("binding") == binding, "TEST completion identity/status differs")
    require(set(done.get("files",{})) == {"predictions.jsonl","metrics.json"}, "TEST completion needs exact file inventory")
    require(all(digest(path/f) == h for f,h in done["files"].items()), "Saved TEST artifacts changed")
    return True


def prediction_calls(images, frames, labels, config, device, guard, image_hashes):
    from . import kit_models as km
    from torch.utils.data import DataLoader

    def detector(model):
        tick = time.perf_counter()
        ds = km.DetectionDataset(images, frames, image_hashes)
        loader = DataLoader(ds, batch_size=config["detector"]["batch"], shuffle=False, num_workers=0, collate_fn=km.collate_detection)
        pred = km.detector_predictions(model, loader, frames, device=device, num_classes=len(labels), guard=guard)
        elapsed = time.perf_counter()-tick
        return {**score_predictions(frames, pred, len(labels)), "inference_seconds": elapsed}, pred

    def heads(model, proposals):
        tick = time.perf_counter()
        h = config["heads"]
        pred = km.classifier_predictions(model, images, frames, proposals, labels, device=device, batch=h["batch"], margin=h["crop_margin"], size=h["crop_size"], score_thresh=h["score_thresh"], nms_thresh=h["nms_thresh"], max_detections=h["detections_per_img"], guard=guard, image_hashes=image_hashes)
        elapsed = time.perf_counter()-tick
        return {**score_predictions(frames, pred, len(labels)), "inference_seconds": elapsed, "inference_scope": "crop_rescoring_only; detector proposal generation measured separately"}, pred
    return detector, heads


def pixel_hashes(receipt):
    # Data helper uses one image-path keyed digest map for runtime byte checks.
    return {p: v["sha256"] for p, v in receipt["files"].items()}


def fit(args, config=None, smoke=False):
    import torch
    from . import kit_data as kd, kit_models as km
    from .labels import LabelMap
    config = config or load_config(args.config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    versions = runtime(args.device)
    identity = kd.dataset_identity(args.manifests)
    if not smoke:
        require(identity["num_classes"] == 36 and {s: identity["splits"][s]["frames"] for s in ("train", "val", "test")} == {"train":58200,"val":14600,"test":18200}, "Production fit requires the prepared full resplit population")
        require(identity["files"]["train"]["sha256"] == kd.RESPLIT_SHA256["train.jsonl"] and identity["files"]["val"]["sha256"] == kd.RESPLIT_SHA256["val.jsonl"] and identity["files"]["test"]["sha256"] == kd.TEST_SHA256, "Production manifest differs from approved split")
        require(identity["files"]["classes"]["sha256"] == kd.CLASSES_SHA256, "Class ontology differs from approved map")
    require(all(identity["files"][s]["path"] == f"{s}.jsonl" for s in ("train","val","test")) and identity["files"]["classes"]["path"] == "classes.json", "Use canonical prepared manifest paths")
    dataset = {s: kd.load_manifest(args.manifests, s) for s in ("train", "val")}
    labels = LabelMap(Path(args.manifests) / "classes.json")
    output = Path(args.output).resolve()
    require(not output.exists() or args.resume, "Use a fresh output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    guard = Guard(args.max_hours, output, config["min_free_disk_gib"])
    with execution_lock(args.device):
        guard()
        receipt = kd.check_pixels(args.manifests, args.images, splits=("train", "val"), guard=guard)
        image_hashes = pixel_hashes(receipt)
        init_files = {} if smoke else {name: digest(Path(args.weights) / name) for name in ("resnet50-0676ba61.pth", "resnet18-f37072fd.pth")}
        study = dict(schema=SCHEMA, config=config, dataset=identity, initialization=init_files, pixel_sha256=canonical(image_hashes), runtime=versions, sources=source_bindings(), smoke=smoke, TEST_used_for_selection=False, protocol="resplit-development-v1; RetinaNet proposal rescoring; not official paper reproduction")
        if (output / "study.json").exists():
            require(read_json(output / "study.json") == study, "Study data/config/weights/code/runtime differs; create a new output")
        else:
            require(not any(output.iterdir()), "Fresh output must be empty")
            atomic_json(output / "study.json", study)
        atomic_json(output / "trainval_pixels.json", receipt)
        val_detector, val_heads = prediction_calls(args.images, dataset["val"], labels, config, args.device, guard, image_hashes)
        try:
            for seed in config["seeds"]:
                seed_root = output / f"seed{seed}"
                detector_out = seed_root / "retinanet"
                binding = binding_for(study, seed, "retinanet")
                if not stage_verified(detector_out, binding):
                    seed_rng(seed, config["cpu_threads"])
                    net = km.build_detector({**config["detector"], "torchvision_version":"0.23.0"}, None if smoke else Path(args.weights)/"resnet50-0676ba61.pth", smoke=smoke, initialize=not (detector_out / "last.pt").exists())
                    trainset = km.DetectionDataset(args.images, dataset["train"], image_hashes)
                    complete = fit_stage(model=net, dataset=trainset, evaluate=val_detector, output=detector_out, binding=binding, config=config, stage_config=config["detector"], device=args.device, seed=seed, guard=guard, resume=args.resume, detector=True, stop_after=args.stop_after_epoch)
                    del net, trainset; gc.collect()
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    if not complete:
                        return {"status": "paused", "stage": str(detector_out)}
                proposals = read_jsonl(detector_out / "val_predictions.jsonl")
                proposal_sha = digest(detector_out / "val_predictions.jsonl")
                for arm in config["arms"]:
                    out = seed_root / arm
                    binding = binding_for(study, seed, arm, {"detector_best_sha256": digest(detector_out / "best.pt"), "val_proposals_sha256": proposal_sha})
                    if stage_verified(out, binding):
                        continue
                    seed_rng(seed, config["cpu_threads"])
                    net = km.build_classifier(arm, seed=seed, backbone_path=None if smoke else Path(args.weights)/"resnet18-f37072fd.pth", smoke=smoke, initialize=not (out/"last.pt").exists())
                    trainset = km.CropDataset(args.images, dataset["train"], training=True, seed=seed, size=config["heads"]["crop_size"], margin=config["heads"]["crop_margin"], image_hashes=image_hashes)
                    presence = km.presence_weights(dataset["train"], labels.states, balance="sqrt" if arm == "G_sqrt" else "none")
                    weights = {"states": torch.tensor(labels.states, dtype=torch.long, device=args.device), "presence": presence}
                    complete = fit_stage(model=net, dataset=trainset, evaluate=lambda m: val_heads(m, proposals), output=out, binding=binding, config=config, stage_config=config["heads"], device=args.device, seed=seed, guard=guard, resume=args.resume, detector=False, stop_after=args.stop_after_epoch, weights=weights)
                    atomic_json(out / "presence_weights.json", presence)
                    del net, trainset; gc.collect()
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    if not complete:
                        return {"status": "paused", "stage": str(out)}
            freeze_selection(output)
            return {"status": "fit_completed_test_not_run", "output": str(output), "selection_sha256": digest(output/"selection.json")}
        except BudgetExceeded as exc:
            atomic_json(output / "paused.json", {"reason": str(exc), "resume": True, "TEST_run": False})
            return {"status": "paused", "reason": str(exc)}


def freeze_selection(output):
    """Predeclare all completed comparisons and select winning head using mean VAL macro."""
    output = Path(output)
    study = read_json(output / "study.json")
    entries, means = [], {}
    for seed in study["config"]["seeds"]:
        for method in ("retinanet", *study["config"]["arms"]):
            p = output / f"seed{seed}" / method
            c = read_json(p / "complete.json")
            expected = expected_stage_binding(study, output, seed, method)
            require(stage_verified(p, expected), "Every preregistered stage must complete before TEST release")
            settings = study["config"]["detector" if method == "retinanet" else "heads"]
            history = read_json(p/"history.json")
            best, best_epoch, stale = history_progress(history, settings)
            require(c["epochs"] == len(history) and c["selected_epoch"] == best_epoch and c["best_score"] == best and c["best_metric"] == settings["selection"], "Completion progress differs from committed history")
            require(c["epochs"] == settings["epochs"] or (method == "retinanet" and stale >= settings["patience"]), "All planned head epochs or detector early stopping required")
            m = read_json(p / "val_metrics.json")
            if method != "retinanet":
                means.setdefault(method, []).append(m["macro_mAP"])
            entries.append(dict(seed=seed, method=method, path=p.relative_to(output).as_posix(), selected_epoch=c["selected_epoch"], best_sha256=c["best_sha256"], completion_sha256=digest(p/"complete.json"), val_metrics_sha256=c["val_metrics_sha256"]))
    winner = sorted(means, key=lambda k: (-sum(means[k])/len(means[k]), k))[0]
    selection = dict(schema=SCHEMA, study_sha256=canonical(study), dataset_sha256=canonical(study["dataset"]), entries=entries, selected_head_by_mean_VAL_macro=winner, TEST_accessed=False, smoke=study["smoke"], test_comparisons="all predeclared methods and seeds; no post-test tuning")
    path = output / "selection.json"
    if path.exists():
        require(read_json(path) == selection, "Existing selection differs; never overwrite a TEST release")
    else:
        atomic_json(path, selection)
    return selection


def final_test(args):
    import torch
    from . import kit_data as kd, kit_models as km
    from .labels import LabelMap
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(args.output).resolve()
    selection = read_json(output / "selection.json")
    require(digest(output/"selection.json") == args.selection_sha256, "Supply the frozen selection SHA256 before TEST")
    study = read_json(output / "study.json")
    require(canonical(study) == selection["study_sha256"], "Study changed after VAL freeze")
    require(study["sources"] == source_bindings(), "Code changed after VAL freeze")
    require(study["runtime"] == runtime(args.device), "Use the fitting runtime/device for frozen inference")
    require(kd.dataset_identity(args.manifests) == study["dataset"], "Dataset changed after VAL freeze")
    require(freeze_selection(output) == selection, "Exact seed/method/path selection changed")
    # Do not open TEST until the checkpoint set is fixed and every hash verified.
    for e in selection["entries"]:
        p = output / e["path"]
        require(digest(p/"best.pt") == e["best_sha256"] and digest(p/"complete.json") == e["completion_sha256"] and digest(p/"val_metrics.json") == e["val_metrics_sha256"], "Frozen selection artifact changed")
    rows = kd.load_manifest(args.manifests, "test")
    labels = LabelMap(Path(args.manifests)/"classes.json")
    config = study["config"]
    seed_rng(0, config["cpu_threads"])
    guard = Guard(args.max_hours, output, config["min_free_disk_gib"])
    with execution_lock(args.device):
        receipt = kd.check_pixels(args.manifests, args.images, splits=("test",), guard=guard)
        hashes = pixel_hashes(receipt)
        trainval = pixel_hashes(read_json(output/"trainval_pixels.json"))
        original_receipt = read_json(output/"trainval_pixels.json")
        require(original_receipt.get("status") == "verified" and original_receipt.get("dataset_json_sha256") == study["dataset"]["dataset_json_sha256"] and set(original_receipt["splits"]) == {"train","val"} and len(trainval) == sum(study["dataset"]["splits"][s]["frames"] for s in ("train","val")) and canonical(trainval) == study["pixel_sha256"], "TRAIN/VAL pixel receipt no longer matches frozen study")
        require(not set(hashes.values()) & set(trainval.values()), "TEST image content overlaps fit/VAL pixels")
        if (output/"test_pixels.json").exists():
            require(pixel_hashes(read_json(output/"test_pixels.json")) == hashes, "TEST pixels changed since first evaluation")
        atomic_json(output/"test_pixels.json", receipt)
        det_eval, heads_eval = prediction_calls(args.images, rows, labels, config, args.device, guard, hashes)
        test_root = output/"test"
        test_root.mkdir(exist_ok=True)
        for seed in config["seeds"]:
            proposals = None
            for method in ("retinanet", *config["arms"]):
                e = next(x for x in selection["entries"] if x["seed"] == seed and x["method"] == method)
                dest = test_root/f"seed{seed}"/method
                binding = {"selection_sha256": args.selection_sha256, "checkpoint_sha256": e["best_sha256"], "test_pixel_sha256": canonical(hashes), "test_manifest_sha256": digest(Path(args.manifests)/"test.jsonl")}
                if (dest/"complete.json").exists():
                    test_completion_verified(dest,binding)
                    if method == "retinanet":
                        proposals = read_jsonl(dest/"predictions.jsonl")
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                # A failed inference may be retried with identical locked bytes; no fitting happens.
                ck = torch.load(output/e["path"]/"best.pt", map_location="cpu", weights_only=True)
                require(ck["schema"] == SCHEMA and ck["binding"] == expected_stage_binding(study,output,seed,method) and ck["selected_epoch"] == e["selected_epoch"], "Selected checkpoint ownership differs")
                net = km.build_detector({**config["detector"], "torchvision_version":"0.23.0"}, None, smoke=study["smoke"], initialize=False) if method == "retinanet" else km.build_classifier(method, seed=seed, backbone_path=None, smoke=study["smoke"], initialize=False)
                net.load_state_dict(ck["state_dict"], strict=True); net.to(args.device); del ck
                metrics, pred = det_eval(net) if method == "retinanet" else heads_eval(net, proposals)
                save_predictions(dest/"predictions.jsonl", pred)
                atomic_json(dest/"metrics.json", metrics)
                atomic_json(dest/"complete.json", dict(schema=SCHEMA, binding=binding, status="completed", files={f:digest(dest/f) for f in ("predictions.jsonl", "metrics.json")}))
                if method == "retinanet":
                    proposals = pred
                del net; gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        atomic_json(output/"test_complete.json", dict(schema=SCHEMA, selection_sha256=args.selection_sha256, status="completed", smoke=study["smoke"], test_frames=len(rows), paper_superiority_established=False, completions={p.relative_to(output).as_posix():digest(p) for p in sorted(test_root.rglob("complete.json"))}))
    return {"status": "test_completed", "smoke": study["smoke"], "test_frames": len(rows)}


def collect_report(output, report_dir):
    from .kit_reporting import write_report
    output = Path(output)
    study = read_json(output/"study.json")
    selection=freeze_selection(output)
    release_sha=digest(output/"selection.json")
    test_hashes=pixel_hashes(read_json(output/"test_pixels.json")) if (output/"test_pixels.json").exists() else None
    records = []
    for seed in study["config"]["seeds"]:
        for method in ("retinanet", *study["config"]["arms"]):
            p = output/f"seed{seed}"/method
            c = read_json(p/"complete.json")
            require(stage_verified(p, expected_stage_binding(study,output,seed,method)), "Invalid stage ownership")
            for split, metrics_path in (("val", p/"val_metrics.json"), ("test", output/"test"/f"seed{seed}"/method/"metrics.json")):
                if not metrics_path.exists():
                    continue
                if split == "test":
                    require(test_hashes is not None, "TEST pixels receipt missing")
                    expected={"selection_sha256":release_sha, "checkpoint_sha256":c["best_sha256"], "test_pixel_sha256":canonical(test_hashes),"test_manifest_sha256":study["dataset"]["files"]["test"]["sha256"]}
                    test_completion_verified(metrics_path.parent,expected)
                m = read_json(metrics_path)
                records.append(dict(seed=seed, method="RetinaNet" if method == "retinanet" else method, split=split, selected_epoch=c["selected_epoch"], train_seconds=c["total_train_val_seconds"], inference_seconds=m["inference_seconds"], checkpoint_sha256=c["best_sha256"], protocol_sha256=canonical(study), metrics=m, smoke=study["smoke"]))
    return write_report(records, report_dir, expected_seeds=study["config"]["seeds"], expected_methods=["RetinaNet", *study["config"]["arms"]])


def fetch_weights(output):
    """Explicit official generic ImageNet downloads, no HELMET-trained checkpoints."""
    import torch
    from torchvision.models import ResNet18_Weights, ResNet50_Weights
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    assets = {"resnet18-f37072fd.pth": ResNet18_Weights.IMAGENET1K_V1.url, "resnet50-0676ba61.pth": ResNet50_Weights.IMAGENET1K_V1.url}
    receipts = {}
    for name, url in assets.items():
        require(url == "https://download.pytorch.org/models/" + name, "Official asset URL differs")
        path = output/name
        prefix = name.split("-")[1].split(".")[0]
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=output, prefix=".imagenet-") as tmp:
                staged = Path(tmp)/name
                torch.hub.download_url_to_file(url, str(staged), hash_prefix=prefix)
                require(digest(staged).startswith(prefix), "ImageNet hash differs")
                os.link(staged, path)
        require(digest(path).startswith(prefix), "Existing ImageNet file differs")
        receipts[name] = dict(sha256=digest(path), bytes=path.stat().st_size, url=url)
    atomic_json(output/"imagenet_receipt.json", receipts)
    return receipts


def doctor(device):
    r = runtime(device)
    import torch
    from torchvision.ops import nms
    x = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device=device)
    keep = nms(x, torch.tensor([.9,.8], device=device), .5)
    require(keep.tolist() == [0], "NMS kernel smoke failed")
    loss = (torch.ones(8, device=device, requires_grad=True)*2).sum(); loss.backward()
    return {**r, "nms_kernel_passed": True, "autograd_kernel_passed": True, "gpu_memory": list(torch.cuda.mem_get_info(device)) if device.startswith("cuda") else None, "full_resolution_fit_verified": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--resplit", type=Path, required=True)
    prepare.add_argument("--test-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    weights = actions.add_parser("weights"); weights.add_argument("--output", type=Path, default=Path("pretrained"))
    doc = actions.add_parser("doctor"); doc.add_argument("--device", default="cuda:0")
    stage = actions.add_parser("stage-images")
    stage.add_argument("--manifests", type=Path, required=True)
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--images", type=Path, required=True)
    stage.add_argument("--trainval-only", action="store_true")
    smoke = actions.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True); smoke.add_argument("--device", default="cpu")
    check = actions.add_parser("check")
    for target in (check,):
        target.add_argument("--manifests", type=Path, required=True); target.add_argument("--images", type=Path, required=True)
    check.add_argument("--include-test", action="store_true"); check.add_argument("--output", type=Path)
    for name in ("fit", "test"):
        sub = actions.add_parser(name)
        sub.add_argument("--manifests", type=Path, required=True); sub.add_argument("--images", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True); sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--max-hours", type=float, required=True)
        if name == "fit":
            sub.add_argument("--config", type=Path, default=Path("configs/helmet2020_training_kit_v1.json"))
            sub.add_argument("--weights", type=Path, default=Path("pretrained")); sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stop-after-epoch", type=int)
        else:
            sub.add_argument("--selection-sha256", required=True)
    report = actions.add_parser("report"); report.add_argument("--output", type=Path, required=True); report.add_argument("--report-dir", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "prepare":
        from .kit_data import prepare_manifests
        result = prepare_manifests(args.resplit, args.test_manifest, args.output)
    elif args.action == "stage-images":
        from .kit_data import stage_images
        result = stage_images(args.manifests, args.source, args.images, splits=("train","val") if args.trainval_only else ("train","val","test"))
        result = {k:v for k,v in result.items() if k not in ("files", "pixel_verification")}
    elif args.action == "smoke":
        from .kit_smoke import run_smoke
        result = run_smoke(args.output, args.device)
    elif args.action == "check":
        from .kit_data import check_pixels
        result = check_pixels(args.manifests, args.images, splits=("train","val","test") if args.include_test else ("train","val"), output=args.output)
        result = {k:v for k,v in result.items() if k not in ("image_sha256", "images", "files")}
    elif args.action == "doctor": result = doctor(args.device)
    elif args.action == "weights": result = fetch_weights(args.output)
    elif args.action == "fit": result = fit(args)
    elif args.action == "test": result = final_test(args)
    else: result = collect_report(args.output, args.report_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 2 if isinstance(result, dict) and result.get("status") == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
