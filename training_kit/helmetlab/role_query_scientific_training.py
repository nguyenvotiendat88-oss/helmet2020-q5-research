"""TRAIN-only matched-arm fit core; callers own data and execution admission.

No model construction, provider access, validation, score selection or launch
logic lives here. Epoch/order seeds are zero-based; committed history is
one-based. A paused fit never selects a model as the final planned model.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import stat
import time

from helmetlab.role_query_study_contract import (
    ARMS, canonical_sha, exposure, valid_sha, validate_binding, validate_recipe,
)

SCHEMA = "helmetlab.helmet2020.role_query_scientific_fit.v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def epoch_order_seed(seed, epoch):
    """Return the fixed TRAIN permutation seed for a zero-based epoch."""
    _require(type(seed) is int and 0 <= seed < 2**63
        and type(epoch) is int and epoch >= 0, "Invalid epoch-order seed/cursor")
    key = f"role_query_epoch_order_v1:{seed}:{epoch}"
    return int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "little") % (2**63 - 1)


def semantic_state_sha256(state):
    """Hash named tensor values, independently of Torch serialization bytes.

    Callers must obtain execution admission before passing CUDA tensors.
    Backbone identity uses its native state-dict names without a prefix.
    """
    torch = importlib.import_module("torch")
    _require(isinstance(state, dict) and state and all(type(name) is str for name in state),
        "A nonempty named tensor state is required")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
            and not tensor.is_meta, "Only materialized dense tensor state is supported")
        tensor = tensor.detach().cpu().contiguous()
        _require(not tensor.is_floating_point() or tensor.dtype == torch.float32,
            "Scientific model floating state must be FP32")
        _require(bool(torch.isfinite(tensor).all()), "Nonfinite initial model state")
        header = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fresh_output(value):
    path = Path(value).expanduser().absolute()
    _require(".." not in path.parts, "Output parent traversal is forbidden")
    for member in (path, *path.parents):
        if not member.exists() and not member.is_symlink():
            continue
        info = member.lstat()
        _require(not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
            "Linked/reparse output path")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Scientific fit requires a fresh output directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path, value):
    with Path(path).open("x", encoding="ascii", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _event(stream, value):
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _validate_model(net, torch, binding, arm):
    _require(getattr(net, "arm", None) == arm and callable(getattr(net, "architecture_metadata", None)),
        "Scientific fit model arm/metadata API differs")
    metadata = net.architecture_metadata()
    _require(canonical_sha(metadata) == canonical_sha(binding["arm_metadata"][arm]),
        "Scientific fit architecture/grouping metadata differs")
    parameters = list(net.parameters())
    _require(parameters and all(parameter.requires_grad and parameter.dtype == torch.float32
        and parameter.device == parameters[0].device and not parameter.is_meta
        and parameter.layout == torch.strided for parameter in parameters),
        "Every model parameter must train on one materialized FP32 device")
    device = parameters[0].device
    _require(device.type in ("cpu", "cuda") and all(buffer.device == device
        and not buffer.is_meta and buffer.layout == torch.strided
        and (not buffer.is_floating_point() or buffer.dtype == torch.float32)
        for buffer in net.buffers()), "Scientific fit model buffer/device contract differs")
    _require(hasattr(net, "backbone") and callable(getattr(net.backbone, "state_dict", None)),
        "Scientific fit requires its separately bound backbone")
    _require(not any(module._load_state_dict_pre_hooks or module._load_state_dict_post_hooks
        for module in net.modules()), "Scientific model load hooks are unsupported")
    return device, metadata


def _configure_rng(torch, np, seed, device, cpu_threads):
    if device.type == "cpu":
        _require(not torch.cuda.is_available(), "CPU fit must be launched with CUDA isolated")
    else:
        _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
            "Set deterministic CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _validate_dataset(dataset, recipe, binding, cached_crops_type):
    _require(type(dataset) is cached_crops_type and getattr(dataset, "training", None) is True
        and type(getattr(dataset, "seed", None)) is int and dataset.seed == recipe["seed"]
        and type(getattr(dataset, "epoch", None)) is int and dataset.epoch >= 0
        and type(dataset.items) is list and len(dataset) == binding["crop_count"]
        and canonical_sha(dataset.items) == binding["items_sha256"],
        "Bound TRAIN CachedCrops population/seed/training contract differs")
    canonical = cached_crops_type(dataset.pixels_path, dataset.items, training=True, seed=recipe["seed"])
    actual, expected = dataset.transform, canonical.transform
    _require(type(actual) is type(expected) and set(vars(actual)) == {"transforms"}
        and type(actual.transforms) is list and len(actual.transforms) == len(expected.transforms)
        and all(type(left) is type(right) and vars(left) == vars(right)
            for left, right in zip(actual.transforms, expected.transforms)),
        "CachedCrops augmentation transform differs from its canonical factory")


def _file_identity(path):
    path = Path(path).expanduser().absolute()
    _require(".." not in path.parts, "Cache file parent traversal is forbidden")
    for member in (path, *path.parents):
        info = member.lstat()
        _require(not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
            "Linked/reparse cache input path")
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
        "Cache pixels must be a singleton regular file")
    return path, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_pixels(dataset, np, binding):
    _require(dataset.pixels is None, "Supply a fresh CachedCrops with unopened pixels")
    path, before = _file_identity(dataset.pixels_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    _require(digest.hexdigest() == binding["cache_pixels_sha256"], "Bound cache pixels SHA256 differs")
    pixels = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        _require(pixels.dtype == np.uint8 and pixels.shape == (binding["crop_count"], 224, 224, 3),
            "Bound cache pixels must be crop_count x 224 x 224 x RGB uint8")
        _require(_file_identity(path)[1] == before, "Cache pixels changed during verification")
    except BaseException:
        if getattr(pixels, "_mmap", None) is not None:
            pixels._mmap.close()
        raise
    dataset.pixels = pixels
    verified = {"path": path, "file_signature": before, "pixels": pixels, "mmap": pixels._mmap,
        "offset": pixels.offset, "strides": pixels.strides}
    receipt = {"pixels_path": str(path), "cache_pixels_sha256": digest.hexdigest(),
        "cache_pixels_bytes": before[2], "items_sha256": binding["items_sha256"],
        "crop_count": binding["crop_count"]}
    return receipt, verified


def _assert_pixel_binding(dataset, np, verified, binding):
    _require(type(dataset.pixels_path) is str
        and Path(dataset.pixels_path).expanduser().absolute() == verified["path"],
        "CachedCrops pixels path changed after verification")
    _require(_file_identity(verified["path"])[1] == verified["file_signature"],
        "Cache pixels file identity/signature changed after verification")
    pixels = dataset.pixels
    _require(pixels is verified["pixels"] and type(pixels) is np.memmap
        and getattr(pixels, "_mmap", None) is verified["mmap"] and not verified["mmap"].closed
        and Path(pixels.filename).absolute() == verified["path"]
        and pixels.mode == "r" and not pixels.flags.writeable and pixels.dtype == np.uint8
        and pixels.shape == (binding["crop_count"], 224, 224, 3)
        and pixels.offset == verified["offset"] and pixels.strides == verified["strides"],
        "CachedCrops opened pixels mapping/read-only provenance changed")


def _batch(dataset, indices, torch, num_classes):
    images, ids = [], []
    for index in indices:
        sample = dataset[index]
        _require(isinstance(sample, (tuple, list)) and len(sample) == 2,
            "CachedCrops must yield exactly an image and class ID")
        image, cid = sample
        _require(isinstance(image, torch.Tensor) and image.device.type == "cpu"
            and image.dtype == torch.float32 and tuple(image.shape) == (3, 224, 224)
            and image.layout == torch.strided and not image.is_meta and not image.requires_grad,
            "CachedCrops must yield CPU FP32 3x224x224 images")
        _require(type(cid) is int and 0 <= cid < num_classes, "Invalid TRAIN crop class ID")
        images.append(image)
        ids.append(cid)
    images = torch.stack(images)
    _require(bool(torch.isfinite(images).all()), "Nonfinite TRAIN crop pixels")
    return images, torch.tensor(ids, dtype=torch.int64)


def _finite_gradients(net, torch):
    gradients = [parameter.grad for parameter in net.parameters()]
    _require(gradients and all(gradient is not None and gradient.layout == torch.strided
        for gradient in gradients), "Every matched model parameter must receive a dense gradient")
    _require(bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()),
        "Nonfinite TRAIN gradient")


def train_arm(net, dataset, states, recipe, binding, arm, output,
              check_deadline, admission_check, resume=None, stop_after_epoch=None):
    """Fit a supplied bound arm using only TRAIN and complete-epoch commits.

    ``resume`` must be ``{"path": path, "sha256": lowercase_sha256}`` and uses
    a newly constructed, bound initial model plus a fresh output directory.
    ``stop_after_epoch`` is an absolute completed-epoch boundary. Data/cache
    manifest/source verification belongs to caller preflight. This core also
    verifies exact CachedCrops type, items, class states and pixel file bytes.
    Both callbacks must raise or return False to deny work.
    """
    validate_recipe(recipe)
    validate_binding(binding)
    _require(type(arm) is str and arm in ARMS, "Unknown matched-study arm")
    _require(callable(check_deadline) and callable(admission_check),
        "Explicit deadline and execution-admission callbacks are required")
    boundary = recipe["epochs"] if stop_after_epoch is None else stop_after_epoch
    _require(type(boundary) is int and 1 <= boundary <= recipe["epochs"],
        "Stop boundary must be an explicit planned completed epoch")
    if resume is not None:
        _require(type(resume) is dict and set(resume) == {"path", "sha256"}
            and isinstance(resume["path"], (str, os.PathLike)) and valid_sha(resume["sha256"]),
            "Resume requires an exact checkpoint path and SHA256")
        resume = {"path": str(resume["path"]), "sha256": resume["sha256"]}
    recipe_sha, binding_sha = canonical_sha(recipe), canonical_sha(binding)
    frozen_recipe, frozen_binding = copy.deepcopy(recipe), copy.deepcopy(binding)
    pixel_binding = None

    def guard():
        _require(check_deadline() is not False, "Scientific fit deadline denied work")
        _require(admission_check() is not False, "Scientific fit admission denied work")
        _require(canonical_sha(recipe) == recipe_sha and canonical_sha(binding) == binding_sha,
            "Scientific recipe/binding changed during fit")
        if pixel_binding is not None:
            _assert_pixel_binding(dataset, np, pixel_binding, frozen_binding)

    guard()
    output = _fresh_output(output)
    history, checkpoints = [], []
    completed_epochs = 0
    epoch_crops = epoch_updates = 0
    started = time.perf_counter()
    events = None
    try:
        _write_json(output / "recipe.json", frozen_recipe)
        _write_json(output / "binding.json", frozen_binding)
        events = (output / "timing_events.jsonl").open("x", encoding="ascii", newline="\n")
        _event(events, {"event": "fit_started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm": arm, "resume": resume})
        preflight_started = time.perf_counter()
        torch, np = importlib.import_module("torch"), importlib.import_module("numpy")
        checkpoint = importlib.import_module("helmetlab.role_query_scientific_checkpoint")
        classification_loss = importlib.import_module("helmetlab.role_queries").classification_loss
        cached_crops_type = importlib.import_module("scripts.train_balanced_crops").CachedCrops
        guard()
        device, metadata = _validate_model(net, torch, frozen_binding, arm)
        _validate_dataset(dataset, frozen_recipe, frozen_binding, cached_crops_type)
        _configure_rng(torch, np, frozen_recipe["seed"], device, frozen_recipe["cpu_threads"])
        guard()
        initial_model = semantic_state_sha256(net.state_dict())
        initial_backbone = semantic_state_sha256(net.backbone.state_dict())
        _require(initial_model == frozen_binding["initial_model_sha256"][arm]
            and initial_backbone == frozen_binding["initial_backbone_sha256"],
            "Bound initial model/backbone semantic SHA256 differs")
        _write_json(output / "initialization.json", {"arm": arm, "device": str(device),
            "architecture_metadata": metadata, "initial_model_sha256": initial_model,
            "initial_backbone_sha256": initial_backbone, "recipe_sha256": recipe_sha,
            "binding_sha256": binding_sha, "initialization_scope": frozen_binding["initialization_scope"]})
        _require(isinstance(states, torch.Tensor) and states.device.type == "cpu"
            and states.dtype in (torch.int32, torch.int64)
            and tuple(states.shape) == (frozen_binding["num_classes"], 5)
            and states.layout == torch.strided and not states.is_meta
            and bool(((states >= 0) & (states <= 2)).all()),
            "Explicit CPU class-map native states differ from bound population")
        class_states_sha = canonical_sha(states.tolist())
        _require(class_states_sha == frozen_binding["class_states_sha256"],
            "Bound class-map native states SHA256 differs")
        data_verification, pixel_binding = _verify_pixels(dataset, np, frozen_binding)
        data_verification["class_states_sha256"] = class_states_sha
        _write_json(output / "data_verification.json", data_verification)
        _event(events, {"event": "input_preflight_verified",
            "preflight_seconds_including_imports": time.perf_counter() - preflight_started,
            "elapsed_seconds": time.perf_counter() - started})
        guard()
        states = states.detach().clone().to(device=device, dtype=torch.int64)
        optimizer = torch.optim.AdamW(net.parameters(), lr=frozen_recipe["lr"],
            weight_decay=frozen_recipe["weight_decay"], betas=tuple(frozen_recipe["optimizer"]["betas"]),
            eps=frozen_recipe["optimizer"]["eps"], amsgrad=False, maximize=False,
            foreach=False, capturable=False, differentiable=False, fused=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=frozen_recipe["epochs"], eta_min=frozen_recipe["scheduler"]["eta_min"])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(frozen_recipe["seed"])
        if resume is not None:
            guard()
            completed_epochs, history = checkpoint.restore_epoch(resume["path"], net, optimizer,
                scheduler, generator, frozen_recipe, frozen_binding, arm, resume["sha256"])
            _require(completed_epochs < boundary,
                "Resume checkpoint already reaches or exceeds the requested boundary")
            checkpoints.append({"completed_epochs": completed_epochs, **resume, "origin": "resume"})
            _event(events, {"event": "resume_restored", "completed_epochs": completed_epochs,
                "elapsed_seconds": time.perf_counter() - started})
        for epoch in range(completed_epochs, boundary):
            guard()
            _validate_dataset(dataset, frozen_recipe, frozen_binding, cached_crops_type)
            dataset.epoch = epoch
            generator.manual_seed(epoch_order_seed(frozen_recipe["seed"], epoch))
            order = torch.randperm(frozen_binding["crop_count"], generator=generator, device="cpu").tolist()
            net.train()
            epoch_started = time.perf_counter()
            epoch_crops = epoch_updates = 0
            loss_sum = 0.
            _event(events, {"event": "epoch_started", "epoch": epoch + 1,
                "order_seed": epoch_order_seed(frozen_recipe["seed"], epoch),
                "elapsed_seconds": epoch_started - started})
            for offset in range(0, len(order), frozen_recipe["batch_size"]):
                guard()
                images, ids = _batch(dataset, order[offset:offset + frozen_recipe["batch_size"]],
                    torch, frozen_binding["num_classes"])
                guard()
                images, ids = images.to(device), ids.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = net(images)
                    _require(isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
                        and logits.device == device and tuple(logits.shape) == (len(ids), 10)
                        and bool(torch.isfinite(logits).all()), "Expected finite native-ten FP32 TRAIN logits")
                    loss = classification_loss(logits, ids, states)
                    _require(loss.ndim == 0 and loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
                        "Nonfinite or malformed native TRAIN loss")
                    loss.backward()
                _finite_gradients(net, torch)
                optimizer.step()
                count = len(ids)
                loss_sum += float(loss.detach().cpu()) * count
                epoch_crops += count
                epoch_updates += 1
                _event(events, {"event": "batch_completed", "epoch": epoch + 1,
                    "updates_in_epoch": epoch_updates, "crops_in_epoch": epoch_crops,
                    "elapsed_seconds": time.perf_counter() - started})
            expected = exposure(frozen_recipe, frozen_binding["crop_count"], 1)
            _require(epoch_crops == expected["seen_crops"] and epoch_updates == expected["completed_updates"],
                "Completed epoch did not consume the exact TRAIN crop population")
            _validate_dataset(dataset, frozen_recipe, frozen_binding, cached_crops_type)
            guard()
            scheduler.step()
            net.eval()
            optimizer.zero_grad(set_to_none=True)
            row = {"epoch": epoch + 1, "mean_loss": loss_sum / epoch_crops,
                "updates": epoch_updates, "crops": epoch_crops, "lr_after": optimizer.param_groups[0]["lr"]}
            candidate_history = history + [row]
            checkpoint_path = output / f"epoch_{epoch + 1:04d}.pt"
            guard()
            checkpoint_sha = checkpoint.save_epoch(checkpoint_path, net, optimizer, scheduler,
                generator, frozen_recipe, frozen_binding, arm, epoch + 1, candidate_history)
            history = candidate_history
            completed_epochs = epoch + 1
            checkpoints.append({"completed_epochs": completed_epochs, "path": str(checkpoint_path),
                "sha256": checkpoint_sha, "origin": "fit"})
            epoch_crops = epoch_updates = 0
            _event(events, {"event": "epoch_committed", "epoch": completed_epochs,
                "epoch_seconds_including_commit": time.perf_counter() - epoch_started,
                "elapsed_seconds": time.perf_counter() - started, "checkpoint_sha256": checkpoint_sha})
        guard()
        status = "completed" if completed_epochs == frozen_recipe["epochs"] else "paused"
        result = {"schema": SCHEMA, "status": status, "arm": arm, "recipe_sha256": recipe_sha,
            "binding_sha256": binding_sha, **exposure(frozen_recipe, frozen_binding["crop_count"], completed_epochs),
            "history": history, "checkpoints": checkpoints, "last_checkpoint": checkpoints[-1],
            "selected_checkpoint": checkpoints[-1] if status == "completed" else None,
            "selection": frozen_recipe["selection"], "fit_scope": frozen_recipe["fit_scope"]}
        _event(events, {"event": "fit_finished", "status": status,
            "elapsed_seconds": time.perf_counter() - started})
        events.close()
        events = None
        _write_json(output / "history.json", history)
        _write_json(output / "result.json", result)
        return result
    except BaseException as error:
        failure = {"schema": SCHEMA, "status": "failed", "arm": arm,
            "recipe_sha256": recipe_sha, "binding_sha256": binding_sha,
            **exposure(frozen_recipe, frozen_binding["crop_count"], completed_epochs),
            "history": history, "checkpoints": checkpoints, "selected_checkpoint": None,
            "uncommitted_epoch_updates": epoch_updates, "uncommitted_epoch_crops": epoch_crops,
            "error_type": type(error).__name__, "error": str(error)}
        try:
            _write_json(output / "failure.json", failure)
            if events is not None:
                _event(events, {"event": "fit_failed", "elapsed_seconds": time.perf_counter() - started,
                    "completed_epochs": completed_epochs})
        except Exception:
            pass
        raise
    finally:
        if events is not None:
            events.close()
