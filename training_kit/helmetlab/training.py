from __future__ import annotations
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import yaml
from .common import write_json, digest
from .data import verify_lock, verify_yolo_export


def source_hashes():
    return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def environment():
    versions={}
    for name in ("torch","torchvision","ultralytics","opencv-python","numpy","Pillow","lap"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    import torch
    return {"python":platform.python_version(),"platform":platform.platform(),"versions":versions,
            "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def train_detector(dataset, config, output, device="0", seed=0, resume=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    class WeightedTrainer(DetectionTrainer):
        def validate(self):
            metrics=self.validator(self)
            if metrics is None: return None,None
            m=self.validator.metrics
            ids=np.asarray(m.box.ap_class_index,dtype=int)
            counts=m.nt_per_class[ids].astype(float)
            score=float(np.dot(counts,m.box.ap50)/counts.sum()) if counts.sum() else 0.
            metrics.pop("fitness",None)
            metrics["metrics/weighted_AP50(B)"]=score
            if self.best_fitness is None or score>self.best_fitness: self.best_fitness=score
            return metrics,score
    dataset=Path(dataset).resolve(); lock=verify_lock(dataset)
    export_check=verify_yolo_export(dataset)
    # Ultralytics label caches are derived files. Force re-reading verified labels.
    for split in ('train','val','test'):
        (dataset/'labels'/f'{split}.cache').unlink(missing_ok=True)
    cfg=yaml.safe_load(Path(config).read_text())
    model_name=cfg.pop("model"); output=Path(output)
    previous=None
    if resume:
        if not (output/'run.json').is_file():raise FileNotFoundError('Resume needs original run.json')
        from .common import read_json
        previous=read_json(output/'run.json')
        if previous.get('dataset_lock_sha256')!=lock or previous.get('source_sha256')!=source_hashes():
            raise ValueError('Dataset/code changed since detector run')
        if previous.get('model')!=model_name or previous.get('config_sha256')!=digest(config):
            raise ValueError('Detector config changed; choose a new run')
        if previous.get('train',{}).get('seed')!=seed or str(previous.get('train',{}).get('device'))!=str(device):
            raise ValueError('Detector seed/device changed; resume the original run configuration')
        if previous.get('environment',{}).get('versions')!=environment()['versions']:
            raise ValueError('Package versions changed since detector run')
        if (output/'complete.json').is_file():return str(output/'weights/best.pt')
        if not (output/'weights/last.pt').is_file():raise FileNotFoundError('No resumable detector checkpoint; choose a new run')
    elif output.exists() and any(output.iterdir()): raise FileExistsError("Choose a new run directory or use --resume")
    output.mkdir(parents=True,exist_ok=True)
    cfg.update(data=str(dataset/"dataset.yaml"),project=str(output.parent.resolve()),name=output.name,
               exist_ok=True,device=device,seed=seed,deterministic=True,amp=True)
    run={"model":model_name,"train":cfg,"dataset_lock_sha256":lock,'config_sha256':digest(config),
                "environment":environment(),"source_sha256":source_hashes(),"export_check":export_check,
                "selection":"validation weighted Ultralytics AP50 (101-point); not historical AP scorer",
                "status":"training"}
    write_json(output/"run.json",run)
    from .common import read_json
    timings=read_json(output/'epoch_timing.json') if (output/'epoch_timing.json').is_file() else []
    clock={}
    def epoch_start(trainer):
        if trainer.device.type=='cuda':torch.cuda.synchronize(trainer.device);torch.cuda.reset_peak_memory_stats(trainer.device)
        clock['start']=time.perf_counter();clock['epoch']=trainer.epoch
    def epoch_end(trainer):
        if trainer.device.type=='cuda':torch.cuda.synchronize(trainer.device)
        if 'start' not in clock or trainer.epoch!=clock['epoch']:return
        row={'epoch':trainer.epoch+1,'train_and_validation_wall_seconds':time.perf_counter()-clock['start'],
             'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type=='cuda' else None}
        timings[:]=[r for r in timings if r['epoch']!=row['epoch']];timings.append(row)
        write_json(output/'epoch_timing.json',timings)
    model=YOLO(str(output/'weights/last.pt') if resume else model_name)
    model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_fit_epoch_end',epoch_end)
    try:
        model.train(trainer=WeightedTrainer,**({**cfg,'resume':str(output/'weights/last.pt')} if resume else cfg))
    except BaseException:
        write_json(output/'run.json',{**run,'status':'interrupted_or_failed'});raise
    write_json(output/"complete.json",{"best":str(output/"weights/best.pt"),"dataset_lock_sha256":lock,
                                     "benchmark_claim":False})
    write_json(output/'run.json',{**run,'status':'completed'})
    return str(output/"weights/best.pt")
