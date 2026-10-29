"""Synthetic mechanics fixture for the portable kit; never a HELMET score."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from .common import digest, write_jsonl
from . import kit_data
from .kit_runner import PROJECT, atomic_json, load_config, fit, final_test, collect_report


def make_fixture(root):
    from PIL import Image, ImageDraw
    root = Path(root)
    if root.exists():
        raise FileExistsError("Synthetic fixture requires a fresh directory")
    root.mkdir(parents=True)
    manifests, pixels = root/"manifests", root/"pixels"
    manifests.mkdir(); pixels.mkdir()
    shutil.copyfile(PROJECT/"configs/classes.json", manifests/"classes.json")
    files = {"classes": {"path":"classes.json", "bytes":(manifests/"classes.json").stat().st_size, "sha256":digest(manifests/"classes.json")}}
    counts = {}
    for n, split in enumerate(("train", "val", "test")):
        rows = []
        for j in range(2):
            cid, frame = f"Fixture_{split}", j+1
            iid = f"{cid}__{frame:03d}"
            objects = [] if j else [{"bbox":[4.,4.,28.,56.],"class_id":0,"track_id":"a"}, {"bbox":[32.,8.,60.,60.],"class_id":2,"track_id":"b"}]
            row = dict(image_id=iid,image=f"images/{split}/{iid}.jpg",clip_id=cid,site_id="Synthetic",frame_index=frame,width=64,height=64,objects=objects)
            path = pixels/row["image"]; path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB",(64,64),(30+n*40,10+j*60,80+n*25))
            draw = ImageDraw.Draw(image); draw.rectangle((4,4,28,56),fill=(200-n*10,60,30+j*40)); draw.line((0,0,63,63),fill=(20,n*60,j*80),width=3)
            image.save(path,"JPEG",quality=90)
            rows.append(row)
        write_jsonl(manifests/f"{split}.jsonl", rows)
        p = manifests/f"{split}.jsonl"
        files[split] = dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p))
        counts[split] = kit_data._counts(rows,36)
    atomic_json(manifests/"dataset.json",dict(schema=kit_data.SCHEMA,num_classes=36,files=files,splits=counts,protocol={"synthetic_only":True,"not_HELMET":True}))
    return manifests,pixels


def smoke_config(arms=None,epochs=1):
    cfg = copy.deepcopy(load_config(PROJECT/"configs/helmet2020_training_kit_v1.json"))
    cfg.update(seeds=[0],cpu_threads=1,min_free_disk_gib=.25,arms=arms or cfg["arms"])
    cfg["detector"].update(epochs=epochs,batch=2,min_size=64,max_size=64,amp=False,score_thresh=.005,detections_per_img=8,topk_candidates=16)
    cfg["heads"].update(epochs=epochs,batch=2,crop_size=64,score_thresh=.000001,detections_per_img=8)
    return cfg


def run_smoke(output, device="cpu"):
    root = Path(output).resolve()
    tick = time.perf_counter()
    manifests,images = make_fixture(root/"fixture")
    args = argparse.Namespace(manifests=manifests,images=images,output=root/"study",device=device,weights=root/"unused",resume=False,stop_after_epoch=None,max_hours=.5)
    cfg = smoke_config()
    result = fit(args, cfg, smoke=True)
    if result["status"] != "fit_completed_test_not_run":
        raise RuntimeError(f"Synthetic fit did not complete: {result}")
    args.selection_sha256 = digest(args.output/"selection.json")
    final_test(args)
    collect_report(args.output,root/"report_SYNTHETIC_ONLY")
    receipt = dict(status="synthetic_end_to_end_passed",device=device,seconds=time.perf_counter()-tick,methods=["RetinaNet",*cfg["arms"]],frames_per_split=2,selection_sha256=args.selection_sha256,real_HELMET_training=False,real_HELMET_AP=False)
    atomic_json(root/"smoke_receipt.json",receipt)
    return receipt


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu")
    args=p.parse_args()
    print(json.dumps(run_smoke(args.output,args.device),indent=2))
