from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import yaml
from .common import digest, safe_child, write_json, read_json, read_jsonl, write_jsonl
from .labels import LabelMap

METADATA = {
    "annotation.zip": "https://osf.io/download/buh57/",
    "data_split.csv": "https://osf.io/download/q7rmb/",
    "F-measure.zip": "https://osf.io/download/8gtfs/",
    "Readme.txt": "https://osf.io/download/pvecy/",
}
IMAGES = {
    "part_1.zip": "https://osf.io/download/452nq/",
    "part_2.zip": "https://osf.io/download/muzgb/",
    "part_3.zip": "https://osf.io/download/9dmw3/",
    "part_4.zip": "https://osf.io/download/39ynb/",
    "part_5.zip": "https://osf.io/download/v3rch/",
    "part_6.zip": "https://osf.io/download/jyg9s/",
    "part_7.zip": "https://osf.io/download/6h5ka/",
}


def extract(archive, root):
    with ZipFile(archive) as z:
        for info in z.infolist():
            safe_child(root, info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symbolic links are unsupported")
        z.extractall(root)


def fetch(root, images=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    selected = dict(METADATA)
    if images: selected.update(IMAGES)
    checksums = {}
    for name, url in selected.items():
        target = root/name
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            temp = target.with_suffix(target.suffix+".partial")
            with urllib.request.urlopen(url, timeout=120) as r, temp.open("wb") as f:
                shutil.copyfileobj(r,f,1024*1024)
            temp.rename(target)
        checksums[name] = {"sha256":digest(target), "bytes":target.stat().st_size, "source":url}
        if name.endswith(".zip"): extract(target,root)
    write_json(root/"download_manifest.json", checksums)
    return checksums


def splits(path):
    result = {}
    with open(path, encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            clip = r["video_id"]
            if clip in result: raise ValueError(f"Duplicate split entry: {clip}")
            split = {"training":"train","validation":"val","test":"test"}[r["Set"]]
            result[clip] = split
    return result


def annotations(path, labels):
    names = {c["name"]:c["id"] for c in labels.classes}
    frames = defaultdict(list); track_labels = {}
    with open(path, encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            frame = int(r["frame_id"])
            if not 1 <= frame <= 100: raise ValueError(f"Frame outside 1..100 in {path}")
            if r["label"] not in names: raise ValueError(f"Unknown label: {r['label']}")
            cid = names[r["label"]]; tid = r["track_id"]
            if tid in track_labels and track_labels[tid] != cid:
                raise ValueError(f"Changing ground-truth class within track: {path}/{tid}")
            track_labels[tid] = cid
            x,y,w,h = [float(r[k]) for k in ("x","y","w","h")]
            if w <= 0 or h <= 0: raise ValueError(f"Invalid box: {path}/{r}")
            if any(o["track_id"] == tid for o in frames[frame]): raise ValueError("Duplicate track/frame")
            frames[frame].append({"class_id":cid,"track_id":tid,"bbox":[x,y,x+w,y+h]})
    return frames, track_labels


def audit(raw, labels_path):
    raw = Path(raw); labels=LabelMap(labels_path); split=splits(raw/"data_split.csv")
    ann = {p.stem:p for p in (raw/"annotation").glob("*.csv")}
    if set(ann) != set(split):
        raise ValueError(f"Annotation/split mismatch: missing={sorted(set(split)-set(ann))[:8]}, extra={sorted(set(ann)-set(split))[:8]}")
    class_frames = Counter(); class_tracks = Counter(); by_split = Counter(); by_site = Counter()
    track_count = 0; frame_count=0
    for clip in sorted(split):
        frames,tracks=annotations(ann[clip],labels)
        track_count += len(tracks)
        class_tracks.update(tracks.values())
        for objs in frames.values():
            class_frames.update(o["class_id"] for o in objs)
            frame_count += len(objs)
        by_split[split[clip]] += 1
        by_site[clip.rsplit("_",1)[0]] += 1
    return {"source":"OSF 4pwj8", "clips":len(split),"frames_expected":100*len(split),
            "tracks":track_count,"annotated_boxes":frame_count,"split_clips":dict(by_split),"sites":dict(by_site),
            "class_frame_support":[class_frames[i] for i in range(len(labels))],
            "class_track_support":[class_tracks[i] for i in range(len(labels))],
            "split_sha256":digest(raw/"data_split.csv"),"class_map_sha256":digest(labels_path),
            "annotation_sha256":{k:digest(p) for k,p in sorted(ann.items())}}


def image_index(root, clips):
    """Find clip folders and numeric frame filenames; reject ambiguous matches."""
    root = Path(root); clips=set(clips); index={}
    for p in root.rglob("*"):
        if p.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
        clip = next((parent.name for parent in p.parents if parent.name in clips),None)
        if clip is None: continue
        m=re.search(r"(\d+)$",p.stem)
        if m is None: continue
        key=(clip,int(m.group(1)))
        if key in index: raise ValueError(f"Duplicate image for {key}")
        index[key]=p
    return index


def prepare(raw, image_root, output, labels_path, clip_limit=None, copy_images=False):
    raw, output=Path(raw),Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Choose an empty output directory to preserve the locked split")
    labels=LabelMap(labels_path); split=splits(raw/"data_split.csv")
    metadata=audit(raw,labels_path)
    selected=sorted(split)
    if clip_limit is not None:
        # Balanced selection only for smoke tests. Never relabel this as a benchmark.
        if clip_limit<1: raise ValueError("clip-limit must be positive")
        selected=[]
        for s in ("train","val","test"):
            selected += [c for c in sorted(split) if split[c]==s][:clip_limit]
    index=image_index(image_root,selected)
    missing=[(c,f) for c in selected for f in range(1,101) if (c,f) not in index]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[:5]}. Expected clip folders with frames numbered 1..100.")
    output.mkdir(parents=True,exist_ok=True)
    manifests={s:[] for s in ("train","val","test")}; clipped=0
    for clip in selected:
        ann,_=annotations(raw/"annotation"/(clip+".csv"),labels)
        subset=split[clip]
        for frame in range(1,101):
            src=index[(clip,frame)]
            with Image.open(src) as im: width,height=im.size
            stem=f"{clip}__{frame:03d}"
            image_rel=f"images/{subset}/{stem}{src.suffix.lower()}"
            dst=output/image_rel; dst.parent.mkdir(parents=True,exist_ok=True)
            if copy_images: shutil.copyfile(src,dst)
            else:
                try: dst.symlink_to(src.resolve())
                except OSError: shutil.copyfile(src,dst)
            objs=[]; lines=[]
            for obj in ann[frame]:
                b=obj["bbox"]
                box=[max(0.,b[0]),max(0.,b[1]),min(float(width),b[2]),min(float(height),b[3])]
                if box[2]<=box[0] or box[3]<=box[1]: raise ValueError(f"Box outside image: {clip}/{frame}")
                if box!=b: clipped+=1
                obj={**obj,"bbox":box};objs.append(obj)
                x1,y1,x2,y2=box
                lines.append(f"{obj['class_id']} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
            label_file=output/"labels"/subset/(stem+".txt");label_file.parent.mkdir(parents=True,exist_ok=True)
            label_file.write_text("\n".join(lines)+("\n" if lines else ""))
            manifests[subset].append({"image_id":stem,"image":image_rel,"clip_id":clip,"site_id":clip.rsplit("_",1)[0],"frame_index":frame,"width":width,"height":height,"objects":objs})
    metadata.update({"subset_only":clip_limit is not None or metadata["clips"]!=910,"selected_clips":selected,"clipped_boxes":clipped})
    for subset,records in manifests.items():
        write_jsonl(output/f"{subset}.jsonl",records)
    shutil.copyfile(labels_path,output/"classes.json")
    config={"path":str(output.resolve()),"train":"images/train","val":"images/val","test":"images/test","names":dict(enumerate(labels.names))}
    (output/"dataset.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    write_json(output/"audit.json",metadata)
    write_json(output/"lock.json",{f:digest(output/f) for f in ["train.jsonl","val.jsonl","test.jsonl","classes.json","audit.json"]})
    return metadata


def verify_lock(root):
    root=Path(root)
    for name,sha in read_json(root/"lock.json").items():
        if digest(root/name)!=sha: raise ValueError(f"Dataset lock changed: {name}")
    clips={s:{r["clip_id"] for r in read_jsonl(root/f"{s}.jsonl")} for s in ("train","val","test")}
    if clips["train"] & clips["val"] or clips["train"] & clips["test"] or clips["val"] & clips["test"]:
        raise ValueError("Clip leakage across splits")
    return digest(root/"lock.json")


def verify_yolo_export(root, *, image_label_splits=("train", "val", "test")):
    """Check what the detector will actually load against the locked manifests.

    Image contents are not hashed here; source archive hashes must be retained.
    This catches altered YOLO labels/YAML, extra files and broken image links.
    Train/val-only file checks must be explicit; all split manifests stay checked.
    """
    import numpy as np
    from .common import validate_frames
    image_label_splits = tuple(image_label_splits)
    if image_label_splits not in (("train", "val", "test"), ("train", "val")):
        raise ValueError("Image/label verification must cover train/val or train/val/test")
    root = Path(root).resolve()
    lock = verify_lock(root)
    labels = LabelMap(root/"classes.json")
    cfg = yaml.safe_load((root/"dataset.yaml").read_text(encoding="utf-8"))
    if Path(cfg.get("path", "")).resolve() != root:
        raise ValueError("dataset.yaml path does not match the locked dataset root")
    names = cfg.get("names")
    expected_names = dict(enumerate(labels.names))
    if names != expected_names and names != labels.names:
        raise ValueError("dataset.yaml class names differ from classes.json")
    counts = {}
    for split in ("train", "val", "test"):
        if cfg.get(split) != f"images/{split}":
            raise ValueError(f"dataset.yaml {split} path differs from the locked export")
        records = read_jsonl(root/f"{split}.jsonl")
        validate_frames(records, len(labels))
        expected_images, expected_labels = set(), set()
        for row in records:
            rel = Path(row["image"])
            if rel.parent != Path("images")/split or rel.is_absolute():
                raise ValueError(f"Unexpected image path: {rel}")
            width, height = row["width"], row["height"]
            if width <= 0 or height <= 0:
                raise ValueError("Nonpositive image dimensions")
            if split not in image_label_splits:
                continue
            image = root/rel
            if not image.is_file():
                raise FileNotFoundError(f"Missing/broken image: {image}")
            expected_images.add(image)
            label = root/"labels"/split/(rel.stem+".txt")
            expected_labels.add(label)
            lines = label.read_text(encoding="utf-8").splitlines()
            values = [[float(v) for v in line.split()] for line in lines if line.strip()]
            expected = []
            for obj in row["objects"]:
                x1,y1,x2,y2 = obj["bbox"]
                expected.append([obj["class_id"],(x1+x2)/(2*width),(y1+y2)/(2*height),
                                 (x2-x1)/width,(y2-y1)/height])
            if len(values) != len(expected) or any(len(v) != 5 for v in values):
                raise ValueError(f"YOLO label mismatch: {label}")
            if values and not np.allclose(values, expected, atol=6e-8, rtol=0):
                raise ValueError(f"YOLO label mismatch: {label}")
        if split in image_label_splits:
            actual_images = {p for p in (root/"images"/split).rglob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
            actual_labels = set((root/"labels"/split).rglob("*.txt"))
            if actual_images != expected_images or actual_labels != expected_labels:
                raise ValueError(f"Extra/missing image or label files in {split}")
        counts[split] = len(records)
    return {"dataset_lock_sha256":lock, "frames":counts,
            "yolo_export_matches_manifests":True, "image_content_hashes_checked":False,
            "image_label_splits_verified":list(image_label_splits)}
