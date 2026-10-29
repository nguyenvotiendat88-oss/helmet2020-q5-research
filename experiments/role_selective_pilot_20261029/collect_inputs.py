"""Collect cached selective-prediction inputs without GPU or remote mutation."""
import hashlib,io,json,tarfile,sys
from pathlib import Path
ROOT=Path("/workspace/HELMET2020_training_kit_v1")
DIAG=Path("/workspace/runs/q5_audit_20261028")
CAL=Path("/workspace/runs/gq5_calibration_pilot_20261028")
BASE=Path("/workspace/runs/gq5_extend24_20261028")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(8*1024*1024),b""):h.update(part)
    return h.hexdigest()

files={"G_val_gt_logits.npz":DIAG/"G_val_gt_logits.npz","Q5_val_gt_logits.npz":DIAG/"Q5_val_gt_logits.npz",
       "samples.json":DIAG/"samples.json","calibration.json":CAL/"calibration.json",
       "classes.json":ROOT/"manifests/classes.json"}
diagnostic=json.loads((DIAG/"diagnostic.json").read_text())
recs=json.loads((BASE/"comparison_result.json").read_text())
meta={"schema":"helmet2020.selective_inputs.v1","scope":"cached_seed0_epoch24_GT_crop_logits",
      "test_accessed":False,"remote_mutations":False,
      "val_manifest_sha256":sha(ROOT/"manifests/val.jsonl"),
      "files":{name:{"sha256":sha(p),"bytes":p.stat().st_size} for name,p in files.items()},"checkpoints":{}}
for arm in ("G","Q5"):
    expected=next(r["last_checkpoint_sha256"] for r in recs["arms"] if r["arm"]==arm)
    assert diagnostic["models"][arm]["checkpoint_sha256"]==expected
    actual=sha(BASE/arm/"epoch24.pt")
    assert actual==expected
    meta["checkpoints"][arm]=actual
payload=json.dumps(meta,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as archive:
    for name,path in files.items():archive.add(path,arcname=name,recursive=False)
    info=tarfile.TarInfo("meta.json");info.size=len(payload)
    archive.addfile(info,io.BytesIO(payload))
