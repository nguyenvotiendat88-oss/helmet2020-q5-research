"""Offline snapshot integrity check; never executes archived experiments."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"provenance/export_manifest.json").read_text())
for row in manifest["files"]:
    path=ROOT/row["path"]
    assert path.is_file(),row["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row["export_sha256"],row["path"]
for path in ROOT.rglob("*.py"):
    if ".git" not in path.parts:ast.parse(path.read_text())
for path in ROOT.rglob("*.json"):
    if ".git" not in path.parts:json.loads(path.read_text())
b=json.loads((ROOT/"evidence/gq5_three_seeds_verified_20261029.json").read_text())
assert b["all_checks_passed"] and b["summary"]["seeds_available"]==[0,1,2]
e=json.loads((ROOT/"evidence/ensemble3_final_verified_20261029.json").read_text())
assert e["all_checks_passed"] and e["result"]["status"]=="completed"
for model in e["result"]["arms"].values():
    for value in model["evaluation"].values():
        assert value["metrics"]["frames"]==9400 and value["metrics"]["supported_classes"]==24
assert not list(ROOT.rglob("*.pt")) and not list(ROOT.rglob("*.pth"))
print(f"PASS: {len(manifest['files'])} exported files, Python/JSON syntax, metrics populations and no checkpoints.")
