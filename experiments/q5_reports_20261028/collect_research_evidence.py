"""Read-only collection and hash verification of completed research artifacts."""
import datetime,hashlib,json
from pathlib import Path
RUNS=Path("/workspace/runs")

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    cal=RUNS/"gq5_calibration_fullval_20261028"
    pilot=RUNS/"q5_objective_pilot_20261028"
    checks=[]
    def verify(path,expected):
        actual=sha(path)
        checks.append({"path":str(path),"sha256":actual,"expected":expected,"ok":actual==expected})
    data={"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "calibration_status":read(cal/"status.json"),"calibration_result":read(cal/"comparison.json"),
          "research_queue":read(pilot/"queue_status.json"),"pilot_status":read(pilot/"status.json"),
          "pilot_variants":{},"test_accessed_by_collector":False}
    if data["calibration_result"]:
        result=data["calibration_result"]
        for arm,record in result["arms"].items():
            verify(cal/f"{arm}_predictions.jsonl",record["predictions_sha256"])
            verify(RUNS/"gq5_extend24_20261028"/arm/"epoch24.pt",record["checkpoint_sha256"])
        plan=result["plan"]
        verify(RUNS/"gq5_calibration_pilot_20261028/calibration.json",plan["calibration_sha256"])
        verify(Path("/workspace/HELMET2020_training_kit_v1/manifests/val.jsonl"),plan["manifest_sha256"])
    for mode in ("omit_constant_presence","valid_state_conditional"):
        folder=pilot/mode
        entry={"status":read(folder/"status.json"),"progress":read(folder/"progress.json"),
               "comparison":read(folder/"comparison_result.json"),"arms":{}}
        for arm in ("G","Q5"):
            d=folder/arm
            receipt=read(d/"complete.json")
            history=read(d/"history.json")
            entry["arms"][arm]={"complete":receipt,"last_epoch":history[-1] if history else None}
            if receipt:
                for evaluation in receipt["evaluations"]:
                    epoch=evaluation["epoch"]
                    for relative,key in ((f"epoch{epoch:02d}.pt","checkpoint_sha256"),
                                         (f"val_epoch{epoch:02d}/metrics.json","metrics_sha256"),
                                         (f"val_epoch{epoch:02d}/predictions.jsonl","predictions_sha256")):
                        verify(d/relative,evaluation[key])
        data["pilot_variants"][mode]=entry
    data["checks"]=checks
    data["completed_artifacts_hashes_passed"]=all(x["ok"] for x in checks) if checks else None
    print(json.dumps(data,indent=2,allow_nan=False))

if __name__=="__main__":main()
