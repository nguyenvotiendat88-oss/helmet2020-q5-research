"""Finite posthoc analysis queue after the authorized residual pilot."""
import fcntl,json,os,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path("/workspace/runs/ensemble3_20261029")
DEP=Path("/workspace/runs/gq5_residual_pilot_20261029")
PY="/workspace/HELMET2020_training_kit_v1/.venv-kit/bin/python"

def write(state,**fields):
    p=OUT/"queue_status.json";tmp=p.with_suffix(".tmp")
    row={"state":state,"pid":os.getpid(),"updated_utc":datetime.now(timezone.utc).isoformat(),**fields}
    tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
    print(json.dumps(row),flush=True)

def is_alive(pid):
    try:return b"run_residual_pilot.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:return False

def dependency_state(d,complete_exists,live):
    if d["status"]=="completed":
        if not complete_exists:raise RuntimeError("Residual completion missing result")
        return "waiting" if live else "ready"
    if d["status"] not in ("running","initializing") or not live:
        raise RuntimeError("Residual dependency failed/stopped; do not start analysis")
    return "waiting"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last=None
        while True:
            d=json.loads((DEP/"status.json").read_text())
            ready=dependency_state(d,(DEP/"comparison_result.json").exists(),is_alive(d["pid"]))
            if ready=="ready":break
            if d!=last:
                write("waiting_for_residual",dependency=d,next="existing_checkpoint_ensemble3_no_training")
                last=d
            time.sleep(30)
        write("running_ensemble_posthoc",models=["G","Q5"],seeds=[0,1,2],test_accessed=False)
        subprocess.run([PY,"-u",str(HERE/"run_ensemble.py")],check=True)
        write("completed",test_accessed=False,training_updates=0,auto_train_longtail=False)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if OUT.exists():write("failed",error=str(exc))
        traceback.print_exc()
        raise
