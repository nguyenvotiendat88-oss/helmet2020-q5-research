"""Immediate finite dependency queue, not a recurring automation."""
import fcntl, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path("/workspace/runs/gq5_confirmation_20261028")
DEPENDENCY=Path("/workspace/runs/gq5_calibration_fullval_20261028")
PYTHON="/workspace/HELMET2020_training_kit_v1/.venv-kit/bin/python"

def write(state,**fields):
    value={"state":state,"pid":os.getpid(),"updated_utc":datetime.now(timezone.utc).isoformat(),**fields}
    tmp=ROOT/"queue_status.json.tmp"
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(ROOT/"queue_status.json")
    print(json.dumps(value),flush=True)

def dependency_state(status,result_exists,pid_alive):
    if status.get("state")=="completed":
        if not result_exists:raise RuntimeError("Dependency reports completed without comparison.json")
        return "waiting" if pid_alive else "ready"
    if status.get("state") in ("failed","paused"):
        raise RuntimeError("Dependency failed or paused; do not start queued training")
    if status.get("state")!="running" or not pid_alive:
        raise RuntimeError("Dependency not running; preserve outputs and inspect")
    return "waiting"

def alive(pid):
    try:
        os.kill(pid,0)
        return b"validate_calibration.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (ProcessLookupError,FileNotFoundError):return False

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last=None
        while True:
            status=json.loads((DEPENDENCY/"status.json").read_text())
            state=dependency_state(status,(DEPENDENCY/"comparison.json").is_file(),alive(status["pid"]))
            if state=="ready":break
            snapshot=(status.get("arm"),status.get("frames"),status.get("phase"))
            if snapshot!=last:
                write("waiting_for_full_VAL",dependency=status,queued_seeds=[1,2],arms=["G","Q5"])
                last=snapshot
            time.sleep(30)
        for seed in (1,2):
            write("running_seed",seed=seed,arms=["G","Q5"],schedule="12+12",primary_epoch=24)
            subprocess.run([PYTHON,"-u",str(HERE/"run_confirmation.py"),"--seed",str(seed)],check=True)
        write("completed",seeds=[1,2],arms=["G","Q5"],test_accessed=False,
              controls="prepared_not_started",calibration_new_seeds="not_started",instance_auto_stop=False)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if ROOT.exists():write("failed",error=str(exc),exception=type(exc).__name__)
        traceback.print_exc()
        raise
