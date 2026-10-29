"""Fixed sequencing: finish existing full VAL, run mechanism pilot, then replication."""
import fcntl,json,os,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path("/workspace/runs/q5_objective_pilot_20261028")
DEPENDENCY=Path("/workspace/runs/gq5_calibration_fullval_20261028")
OLD_QUEUE=Path("/workspace/runs/gq5_confirmation_20261028")
PY="/workspace/HELMET2020_training_kit_v1/.venv-kit/bin/python"

def status(state,**fields):
    row={"state":state,"pid":os.getpid(),"updated_utc":datetime.now(timezone.utc).isoformat(),**fields}
    p=ROOT/"queue_status.json";tmp=p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
    print(json.dumps(row),flush=True)

def alive(pid):
    try:return b"validate_calibration.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:return False

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"queue.lock").open("a") as lock, (OLD_QUEUE/"queue.lock").open("a") as old_lock:
        # Preserve the old queue's exclusion: no duplicate replication scheduler.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last=None
        while True:
            d=json.loads((DEPENDENCY/"status.json").read_text())
            if d["state"]=="completed" and not alive(d["pid"]):
                if not (DEPENDENCY/"comparison.json").exists():raise RuntimeError("Missing VAL result")
                break
            if d["state"] not in ("running","completed") or (d["state"]=="running" and not alive(d["pid"])):
                raise RuntimeError("Full VAL dependency failed/stopped")
            state=(d.get("arm"),d.get("frames"),d.get("phase"))
            if state!=last:
                status("waiting_for_full_VAL",dependency=d,next="objective_pilot_then_seed1_seed2")
                last=state
            time.sleep(30)
        status("running_objective_pilot",variants=["omit_constant_presence","valid_state_conditional"])
        subprocess.run([PY,"-u",str(HERE/"run_pilot.py")],check=True)
        for seed in (1,2):
            status("running_confirmation_seed",seed=seed)
            subprocess.run([PY,"-u","/workspace/gq5_confirmation_20261028/run_confirmation.py","--seed",str(seed)],check=True)
        status("completed",test_accessed=False,instance_auto_stop=False)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if ROOT.exists():status("failed",error=str(exc))
        traceback.print_exc()
        raise
