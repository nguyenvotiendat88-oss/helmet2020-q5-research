"""Replace only an idle waiting scheduler, preserving all running GPU work."""
import json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
OLD=Path("/workspace/runs/gq5_confirmation_20261028")
NEW=Path("/workspace/runs/q5_objective_pilot_20261028")
HERE=Path("/workspace/q5_objective_pilot_20261028")
SESSION="q5research20261028"

def atomic(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)

def main():
    verification=json.loads((HERE/"pilot_verification.json").read_text())
    if verification["status"]!="passed":raise RuntimeError("Pilot checks not passed")
    if subprocess.run(["tmux","has-session","-t",SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError("Research session already exists; do not replace again")
    state=json.loads((OLD/"queue_status.json").read_text())
    if state["state"]!="waiting_for_full_VAL":raise RuntimeError("Old scheduler already started work; leave it intact")
    pid=state["pid"]
    if b"queue_confirmation.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
        raise RuntimeError("PID no longer identifies the known waiting scheduler")
    os.kill(pid,signal.SIGSTOP)
    stopped=True
    try:
        latest=json.loads((OLD/"queue_status.json").read_text())
        children=Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
        if latest["state"]!="waiting_for_full_VAL" or children:
            raise RuntimeError("Scheduler has started a child; do not interrupt")
        os.kill(pid,signal.SIGTERM)
        os.kill(pid,signal.SIGCONT);stopped=False
        for _ in range(50):
            if not Path(f"/proc/{pid}").exists():break
            time.sleep(.1)
        else:raise RuntimeError("Old scheduler did not exit")
    finally:
        if stopped:os.kill(pid,signal.SIGCONT)
    NEW.mkdir(parents=True,exist_ok=True)
    receipt={"previous_queue":latest,"previous_waiting_pid":pid,
             "active_training_interrupted":False,"time_utc":datetime.now(timezone.utc).isoformat(),
             "new_order":["finish_current_full_VAL","four_objective_pilot_fits","G_Q5_seed1","G_Q5_seed2"],
             "reason":"User prioritized testable research contribution; preserve previous results and resume planned seed replication"}
    atomic(NEW/"queue_switch_receipt.json",receipt)
    command=("/workspace/HELMET2020_training_kit_v1/.venv-kit/bin/python -u "
             "/workspace/q5_objective_pilot_20261028/research_queue.py "
             "> /workspace/runs/q5_objective_pilot_20261028.console.log 2>&1")
    subprocess.run(["tmux","new-session","-d","-s",SESSION,command],check=True)
    for _ in range(50):
        if (NEW/"queue_status.json").exists():break
        time.sleep(.1)
    else:raise RuntimeError("New scheduler has not produced status; inspect new log")
    new_status=json.loads((NEW/"queue_status.json").read_text())
    if new_status["state"]=="failed":raise RuntimeError("New scheduler failed: "+str(new_status))
    atomic(OLD/"queue_status.json",{"state":"superseded_by_research_queue","new_status":str(NEW/"queue_status.json"),
                                  "previous_waiting_pid":pid,"training_interrupted":False})
    print(json.dumps({"status":"activated","receipt":receipt,"queue":new_status},indent=2))

if __name__=="__main__":main()
