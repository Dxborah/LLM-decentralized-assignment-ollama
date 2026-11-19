"""Detach bounded CPU workers and retain per-job logs and exit status."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "BASELINE_PLAN.json").read_text())


def main():
    assert socket.gethostname().lower() == PLAN["host"].lower()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = [j for j in PLAN["jobs"] if not (Path(j["directory"]) / "RUN_COMPLETE.json").exists()]
    running = {}
    exited = {}
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    while pending or running:
        while pending and len(running) < PLAN["workers"]:
            job = pending.pop(0)
            d = Path(job["directory"])
            with (d / "evaluation.log").open("ab", buffering=0) as stream:
                child = subprocess.Popen([PLAN["python"], "-u", str(ROOT / "baseline_worker.py"), job["id"]],
                                         cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            running[job["id"]] = child
            (d / "LAUNCHER.json").write_text(json.dumps({
                "worker_pid": child.pid, "host": socket.gethostname(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "detached": True, "launched_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        for job_id, child in list(running.items()):
            rc = child.poll()
            if rc is not None:
                exited[job_id] = rc
                del running[job_id]
        temp = ROOT / "CONTROLLER_STATUS.tmp"
        temp.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "controller_pid": os.getpid(),
            "running": {j: p.pid for j, p in running.items()},
            "queued": [j["id"] for j in pending], "exit_codes": exited}, indent=2) + "\n")
        temp.replace(ROOT / "CONTROLLER_STATUS.json")
        if pending or running:
            time.sleep(5)


if __name__ == "__main__":
    main()
