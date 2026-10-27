"""One cancellable local process at a time; persisted state survives page reloads."""
from pathlib import Path
import json
import subprocess
import sys
import os
import time
import uuid
from filelock import FileLock, Timeout
import config
from src.services.storage import atomic_json, check_storage_budget

ACTIVE_STATES=('queued','running','cancelling')

def read_status(run_dir):
    path=Path(run_dir)/'status.json'
    if not path.exists(): return {}
    try:
        status=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,OSError): return {}
    # An interrupted OS process must not leave a permanent running banner.
    if status.get('state') in ACTIVE_STATES and time.time()-status.get('updated',0)>120:
        lock=FileLock(str(Path(config.OUTPUTS_ROOT)/'.analysis.lock'))
        try:
            with lock.acquire(timeout=0):
                status.update(state='failed',error='The worker stopped unexpectedly. Run the analysis again.')
                atomic_json(path,status)
        except Timeout: pass
    return status

def start_job(run_dir, payload):
    root=Path(config.OUTPUTS_ROOT); root.mkdir(parents=True,exist_ok=True)
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'.dispatch.lock'),timeout=5):
        # This lock protects checks and launches across browser sessions.
        for p in (root/'sessions').glob('*/*/status.json'):
            if read_status(p.parent).get('state') in ACTIVE_STATES:
                raise ValueError('Another analysis is running. Open its saved session or wait for it to finish.')
        check_storage_budget(root, 150*1024**2)
        (run_dir/'cancel').unlink(missing_ok=True)
        atomic_json(run_dir/'job.json',payload)
        atomic_json(run_dir/'status.json',dict(state='queued',updated=time.time(),progress=0.,message='Starting analysis'))
        log=open(run_dir/'worker.log','w',encoding='utf-8')
        try:
            proc=subprocess.Popen([sys.executable,'-B','-m','src.services.worker',str(run_dir.resolve())],
                                  cwd=config.PROJECT_ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:
            atomic_json(run_dir/'status.json',dict(state='failed',updated=time.time(),error='Could not start worker.'))
            raise
        finally: log.close()
    return proc.pid

def cancel_job(run_dir):
    (Path(run_dir)/'cancel').touch()
