"""Immutable per-run artifacts and bounded, explicitly removable session storage."""
from pathlib import Path
import json
import os
import shutil
import tempfile
from src.utils.privacy import redact_sensitive


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(redact_sensitive(value), f, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def storage_bytes(path):
    root = Path(path)
    size = 0
    if root.exists():
        for p in root.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink(): size += p.stat().st_size
            except FileNotFoundError:
                pass
    return size


def session_inventory(root):
    """Only app-owned UUID session folders; never traverse linked directories."""
    import re
    root=Path(root)
    if not root.exists(): return []
    sessions=[]
    for directory in root.iterdir():
        if directory.is_dir() and not directory.is_symlink() and re.fullmatch(r'[a-f0-9]{32}',directory.name):
            runs=[]
            for record in directory.glob('*/result.json'):
                if record.parent.is_symlink(): continue
                try:
                    data=json.loads(record.read_text(encoding='utf-8'))
                    output=Path(data['annotated_path']).resolve()
                    if output.parent==record.parent.resolve() and output.is_file():
                        runs.append(data)
                except (OSError,ValueError,KeyError): pass
            sessions.append(dict(id=directory.name,bytes=storage_bytes(directory),runs=runs,modified=directory.stat().st_mtime))
    return sorted(sessions,key=lambda x:x['modified'],reverse=True)


def check_storage_budget(root, additional_bytes, limit=2*1024**3):
    """Fail before writing; callers serialize writes/jobs with the local workspace lock."""
    reserved=0
    for path in (Path(root)/'sessions').glob('*/*/status.json'):
        try:
            state=json.loads(path.read_text(encoding='utf-8')).get('state')
            if state in ('queued','running','cancelling'):
                reserved+=max(0,150*1024**2-storage_bytes(path.parent))
        except (OSError,ValueError): pass
    if storage_bytes(root)+reserved+additional_bytes > limit:
        raise ValueError('The 2 GB workspace limit would be exceeded. Remove saved sessions in Storage before continuing.')


def clear_session(root, session_id):
    import re
    root = Path(root).resolve()
    original=root/session_id
    if not re.fullmatch(r'[a-f0-9]{32}',session_id) or original.is_symlink() or original.is_junction():
        raise ValueError('Invalid session path.')
    target = (root / session_id).resolve()
    if target.parent != root or not session_id or target == root:
        raise ValueError('Invalid session path.')
    if target.exists():
        shutil.rmtree(target)


def retain_run(session_dir, signature):
    """Keep one completed run per session. Only remove our signature-named run folders."""
    import re
    root = Path(session_dir).resolve()
    for path in root.iterdir():
        if path.is_dir() and not path.is_symlink() and re.fullmatch(r"[0-9a-f]{24}", path.name) and path.name != signature:
            if path.resolve().parent == root:
                shutil.rmtree(path)


LEGACY_FOLDERS = ('streamlit_videos', 'streamlit_images', 'rtsp_recordings', '_uploads', 'csv', 'reports')

def clear_legacy_exports(outputs_root, folders):
    """Delete only explicitly selected direct legacy exports, never source media elsewhere."""
    root = Path(outputs_root).resolve()
    if any(folder not in LEGACY_FOLDERS for folder in folders):
        raise ValueError('Select a supported generated-output folder.')
    removed = 0
    for folder in folders:
        original = root / folder
        target = original.resolve()
        if original.is_symlink() or target.parent != root:
            raise ValueError('Unsafe cleanup path; nothing deleted in this folder.')
        if target.exists():
            for path in target.iterdir():
                if path.is_file() and not path.is_symlink():
                    removed += path.stat().st_size
                    path.unlink()
    return removed
