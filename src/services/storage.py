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
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def clear_session(root, session_id):
    root = Path(root).resolve()
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
