"""
history_manager.py

Lightweight JSON-backed storage for past analysis runs (image or video),
used by the History page and by the Reports page (to re-generate a report
for a previous run) and the Analytics page (to show the latest run).

Deliberately dependency-free (stdlib json only) so it works the same
whether the caller is app.py or any Streamlit page.
"""

from __future__ import annotations

import json
import os
import uuid
import threading
from src.utils.privacy import redact_sensitive
from src.services.storage import atomic_json
_HISTORY_LOCK = threading.RLock()
from datetime import datetime
from typing import Any, Dict, List, Optional

import config
from src.utils.helpers import ensure_dir, setup_logger

logger = setup_logger(__name__)


def _load_all() -> List[Dict[str, Any]]:
    ensure_dir(config.HISTORY_DIR)
    if not os.path.isfile(config.HISTORY_FILE):
        return []
    try:
        with open(config.HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return redact_sensitive(data) if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read history file (%s) -- starting fresh: %s", config.HISTORY_FILE, exc)
        return []


def _save_all(records: List[Dict[str, Any]]) -> None:
    ensure_dir(config.HISTORY_DIR)
    # Keep only the most recent N entries so the file doesn't grow forever.
    trimmed = records[-config.MAX_HISTORY_ENTRIES:]
    try:
        atomic_json(config.HISTORY_FILE, trimmed)
    except OSError as exc:
        logger.error("Failed to write history file: %s", exc)


def add_record(
    media_type: str,
    file_name: str,
    max_head_count: int,
    avg_head_count: float,
    final_head_count: int,
    crowd_level: str,
    risk_level: str,
    processing_time_seconds: float,
    frames_processed: Optional[int] = None,
    fps_processed: Optional[float] = None,
    annotated_path: Optional[str] = None,
    per_frame_counts: Optional[List[int]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a new analysis record and persist it. Returns the saved record."""
    record = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "media_type": media_type,  # "image" or "video"
        "file_name": file_name,
        "max_head_count": max_head_count,
        "avg_head_count": round(avg_head_count, 2),
        "final_head_count": final_head_count,
        "crowd_level": crowd_level,
        "risk_level": risk_level,
        "processing_time_seconds": round(processing_time_seconds, 2),
        "frames_processed": frames_processed,
        "fps_processed": round(fps_processed, 2) if fps_processed else None,
        "annotated_path": annotated_path,
        "per_frame_counts": per_frame_counts or [],
        "settings": settings or {},
    }
    record = redact_sensitive(record)
    with _HISTORY_LOCK:
        records = _load_all()
        records.append(record)
        _save_all(records)
    logger.info("History record saved: %s", record["id"])
    return record


def get_all_records() -> List[Dict[str, Any]]:
    """Return all history records, most recent first."""
    return list(reversed(_load_all()))


def get_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    for record in _load_all():
        if record.get("id") == record_id:
            return record
    return None


def delete_record(record_id: str) -> bool:
    records = _load_all()
    filtered = [r for r in records if r.get("id") != record_id]
    if len(filtered) == len(records):
        return False
    _save_all(filtered)
    return True


def clear_history() -> None:
    _save_all([])
