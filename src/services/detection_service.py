"""
detection_service.py

Thin service layer sitting between the Streamlit pages and the existing
detection core (detect.py / video_detect.py). Centralizes model caching
(so the YOLO model is loaded once and shared by every page in the same
session, instead of every page reloading it) and common validation, so
each page file stays focused on UI code only.

This module intentionally does NOT change any detection logic in
detect.py / video_detect.py / utils.py -- it only wraps them.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import streamlit as st
from ultralytics import YOLO

import config
from src.core.detector import run_detection
from src.utils.helpers import is_valid_weights_reference, setup_logger
from src.core.video_processor import (
    VideoProcessingResult,
    process_video,
)

logger = setup_logger(__name__)


class WeightsNotFoundError(FileNotFoundError):
    """Raised when the configured model weights path doesn't resolve to a usable model."""


@st.cache_resource(show_spinner=False)
def _cached_model(weights_path: str, mtime: float) -> YOLO:
    """
    Cache the loaded YOLO model per (weights_path, mtime) so:
      - the model is loaded only once per Streamlit session, not once per
        page / per click, and
      - if the user retrains and swaps best.pt, the new file's mtime
        invalidates the cache automatically instead of serving a stale model.
    """
    return YOLO(weights_path)


def get_model(weights_path: str) -> YOLO:
    """Return a cached YOLO model instance for the given weights path."""
    if not is_valid_weights_reference(weights_path):
        raise WeightsNotFoundError(
            f"Model weights not found at '{weights_path}'. Train the model first (train.py) "
            f"or update the path in Settings."
        )
    mtime = os.path.getmtime(weights_path) if os.path.isfile(weights_path) else 0.0
    return _cached_model(weights_path, mtime)


def validate_weights(weights_path: str) -> Optional[str]:
    """Return a user-friendly error message if weights are invalid, else None."""
    if not is_valid_weights_reference(weights_path):
        return (
            f"Model weights not found at '{weights_path}'. Train the model first (train.py) "
            f"or update the path in Settings."
        )
    return None


def run_image_detection(
    weights_path: str, source_path: str, conf: float, iou: float, imgsz: int,
    classification_mode: str = "manual", max_capacity: Optional[int] = None,
    show_body_box_estimate: bool = config.SHOW_BODY_BOX_ESTIMATE,
):
    """
    Run detection on a single image. Delegates to detect.run_detection, but
    warms the shared model cache first so subsequent calls in the same
    session are fast.
    """
    model = get_model(weights_path)  # warm/reuse cache; raises WeightsNotFoundError early
    return run_detection(
        weights_path=weights_path, source_path=source_path,
        conf=conf, iou=iou, imgsz=imgsz, device="cpu",
        preloaded_model=model,
        classification_mode=classification_mode, max_capacity=max_capacity,
        show_body_box_estimate=show_body_box_estimate,
    )  # -> (annotated, head_count, crowd_level, auto_crowd_level, auto_risk_level)


def run_video_detection(
    weights_path: str,
    source_path: str,
    conf: float,
    iou: float,
    imgsz: int,
    output_path: str,
    frame_callback: Optional[Callable] = None,
    classification_mode: str = "manual",
    max_capacity: Optional[int] = None,
    show_body_box_estimate: bool = config.SHOW_BODY_BOX_ESTIMATE,
    analysis_fps: float = 5.0,
    max_width: int = 960,
) -> VideoProcessingResult:
    """Run frame-by-frame detection on a video. Thin pass-through to video_detect.process_video."""
    model = get_model(weights_path)
    return process_video(
        weights_path=weights_path, source_path=source_path,
        conf_threshold=conf, iou_threshold=iou, imgsz=imgsz, device="cpu",
        output_path=output_path, frame_callback=frame_callback,
        analysis_fps=analysis_fps, max_width=max_width,
        preloaded_model=model,
        classification_mode=classification_mode, max_capacity=max_capacity,
        show_body_box_estimate=show_body_box_estimate,
    )
