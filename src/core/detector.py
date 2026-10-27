"""
detect.py

Runs head detection using a YOLO11 model (custom-trained head model, or
the pretrained person-detecting yolo11n.pt/yolo11s.pt with head-region
estimation). Draws blue bounding boxes around detected heads, counts
them, classifies crowd level, and saves the annotated result.

Exposes `load_model()` and `detect_heads_in_frame()` so video_detect.py
can reuse the same detection logic per-frame without reloading the model
each time. `run_detection()` (single image, unchanged signature) is kept
for backward compatibility with app.py's existing image-upload flow.

Usage:
    python detect.py --weights yolov8n.pt --source image.jpg
    python detect.py --weights best.pt --source image.jpg --conf 0.4 --save-dir outputs
"""

from __future__ import annotations

import argparse
import os
import threading

_INFERENCE_LOCK = threading.RLock()
from typing import List, Optional

import cv2
from ultralytics import YOLO  # noqa: F401 (re-exported / used as type hint below)

import config
from src.utils.helpers import (
    Detection,
    draw_detections,
    draw_overlay,
    ensure_dir,
    estimate_body_box_from_head,
    estimate_head_box,
    get_auto_crowd_level,
    get_auto_risk_level,
    get_crowd_level,
    is_plausible_head_box,
    is_valid_weights_reference,
    resolve_class_ids,
    setup_logger,
)

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO11 head detection on a single image.")
    parser.add_argument("--weights", type=str, default=config.BEST_WEIGHTS_PATH, help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=config.INFERENCE_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=config.INFERENCE_IOU_THRESHOLD, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMGSZ, help="Inference image size")
    parser.add_argument("--device", type=str, default=config.INFERENCE_DEVICE, help="Inference device")
    parser.add_argument("--save-dir", type=str, default=config.OUTPUT_IMAGES_DIR, help="Directory to save result")
    return parser.parse_args()


def load_model(weights_path: str) -> YOLO:
    """Load a YOLO model once, for reuse across many frames (images or video)."""
    if not is_valid_weights_reference(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(weights_path)


def _model_has_native_head_class(model: YOLO) -> bool:
    """True if the loaded model already has a dedicated 'head' class."""
    return resolve_class_ids(model.names, config.NATIVE_HEAD_CLASS_NAMES, allow_fallback=False) is not None


def detect_heads_in_frame(
    model: YOLO,
    frame,
    conf: float = config.INFERENCE_CONF_THRESHOLD,
    iou: float = config.INFERENCE_IOU_THRESHOLD,
    imgsz: int = config.INFERENCE_IMGSZ,
    device: str = config.INFERENCE_DEVICE,
    class_ids: Optional[list] = None,
    native_head_class: Optional[bool] = None,
    show_body_box_estimate: bool = config.SHOW_BODY_BOX_ESTIMATE,
) -> List[Detection]:
    """
    Run detection on a single frame (numpy BGR array) and return a list of
    HEAD bounding boxes.

    If the model has a native "head" class, its boxes are used directly.
    Otherwise (e.g. pretrained yolo11n.pt/yolo11s.pt detecting "person"),
    each person box is narrowed down to an estimated head region via
    utils.estimate_head_box().
    """
    if native_head_class is None:
        native_head_class = _model_has_native_head_class(model)
    # A model containing both classes must not count a head and its person twice.
    class_ids = resolve_class_ids(model.names, config.NATIVE_HEAD_CLASS_NAMES if native_head_class else ('person',))
    if getattr(model, 'task', 'detect') != 'detect':
        raise ValueError('Choose an object-detection model with head or person classes.')

    # The cached model is shared by Streamlit sessions; serialize mutable predictor state.
    with _INFERENCE_LOCK:
        results = model.predict(
            source=frame, conf=conf, iou=iou, imgsz=imgsz, device=device,
            classes=class_ids, max_det=2000, verbose=False,
        )

    detections: List[Detection] = []
    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)

    frame_h, frame_w = frame.shape[:2]

    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        x1i = max(0, min(int(round(x1)), frame_w - 1))
        y1i = max(0, min(int(round(y1)), frame_h - 1))
        x2i = max(x1i + 1, min(int(round(x2)), frame_w))
        y2i = max(y1i + 1, min(int(round(y2)), frame_h))

        display_box = None
        if native_head_class:
            if not is_plausible_head_box(x1i, y1i, x2i, y2i):
                continue
            hx1, hy1, hx2, hy2 = x1i, y1i, x2i, y2i
            if show_body_box_estimate:
                display_box = estimate_body_box_from_head(
                    hx1, hy1, hx2, hy2, frame_width=frame_w, frame_height=frame_h,
                    height_multiplier=config.HEAD_TO_BODY_HEIGHT_MULTIPLIER,
                    width_multiplier=config.HEAD_TO_BODY_WIDTH_MULTIPLIER,
                )
        else:
            # Person detection mode: x1i, y1i, x2i, y2i is the full person box
            hx1, hy1, hx2, hy2 = estimate_head_box(
                x1i, y1i, x2i, y2i,
                top_fraction=config.HEAD_TOP_FRACTION,
                width_shrink=config.HEAD_WIDTH_SHRINK,
            )
            if show_body_box_estimate:
                # User wants to draw body boxes -> draw the original detected person box
                display_box = (x1i, y1i, x2i, y2i)

        detections.append(Detection(hx1, hy1, hx2, hy2, float(c), display_box=display_box))

    return detections


def run_detection(
    weights_path: str,
    source_path: str,
    conf: float = config.INFERENCE_CONF_THRESHOLD,
    iou: float = config.INFERENCE_IOU_THRESHOLD,
    imgsz: int = config.INFERENCE_IMGSZ,
    device: str = config.INFERENCE_DEVICE,
    preloaded_model: Optional[YOLO] = None,
    classification_mode: str = "manual",
    max_capacity: Optional[int] = None,
    show_body_box_estimate: bool = config.SHOW_BODY_BOX_ESTIMATE,
    roi_polygon=None,
) -> tuple:
    """
    Detect heads in a single image. Returns
    (annotated_frame, head_count, crowd_level, auto_crowd_level, auto_risk_level).

    `preloaded_model` is optional: pass an already-loaded YOLO instance
    (e.g. from detection_service's cached model) to skip reloading the
    model from disk on every call. If omitted, the model is loaded fresh
    exactly as before.

    `classification_mode` ("auto" or "manual") controls which crowd level
    is BURNED INTO the returned annotated image, not just what's reported
    back to the caller -- it must match whatever the dashboard/UI is using,
    or the image overlay and the on-screen numbers will disagree.
    """
    if not is_valid_weights_reference(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source image not found: {source_path}")

    frame = cv2.imread(source_path)
    if frame is None:
        raise IOError(f"Failed to read image: {source_path}")

    model = preloaded_model if preloaded_model is not None else load_model(weights_path)
    class_ids = resolve_class_ids(model.names, config.TARGET_CLASS_NAMES)
    native_head_class = _model_has_native_head_class(model)

    detections = detect_heads_in_frame(
        model, frame, conf=conf, iou=iou, imgsz=imgsz, device=device,
        class_ids=class_ids, native_head_class=native_head_class,
        show_body_box_estimate=show_body_box_estimate,
    )

    from src.core.scene import in_zone, draw_scene
    detections = in_zone(detections, frame.shape, roi_polygon)
    head_count = len(detections)
    crowd_level = get_crowd_level(head_count, max_capacity=max_capacity if classification_mode == "manual" else None)
    auto_crowd_level = get_auto_crowd_level(detections, frame.shape[:2])
    auto_risk_level = get_auto_risk_level(detections, frame.shape[:2])

    # This is the level actually burned into the returned image -- must
    # match the dashboard's chosen mode (see docstring above).
    display_crowd_level = auto_crowd_level if classification_mode == "auto" else crowd_level

    annotated = draw_detections(frame.copy(), detections)
    annotated = draw_overlay(annotated, head_count, display_crowd_level)
    annotated = draw_scene(annotated, roi_polygon)

    return annotated, head_count, crowd_level, auto_crowd_level, auto_risk_level


def main() -> None:
    args = parse_args()
    ensure_dir(args.save_dir)

    try:
        annotated, head_count, crowd_level, auto_crowd_level, auto_risk_level = run_detection(
            weights_path=args.weights,
            source_path=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
    except (FileNotFoundError, IOError) as exc:
        logger.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during detection: %s", exc)
        return

    filename = os.path.splitext(os.path.basename(args.source))[0]
    out_path = os.path.join(args.save_dir, f"{filename}_detected.jpg")
    cv2.imwrite(out_path, annotated)

    logger.info("Head Count: %d", head_count)
    logger.info("Crowd Level: %s", crowd_level)
    logger.info("Auto (density-based) Crowd Level: %s | Risk: %s", auto_crowd_level, auto_risk_level)
    logger.info("Saved annotated image to: %s", out_path)


if __name__ == "__main__":
    main()
