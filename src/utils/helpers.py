"""
utils.py

Shared utility functions for the Crowd Management System.
Includes crowd-level classification, drawing helpers, logging setup,
and misc file / time helpers. CPU-friendly, no heavy dependencies
beyond OpenCV and NumPy.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

import config

# --------------------------------------------------------------------------- #
# Constants (sourced from config.py -- edit values there, not here)
# --------------------------------------------------------------------------- #

BOX_COLOR: Tuple[int, int, int] = config.BOX_COLOR_BGR
TEXT_COLOR: Tuple[int, int, int] = config.TEXT_COLOR_BGR
BOX_THICKNESS: int = config.BOX_THICKNESS
FONT = cv2.FONT_HERSHEY_SIMPLEX

LOW_CROWD_MAX = config.LOW_CROWD_MAX
MODERATE_CROWD_MAX = config.MODERATE_CROWD_MAX

CROWD_LEVEL_COLORS = config.CROWD_LEVEL_COLORS_BGR


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logger(name: str = "crowd_management", level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a configured logger with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured -> avoid duplicate handlers on repeated imports
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------- #
# Automatic, density-based classification
# --------------------------------------------------------------------------- #
#
# Raw head COUNT can't tell a tightly-packed close-up from a stadium shot
# apart -- 50 heads filling a small room is a crush, 50 heads scattered
# across a stadium's upper tier (or seen from a wide/crooked overhead
# angle) is nothing. Counting alone also has no notion of "how full does
# this frame actually look".
#
# Instead, PACKING DENSITY -- what fraction of the visible frame area the
# detected heads themselves cover -- is computed directly from the boxes
# YOLO already returns and the frame's own dimensions. This scales itself
# automatically to whatever is on screen: a distant stadium tier, a close
# hallway, an oblique/crooked security-camera angle -- no manual "capacity"
# number to type in and get wrong. Any Optional[int] capacity is now only
# an *override* for someone who genuinely wants a fixed-headcount rule for
# a specific, known space; automatic mode is the default in the app.

# Fraction of total frame area covered by head boxes at each boundary.
# Tuned so a handful of close-up faces (large boxes) and a genuine crowd
# of small, distant boxes both land in a sensible bucket.
DENSITY_LOW_MAX = 0.015     # <=1.5% of frame covered by heads -> Low
DENSITY_MODERATE_MAX = 0.05  # <=5% of frame covered by heads -> Moderate


def compute_packing_density(detections: Sequence["Detection"], frame_shape: Tuple[int, int]) -> float:
    """
    Fraction (0-1) of the frame's area covered by detected head boxes.

    `frame_shape` is (height, width) -- e.g. `frame.shape[:2]` from OpenCV.
    Overlapping boxes are summed rather than de-duplicated, which is
    intentional: overlap itself is a signal of crowding, not noise.
    """
    height, width = frame_shape[0], frame_shape[1]
    frame_area = max(height * width, 1)
    covered = 0
    for det in detections:
        w = max(det.x2 - det.x1, 0)
        h = max(det.y2 - det.y1, 0)
        covered += w * h
    return covered / frame_area


def get_auto_crowd_level(detections: Sequence["Detection"], frame_shape: Tuple[int, int]) -> str:
    """Classify Low/Moderate/High crowd level from packing density, not raw count."""
    density = compute_packing_density(detections, frame_shape)
    if density <= DENSITY_LOW_MAX:
        return "Low Crowd"
    if density <= DENSITY_MODERATE_MAX:
        return "Moderate Crowd"
    return "High Crowd"


def get_auto_risk_level(detections: Sequence["Detection"], frame_shape: Tuple[int, int]) -> str:
    """Safe/Warning/Critical counterpart of get_auto_crowd_level(), same thresholds."""
    level = get_auto_crowd_level(detections, frame_shape)
    return {"Low Crowd": "Safe", "Moderate Crowd": "Warning", "High Crowd": "Critical"}[level]


def get_crowd_level(count: int, max_capacity: Optional[int] = None) -> str:
    """
    Classify a head-count into Low / Moderate / High crowd level.

    If `max_capacity` is given, thresholds scale with it (as a % of
    capacity) instead of using the fixed absolute defaults below. This
    matters for venues like a stadium: a fixed "40 heads = High Crowd"
    rule makes every wide/crooked-angle stadium shot register as
    permanently critical, even at normal occupancy. Pass the scene's
    configured capacity (e.g. from Settings) to get a level that's
    meaningful for that specific venue.
    """
    if count < 0:
        raise ValueError("count must be a non-negative integer")

    if max_capacity and max_capacity > 0:
        # Explicit occupancy bands; heuristic rules, not validated safety thresholds.
        low_max = max_capacity * 0.60
        moderate_max = max_capacity * 0.85
        if count <= low_max:
            return "Low Crowd"
        if count <= moderate_max:
            return "Moderate Crowd"
        return "High Crowd"

    if count <= LOW_CROWD_MAX:
        return "Low Crowd"
    if count <= MODERATE_CROWD_MAX:
        return "Moderate Crowd"
    return "High Crowd"


def get_crowd_level_color(level: str) -> Tuple[int, int, int]:
    """Return a BGR color associated with a crowd level for overlay text."""
    return CROWD_LEVEL_COLORS.get(level, (255, 255, 255))


def get_risk_level(count: int, max_capacity: Optional[int] = None) -> str:
    """
    Classify a head-count into a safety-oriented risk level: Safe / Warning /
    Critical. Uses the same thresholds as get_crowd_level() (fixed by
    default, or scaled to `max_capacity` when provided) so the two stay
    consistent, but is exposed separately since "risk" is the framing used
    in the Analytics / Reports pages.
    """
    if count < 0:
        raise ValueError("count must be a non-negative integer")

    if max_capacity and max_capacity > 0:
        level = get_crowd_level(count, max_capacity=max_capacity)
        return {"Low Crowd": "Safe", "Moderate Crowd": "Warning", "High Crowd": "Critical"}[level]

    if count <= config.RISK_LOW_MAX:
        return "Safe"
    if count <= config.RISK_MODERATE_MAX:
        return "Warning"
    return "Critical"


def get_risk_level_icon(level: str) -> str:
    """Return the emoji icon associated with a risk level."""
    return config.RISK_LEVEL_ICONS.get(level, "⚪")


def get_density_percentage(count: int, max_capacity: Optional[int] = None) -> float:
    """
    Compute an occupancy percentage relative to an assumed maximum capacity.
    Values above 100 are intentionally not clamped -- the UI uses that to
    flag genuine overcrowding rather than silently capping it at 100%.
    """
    capacity = max_capacity or config.ASSUMED_MAX_CAPACITY
    if capacity <= 0:
        return 0.0
    return round((count / capacity) * 100.0, 1)


# --------------------------------------------------------------------------- #
# Data container for a single detection
# --------------------------------------------------------------------------- #

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: Optional[int] = None
    # Optional alternate box used ONLY for drawing (see
    # estimate_body_box_from_head). When set, draw_detections() draws this
    # instead of (x1, y1, x2, y2). Head-count and packing-density
    # calculations always use the real (x1, y1, x2, y2) box, never this one.
    display_box: Optional[Tuple[int, int, int, int]] = None


class BoxSmoother:
    """
    Exponential Moving Average (EMA) bounding box coordinate smoother across video frames.
    Prevents visual box jittering and flickering in live streams / video playback.
    """

    def __init__(self, alpha: float = 0.45) -> None:
        self.alpha = alpha
        self.prev_boxes: dict[int, Tuple[float, float, float, float]] = {}

    def reset(self) -> None:
        self.prev_boxes.clear()

    def smooth(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            self.prev_boxes.clear()
            return detections

        smoothed_detections: List[Detection] = []
        current_ids: set[int] = set()

        for idx, det in enumerate(detections):
            # Key by track_id if available, otherwise by detection index
            if det.track_id is None:
                # Detection order is unstable. Never mix coordinates across people.
                smoothed_detections.append(det)
                continue
            key = det.track_id
            current_ids.add(key)

            x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)

            if key in self.prev_boxes:
                px1, py1, px2, py2 = self.prev_boxes[key]
                sx1 = self.alpha * x1 + (1.0 - self.alpha) * px1
                sy1 = self.alpha * y1 + (1.0 - self.alpha) * py1
                sx2 = self.alpha * x2 + (1.0 - self.alpha) * px2
                sy2 = self.alpha * y2 + (1.0 - self.alpha) * py2
            else:
                sx1, sy1, sx2, sy2 = x1, y1, x2, y2

            self.prev_boxes[key] = (sx1, sy1, sx2, sy2)

            # Smooth display_box if present
            disp = None
            if det.display_box is not None:
                dx1, dy1, dx2, dy2 = (float(v) for v in det.display_box)
                disp = (
                    int(round(dx1)),
                    int(round(dy1)),
                    int(round(dx2)),
                    int(round(dy2)),
                )

            smoothed_detections.append(
                Detection(
                    x1=int(round(sx1)),
                    y1=int(round(sy1)),
                    x2=int(round(sx2)),
                    y2=int(round(sy2)),
                    confidence=det.confidence,
                    track_id=det.track_id,
                    display_box=disp,
                )
            )

        # Evict missing track IDs
        stale_keys = [k for k in self.prev_boxes if k not in current_ids]
        for k in stale_keys:
            del self.prev_boxes[k]

        return smoothed_detections


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw blue bounding boxes + track-id labels for every detected head."""
    h, w = frame.shape[:2]
    for det in detections:
        raw_box = det.display_box if det.display_box is not None else (det.x1, det.y1, det.x2, det.y2)
        bx1 = max(0, min(int(round(raw_box[0])), w - 1))
        by1 = max(0, min(int(round(raw_box[1])), h - 1))
        bx2 = max(bx1 + 1, min(int(round(raw_box[2])), w))
        by2 = max(by1 + 1, min(int(round(raw_box[3])), h))

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), BOX_COLOR, BOX_THICKNESS)

        label = f"ID {det.track_id}" if det.track_id is not None else f"{det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, config.FONT_SCALE, 1)

        label_y1 = max(by1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            frame,
            (bx1, label_y1),
            (min(bx1 + text_w + 4, w), min(label_y1 + text_h + baseline + 4, h)),
            BOX_COLOR,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (bx1 + 2, min(label_y1 + text_h + 2, h - 2)),
            FONT,
            config.FONT_SCALE,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return frame



def draw_overlay(frame: np.ndarray, head_count: int, crowd_level: str) -> np.ndarray:
    """Draw a semi-transparent info panel with head count and crowd level."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 300, 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    level_color = get_crowd_level_color(crowd_level)

    cv2.putText(
        frame, f"Head Count: {head_count}", (10, 30),
        FONT, config.OVERLAY_FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Crowd Level: {crowd_level}", (10, 62),
        FONT, config.OVERLAY_FONT_SCALE, level_color, 2, cv2.LINE_AA,
    )
    return frame


def generate_density_heatmap(
    frame_shape: Tuple[int, int],
    all_detections: Sequence[Sequence["Detection"]],
    blur_ksize: int = 51,
) -> np.ndarray:
    """
    Build a BGR heatmap image (same size as the source frame) showing where
    heads were most frequently detected across many frames (or a single
    frame for images). Each detected head center contributes a small
    "heat" blob; blobs are accumulated, normalized, then colorized.

    Args:
        frame_shape: (height, width) of the source image/video frame.
        all_detections: a list of per-frame Detection lists. For a still
            image this is just `[detections]`.
        blur_ksize: Gaussian blur kernel size (odd number) used to spread
            each point into a smooth blob.

    Returns:
        A BGR heatmap image ready to be blended with the source frame or
        saved standalone.
    """
    h, w = frame_shape
    accumulator = np.zeros((h, w), dtype=np.float32)

    for detections in all_detections:
        for det in detections:
            cx = int((det.x1 + det.x2) / 2)
            cy = int((det.y1 + det.y2) / 2)
            if 0 <= cy < h and 0 <= cx < w:
                accumulator[cy, cx] += 1.0

    if accumulator.max() > 0:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        accumulator = cv2.GaussianBlur(accumulator, (ksize, ksize), 0)
        accumulator = accumulator / accumulator.max()

    heat_uint8 = np.uint8(255 * accumulator)
    heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return heatmap


def blend_heatmap_with_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay a colorized heatmap on top of the original frame."""
    return cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)


def export_frame_counts_to_csv(
    csv_path: str,
    per_frame_counts: Sequence[int],
    fps: float = 25.0,
    classification_mode: str = "auto",
    max_capacity: Optional[int] = None,
    auto_crowd_levels: Optional[Sequence[str]] = None,
) -> str:
    """
    Write a CSV with columns: Frame Number, Head Count, Crowd Level,
    Risk Level, Timestamp (seconds, derived from fps). Returns the path
    written to.

    BUG FIX: this used to always call get_crowd_level(count) / get_risk_level(count)
    with NO capacity argument, which silently applied the fixed absolute
    thresholds (LOW_CROWD_MAX / MODERATE_CROWD_MAX) regardless of the mode
    selected in Settings. A confined space (e.g. a bus, a small room)
    analyzed in "Automatic (density-based)" mode -- where the on-screen
    dashboard correctly reported density-based levels -- could export a
    CSV whose "Crowd Level" column disagreed with the dashboard, and a
    "Manual" run with a small custom max_capacity would still get
    exported using the generic default thresholds instead of that capacity.

    Fix: the caller now tells this function which mode was actually used.
      - classification_mode == "auto": prefer the exact, already-computed
        per-frame density-based levels via `auto_crowd_levels` (the same
        values shown in the live dashboard / Analytics donut chart), so
        the CSV always matches what the user saw on screen. Falls back to
        recomputing from count only if that list isn't available.
      - classification_mode == "manual": use `max_capacity` so a small,
        confined space is classified relative to ITS capacity instead of
        the generic global thresholds.
    """
    import csv as csv_module

    ensure_dir(os.path.dirname(csv_path) or ".")

    use_manual = classification_mode == "manual"
    has_precomputed_auto = auto_crowd_levels is not None and len(auto_crowd_levels) == len(per_frame_counts)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(["Frame Number", "Head Count", "Crowd Level", "Risk Level", "Timestamp (s)"])
        for idx, count in enumerate(per_frame_counts, start=1):
            timestamp = round(idx / fps, 2) if fps > 0 else 0.0

            if use_manual:
                crowd_level = get_crowd_level(count, max_capacity=max_capacity)
                risk_level = get_risk_level(count, max_capacity=max_capacity)
            elif has_precomputed_auto:
                crowd_level = auto_crowd_levels[idx - 1]
                risk_level = {"Low Crowd": "Safe", "Moderate Crowd": "Warning", "High Crowd": "Critical"}[crowd_level]
            else:
                crowd_level = get_crowd_level(count)
                risk_level = get_risk_level(count)

            writer.writerow([idx, count, crowd_level, risk_level, timestamp])
    return csv_path


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't already exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """Format seconds into a human readable mm:ss.ms string."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


class Timer:
    """Simple context-manager based timer for measuring processing time."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start


def get_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    fourcc_str: str = "mp4v",
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter with sane defaults for mp4 output."""
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise IOError(f"Failed to open VideoWriter for path: {output_path}")
    return writer


def resolve_class_ids(
    model_names: dict, target_names: Sequence[str] = ("head",), allow_fallback: bool = True,
) -> Optional[list]:
    """
    Resolve class-name(s) to class-id(s) using the model's `names` dict.

    If none of `target_names` match the model's actual class names, this
    used to fall back to `None` ("use all classes" for Ultralytics'
    `classes=` filter). That fallback is dangerous for THIS app: this
    project only ever wants head/person detection, never general object
    detection, so silently detecting every class the model knows (chairs,
    laptops, bottles, etc. for a COCO-style checkpoint) produces exactly
    the "boxes on furniture" bug -- wrong, misaligned-looking boxes that
    have nothing to do with heads.

    Instead: if there's no name match AND `allow_fallback` is True, fall
    back to class id 0. The overwhelming majority of custom head/person/
    crowd-detection exports (Ultralytics `yolo export`, Roboflow, CVAT,
    etc.) are single-class with the target class as id 0 regardless of
    what it's literally named in `model.names` -- so this is a much safer
    default than "everything."

    `allow_fallback=False` is for callers that need a STRICT match (e.g.
    "does this model have a real, explicitly named 'head' class?" --
    used to decide native-head vs person-with-estimation drawing mode.
    Falling back to id 0 there would wrongly mark every model as a native
    head model, since id 0 always exists.
    """
    name_to_id = {v.lower(): k for k, v in model_names.items()}
    ids = [name_to_id[n.lower()] for n in target_names if n.lower() in name_to_id]
    if ids:
        return ids
    if allow_fallback:
        raise ValueError("This checkpoint has no head/person class. Select the documented crowd model.")
    return None


def is_stock_pretrained_checkpoint(model_names: dict) -> bool:
    """
    Heuristic: True if `model_names` looks like an UNTOUCHED, plain
    Ultralytics pretrained checkpoint (all 80 COCO classes) rather than a
    model fine-tuned on a custom head/person dataset.

    This matters because a stock checkpoint still "works" in this app --
    it has a real 'person' class, so detection + the head-crop heuristic
    still run and can look fine at a glance -- but it was never trained on
    YOUR head dataset, so its accuracy on your specific cameras/scenes is
    whatever generic COCO training happened to generalize to, not
    anything tuned for your use case. Silently treating it as "the head
    model" is exactly what caused the earlier best.pt/model.pt mix-up.
    """
    names_lower = {v.lower() for v in model_names.values()}
    coco_markers = {"chair", "laptop", "cell phone", "keyboard", "bicycle", "giraffe"}
    return len(model_names) >= 20 and "person" in names_lower and len(names_lower & coco_markers) >= 3


def describe_model_classes(model_names: dict, target_names: Sequence[str] = ("person", "head")) -> str:
    """
    Human-readable one-liner describing whether a loaded model's classes
    match what this app expects, for display in Settings so a mismatched
    model (e.g. a generic/pretrained checkpoint dropped in by mistake) is
    obvious instead of silently producing wrong-looking boxes.
    """
    if is_stock_pretrained_checkpoint(model_names):
        return (
            f"⚠️ This is the stock pretrained checkpoint ({len(model_names)} generic COCO "
            f"classes: person, car, chair, laptop, ...) -- NOT fine-tuned on your head "
            f"dataset. It will still detect 'person' and draw estimated head boxes, but "
            f"accuracy on your cameras is whatever generic COCO training happened to "
            f"generalize to. Use train.py to fine-tune it on your dataset before treating "
            f"it as a real head-detection model."
        )
    names_lower = {v.lower() for v in model_names.values()}
    matched = [n for n in target_names if n.lower() in names_lower]
    if matched:
        return f"✅ Model classes match expected: {', '.join(matched)}."
    class_list = ", ".join(f"{k}: {v}" for k, v in sorted(model_names.items()))
    if len(model_names) == 1:
        return (
            f"⚠️ Model's single class isn't named 'person'/'head' ({class_list}) -- "
            f"assuming it's still the intended head/person class (id 0)."
        )
    return (
        f"❌ None of this model's classes are 'person' or 'head' -- it has {len(model_names)} "
        f"classes ({class_list}). This looks like a general-purpose/pretrained model, not a "
        f"head-detection model, and will likely draw boxes on unrelated objects (chairs, "
        f"laptops, etc). Double-check you selected the right weights file."
    )


def is_valid_weights_reference(weights_path: str) -> bool:
    """
    True if `weights_path` is either:
      - an existing file on disk (a real custom-trained .pt path), or
      - a bare official model name with no directory component (e.g.
        "yolov8n.pt") that Ultralytics resolves/downloads automatically.

    A plain `os.path.isfile()` check would incorrectly reject official
    pretrained model names on a machine where they haven't been
    downloaded yet, since the file doesn't exist until YOLO() fetches it.
    """
    if os.path.isfile(weights_path):
        return True
    has_no_dir = os.path.dirname(weights_path) == ""
    return has_no_dir and weights_path.lower().endswith(".pt")


def discover_weight_files() -> list:
    """
    Scan the project for candidate .pt model files so the UI can offer a
    dropdown instead of requiring the user to type a path by hand.

    Looks in:
      - the project root (e.g. best.pt sitting next to app.py)
      - runs/train/<run_name>/weights/*.pt for every training run
        Ultralytics has produced (best.pt, last.pt, or any other .pt saved
        there -- e.g. a manually renamed "model.pt")

    Returns a list of dicts: [{"label": <friendly name>, "path": <abs path>}],
    de-duplicated by absolute path, sorted with the configured default
    weights first, then alphabetically by label.
    """
    candidates = {}

    def _add(path: str, label: str) -> None:
        abspath = os.path.abspath(path)
        if os.path.isfile(abspath):
            candidates[abspath] = label

    # Project-root-level weights (e.g. best.pt shipped alongside app.py)
    try:
        for fname in os.listdir(config.PROJECT_ROOT):
            if fname.lower().endswith(".pt"):
                label = fname
                if os.path.abspath(os.path.join(config.PROJECT_ROOT, fname)) == os.path.abspath(
                    config.PRETRAINED_BASE_CHECKPOINT
                ):
                    label = f"{fname}  (stock pretrained -- not fine-tuned on your dataset)"
                _add(os.path.join(config.PROJECT_ROOT, fname), label)
    except OSError:
        pass

    # Every runs/train/<run_name>/weights/*.pt
    train_root = config.TRAIN_PROJECT_DIR
    if os.path.isdir(train_root):
        for run_name in sorted(os.listdir(train_root)):
            weights_dir = os.path.join(train_root, run_name, "weights")
            if not os.path.isdir(weights_dir):
                continue
            try:
                for fname in sorted(os.listdir(weights_dir)):
                    if fname.lower().endswith(".pt"):
                        stem = os.path.splitext(fname)[0]
                        _add(os.path.join(weights_dir, fname), f"{run_name} / {stem}")
            except OSError:
                pass

    default_path = os.path.abspath(config.BEST_WEIGHTS_PATH)
    items = [{"label": label, "path": path} for path, label in candidates.items()]
    items.sort(key=lambda d: (d["path"] != default_path, d["label"].lower()))
    return items


def estimate_head_box(
    x1: int, y1: int, x2: int, y2: int,
    top_fraction: float = 0.22,
    width_shrink: float = 0.15,
) -> Tuple[int, int, int, int]:
    """
    Estimate a head bounding box from a full-body person bounding box.

    Used when the loaded model only knows "person" (e.g. pretrained COCO
    yolov8n.pt) and has no dedicated "head" class -- the top portion of
    the person box, narrowed horizontally, approximates the head region.

    Args:
        x1, y1, x2, y2: the person's full bounding box.
        top_fraction: fraction of the person box height counted as head.
        width_shrink: fraction shaved off each side (shoulders are wider
            than the head).

    Returns:
        (hx1, hy1, hx2, hy2) estimated head box, guaranteed valid
        (hx2 > hx1, hy2 > hy1).
    """
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return x1, y1, x2, y2

    head_height = max(1, int(round(height * top_fraction)))
    dx = int(round(width * width_shrink))

    hx1 = x1 + dx
    hx2 = x2 - dx
    hy1 = y1
    hy2 = y1 + head_height

    # Guard against over-shrinking on very narrow boxes.
    if hx2 <= hx1:
        hx1, hx2 = x1, x2
    if hy2 <= hy1:
        hy2 = hy1 + 1

    return hx1, hy1, hx2, hy2


def estimate_body_box_from_head(
    x1: int, y1: int, x2: int, y2: int,
    frame_width: int,
    frame_height: int,
    height_multiplier: float = 2.3,
    width_multiplier: float = 1.7,
) -> Tuple[int, int, int, int]:
    """
    Estimate a body-sized bounding box from a HEAD bounding box.

    Needed when the loaded model natively detects HEADS ONLY (e.g. a
    head-trained YOLO11 model, `model.names == {0: "head"}`), which is
    great for accurate small/distant-person counting but produces tiny
    boxes that look wrong for a crowd-management display expecting
    larger, person-sized boxes.

    NOTE: this is intentionally NOT the mathematical inverse of
    estimate_head_box() above. That function's ratios describe a
    STANDING person's full-body proportions (head ~= 22% of total height),
    which is the wrong model for typical office/desk CCTV footage where
    most people are SEATED and only visible from the head down to roughly
    the chest/shoulders before a desk or chair occludes the rest --
    inverting the standing-body ratio there produces boxes that extend
    well past the person into whatever is behind them. `height_multiplier`
    / `width_multiplier` here instead approximate "head + neck + shoulders
    + upper chest", a conservative box that stays visually anchored to the
    person regardless of whether they're seated or standing (see
    config.HEAD_TO_BODY_HEIGHT_MULTIPLIER / _WIDTH_MULTIPLIER to tune).

    This is a DISPLAY-ONLY estimate: it is intentionally not used for
    head-count or packing-density calculations, which stay based on the
    real, accurate head box (inflating those to body-sized boxes would
    make the auto density/crowd-level classifier -- calibrated on head-box
    sizes -- wildly overestimate crowding).

    Args:
        x1, y1, x2, y2: the detected head's bounding box.
        frame_width, frame_height: dimensions of the source frame, so the
            estimated body box can be clipped to stay on-screen.
        height_multiplier: body box height = head box height * this.
        width_multiplier: body box width = head box width * this.

    Returns:
        (bx1, by1, bx2, by2) estimated body box, clipped to the frame and
        guaranteed valid (bx2 > bx1, by2 > by1).
    """
    head_w = max(x2 - x1, 1)
    head_h = max(y2 - y1, 1)

    body_height = head_h * height_multiplier
    body_width = head_w * width_multiplier

    # Body is centered on the same horizontal midpoint as the head, and
    # extends downward starting at the top of the head.
    cx = (x1 + x2) / 2.0
    bx1 = int(round(cx - body_width / 2.0))
    bx2 = int(round(cx + body_width / 2.0))
    by1 = y1
    by2 = int(round(y1 + body_height))

    # Clip to the frame so an estimated body near an edge doesn't draw
    # off-screen.
    bx1 = max(0, min(bx1, frame_width - 1))
    bx2 = max(bx1 + 1, min(bx2, frame_width))
    by1 = max(0, min(by1, frame_height - 1))
    by2 = max(by1 + 1, min(by2, frame_height))

    return bx1, by1, bx2, by2


def is_plausible_head_box(
    x1: int, y1: int, x2: int, y2: int,
    min_aspect: float = 0.35,
    max_aspect: float = 2.2,
) -> bool:
    """
    Sanity-check a raw detection's aspect ratio (width / height) against
    what a real human head can plausibly look like, regardless of viewing
    angle (roughly circular/oval -> close to 1:1, with some allowance for
    tilt, hair, or a side profile).

    Needed because a native head-only model can occasionally fire on a
    non-head object that happens to clear the confidence threshold (a
    hand resting on a laptop, a chair silhouette, etc.) -- these tend to
    produce boxes with an aspect ratio far outside anything a head can
    have (e.g. ~4:1, wide and short), even at high confidence. Confidence
    alone does not catch this, since the model can be very confident about
    a wrong detection; shape does.

    Deliberately permissive by default (0.35-2.2) so it only rejects clear
    outliers, not merely unusual real heads (tilted, partially cropped,
    side-on, etc.).
    """
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    aspect = w / h
    return min_aspect <= aspect <= max_aspect


# --------------------------------------------------------------------------- #
# Dataset validation / cache handling (training bug fix)
# --------------------------------------------------------------------------- #

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _images_dir_to_labels_dir(images_dir: str) -> str:
    """
    Mirror Ultralytics' own path convention: an images directory is mapped
    to its labels directory by replacing the LAST path segment named
    'images' with 'labels'. Works regardless of OS path separator.
    """
    norm = images_dir.replace("\\", "/").rstrip("/")
    parts = norm.split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return "/".join(parts)
    # Fallback: no 'images' segment found, assume labels dir is a sibling.
    return norm


def _load_data_yaml(data_yaml_path: str) -> dict:
    import yaml  # local import: keeps this optional dependency out of the hot path

    if not os.path.isfile(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clear_dataset_cache(data_yaml_path: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Delete stale Ultralytics .cache files (train.cache / val.cache) for the
    splits referenced in data.yaml. Ultralytics trusts an existing cache
    file blindly and will NOT re-scan labels on disk if a (possibly stale
    or wrong) cache is present -- this forces a fresh scan.
    """
    log = logger or setup_logger(__name__)
    data_cfg = _load_data_yaml(data_yaml_path)
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    for split in ("train", "val"):
        images_dir = data_cfg.get(split)
        if not images_dir:
            continue
        if not os.path.isabs(images_dir):
            images_dir = os.path.join(yaml_dir, images_dir)

        labels_dir = _images_dir_to_labels_dir(images_dir)
        cache_path = labels_dir.rstrip("/").replace("/", os.sep) + ".cache"

        if os.path.isfile(cache_path):
            try:
                os.remove(cache_path)
                log.info("Removed stale cache: %s", cache_path)
            except OSError as exc:
                log.warning("Could not remove cache file %s: %s", cache_path, exc)


def _parse_label_file(label_path: str, nc: int) -> dict:
    """
    Parse a single YOLO-format label .txt file the way Ultralytics does,
    and report exactly why rows are or aren't counted as valid.

    Returns a dict: {
        'valid_boxes': int,          # rows that pass every check
        'total_rows': int,           # non-blank lines in the file
        'malformed_rows': int,       # wrong token count / non-numeric
        'class_out_of_range': int,   # class id not in [0, nc-1]
        'bad_coords': int,           # coords outside [0,1] or w/h <= 0
        'reasons': list[str],        # human-readable first few problems
    }
    """
    result = {
        "valid_boxes": 0,
        "total_rows": 0,
        "malformed_rows": 0,
        "class_out_of_range": 0,
        "bad_coords": 0,
        "reasons": [],
    }

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        result["reasons"].append(f"could not read file: {exc}")
        return result

    for line_no, line in enumerate(lines, start=1):
        result["total_rows"] += 1
        tokens = line.split()

        if len(tokens) != 5:
            result["malformed_rows"] += 1
            if len(result["reasons"]) < 3:
                result["reasons"].append(
                    f"line {line_no}: expected 5 values (class x y w h), got {len(tokens)}: '{line}'"
                )
            continue

        try:
            class_id = int(float(tokens[0]))
            x, y, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            result["malformed_rows"] += 1
            if len(result["reasons"]) < 3:
                result["reasons"].append(f"line {line_no}: non-numeric value(s): '{line}'")
            continue

        if class_id < 0 or class_id >= nc:
            result["class_out_of_range"] += 1
            if len(result["reasons"]) < 3:
                result["reasons"].append(
                    f"line {line_no}: class id {class_id} is outside valid range [0, {nc - 1}] "
                    f"(data.yaml declares nc={nc})"
                )
            continue

        # Ultralytics requires normalized coords in (0, 1] and positive width/height.
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            result["bad_coords"] += 1
            if len(result["reasons"]) < 3:
                result["reasons"].append(f"line {line_no}: x/y center out of [0,1] range: '{line}'")
            continue
        if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
            result["bad_coords"] += 1
            if len(result["reasons"]) < 3:
                result["reasons"].append(f"line {line_no}: invalid width/height: '{line}'")
            continue

        result["valid_boxes"] += 1

    return result


def log_resolved_dataset_paths(data_yaml_path: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Print the exact, fully-resolved image/label directories that will be
    used for this run, for debugging path-resolution issues before training
    starts (see normalize_data_yaml() for why this matters on Windows).
    """
    log = logger or setup_logger(__name__)
    data_cfg = _load_data_yaml(data_yaml_path)
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    log.info("Resolved data.yaml: %s", os.path.abspath(data_yaml_path))
    for split in ("train", "val"):
        raw = data_cfg.get(split)
        if not raw:
            continue
        images_dir = raw if os.path.isabs(raw) else os.path.join(yaml_dir, raw)
        images_dir = os.path.normpath(images_dir)
        labels_dir = os.path.normpath(_images_dir_to_labels_dir(images_dir).replace("/", os.sep))
        log.info("[%s] raw path in yaml : %s", split, raw)
        log.info("[%s] resolved images  : %s", split, images_dir)
        log.info("[%s] resolved labels  : %s", split, labels_dir)


def normalize_data_yaml(data_yaml_path: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Root-cause fix for the misleading "All labels empty in .../train.cache"
    error on Windows.

    Ultralytics resolves each image's label path with a hardcoded,
    OS-specific string substitution:
        f"{os.sep}images{os.sep}"  ->  f"{os.sep}labels{os.sep}"
    On Windows, os.sep == '\\'. If data.yaml specifies paths with forward
    slashes (e.g. "E:/project/dataset/images/train"), that substring never
    matches a '\\images\\' pattern, so Ultralytics silently fails to locate
    ANY label file for ANY image -- even when the images and labels are
    100% correctly paired on disk. It then reports every image as having
    an empty label set.

    This function reads the user's original data.yaml (never modified),
    resolves the train/val paths to absolute paths using the current OS's
    separator, and writes a normalized copy to a temp file for Ultralytics
    to consume. If no normalization was actually needed, the original path
    is returned unchanged.
    """
    import tempfile
    import yaml

    log = logger or setup_logger(__name__)
    data_cfg = _load_data_yaml(data_yaml_path)
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    changed = False
    for split in ("train", "val", "test"):
        raw = data_cfg.get(split)
        if not raw:
            continue
        abs_path = raw if os.path.isabs(raw) else os.path.join(yaml_dir, raw)
        normalized = os.path.normpath(abs_path)
        if normalized != raw:
            changed = True
        data_cfg[split] = normalized

    if not changed:
        log.info("data.yaml paths already OS-normalized -- using original file.")
        return data_yaml_path

    tmp_path = os.path.join(tempfile.gettempdir(), "crowd_mgmt_data_normalized.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    log.info(
        "data.yaml paths were not OS-normalized (likely mixed '/' separators on "
        "Windows) -- Ultralytics' internal images->labels path substitution "
        "requires exact os.sep matches, or it silently can't find your labels. "
        "Wrote a normalized copy for training (your original data.yaml is untouched): %s",
        tmp_path,
    )
    return tmp_path


def validate_dataset(data_yaml_path: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Verify that images and labels are correctly paired AND that label files
    actually contain valid, parseable bounding boxes, for every split
    listed in data.yaml. This mirrors the checks Ultralytics performs
    internally, so a broken dataset fails HERE with a specific, actionable
    reason instead of surfacing later as the opaque
    "All labels empty in ...cache" ValueError from Ultralytics.
    """
    log = logger or setup_logger(__name__)
    data_cfg = _load_data_yaml(data_yaml_path)
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    nc = data_cfg.get("nc")
    names = data_cfg.get("names")
    if nc is None and isinstance(names, dict):
        nc = len(names)
    if nc is None:
        raise ValueError(f"data.yaml at '{data_yaml_path}' has no 'nc' (or 'names') field -- cannot validate.")

    log.info("Resolved data.yaml: %s", os.path.abspath(data_yaml_path))
    log.info("nc=%s names=%s", nc, names)

    for split in ("train", "val"):
        images_dir = data_cfg.get(split)
        if not images_dir:
            log.warning("data.yaml has no '%s' entry -- skipping validation for it.", split)
            continue
        if not os.path.isabs(images_dir):
            images_dir = os.path.join(yaml_dir, images_dir)
        images_dir = os.path.normpath(images_dir)

        labels_dir = os.path.normpath(_images_dir_to_labels_dir(images_dir).replace("/", os.sep))

        log.info("[%s] images_dir = %s", split, images_dir)
        log.info("[%s] labels_dir = %s", split, labels_dir)

        if not os.path.isdir(images_dir):
            raise ValueError(f"[{split}] Images directory does not exist: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise ValueError(f"[{split}] Labels directory does not exist: {labels_dir}")

        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(_IMAGE_EXTENSIONS)
        ]
        label_files_on_disk = {
            f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")
        }
        label_stems = {os.path.splitext(f)[0] for f in label_files_on_disk}

        if not image_files:
            raise ValueError(f"[{split}] No image files found in: {images_dir}")

        matched, unmatched = 0, []
        total_valid_boxes = 0
        files_with_zero_valid_boxes = []
        malformed_total = 0
        class_range_total = 0
        bad_coord_total = 0
        first_bad_example = None

        for img_name in image_files:
            stem = os.path.splitext(img_name)[0]
            if stem not in label_stems:
                unmatched.append(img_name)
                continue
            matched += 1

            label_path = os.path.join(labels_dir, stem + ".txt")
            stats = _parse_label_file(label_path, nc=nc)

            total_valid_boxes += stats["valid_boxes"]
            malformed_total += stats["malformed_rows"]
            class_range_total += stats["class_out_of_range"]
            bad_coord_total += stats["bad_coords"]

            if stats["valid_boxes"] == 0:
                files_with_zero_valid_boxes.append(stem)
                if first_bad_example is None and stats["reasons"]:
                    first_bad_example = (stem, stats)

        log.info(
            "[%s] images=%d matched_label_files=%d unmatched=%d",
            split, len(image_files), matched, len(unmatched),
        )
        log.info(
            "[%s] total_valid_boxes=%d files_with_zero_valid_boxes=%d "
            "malformed_rows=%d class_id_out_of_range=%d bad_coords=%d",
            split, total_valid_boxes, len(files_with_zero_valid_boxes),
            malformed_total, class_range_total, bad_coord_total,
        )

        if unmatched[:5]:
            log.warning("[%s] Example images with NO matching label file: %s", split, unmatched[:5])

        if matched == 0:
            raise ValueError(
                f"[{split}] 0 of {len(image_files)} images have a matching label file in "
                f"'{labels_dir}'. Every image 'X.jpg' needs a same-named 'X.txt' in the "
                f"labels folder. Fix the pairing before training."
            )

        if total_valid_boxes == 0:
            detail = ""
            if first_bad_example:
                stem, stats = first_bad_example
                detail = (
                    f"\nExample failing file: {stem}.txt\n"
                    f"  total_rows={stats['total_rows']} malformed={stats['malformed_rows']} "
                    f"class_out_of_range={stats['class_out_of_range']} bad_coords={stats['bad_coords']}\n"
                    f"  reasons: {stats['reasons']}"
                )
            raise ValueError(
                f"[{split}] Labels and images are correctly paired ({matched} files matched), "
                f"but ZERO valid bounding boxes were found across all {matched} label files. "
                f"This is the actual cause of Ultralytics' 'All labels empty' error -- it is a "
                f"label CONTENT problem, not a pairing/cache problem.\n"
                f"Breakdown across {matched} files: malformed_rows={malformed_total}, "
                f"class_id_out_of_range={class_range_total} (valid range is 0-{nc - 1}), "
                f"invalid_coords_or_size={bad_coord_total}."
                f"{detail}"
            )

        if files_with_zero_valid_boxes:
            log.warning(
                "[%s] %d/%d matched label files have ZERO valid boxes (background images). "
                "This is normal in small numbers, but check if this count looks unexpectedly high.",
                split, len(files_with_zero_valid_boxes), matched,
            )
