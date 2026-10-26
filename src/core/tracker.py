

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.utils.helpers import Detection, resolve_class_ids, setup_logger

logger = setup_logger(__name__)

# Path to the ByteTrack config shipped with Ultralytics.
# Ultralytics resolves this by name automatically, no need for an absolute path.
BYTETRACK_CONFIG = "bytetrack.yaml"


class HeadTracker:
    """
    Wraps a YOLO11 model + ByteTrack to detect and track heads across
    a video stream. Call `reset()` between independent videos/sessions
    so track IDs don't leak across unrelated streams.
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
        target_class_names: tuple = ("head",),
    ) -> None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")

        logger.info("Loading YOLO model from %s", weights_path)
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.class_ids = resolve_class_ids(self.model.names, target_class_names)
        self._started = False

    def reset(self) -> None:
        """Reset internal ByteTrack state before processing a new video."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()
        self._started = False

    def update(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single frame.
        Returns a list of Detection objects (head boxes with track IDs).
        """
        try:
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                classes=self.class_ids,
                tracker=BYTETRACK_CONFIG,
                persist=True,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Tracking inference failed; crowd count is unknown.") from exc

        self._started = True
        return self._parse_results(results)

    @staticmethod
    def _parse_results(results) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

        for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
            detections.append(
                Detection(
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    confidence=float(conf),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return detections
