"""Threaded, low-latency RTSP and live stream ingestion for real-time video analytics."""
from __future__ import annotations
import os
import sys
import time
import threading
from typing import Optional, Tuple, Union
import cv2
import numpy as np


def is_stream_url(source: Union[str, int]) -> bool:
    """True if source is an RTSP/HTTP URL, network transport, or local camera index."""
    if isinstance(source, int):
        return True
    s = str(source).strip()
    if s.isdigit():
        return True
    lower = s.lower()
    return lower.startswith(('rtsp://', 'rtsps://', 'http://', 'https://', 'udp://', 'tcp://', 'rtmp://'))


def normalize_stream_source(source: Union[str, int]) -> Union[str, int]:
    """Normalize webcam index strings like '0' to integer 0, or clean URL."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s


def configure_stream_backend():
    """Configure OpenCV FFmpeg backend for reliable TCP RTSP streaming without packet loss."""
    # Force TCP for RTSP (prevents UDP frame corruption and grey blocks) and minimize delay
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|max_delay;500000|buffer_size;1024000'


class StreamCapture:
    """
    Threaded, non-blocking RTSP / camera capture.
    
    Standard cv2.VideoCapture buffers 30+ frames internally. When YOLO detection takes
    50-100ms per frame, naive Capture falls behind real-time by seconds or minutes.
    
    StreamCapture runs a lightweight daemon thread continuously draining the network
    buffer and holding the single freshest frame in memory. When the inference loop
    calls read(), it gets the live camera frame with near-zero latency (<30ms).
    """

    def __init__(
        self,
        source: Union[str, int],
        reconnect_timeout: float = 15.0,
        target_fps: Optional[float] = None,
    ):
        configure_stream_backend()
        self.source = normalize_stream_source(source)
        self.reconnect_timeout = reconnect_timeout
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_pts: float = 0.0
        self._frame_counter: int = 0
        self._is_opened = False
        self._last_read_success_time = time.time()

        self.width = 0
        self.height = 0
        self.fps = 25.0

        self._connect()
        self.start()

    def _connect(self) -> bool:
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY)
            if not self.cap.isOpened():
                # Fallback for webcam or standard backend
                self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                # Request 1-frame buffer where supported
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                f = float(self.cap.get(cv2.CAP_PROP_FPS))

                if w > 0 and h > 0:
                    self.width = w
                    self.height = h
                if f > 0 and np.isfinite(f):
                    self.fps = f
                elif self.target_fps and self.target_fps > 0:
                    self.fps = float(self.target_fps)

                # Read test frame to confirm decoding
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._latest_frame = frame
                    self._latest_pts = time.perf_counter()
                    self.width = frame.shape[1]
                    self.height = frame.shape[0]
                    self._is_opened = True
                    self._last_read_success_time = time.time()
                    return True

        self._is_opened = False
        return False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name='RTSPReaderWorker')
        self._thread.start()

    def _worker(self):
        while self._running:
            if not self._is_opened or self.cap is None:
                # Attempt auto-reconnect if dropped
                if time.time() - self._last_read_success_time > self.reconnect_timeout:
                    time.sleep(1.0)
                    continue
                time.sleep(0.5)
                self._connect()
                continue

            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None

            now = time.perf_counter()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_pts = now
                    self._frame_counter += 1
                    self._last_read_success_time = time.time()
            else:
                # Stream stalled or connection lost
                if time.time() - self._last_read_success_time > 3.0:
                    # Trigger reconnection
                    self._is_opened = False
                time.sleep(0.01)

    def read(self, timeout: float = 2.0) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Return (success, frame, timestamp).
        Always delivers the freshest frame from the camera with zero buffer latency.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_pts
            time.sleep(0.005)
        return False, None, 0.0

    def isOpened(self) -> bool:
        return self._is_opened or (self.cap is not None and self.cap.isOpened())

    def get(self, prop_id: int):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return -1
        if self.cap is not None:
            try:
                return self.cap.get(prop_id)
            except Exception:
                pass
        return 0

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self._is_opened = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
