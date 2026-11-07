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
    """Configure OpenCV FFmpeg backend for reliable TCP RTSP streaming with bounded timeouts."""
    # Force TCP for RTSP (prevents UDP frame corruption and grey blocks) and bound network socket timeouts
    # stimeout: RTSP/TCP socket timeout in microseconds (3,000,000 us = 3.0s)
    # rw_timeout: Read/write network timeout in microseconds
    # max_delay: Maximum demuxing delay
    # buffer_size: Demuxing buffer size
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|stimeout;3000000|rw_timeout;3000000|timeout;3000000|max_delay;500000|buffer_size;1024000'
    )


class StreamCapture:
    """
    Threaded, non-blocking RTSP / camera capture with fresh frame validation,
    controlled exponential backoff auto-reconnection, and safe resource ownership.

    Standard cv2.VideoCapture buffers 30+ frames internally. When YOLO detection takes
    50-100ms per frame, naive Capture falls behind real-time by seconds or minutes.

    StreamCapture runs a lightweight daemon thread continuously draining network buffers
    and holding the single freshest frame in memory with capture timestamp verification.
    """

    def __init__(
        self,
        source: Union[str, int],
        reconnect_timeout: float = 15.0,
        target_fps: Optional[float] = None,
        stale_timeout: float = 1.0,
        min_reconnect_interval: float = 0.5,
        max_reconnect_interval: float = 10.0,
        backoff_factor: float = 1.5,
        shutdown_timeout: float = 2.0,
    ):
        configure_stream_backend()
        self.source = normalize_stream_source(source)
        self.reconnect_timeout = reconnect_timeout
        self.target_fps = target_fps
        self.stale_timeout = stale_timeout
        self.min_reconnect_interval = min_reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.backoff_factor = backoff_factor
        self.shutdown_timeout = shutdown_timeout

        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._running = False
        self._released = False
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_pts: float = 0.0
        self._last_delivered_pts: float = 0.0
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
            if not self._running and self._released:
                return False

            old_cap = self.cap
            self.cap = None
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass

            # Reset frame state and timestamps on connection / reconnection
            self._latest_frame = None
            self._latest_pts = 0.0
            self._last_delivered_pts = 0.0
            self._is_opened = False

        # OpenCV VideoCapture open performed outside self._lock to prevent blocking lock during network operations
        new_cap = None
        try:
            new_cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY)
            if not new_cap.isOpened():
                new_cap.release()
                new_cap = cv2.VideoCapture(self.source)
        except Exception:
            if new_cap is not None:
                try:
                    new_cap.release()
                except Exception:
                    pass
            new_cap = None

        with self._lock:
            # Check if shutdown occurred while opening
            if self._released or (self._thread is not None and not self._running):
                if new_cap is not None:
                    try:
                        new_cap.release()
                    except Exception:
                        pass
                return False

            if new_cap is not None and new_cap.isOpened():
                self.cap = new_cap
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
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    ret, frame = False, None

                if ret and frame is not None:
                    self._latest_frame = frame
                    self._latest_pts = time.perf_counter()
                    self._last_delivered_pts = 0.0
                    self.width = frame.shape[1]
                    self.height = frame.shape[0]
                    self._is_opened = True
                    self._last_read_success_time = time.time()
                    return True

            if new_cap is not None:
                try:
                    new_cap.release()
                except Exception:
                    pass
            self.cap = None
            self._is_opened = False
            self._latest_frame = None
            self._latest_pts = 0.0
            return False

    def start(self):
        if self._running or self._released:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name='RTSPReaderWorker')
        self._thread.start()

    def _worker(self):
        backoff = self.min_reconnect_interval
        try:
            while self._running:
                cap_handle = None
                is_open = False
                with self._lock:
                    if not self._running:
                        break
                    cap_handle = self.cap
                    is_open = self._is_opened

                if not is_open or cap_handle is None:
                    # Stream disconnected: ensure stale frame is cleared
                    with self._lock:
                        self._latest_frame = None
                        self._latest_pts = 0.0
                        self._is_opened = False

                    # Controlled exponential backoff sleep with responsive interruption check
                    slept = 0.0
                    while self._running and slept < backoff:
                        step = min(0.05, backoff - slept)
                        time.sleep(step)
                        slept += step

                    if not self._running:
                        break

                    # Attempt reconnection
                    reconnected = self._connect()
                    if reconnected:
                        backoff = self.min_reconnect_interval
                    else:
                        backoff = min(backoff * self.backoff_factor, self.max_reconnect_interval)
                    continue

                try:
                    ret, frame = cap_handle.read()
                except Exception:
                    ret, frame = False, None

                now = time.perf_counter()
                if ret and frame is not None and self._running:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_pts = now
                        self._frame_counter += 1
                        self._last_read_success_time = time.time()
                    backoff = self.min_reconnect_interval
                else:
                    # Stream stalled or connection lost: clear stale frames
                    with self._lock:
                        self._is_opened = False
                        self._latest_frame = None
                        self._latest_pts = 0.0
                    time.sleep(0.02)
        finally:
            with self._lock:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                self._is_opened = False
                self._latest_frame = None
                self._latest_pts = 0.0

    def read(
        self,
        timeout: float = 2.0,
        max_age: Optional[float] = None,
        require_new: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Return (success, frame, timestamp).
        Verifies frame freshness using capture timestamp against max_age (default: self.stale_timeout).
        If require_new=True, ensures caller does not receive the same frame repeatedly.
        Returns (False, None, 0.0) if frame is stale, feed is disconnected, or timeout expires.
        """
        allowed_age = max_age if max_age is not None else self.stale_timeout
        deadline = time.time() + timeout

        while time.time() <= deadline:
            with self._lock:
                if not self._running or self._released or not self._is_opened:
                    self._latest_frame = None
                    self._latest_pts = 0.0
                elif self._latest_frame is not None:
                    age = time.perf_counter() - self._latest_pts
                    is_fresh = age <= allowed_age
                    is_new = (self._latest_pts > self._last_delivered_pts) if require_new else True

                    if is_fresh and is_new:
                        self._last_delivered_pts = self._latest_pts
                        return True, self._latest_frame.copy(), self._latest_pts
                    elif not is_fresh:
                        # Invalidate stale frame
                        self._latest_frame = None
                        self._latest_pts = 0.0

            time.sleep(0.005)

        return False, None, 0.0

    def isOpened(self) -> bool:
        with self._lock:
            return not self._released and self._is_opened and (self.cap is not None and self.cap.isOpened())

    def get(self, prop_id: int):
        with self._lock:
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
        """
        Idempotent, thread-safe shutdown of worker thread and capture resource.
        Guarantees that native VideoCapture is not concurrently touched while worker is active.
        """
        with self._shutdown_lock:
            if self._released:
                return
            self._released = True
            self._running = False
            with self._lock:
                self._is_opened = False
                self._latest_frame = None
                self._latest_pts = 0.0

        # Wait for worker thread to terminate if it can
        # Note: If the underlying native backend blocks indefinitely in an un-interruptible socket call,
        # the thread may not exit within join timeout.
        if self._thread and self._thread.is_alive():
            if threading.current_thread() != self._thread:
                self._thread.join(timeout=self.shutdown_timeout)

        # Safe resource ownership:
        # We do NOT touch self.cap concurrently while the worker thread is still alive and reading from it.
        # If the worker has terminated (or never started), clean up any lingering handle.
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
