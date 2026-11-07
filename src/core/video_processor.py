"""Sample video on its source timeline and export high-definition H.264 directly with RTSP stream support."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
import time
from typing import Optional, Callable
import cv2
import numpy as np
import config
from src.core.encoding import CompactVideoWriter
from src.core.detector import detect_heads_in_frame, load_model, _model_has_native_head_class
from src.utils.helpers import draw_detections, draw_overlay, get_auto_crowd_level, get_crowd_level, resolve_class_ids, BoxSmoother
from src.core.scene import in_zone, draw_scene, validate_polygon, FlowTracker, capacity_events
from src.core.zones import ZoneAnalytics
from src.core.stream_reader import is_stream_url, StreamCapture


class AnalysisCancelled(RuntimeError):
    pass


@dataclass
class VideoProcessingResult:
    output_path: Optional[str]
    total_frames: int
    processed_frames: int
    max_head_count: int
    avg_head_count: float
    final_head_count: int
    final_crowd_level: str
    processing_time_seconds: float
    fps_processed: float
    per_frame_counts: list[int] = field(default_factory=list)
    final_auto_crowd_level: str = 'Low Crowd'
    final_auto_risk_level: str = 'Safe'
    per_frame_auto_crowd_levels: list[str] = field(default_factory=list)
    source_fps: float = 0.0
    sample_fps: float = 0.0
    timestamps: list[float] = field(default_factory=list)
    peak_crowd_level: str = 'Low Crowd'
    source_duration: float = 0.0
    input_bytes: int = 0
    output_bytes: int = 0
    output_width: int = 0
    output_height: int = 0
    warnings: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    crossings: list[dict] = field(default_factory=list)
    forward_crossings: int = 0
    reverse_crossings: int = 0
    timestamp_basis: str = 'decoder presentation timestamps'
    zones: list[dict] = field(default_factory=list)
    heatmap: dict = field(default_factory=dict)


def calculate_optimal_bitrate(width: int, height: int, fps: float) -> int:
    """Calculate high-definition video bitrate that avoids compression artifacts."""
    pixels = width * height
    if pixels >= 1920 * 1080:
        base = 8_000_000
    elif pixels >= 1280 * 720:
        base = 5_000_000
    elif pixels >= 960 * 540:
        base = 3_500_000
    else:
        base = 2_000_000
    if fps > 30:
        base = int(base * 1.3)
    return base


def process_video(
    weights_path: str, source_path: str,
    conf_threshold: float = config.INFERENCE_CONF_THRESHOLD,
    iou_threshold: float = config.INFERENCE_IOU_THRESHOLD,
    imgsz: int = config.INFERENCE_IMGSZ, device: str = 'cpu',
    output_path: Optional[str] = None, frame_callback: Optional[Callable] = None,
    preloaded_model=None, classification_mode: str = 'manual',
    max_capacity: Optional[int] = None, show_body_box_estimate: bool = False,
    analysis_fps: float = 5.0, max_width: int = 960,
    roi_polygon=None, flow_line=None, event_seconds: float = 3.,
    cancel_check=None, max_duration_seconds: float = 600., max_samples: int = 9000,
    zones=None, smooth_playback: bool = False,
) -> VideoProcessingResult:
    if classification_mode not in ('manual', 'auto'):
        raise ValueError('Classification mode must be manual or auto.')
    if not math.isfinite(analysis_fps) or not 0 < analysis_fps <= 30:
        raise ValueError('Analysis rate must be between 0 and 30 FPS.')
    if max_width is not None and 0 < max_width < 64:
        raise ValueError('Output width must be at least 64 pixels.')
    roi_polygon = validate_polygon(roi_polygon)
    spatial = ZoneAnalytics(zones) if zones else None
    if flow_line is not None and analysis_fps < 5:
        raise ValueError('Flow estimation requires at least 5 samples per second.')
    tracker = FlowTracker(flow_line) if flow_line is not None else None

    is_stream = is_stream_url(source_path)
    if output_path and not is_stream and os.path.abspath(output_path) == os.path.abspath(source_path):
        raise ValueError('Input and output paths must be different.')
    if not is_stream and not os.path.isfile(source_path):
        raise FileNotFoundError('Source video not found.')

    if is_stream:
        cap = StreamCapture(source_path, target_fps=analysis_fps)
    else:
        cap = cv2.VideoCapture(source_path)

    writer = None
    success = False
    started = time.perf_counter()
    counts, levels, timestamps = [], [], []
    smoother = BoxSmoother(alpha=config.BOX_SMOOTHING_ALPHA) if config.ENABLE_BOX_SMOOTHING else None

    try:
        if not cap.isOpened():
            raise IOError('Cannot decode this video or stream. Verify the source or URL.')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if is_stream and (not math.isfinite(fps) or fps <= 0):
            fps = 25.0
        elif not math.isfinite(fps) or fps <= 0:
            raise ValueError('The video has no valid frame rate; re-export it before analysis.')

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if min(width, height) <= 0:
            raise ValueError('The video has invalid dimensions.')

        sample_fps = min(fps, analysis_fps)
        if max_width is None or max_width <= 0:
            scale = 1.0
        else:
            scale = min(1.0, max_width / width)
        out_w = max(2, int(width * scale) // 2 * 2)
        out_h = max(2, int(height * scale) // 2 * 2)

        duration = total / fps if total else 0
        if not is_stream and duration > max_duration_seconds:
            raise ValueError(f'Clip exceeds the {max_duration_seconds/60:.0f}-minute limit. Trim it before analysis.')

        size = os.path.getsize(source_path) if not is_stream else 0
        recommended_bitrate = calculate_optimal_bitrate(out_w, out_h, fps)
        if not is_stream and size > 0 and duration > 0:
            source_bitrate = int(size * 8 / duration)
            bitrate = min(12_000_000, max(source_bitrate, recommended_bitrate))
        else:
            bitrate = recommended_bitrate

        model = preloaded_model if preloaded_model is not None else load_model(weights_path)
        ids = resolve_class_ids(model.names, config.TARGET_CLASS_NAMES)
        native = _model_has_native_head_class(model)

        output_fps = fps if smooth_playback else sample_fps
        max_bytes = max(512 * 1024**2, int(size * 3) + 50 * 1024**2) if not is_stream else 1024 * 1024**2
        if output_path:
            writer = CompactVideoWriter(
                output_path, out_w, out_h, output_fps,
                bitrate=bitrate, max_bytes=max_bytes, crf=21, preset='fast',
            )

        frame_idx = 0
        next_sample = 0.0
        previous_time = -1.0
        first_pts = None
        last_annotated = None
        written_frames = 0
        current_detections = []
        current_count = 0
        current_auto_level = 'Low Crowd'
        current_level = 'Low Crowd'

        while True:
            if cancel_check and cancel_check():
                raise AnalysisCancelled('Analysis cancelled. Partial video removed.')
            if time.perf_counter() - started > (1800 if not is_stream else max_duration_seconds):
                if is_stream:
                    break
                raise RuntimeError('Analysis exceeded 30 minutes. Use a shorter clip or fewer samples.')

            if is_stream:
                ok, frame, _ = cap.read(timeout=1.0)
                if not ok or frame is None:
                    if time.perf_counter() - started > 3.0:
                        break
                    continue
                index = frame_idx
                frame_idx += 1
                timestamp = time.perf_counter() - started
            else:
                ret = cap.grab()
                if not ret:
                    break
                index = frame_idx
                frame_idx += 1
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if first_pts is None:
                    first_pts = pts
                timestamp = pts - first_pts
                if not math.isfinite(timestamp) or (index > 0 and timestamp <= previous_time):
                    raise ValueError('Decoder timestamps are missing or non-monotonic. Re-export as H.264 MP4 before analysis.')
                previous_time = timestamp

            if timestamp > max_duration_seconds:
                if is_stream:
                    break
                raise ValueError('Decoded clip exceeds the duration limit.')

            is_sample = (timestamp + 1e-6 >= next_sample)
            if not smooth_playback and not is_sample:
                continue

            if not is_stream:
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    raise IOError(f'Cannot decode video frame {index + 1}.')

            if is_sample:
                if len(counts) >= max_samples:
                    if is_stream:
                        break
                    raise ValueError('Sample limit reached. Trim the clip or lower the analysis rate.')
                try:
                    detections = detect_heads_in_frame(
                        model, frame, conf=conf_threshold, iou=iou_threshold,
                        imgsz=imgsz, device=device, class_ids=ids, native_head_class=native,
                        show_body_box_estimate=show_body_box_estimate,
                    )
                except Exception as exc:
                    raise RuntimeError(f'Detection failed at {timestamp:.2f}s. No valid result was saved.') from exc

                detections = spatial.observe(detections, frame.shape) if spatial else in_zone(detections, frame.shape, roi_polygon)
                if tracker:
                    if len(detections) > 500:
                        raise ValueError('Flow estimation supports at most 500 heads per sample; choose a smaller area or disable flow.')
                    detections = tracker.update(detections, frame.shape, timestamp)

                count = len(detections)
                auto = get_auto_crowd_level(detections, frame.shape[:2])
                level = auto if classification_mode == 'auto' else get_crowd_level(count, max_capacity=max_capacity)

                current_detections = detections
                current_count = count
                current_auto_level = auto
                current_level = level

                counts.append(count)
                levels.append(auto)
                timestamps.append(timestamp)

                next_sample = (math.floor(timestamp * sample_fps + 1e-6) + 1) / sample_fps

            display_dets = current_detections
            if smoother and display_dets:
                display_dets = smoother.smooth(display_dets)

            annotated = draw_detections(frame.copy(), display_dets)
            if spatial:
                cv2.rectangle(annotated, (8, 8), (min(width-1, 430), 45), (30, 43, 48), -1)
                cv2.putText(annotated, f'Observed heads in selected zones: {current_count}', (16, 33), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 245, 245), 2)
            else:
                annotated = draw_overlay(annotated, current_count, current_level)

            annotated = draw_scene(annotated, roi_polygon, flow_line)
            if spatial:
                for zone in spatial.zones:
                    annotated = draw_scene(annotated, zone['polygon'])

            if scale < 1.0 or annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if writer:
                if smooth_playback:
                    writer.write(annotated)
                    written_frames += 1
                else:
                    target_frame = int(math.floor(timestamp * sample_fps + 1e-6))
                    while last_annotated is not None and written_frames < target_frame:
                        writer.write(last_annotated)
                        written_frames += 1
                    writer.write(annotated)
                    written_frames += 1

            last_annotated = annotated
            if frame_callback and (is_sample or is_stream):
                frame_callback(frame_idx, total if not is_stream else 0, annotated, current_count, current_level, current_auto_level)

        if not counts:
            raise IOError('No readable frames were found. No analysis was saved.')
        if total > 0 and not is_stream and frame_idx < total - max(10, math.ceil(total * 0.02)):
            raise IOError(f'Video ended early ({frame_idx}/{total} frames). Re-export the source video.')
        success = True
    finally:
        cap.release()
        if writer:
            writer.close(success=success)

    elapsed = time.perf_counter() - started
    chosen = levels if classification_mode == 'auto' else [get_crowd_level(c, max_capacity=max_capacity) for c in counts]
    order = {'Low Crowd': 0, 'Moderate Crowd': 1, 'High Crowd': 2}
    result_duration = (previous_time + 1/fps) if not is_stream else (timestamps[-1] if timestamps else elapsed)
    result = VideoProcessingResult(
        output_path, total if not is_stream else frame_idx, len(counts), max(counts), sum(counts) / len(counts), counts[-1], chosen[-1],
        elapsed, len(counts) / elapsed if elapsed else 0, counts,
        levels[-1], {'Low Crowd': 'Safe', 'Moderate Crowd': 'Warning', 'High Crowd': 'Critical'}[levels[-1]], levels,
        fps, sample_fps, timestamps, max(chosen, key=order.get), result_duration,
        size, os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0, out_w, out_h,
        [f"Container reported {total} frames; decoded {frame_idx}. Timeline uses decoded frames."] if total and frame_idx != total and not is_stream else [],
        capacity_events(counts, timestamps, max_capacity or config.ASSUMED_MAX_CAPACITY, event_seconds, max(1., 2/sample_fps)),
        tracker.events if tracker else [], tracker.forward if tracker else 0, tracker.reverse if tracker else 0,
        timestamp_basis='wall-clock live timestamps' if is_stream else 'decoder presentation timestamps',
    )
    if spatial:
        result.zones, result.heatmap = spatial.finish(timestamps, event_seconds, max(1., 2/sample_fps))
        result.events = [dict(e, zone=z['name']) for z in result.zones for e in z['events']]
        result.events.sort(key=lambda e: e['start_seconds'])
    return result
