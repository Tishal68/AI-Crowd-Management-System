"""
rtsp_stream.py

High-performance, low-latency live crowd monitoring for RTSP IP cameras,
HTTP streams, and local webcams with full counting area & flow line selection.

Features:
- Threaded non-blocking capture (zero network buffer lag)
- Custom counting area (ROI polygon) & interactive mouse ROI selection (--select-roi)
- Directional flow estimation (tripwire line crossing detection A -> B)
- High-definition smooth video encoding and real-time bounding box tracking
- Live OpenCV window display with head count, crossings & crowd status HUD
- Automated capacity warnings and live statistics
- Optional MP4 recording with crisp H.264 compression

Usage:
    # Monitor an RTSP camera with live display:
    python scripts/rtsp_stream.py --source rtsp://admin:password@192.168.1.100:554/live --display

    # Interactively select a counting area with mouse drag-and-drop:
    python scripts/rtsp_stream.py --source 0 --display --select-roi

    # Monitor with directional flow line:
    python scripts/rtsp_stream.py --source 0 --display --flow-line "[[0.2, 0.5], [0.8, 0.5]]"

    # Run headless and record:
    python scripts/rtsp_stream.py --source rtsp://camera:554/h264 --record outputs/cam1.mp4 --duration 60
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.core.stream_reader import StreamCapture, is_stream_url, normalize_stream_source
from src.core.detector import load_model, detect_heads_in_frame, _model_has_native_head_class
from src.utils.helpers import (
    draw_detections,
    draw_overlay,
    get_crowd_level,
    get_auto_crowd_level,
    resolve_class_ids,
    BoxSmoother,
    setup_logger,
)
from src.core.scene import in_zone, draw_scene, validate_polygon, validate_line, FlowTracker
from src.core.encoding import CompactVideoWriter

logger = setup_logger('rtsp_stream')


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP & Live Stream Crowd Management Monitor')
    parser.add_argument('--source', type=str, default='0', help='RTSP URL (rtsp://...) or webcam index (0, 1)')
    parser.add_argument('--weights', type=str, default=config.BEST_WEIGHTS_PATH, help='Path to YOLO weights')
    parser.add_argument('--conf', type=float, default=config.INFERENCE_CONF_THRESHOLD, help='Detection confidence')
    parser.add_argument('--analysis-fps', type=float, default=10.0, help='Inference rate (samples/sec)')
    parser.add_argument('--capacity', type=int, default=config.ASSUMED_MAX_CAPACITY, help='Monitored area capacity')
    parser.add_argument('--roi', type=str, default=None, help='JSON array of polygon coordinates (e.g. "[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]")')
    parser.add_argument('--flow-line', type=str, default=None, help='JSON array of line coordinates (e.g. "[[0.2,0.5],[0.8,0.5]]")')
    parser.add_argument('--select-roi', action='store_true', help='Interactively drag and select counting box on first frame')
    parser.add_argument('--max-width', type=int, default=1280, help='Max width for display/recording (0 for native)')
    parser.add_argument('--record', type=str, default=None, help='Output path for recorded MP4 video')
    parser.add_argument('--display', action='store_true', help='Open real-time OpenCV preview window')
    parser.add_argument('--duration', type=float, default=0.0, help='Max duration in seconds (0 = run indefinitely)')
    parser.add_argument('--device', type=str, default='cpu', help='Inference device (cpu or cuda)')
    return parser.parse_args()


def run_stream(args):
    source = normalize_stream_source(args.source)
    logger.info(f"Connecting to live stream source: {source}")

    stream = StreamCapture(source, target_fps=args.analysis_fps)
    if not stream.isOpened():
        logger.error(f"Failed to open live stream source: {source}")
        return

    logger.info(f"Connected to stream: {stream.width}x{stream.height} @ {stream.fps:.1f} FPS")

    # Geometry setup
    polygon = None
    if args.roi:
        try:
            pts = json.loads(args.roi)
            polygon = validate_polygon(pts)
            logger.info(f"Configured counting area: {len(polygon)} vertices")
        except Exception as e:
            logger.warning(f"Invalid --roi argument ({e}). Using full frame.")

    if args.select_roi and polygon is None:
        logger.info("Opening interactive ROI selection. Drag a rectangle and press ENTER or SPACE.")
        ok, frame, _ = stream.read(timeout=2.0)
        if ok and frame is not None:
            r = cv2.selectROI("Select Monitored Counting Area", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Monitored Counting Area")
            rx, ry, rw, rh = r
            if rw > 10 and rh > 10:
                h, w = frame.shape[:2]
                polygon = validate_polygon([
                    [rx / w, ry / h],
                    [(rx + rw) / w, ry / h],
                    [(rx + rw) / w, (ry + rh) / h],
                    [rx / w, (ry + rh) / h],
                ])
                logger.info(f"Interactive area set: {polygon}")

    flow_line = None
    if args.flow_line:
        try:
            line_pts = json.loads(args.flow_line)
            flow_line = validate_line(line_pts)
            logger.info(f"Configured directional flow line: {flow_line}")
        except Exception as e:
            logger.warning(f"Invalid --flow-line argument ({e}). Directional flow disabled.")

    tracker = FlowTracker(flow_line) if flow_line is not None else None

    logger.info(f"Loading detector weights: {args.weights}")
    model = load_model(args.weights)
    class_ids = resolve_class_ids(model.names, config.TARGET_CLASS_NAMES)
    native = _model_has_native_head_class(model)

    scale = 1.0 if (args.max_width <= 0 or args.max_width >= stream.width) else (args.max_width / stream.width)
    out_w = max(2, int(stream.width * scale) // 2 * 2)
    out_h = max(2, int(stream.height * scale) // 2 * 2)

    writer = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        writer = CompactVideoWriter(
            str(rec_path), out_w, out_h, stream.fps,
            bitrate=6_000_000, crf=21, preset='fast',
        )
        logger.info(f"Recording live annotated video to: {rec_path}")

    smoother = BoxSmoother(alpha=config.BOX_SMOOTHING_ALPHA)
    started = time.perf_counter()
    last_inference_time = 0.0
    inference_interval = 1.0 / max(0.5, args.analysis_fps)

    current_detections = []
    current_count = 0
    current_level = 'Low Crowd'
    processed_frames = 0
    total_detections_sum = 0
    peak_count = 0

    window_name = f"CrowdVision RTSP Monitor: {source}"

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            if args.duration > 0 and elapsed >= args.duration:
                logger.info(f"Reached configured duration of {args.duration:.1f}s. Stopping stream.")
                break

            ok, frame, frame_pts = stream.read(timeout=1.0)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            processed_frames += 1

            # Run detection at configured cadence
            if now - last_inference_time >= inference_interval:
                last_inference_time = now
                detections = detect_heads_in_frame(
                    model, frame, conf=args.conf, iou=0.5,
                    imgsz=640, device=args.device,
                    class_ids=class_ids, native_head_class=native,
                )

                # Filter heads to monitored counting area
                detections = in_zone(detections, frame.shape, polygon)
                if tracker:
                    detections = tracker.update(detections, frame.shape, now)

                current_detections = detections
                current_count = len(detections)
                current_level = get_crowd_level(current_count, max_capacity=args.capacity)
                total_detections_sum += current_count
                if current_count > peak_count:
                    peak_count = current_count

            # Smooth box transitions
            display_dets = smoother.smooth(current_detections) if current_detections else []

            annotated = draw_detections(frame.copy(), display_dets)
            annotated = draw_overlay(annotated, current_count, current_level)
            annotated = draw_scene(annotated, polygon, flow_line)

            # Add live status HUD
            hud_text = f"LIVE RTSP | COUNT: {current_count}/{args.capacity}"
            if tracker:
                hud_text += f" | IN: {tracker.forward} | OUT: {tracker.reverse}"
            else:
                hud_text += f" | PEAK: {peak_count}"

            cv2.putText(
                annotated, hud_text,
                (annotated.shape[1] - 460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA,
            )

            if scale < 1.0 or annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if writer:
                writer.write(annotated)

            if args.display:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("Exit requested by user.")
                    break

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")
    finally:
        stream.release()
        if writer:
            writer.close(success=True)
            logger.info(f"Recorded video saved successfully to: {args.record}")
        if args.display:
            cv2.destroyAllWindows()

        runtime = time.perf_counter() - started
        avg_count = total_detections_sum / max(1, processed_frames)
        msg = f"Stream finished. Monitored {runtime:.1f}s | {processed_frames} frames | Area Peak: {peak_count} | Avg: {avg_count:.1f}"
        if tracker:
            msg += f" | Total In: {tracker.forward} | Total Out: {tracker.reverse}"
        logger.info(msg)


if __name__ == '__main__':
    run_stream(parse_args())
