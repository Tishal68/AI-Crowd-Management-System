"""
src.core package initialization
"""
from src.core.detector import load_model, detect_heads_in_frame, run_detection  # noqa: F401
from src.core.video_processor import process_video, VideoProcessingResult  # noqa: F401
