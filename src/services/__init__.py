"""
src.services package initialization
"""
from src.services.detection_service import get_model, validate_weights, run_image_detection, run_video_detection  # noqa: F401
from src.services.history_manager import add_record, get_all_records, get_record_by_id  # noqa: F401
from src.services.report_generator import generate_pdf_report  # noqa: F401
