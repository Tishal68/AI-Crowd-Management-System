"""
app_state.py

Shared Streamlit session-state initialization and CSS theming for the
single-page CrowdVision AI app. Premium dark glassmorphism theme.
"""

from __future__ import annotations

import streamlit as st

import config

SETTINGS_KEY = "app_settings"


def init_app_state() -> None:
    """Initialize all session-state defaults exactly once."""
    if SETTINGS_KEY not in st.session_state:
        st.session_state[SETTINGS_KEY] = {
            "weights_path": config.BEST_WEIGHTS_PATH,
            "conf": config.INFERENCE_CONF_THRESHOLD,
            "iou": config.INFERENCE_IOU_THRESHOLD,
            "imgsz": config.INFERENCE_IMGSZ,
            "max_capacity": config.ASSUMED_MAX_CAPACITY,
            "classification_mode": "auto",
            "show_body_box_estimate": config.SHOW_BODY_BOX_ESTIMATE,
        }

    defaults = {
        "annotated_path": None,
        "head_count": None,
        "crowd_level": None,
        "risk_level": None,
        "processing_time": None,
        "error": None,
        "last_image_record": None,
        "last_uploaded_name": None,
        "video_output_path": None,
        "video_final_head_count": None,
        "video_final_crowd_level": None,
        "video_final_risk_level": None,
        "video_max_head_count": None,
        "video_avg_head_count": None,
        "video_processing_time": None,
        "video_per_frame_counts": None,
        "video_per_frame_auto_levels": None,
        "video_frames_processed": None,
        "video_fps_processed": None,
        "video_error": None,
        "last_video_record": None,
        "last_csv_path": None,
        "last_heatmap_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_settings() -> dict:
    init_app_state()
    return st.session_state[SETTINGS_KEY]


def update_settings(**kwargs) -> None:
    init_app_state()
    st.session_state[SETTINGS_KEY].update(kwargs)


def configure_page() -> None:
    """st.set_page_config + CSS injection. Called once at the top of app.py."""
    st.set_page_config(
        page_title=f"CrowdVision AI · {config.PROJECT_DISPLAY_NAME}",
        page_icon="🧍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_app_state()
    _inject_css()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            color: #F1F3F9;
        }

        :root {
            --bg: #070A12;
            --card-bg: rgba(255, 255, 255, 0.035);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent-blue: #4F6BFF;
            --accent-purple: #9B5CFF;
            --accent-green: #10B981;
            --accent-orange: #F59E0B;
            --accent-red: #EF4444;
        }

        /* Hide Streamlit chrome */
        #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {
            visibility: hidden; height: 0; }
        div[data-testid="stDecoration"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }

        .stApp {
            background:
                radial-gradient(1400px 700px at 15% -15%, rgba(79,107,255,0.12), transparent 70%),
                radial-gradient(1200px 600px at 85% 10%, rgba(155,92,255,0.12), transparent 70%),
                radial-gradient(800px 400px at 50% 100%, rgba(16,185,129,0.05), transparent 60%),
                var(--bg);
            background-attachment: fixed;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* ---- Glass card ---- */
        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 22px 24px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.14);
        }
        .card-title {
            display: flex; align-items: center; gap: 10px;
            font-size: 1rem; font-weight: 700; color: var(--text-primary);
            margin-bottom: 16px; letter-spacing: -0.01em;
        }

        /* ---- Top bar ---- */
        .topbar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 4px 2px 18px 2px;
        }
        .topbar .brand { display: flex; align-items: center; gap: 14px; }
        .topbar .logo-badge {
            width: 44px; height: 44px; border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem; box-shadow: 0 6px 20px rgba(79,107,255,0.4);
        }
        .topbar .brand-title { font-size: 1.45rem; font-weight: 800; color: var(--text-primary); line-height: 1.1; letter-spacing: -0.02em; }
        .topbar .brand-title span { color: var(--accent-purple); }
        .topbar .brand-subtitle { font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px; font-weight: 500; }

        /* ---- Hero / info strip ---- */
        .info-strip {
            display: flex; align-items: center; justify-content: space-between;
            gap: 20px; flex-wrap: wrap;
        }
        .info-strip .info-left { max-width: 680px; }
        .info-strip h2 { margin: 0 0 6px 0; font-size: 1.4rem; color: var(--text-primary); font-weight: 800; letter-spacing: -0.02em; }
        .info-strip p { margin: 0; color: var(--text-secondary); font-size: 0.92rem; line-height: 1.5; }
        .badge-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .pill {
            padding: 7px 16px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
            border: 1px solid var(--card-border); white-space: nowrap; backdrop-filter: blur(8px);
            transition: all 0.2s ease;
        }
        .pill-blue   { background: rgba(79,107,255,0.14);  color: #A5B4FC; border-color: rgba(79,107,255,0.25); }
        .pill-purple { background: rgba(155,92,255,0.14);  color: #DDD6FE; border-color: rgba(155,92,255,0.25); }
        .pill-green  { background: rgba(16,185,129,0.14);   color: #6EE7B7; border-color: rgba(16,185,129,0.25); }
        .pill-orange { background: rgba(245,158,11,0.14);  color: #FDE68A; border-color: rgba(245,158,11,0.25); }

        /* ---- Metric tiles ---- */
        .metric-tile {
            display: flex; align-items: center; gap: 14px;
            background: rgba(255,255,255,0.025);
            border: 1px solid var(--card-border);
            border-radius: 16px; padding: 16px 18px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-tile:hover {
            border-color: rgba(255,255,255,0.15);
            transform: translateY(-2px);
        }
        .metric-tile .icon-badge {
            width: 42px; height: 42px; min-width: 42px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center; font-size: 1.15rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .metric-tile .m-label { font-size: 0.76rem; color: var(--text-secondary); margin-bottom: 3px; font-weight: 500; }
        .metric-tile .m-value { font-size: 1.4rem; font-weight: 800; color: var(--text-primary); line-height: 1.1; letter-spacing: -0.02em; }
        .metric-tile .m-sub { font-size: 0.74rem; color: var(--text-secondary); margin-top: 3px; font-weight: 500; }

        .risk-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 999px; font-size: 0.84rem; font-weight: 700;
        }
        .risk-safe     { background: rgba(16,185,129,0.16);  color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); }
        .risk-warning  { background: rgba(245,158,11,0.16); color: #FDE68A; border: 1px solid rgba(245,158,11,0.3); }
        .risk-critical { background: rgba(239,68,68,0.18);  color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }

        /* Buttons */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: 1px solid var(--card-border) !important;
            transition: all 0.2s ease !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            padding: 0.55rem 1.2rem !important;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: rgba(255, 255, 255, 0.25) !important;
            transform: translateY(-1px);
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
            border: none !important;
            box-shadow: 0 4px 18px rgba(79,107,255,0.35) !important;
        }
        div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 24px rgba(79,107,255,0.5) !important;
            transform: translateY(-1px);
        }

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.018) !important;
            border: 1.5px dashed rgba(255,255,255,0.16) !important;
            border-radius: 16px !important;
            transition: all 0.2s ease !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-blue) !important;
            background: rgba(79,107,255,0.04) !important;
        }

        hr { border-color: var(--card-border); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_level: str) -> str:
    css_class = {"Safe": "risk-safe", "Warning": "risk-warning", "Critical": "risk-critical"}.get(
        risk_level, "risk-warning"
    )
    icon = config.RISK_LEVEL_ICONS.get(risk_level, "⚪")
    return f'<span class="risk-badge {css_class}">{icon} {risk_level}</span>'
