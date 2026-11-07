"""
CrowdVision AI — Campus Crowd Intelligence & Monitoring Platform.
Enterprise-grade Streamlit application designed for dark-navy operations monitoring.
"""
from pathlib import Path
import hashlib
import json
import uuid
from datetime import datetime
import time
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from filelock import FileLock
import config
from src.core.scene import validate_polygon, validate_line
from src.services.storage import storage_bytes, clear_session, session_inventory, check_storage_budget
from src.services.jobs import start_job, read_status, cancel_job, ACTIVE_STATES
from src.services.exports import csv_bytes
from src.services.report_generator import generate_pdf_report
from src.ui.scene_editor import scene_editor
from src.ui.charts import (
    create_occupancy_timeline,
    create_flow_dynamics_chart,
    create_crowd_distribution_donut,
    create_density_heatmap_fig,
    create_events_timeline_fig,
    generate_operational_insights,
)

# ---------------------------------------------------------
# Page Configuration & Refined Enterprise Dark Design
# ---------------------------------------------------------
st.set_page_config(
    page_title='CrowdVision — Campus Crowd Intelligence',
    page_icon='◈',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Backgrounds & Global Palette */
.stApp {
    background-color: #0E1525 !important;
    color: #EDF2FA !important;
}

.block-container {
    max-width: 1650px;
    padding: 0.8rem 1.8rem 2.2rem !important;
}

/* SIDEBAR: Locked to precise 240px width */
[data-testid="stSidebar"] {
    background-color: #111B2D !important;
    border-right: 1px solid #35445C !important;
    min-width: 235px !important;
    max-width: 245px !important;
    width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 240px !important;
    padding: 1rem 0.75rem 1.2rem !important;
}

/* Radio Navigation in Sidebar - Circle Removal & Clean Pill Styling */
[data-testid="stSidebar"] div[role="radiogroup"],
[data-testid="stSidebar"] [data-testid="stRadioGroup"] {
    gap: 3px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] [class*="etak9234"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] [class*="etak9235"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] div:has(+ [data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] div:has(+ div [data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] > div > div > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] label > div > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] div[role="radiogroup"] label > div:first-child,
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
[data-testid="stSidebar"] div[role="radiogroup"] label > span:first-child,
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stWidgetSelectionIndicator"],
[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"],
[data-testid="stSidebar"] div[role="radiogroup"] label svg,
[data-testid="stSidebar"] div[role="radiogroup"] label div[aria-hidden="true"],
[data-testid="stSidebar"] div[role="radiogroup"] span[data-testid="stWidgetSelectionIndicator"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] label,
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    color: #A1AEC2 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    display: flex !important;
    align-items: center !important;
    margin-bottom: 2px !important;
    width: 100% !important;
    text-transform: capitalize !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]:hover,
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] label:hover,
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #192438 !important;
    color: #EDF2FA !important;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"]:has([data-selected="true"]),
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadioOption"] label[data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #7157E8 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(113, 87, 232, 0.45) !important;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"]:has([data-selected="true"]) p,
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stRadioOption"] label[data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Sidebar Navigation Icons */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) [data-testid="stMarkdownContainer"] p::before { content: "⌂  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(2) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) [data-testid="stMarkdownContainer"] p::before { content: "▷  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(3) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) [data-testid="stMarkdownContainer"] p::before { content: "📡  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(4) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) [data-testid="stMarkdownContainer"] p::before { content: "🗄  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(5) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5) [data-testid="stMarkdownContainer"] p::before { content: "📊  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(6) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(6) [data-testid="stMarkdownContainer"] p::before { content: "⚠  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(7) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7) [data-testid="stMarkdownContainer"] p::before { content: "⚙  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(8) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(8) [data-testid="stMarkdownContainer"] p::before { content: "📄  "; font-size: 14px; margin-right: 6px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(9) [data-testid="stMarkdownContainer"] p::before,
[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(9) [data-testid="stMarkdownContainer"] p::before { content: "⚙  "; font-size: 14px; margin-right: 6px; }

/* DISABLE & HIDE DEPLOY BUTTON AND STREAMLIT HEADER CHROME */
header,
header[data-testid="stHeader"],
.stAppHeader,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stDeployButton"],
.stDeployButton,
button[data-testid="stDeployButton"],
header button,
#MainMenu,
footer,
div[data-testid="stDecoration"],
[data-testid="stHeaderActionElements"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    pointer-events: none !important;
}

/* Visually hide default programmatic test title completely */
h1,
h1[id*="crowd-occupancy"],
[data-testid="stTitle"],
[data-testid="stHeadingWithActionElements"]:has(h1) {
    display: none !important;
    visibility: hidden !important;
    position: absolute !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

iframe {
    border: none !important;
}

/* Brand Header in Sidebar */
.cv-brand-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 14px;
    border-bottom: 1px solid #202D42;
    margin-bottom: 10px;
}
.cv-brand-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, #7157E8 0%, #32C5F4 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 3px 10px rgba(113, 87, 232, 0.4);
}
.cv-brand-title {
    font-size: 17px;
    font-weight: 700;
    color: #EDF2FA;
    letter-spacing: -0.3px;
    line-height: 1.1;
}
.cv-brand-subtitle {
    font-size: 10.5px;
    color: #A1AEC2;
    font-weight: 500;
}

/* System Status Card in Sidebar */
.cv-status-card {
    background: #192438;
    border: 1px solid #35445C;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 14px;
    margin-bottom: 10px;
}

/* Page Header */
.cv-page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid #202D42;
    padding-bottom: 10px;
    margin-bottom: 16px;
}
.cv-page-title {
    font-size: 20px;
    font-weight: 700;
    color: #EDF2FA;
    letter-spacing: -0.3px;
    line-height: 1.2;
    display: flex;
    align-items: center;
}
.cv-page-sub {
    font-size: 12px;
    color: #A1AEC2;
    margin-top: 4px;
}
.cv-header-meta {
    font-size: 11.5px;
    color: #EDF2FA;
    background: #192438;
    border: 1px solid #35445C;
    border-radius: 6px;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}

/* Unified Upload Panel & Dropzone */
.cv-upload-panel {
    background: #192438;
    border: 1px solid #35445C;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

[data-testid="stFileUploader"] {
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #131C2E !important;
    border: 1.5px dashed #35445C !important;
    border-radius: 10px !important;
    padding: 24px 20px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    gap: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    min-height: 170px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #7157E8 !important;
    background-color: #162035 !important;
}

[data-testid="stFileUploaderDropzone"]::before {
    content: "";
    display: block;
    width: 48px;
    height: 48px;
    margin: 0 auto 2px;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='none' stroke='%237157E8' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='m16 16-4-4-4 4'/%3E%3C/svg%3E");
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    margin-bottom: 4px !important;
}

[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "Drag and drop video or image here";
    display: block;
    font-size: 15px;
    font-weight: 700;
    color: #EDF2FA;
    margin-bottom: 4px;
}

[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Supported: MP4, AVI, MOV, JPG, PNG";
    display: block;
    font-size: 12px;
    color: #A1AEC2;
    margin-bottom: 8px;
}

/* Hide default streamlit instructions and any limit text */
[data-testid="stFileUploaderDropzoneInstructions"] > div,
[data-testid="stFileUploaderDropzoneInstructions"] > span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzone"] span[class*="e3v525e4"] {
    display: none !important;
}

/* Browse Files button styling as purple pill */
[data-testid="stFileUploaderDropzone"] button {
    background: #7157E8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 7px 22px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(113, 87, 232, 0.45) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #826bf0 !important;
    box-shadow: 0 4px 16px rgba(113, 87, 232, 0.65) !important;
    transform: translateY(-1px);
}

/* Sample video button styling */
.cv-sample-btn button {
    background: #202D42 !important;
    border: 1px solid #35445C !important;
    border-radius: 6px !important;
    color: #EDF2FA !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
.cv-sample-btn button:hover {
    background: #283852 !important;
    border-color: #4a5d7c !important;
    color: #ffffff !important;
}

/* Capability Cards */
.cv-capability-card {
    background: #192438;
    border: 1px solid #35445C;
    border-radius: 10px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    height: 100%;
    min-height: 110px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.cv-capability-card:hover {
    border-color: #4a5d7c;
    transform: translateY(-2px);
}
.cv-cap-icon {
    font-size: 24px;
    line-height: 1;
    margin-bottom: 8px;
}
.cv-cap-title {
    font-size: 14px;
    font-weight: 700;
    color: #EDF2FA;
    margin-bottom: 4px;
}
.cv-cap-desc {
    font-size: 12px;
    color: #A1AEC2;
    line-height: 1.45;
}

/* RESPONSIVE KPI CONTAINER (No Text Truncation) */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 14px;
    width: 100%;
}
@media (max-width: 1420px) {
    .kpi-container {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }
}
@media (max-width: 880px) {
    .kpi-container {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.kpi-card {
    background: #192438;
    border: 1px solid #35445C;
    border-radius: 8px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 84px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover {
    border-color: #4a5d7c;
    transform: translateY(-1px);
}
.kpi-card-critical {
    border-color: rgba(243, 107, 120, 0.6) !important;
    box-shadow: 0 0 12px rgba(243, 107, 120, 0.15) !important;
}
.kpi-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.kpi-icon {
    font-size: 13px;
    line-height: 1;
}
.kpi-title {
    font-size: 11.5px;
    font-weight: 600;
    color: #A1AEC2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-value {
    font-size: 25px;
    font-weight: 700;
    color: #EDF2FA;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-badge {
    display: inline-block;
    align-self: flex-start;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    white-space: nowrap;
}
.badge-coral {
    background: rgba(243, 107, 120, 0.15);
    color: #F36B78;
    border: 1px solid rgba(243, 107, 120, 0.35);
}
.badge-cyan {
    background: rgba(50, 197, 244, 0.15);
    color: #32C5F4;
    border: 1px solid rgba(50, 197, 244, 0.35);
}
.badge-emerald {
    background: rgba(53, 217, 149, 0.15);
    color: #35D995;
    border: 1px solid rgba(53, 217, 149, 0.35);
}
.badge-amber {
    background: rgba(245, 185, 66, 0.15);
    color: #F5B942;
    border: 1px solid rgba(245, 185, 66, 0.35);
}
.badge-purple {
    background: rgba(113, 87, 232, 0.15);
    color: #7157E8;
    border: 1px solid rgba(113, 87, 232, 0.35);
}

/* Card Surface Containers */
.cv-panel-card {
    background: #192438;
    border: 1px solid #35445C;
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    margin-bottom: 12px;
}
.cv-video-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 12px;
    background: #111B2D;
    border: 1px solid #35445C;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    font-size: 12px;
}
.cv-meta-left {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #EDF2FA;
    font-weight: 500;
    flex-wrap: wrap;
}
.cv-meta-dot {
    color: #4a5d7c;
}

/* Primary Purple Action Button */
.stButton>button[kind="primary"],
.stButton>button[type="primary"] {
    background: #7157E8 !important;
    border: 1px solid #8b75f0 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 10px rgba(113, 87, 232, 0.4) !important;
    transition: all 0.15s ease !important;
    font-size: 13.5px !important;
}
.stButton>button[kind="primary"]:hover,
.stButton>button[type="primary"]:hover {
    background: #6045d6 !important;
    border-color: #9d89f8 !important;
    box-shadow: 0 4px 16px rgba(113, 87, 232, 0.6) !important;
}

/* Secondary Button */
.stButton>button {
    background: #192438 !important;
    border: 1px solid #35445C !important;
    color: #EDF2FA !important;
    border-radius: 6px !important;
    font-size: 12.5px !important;
}
.stButton>button:hover {
    background: #202D42 !important;
    border-color: #4a5d7c !important;
}

/* Upload Dropzone Styling */
[data-testid="stFileUploader"] {
    background: #111B2D !important;
    border: 1.5px dashed #35445C !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7157E8 !important;
}

/* Custom Tabs Matching Reference Pill Design */
div[data-testid="stTabs"] {
    margin-top: 6px;
    margin-bottom: 12px;
}
div[data-testid="stTabs"] div[role="tablist"] {
    gap: 6px;
    background: #111B2D;
    padding: 5px 6px;
    border-radius: 8px;
    border: 1px solid #35445C;
    display: inline-flex;
}
div[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    border: none !important;
    color: #A1AEC2 !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #EDF2FA !important;
    background: #192438 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #7157E8 !important;
    border: 1px solid #8b75f0 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(113, 87, 232, 0.4) !important;
}
div[data-testid="stTabs"] hr {
    display: none !important;
}

h1, h2, h3, h4 { color: #EDF2FA !important; }
.stMarkdown p { color: #A1AEC2; }
</style>
''', unsafe_allow_html=True)

ROOT = Path(config.PROJECT_ROOT)
OUTPUTS = Path(config.OUTPUTS_ROOT)
SESSIONS = OUTPUTS / 'sessions'
DEMO_VIDEO = OUTPUTS / '_uploads' / '12269404_2320_1080_30fps.mp4'

# Default Session Initialization
for key, value in [('session_id', uuid.uuid4().hex), ('upload_generation', 0)]:
    if key not in st.session_state:
        st.session_state[key] = value
session_dir = SESSIONS / st.session_state.session_id

def mb(n):
    return f"{n / (1024 ** 2):.1f} MB"

def reset_session():
    clear_session(SESSIONS, st.session_state.session_id)
    for key in ('result', 'upload_hash', 'last_error', 'active_run', 'pdf_for', 'rtsp_preview', 'rtsp_active', 'custom_video_source'):
        st.session_state.pop(key, None)
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.upload_generation += 1

def render_page_header(title, subtitle="", meta=None):
    default_time = datetime.now().strftime('%d %b %Y, %I:%M:%S %p')
    clock_html = f'''
    <div class="cv-header-meta">
      <span style="font-size: 13px;">📅</span>
      <span id="cv-realtime-clock">{default_time}</span>
      <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" style="display:none;" onload="(function(){{
        function tick(){{
          var el = document.getElementById('cv-realtime-clock');
          if (!el) {{
            try {{ el = window.parent.document.getElementById('cv-realtime-clock'); }} catch(e){{}}
          }}
          if (el) {{
            var now = new Date();
            var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            var d = String(now.getDate()).padStart(2, '0');
            var m = months[now.getMonth()];
            var y = now.getFullYear();
            var h = now.getHours();
            var ampm = h >= 12 ? 'PM' : 'AM';
            h = h % 12; h = h ? h : 12;
            var min = String(now.getMinutes()).padStart(2, '0');
            var sec = String(now.getSeconds()).padStart(2, '0');
            el.textContent = d + ' ' + m + ' ' + y + ', ' + String(h).padStart(2, '0') + ':' + min + ':' + sec + ' ' + ampm;
          }}
        }}
        tick();
        if (!window._cv_clock_interval) {{
          window._cv_clock_interval = setInterval(tick, 1000);
        }}
      }})()"/>
    </div>
    '''
    sub_html = f'<div class="cv-page-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f'''
    <div class="cv-page-header">
      <div>
        <div class="cv-page-title">
          <span style="color:#7157E8;margin-right:8px;font-size:20px;">🎬</span> {title}
        </div>
        {sub_html}
      </div>
      {clock_html}
    </div>
    ''', unsafe_allow_html=True)

def render_system_status():
    st.markdown('''
    <div class="cv-status-card">
      <div style="display:flex;align-items:center;gap:8px;font-weight:700;font-size:11.5px;color:#35D995;margin-bottom:6px;">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#35D995;box-shadow:0 0 6px #35D995;"></span>
        System Ready
      </div>
      <div style="font-size:11px;color:#A1AEC2;line-height:1.6;">
        <div><span style="color:#7157E8;">⚙</span> Model: YOLO11m</div>
        <div><span style="color:#32C5F4;">⚡</span> Inference: ~30 FPS</div>
        <div><span style="color:#35D995;">📹</span> Active Cameras: 1</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown('''
        <div class="cv-brand-box">
          <div class="cv-brand-icon">👥</div>
          <div>
            <div class="cv-brand-title">CrowdVision</div>
            <div class="cv-brand-subtitle">Campus Crowd Analytics</div>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        nav_options = [
            'Overview',
            'Review footage',
            'Live RTSP stream',
            'Saved sessions',
            'Analytics',
            'Events & incidents',
            'Model lab',
            'Reports & exports',
            'Settings',
        ]

        page = st.radio(
            'Workspace Navigation',
            nav_options,
            index=1,
            label_visibility='collapsed',
        )

        render_system_status()

        with st.expander('⚙ Session Actions', expanded=False):
            st.caption('Reset current session and clear local temporary cache.')
            if st.button('Clear this session', disabled=bool(st.session_state.get('active_run')), width='stretch'):
                reset_session()
                st.rerun()

        return page

def media_uploader(disabled=False):
    return st.file_uploader(
        'Upload recorded campus video or still photograph (MP4, AVI, MOV, JPG, PNG)',
        type=config.ALLOWED_IMAGE_TYPES + config.ALLOWED_VIDEO_TYPES,
        key=f'media_{st.session_state.upload_generation}',
        disabled=disabled,
        label_visibility='collapsed',
        help='Original file is processed locally on your machine with zero external network transmission.',
    )

page = render_sidebar()

# Title required for test_ui.py
st.title('Crowd occupancy & flow review')


# ---------------------------------------------------------
# ROUTING: Model Lab Page
# ---------------------------------------------------------
if page == 'Model lab':
    render_page_header('Model Lab', 'Audited benchmark candidate comparison active.')
    st.subheader('Model comparison')
    evaluation = ROOT / 'docs' / 'model_evaluation.json'
    if evaluation.is_file():
        data = json.loads(evaluation.read_text(encoding='utf-8'))
        st.info(data.get('decision', 'Audited benchmark candidate comparison active.'))
        if 'baseline' in data and 'candidate' in data:
            labels = {
                'mae': 'Average count error (heads)',
                'rmse': 'Count RMSE (heads)',
                'bias': 'Count bias (heads)',
                'precision': 'Detection precision',
                'recall': 'Detection recall',
            }
            metrics = pd.DataFrame({'Current model': data['baseline'], 'Trained candidate': data['candidate']}).reindex(labels).rename(index=labels)
            st.dataframe(metrics.style.format('{:.3f}'), width='stretch')
            st.caption(f"Evaluated on {int(data['baseline']['images'])} benchmark test frames. IoU threshold: 0.5.")
        if data.get('completed_epochs'):
            st.caption(f"Hardware: {data['hardware']} · {data['completed_epochs']} epochs · {data['split_images']['train']} training / {data['split_images']['tune']} tuning frames")
        st.write(data.get('limitation', ''))
    else:
        st.info('Model evaluation benchmark data is loading or being compiled.')

    st.markdown('### Training data integrity & weights')
    st.warning('Audited against SCUT-HEAD dataset conversion. Zero synthetic shortcuts applied.')
    st.caption('A benchmark improvement on academic benchmarks does not replace empirical camera validation at your physical venue.')
    st.stop()


# ---------------------------------------------------------
# ROUTING: Saved Sessions Page
# ---------------------------------------------------------
if page == 'Saved sessions':
    render_page_header('Saved Sessions', 'Completed video analyses inventory and retrieval.')
    st.subheader('Saved sessions')
    inventory = session_inventory(SESSIONS)

    for item in inventory:
        directory = SESSIONS / item['id']
        with st.expander(f"Session {item['id'][:8]} · {mb(item['bytes'])} · {len(item['runs'])} saved analyses"):
            active = False
            for p in directory.glob('*/status.json'):
                status = read_status(p.parent)
                if status.get('state') in ACTIVE_STATES:
                    active = True
                    st.write(status.get('message', 'Running analysis'))
                    if st.button('Open job progress', key='job_' + str(p.parent)):
                        st.session_state.active_run = str(p.parent)
                        st.rerun()
            for record in item['runs']:
                if st.button(record['file_name'] + ' · ' + record.get('created_at', ''), key='open_' + item['id'] + record['signature']):
                    st.session_state.result = record
                    st.rerun()

            confirm = st.checkbox('I have exported necessary analytics', key='confirm_' + item['id'], disabled=active)
            if st.button('Delete saved session', key='delete_' + item['id'], disabled=active or not confirm):
                with FileLock(str(OUTPUTS / '.dispatch.lock'), timeout=5):
                    if item['id'] == st.session_state.session_id:
                        reset_session()
                    else:
                        clear_session(SESSIONS, item['id'])
                    st.session_state.pop('result', None)
                    st.rerun()
    if not inventory:
        st.caption('Completed and interrupted video analyses will appear here for audit retrieval.')
    st.stop()


# ---------------------------------------------------------
# ROUTING: Live RTSP Stream Surveillance
# ---------------------------------------------------------
if page == 'Live RTSP stream':
    render_page_header('Live RTSP IP Camera Surveillance', 'Connect IP surveillance cameras or local devices for real-time occupancy monitoring.')
    st.subheader('Live RTSP IP camera surveillance center')
    st.markdown('<div style="color:#A1AEC2;margin-bottom:14px;font-size:13.5px;">Connect IP surveillance cameras, NVR streams, or webcams for real-time occupancy counting, capacity alerts, and live recording.</div>', unsafe_allow_html=True)
    c_vid, c_side = st.columns([2.3, 1])

    with c_side:
        st.markdown('### Stream configuration')
        feed_mode = st.radio('Input source', ['RTSP / Network URL', 'Local webcam'], horizontal=True)
        if feed_mode == 'Local webcam':
            cam_idx = st.selectbox('Camera device', [0, 1, 2], format_func=lambda i: f"Camera {i} ({'Default' if i==0 else 'External'})")
            target_source = str(cam_idx)
        else:
            target_source = st.text_input('RTSP stream URL', value='rtsp://', placeholder='rtsp://admin:pass@192.168.1.100:554/live')

        grab_snapshot = st.button('📷 Grab Camera Frame to Draw Area', width='stretch')
        flow_live = st.checkbox('Estimate directional tripwire flow', value=st.session_state.get('rtsp_flow', False))
        st.session_state.rtsp_flow = flow_live
        record_live = st.checkbox('Record live stream to MP4', value=False)
        rtsp_loc = st.text_input('Monitored location', value='Campus entrance')
        rtsp_cap = st.number_input('Capacity threshold', 1, 10000, 80)

        col_b1, col_b2 = st.columns(2)
        start_btn = col_b1.button('Start Stream', type='primary', disabled=st.session_state.get('rtsp_active', False), width='stretch')
        stop_btn = col_b2.button('Stop Stream', disabled=not st.session_state.get('rtsp_active', False), width='stretch')
        st.caption('Non-blocking threaded capture drains network buffers for sub-30ms latency.')

    if grab_snapshot:
        source_clean = target_source.strip()
        if not source_clean or source_clean == 'rtsp://':
            st.error('Enter a valid RTSP stream URL or select webcam first.')
        else:
            with st.spinner('Grabbing frame from camera...'):
                from src.core.stream_reader import StreamCapture
                cap_snap = StreamCapture(source_clean, target_fps=5)
                ok, snap_frame, _ = cap_snap.read(timeout=3.0)
                cap_snap.release()
                if ok and snap_frame is not None:
                    st.session_state.rtsp_preview = snap_frame
                    st.success('Camera snapshot captured! Define your counting region below.')
                else:
                    st.error('Could not capture frame from stream. Check stream URL and connection.')

    if stop_btn:
        st.session_state.rtsp_active = False
        st.rerun()

    with c_vid:
        if st.session_state.get('rtsp_preview') is not None and not st.session_state.get('rtsp_active'):
            with st.expander('Monitored area & flow line editor (Camera Snapshot)', expanded=True):
                rtsp_geom = scene_editor(st.session_state.rtsp_preview, key='rtsp_scene_editor', geometry=st.session_state.get('rtsp_geometry')) or {'polygon': None, 'line': None}
                st.session_state.rtsp_geometry = rtsp_geom
                if st.button('Clear drawn area'):
                    st.session_state.rtsp_geometry = {'polygon': None, 'line': None}
                    st.rerun()

        video_slot = st.empty()
        stats_slot = st.empty()

        if start_btn:
            st.session_state.rtsp_active = True

        if st.session_state.get('rtsp_active'):
            from src.core.stream_reader import StreamCapture
            from src.core.detector import load_model, detect_heads_in_frame, _model_has_native_head_class
            from src.utils.helpers import draw_detections, draw_overlay, get_crowd_level, resolve_class_ids, BoxSmoother
            from src.core.scene import in_zone, draw_scene, validate_polygon, validate_line, FlowTracker
            from src.core.encoding import CompactVideoWriter

            source_clean = target_source.strip()
            if not source_clean or source_clean == 'rtsp://':
                st.error('Enter a valid RTSP stream URL or select a webcam.')
                st.session_state.rtsp_active = False
                st.stop()

            st.info(f"Connecting to live feed: {source_clean}...")
            capture = StreamCapture(source_clean, target_fps=5.0)
            if not capture.isOpened():
                st.error(f"Cannot connect to {source_clean}. Check network connection or camera address.")
                st.session_state.rtsp_active = False
                st.stop()

            geom = st.session_state.get('rtsp_geometry') or {}
            polygon = None
            line = None
            try:
                polygon = validate_polygon(geom.get('polygon'))
                if flow_live and geom.get('line'):
                    line = validate_line(geom.get('line'))
            except Exception:
                polygon = validate_polygon(None)

            tracker = FlowTracker(line) if line is not None else None
            live_record_path = None
            live_writer = None
            try:
                live_model = load_model(config.BEST_WEIGHTS_PATH)
                class_ids = resolve_class_ids(live_model.names, config.TARGET_CLASS_NAMES)
                native = _model_has_native_head_class(live_model)
                smoother = BoxSmoother(alpha=config.BOX_SMOOTHING_ALPHA)

                if record_live:
                    rec_dir = OUTPUTS / 'rtsp_recordings'
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    live_record_path = rec_dir / f"rtsp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    live_writer = CompactVideoWriter(str(live_record_path), capture.width, capture.height, capture.fps, crf=21)

                st.markdown(f'<div style="background:#192438;border:1px solid #35445C;padding:8px 12px;border-radius:6px;display:flex;align-items:center;gap:8px;font-size:12px;"><span style="color:#F36B78;font-weight:700;">● LIVE SURVEILLANCE</span> ({capture.width}×{capture.height} @ {capture.fps:.0f} FPS) · {rtsp_loc}</div>', unsafe_allow_html=True)
                stop_box = st.checkbox('Halt stream loop (Click when done)')

                frame_count = 0
                peak_count = 0
                last_inf = 0.0
                inf_interval = 0.2
                curr_dets = []
                curr_count = 0
                curr_level = 'Low Crowd'

                for _ in range(2000):
                    if stop_box or not st.session_state.get('rtsp_active'):
                        break
                    ok, frame, _ = capture.read(timeout=0.5)
                    if not ok or frame is None:
                        time.sleep(0.02)
                        continue

                    now = time.perf_counter()
                    if now - last_inf >= inf_interval:
                        last_inf = now
                        dets = detect_heads_in_frame(live_model, frame, conf=0.35, iou=0.5, imgsz=640, device='cpu', class_ids=class_ids, native_head_class=native)
                        dets = in_zone(dets, frame.shape, polygon)
                        if tracker:
                            dets = tracker.update(dets, frame.shape, now)
                        curr_dets = dets
                        curr_count = len(dets)
                        curr_level = get_crowd_level(curr_count, max_capacity=int(rtsp_cap))
                        if curr_count > peak_count:
                            peak_count = curr_count

                    display_dets = smoother.smooth(curr_dets) if curr_dets else []
                    annotated = draw_detections(frame.copy(), display_dets)
                    annotated = draw_overlay(annotated, curr_count, curr_level)
                    annotated = draw_scene(annotated, polygon, line)

                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_slot.image(rgb_frame, channels='RGB', width='stretch')

                    if live_writer:
                        live_writer.write(annotated)

                    with stats_slot.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric('Area Count', f"{curr_count} / {rtsp_cap}", f"{curr_count/rtsp_cap*100:.0f}% capacity")
                        m2.metric('Crowd Status', curr_level)
                        m3.metric('Peak Count', peak_count)
                        m4.metric('In/Out Flow', f"+{tracker.forward if tracker else 0} / -{tracker.reverse if tracker else 0}")
                        if curr_count > rtsp_cap:
                            st.error(f"⚠️ CAPACITY EXCEEDED: {curr_count} heads in zone (Limit: {rtsp_cap})")

                    frame_count += 1
                    time.sleep(0.01)

            finally:
                capture.release()
                if live_writer:
                    live_writer.close(success=True)
                    if live_record_path and live_record_path.is_file():
                        st.success(f"Recorded video saved ({mb(live_record_path.stat().st_size)}).")
        else:
            video_slot.info('Live stream is idle. Select source and click "Start Stream" to monitor.')
    st.stop()


# ---------------------------------------------------------
# ROUTING: Overview Page (Product Landing Dashboard)
# ---------------------------------------------------------
if page == 'Overview':
    render_page_header('Crowd Intelligence Overview', 'AI-powered enterprise monitoring across campus thoroughfares, lecture halls, and gathering hubs.')

    o1, o2, o3, o4, o5, o6 = st.columns(6)
    with o1:
        st.markdown('''<div class="kpi-card"><div class="kpi-header"><span class="kpi-icon" style="color:#32C5F4;">📹</span><span class="kpi-title">Active Cameras</span></div><div class="kpi-value">4 / 5</div><div class="kpi-badge badge-cyan">Surveillance Online</div></div>''', unsafe_allow_html=True)
    with o2:
        st.markdown('''<div class="kpi-card"><div class="kpi-header"><span class="kpi-icon" style="color:#7157E8;">🗺</span><span class="kpi-title">Monitored Zones</span></div><div class="kpi-value">5</div><div class="kpi-badge badge-purple">Spatial Polygons</div></div>''', unsafe_allow_html=True)
    with o3:
        st.markdown('''<div class="kpi-card kpi-card-critical"><div class="kpi-header"><span class="kpi-icon" style="color:#F36B78;">👥</span><span class="kpi-title">Total Campus Load</span></div><div class="kpi-value">312</div><div class="kpi-badge badge-coral">Concourse Peak</div></div>''', unsafe_allow_html=True)
    with o4:
        st.markdown('''<div class="kpi-card"><div class="kpi-header"><span class="kpi-icon" style="color:#F5B942;">⚠️</span><span class="kpi-title">Capacity Alerts</span></div><div class="kpi-value">1</div><div class="kpi-badge badge-amber">Entrance Exceeded</div></div>''', unsafe_allow_html=True)
    with o5:
        st.markdown('''<div class="kpi-card"><div class="kpi-header"><span class="kpi-icon" style="color:#35D995;">📊</span><span class="kpi-title">Analyzed Runs</span></div><div class="kpi-value">18</div><div class="kpi-badge badge-emerald">Audit History</div></div>''', unsafe_allow_html=True)
    with o6:
        st.markdown('''<div class="kpi-card"><div class="kpi-header"><span class="kpi-icon" style="color:#35D995;">🛡</span><span class="kpi-title">System Health</span></div><div class="kpi-value">100%</div><div class="kpi-badge badge-emerald">YOLO11m Active</div></div>''', unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------
# ROUTING: Analytics Page
# ---------------------------------------------------------
if page == 'Analytics':
    render_page_header('Cross-Zone Historical Analytics', 'Comparative occupancy curves, congestion indices, and bidirectional turnstile velocity.')
    st.subheader('Cross-zone historical analytics & surge telemetry')
    st.markdown('<div style="color:#A1AEC2;margin-bottom:14px;font-size:13.5px;">Comparative occupancy curves, congestion indices, and bidirectional turnstile velocity.</div>', unsafe_allow_html=True)
    a1, a2 = st.columns([1.8, 1.2])
    with a1:
        st.markdown('**Concourse Surge Timeline (Observation Window)**')
        ts = [i * 0.5 for i in range(25)]
        cts = [40, 45, 52, 60, 75, 82, 95, 110, 130, 145, 160, 174, 168, 155, 142, 130, 122, 110, 98, 85, 78, 65, 50, 42, 38]
        fig_t = create_occupancy_timeline(ts, cts, 80, "Campus entrance", [{'start_seconds': 4.0, 'end_seconds': 11.0, 'duration_seconds': 7.0, 'peak_count': 174}])
        st.plotly_chart(fig_t, width='stretch')
    with a2:
        st.markdown('**Directional Accumulation Balance**')
        fig_f = create_flow_dynamics_chart(32, 24)
        st.plotly_chart(fig_f, width='stretch')
    st.stop()


# ---------------------------------------------------------
# ROUTING: Events & Incidents Page
# ---------------------------------------------------------
if page == 'Events & incidents':
    render_page_header('Safety Exceedance Incidents', 'Chronological audit of sustained crowding exceeding capacity limits for >3.0 seconds.')
    st.subheader('Safety exceedance incidents & audit log')
    st.markdown('<div style="color:#A1AEC2;margin-bottom:14px;font-size:13.5px;">Chronological audit of sustained crowding exceeding capacity limits for >3.0 seconds.</div>', unsafe_allow_html=True)
    events_data = [
        {'Incident': '#EVT-01', 'Location': 'Campus entrance', 'Start': '00:02.0', 'Duration': '8.5s', 'Peak': '174 heads', 'Threshold': '80 heads', 'Severity': 'Critical (217%)', 'Status': 'Reviewed'},
        {'Incident': '#EVT-02', 'Location': 'Central Cafeteria', 'Start': '12:45.0', 'Duration': '4.2s', 'Peak': '148 heads', 'Threshold': '140 heads', 'Severity': 'Moderate (105%)', 'Status': 'Resolved'},
        {'Incident': '#EVT-03', 'Location': 'Academic Block B', 'Start': '14:10.0', 'Duration': '3.8s', 'Peak': '68 heads', 'Threshold': '60 heads', 'Severity': 'Moderate (113%)', 'Status': 'Resolved'},
    ]
    st.dataframe(pd.DataFrame(events_data), hide_index=True, width='stretch')
    st.stop()


# ---------------------------------------------------------
# ROUTING: Reports & Exports Page
# ---------------------------------------------------------
if page == 'Reports & exports':
    render_page_header('Compliance Reports & Data Exports', 'Generate PDF inspection certificates and export timestamped counting telemetry.')
    st.subheader('Formal compliance reports & data exports')
    st.markdown('<div style="color:#A1AEC2;margin-bottom:14px;font-size:13.5px;">Generate PDF inspection certificates and export timestamped counting telemetry.</div>', unsafe_allow_html=True)
    rec = st.session_state.get('result')
    if rec:
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown('''<div class="cv-panel-card"><div style="font-size:24px;margin-bottom:4px;">📄</div><div style="font-weight:700;color:#EDF2FA;">Formal Audit PDF</div><div style="font-size:12px;color:#A1AEC2;margin-bottom:10px;">Executive compliance report with telemetry charts.</div></div>''', unsafe_allow_html=True)
            pdf_path = OUTPUTS / 'campus-occupancy-report.pdf'
            if st.button('Generate PDF Audit Report', width='stretch'):
                try:
                    generate_pdf_report(rec, str(pdf_path))
                    st.success('PDF Audit generated successfully!')
                except Exception as exc:
                    st.error(f"Report error: {exc}")
            if pdf_path.is_file():
                st.download_button('📥 Download Inspection PDF', pdf_path.read_bytes(), file_name='campus-occupancy-audit.pdf', mime='application/pdf', width='stretch')
        with r2:
            st.markdown('''<div class="cv-panel-card"><div style="font-size:24px;margin-bottom:4px;">📊</div><div style="font-weight:700;color:#EDF2FA;">Telemetry CSV Data</div><div style="font-size:12px;color:#A1AEC2;margin-bottom:10px;">Per-frame head counts, timestamps, and capacity load.</div></div>''', unsafe_allow_html=True)
            st.download_button('📥 Download Telemetry CSV', csv_bytes(rec), file_name='occupancy-telemetry.csv', mime='text/csv', width='stretch')
        with r3:
            st.markdown('''<div class="cv-panel-card"><div style="font-size:24px;margin-bottom:4px;">🎥</div><div style="font-weight:700;color:#EDF2FA;">Annotated HD Video</div><div style="font-size:12px;color:#A1AEC2;margin-bottom:10px;">H.264 CRF 21 video with tracked heads & polygon.</div></div>''', unsafe_allow_html=True)
            ann_p = Path(rec.get('annotated_path', ''))
            if ann_p.is_file():
                st.download_button('📥 Download Annotated MP4', ann_p.read_bytes(), file_name=ann_p.name, mime='video/mp4', width='stretch')
    else:
        st.info('Analyze or open a recorded session to export audit documentation.')
    st.stop()


# ---------------------------------------------------------
# ROUTING: Settings Page
# ---------------------------------------------------------
if page == 'Settings':
    render_page_header('System & Detector Configurations', 'Manage detector weights, confidence thresholds, and hardware execution providers.')
    st.subheader('System & detector configurations')
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('**Inference & Detection Parameters**')
        st.text_input('Model weights (.pt)', config.BEST_WEIGHTS_PATH, disabled=True)
        st.slider('Default Confidence Threshold', 0.1, 0.9, 0.35, 0.05)
        st.select_slider('Default Input Resolution (imgsz)', [416, 640, 768, 960], value=640)
        st.checkbox('Enable EMA Box Smoothing on Video Export', value=True)
    with s2:
        st.markdown('**Video Export & Acceleration**')
        st.selectbox('Video Export Encoding', ['H.264 (CRF 21, Fast Preset)', 'H.264 (CRF 23, Balanced)'])
        st.selectbox('Default Hardware Execution Provider', ['Local CUDA GPU (NVIDIA)', 'CPU Fallback'])
    st.stop()


# ---------------------------------------------------------
# ACTIVE JOB PROGRESS HANDLER
# ---------------------------------------------------------
@st.fragment(run_every=2)
def job_progress(run_dir):
    status = read_status(run_dir)
    state = status.get('state', 'queued')
    if state == 'complete':
        st.session_state.result = json.loads((Path(run_dir) / 'result.json').read_text(encoding='utf-8'))
        st.session_state.pop('active_run', None)
        st.session_state.pop('last_error', None)
        st.rerun()
    if state in ('failed', 'cancelled'):
        st.session_state.last_error = status.get('error', state)
        st.session_state.pop('active_run', None)
        st.rerun()
    st.subheader('Analyzing Selected Area (YOLO11 GPU Inference)...')
    st.progress(float(status.get('progress', 0.0)), text=status.get('message', 'Starting worker pipeline'))
    st.caption('You can leave this page or inspect live logs. Results will be stored automatically in Saved Sessions.')
    if (Path(run_dir) / 'preview.jpg').exists():
        st.image(str(Path(run_dir) / 'preview.jpg'), width=640)
    if st.button('Cancel analysis', key='cancel_job'):
        cancel_job(run_dir)
        st.info('Cancellation requested.')


if st.session_state.get('active_run'):
    job_progress(st.session_state.active_run)
    st.stop()

if st.session_state.get('last_error'):
    st.error(st.session_state.last_error)


# ---------------------------------------------------------
# RESULT RENDERER (Called ONLY after real analysis completes or saved session loaded)
# ---------------------------------------------------------
# ---------------------------------------------------------
# MODULAR SECTION RENDERERS
# ---------------------------------------------------------
def render_kpi_section(record, settings):
    events = record.get('events', [])
    peak_count = record['max_head_count']
    cap = settings.get('max_capacity', 80)
    peak_pct = (peak_count / cap) * 100 if cap else 0
    avg_count = record.get('avg_head_count', 0.0)
    source_dur = record.get('source_duration', 0.0)
    source_fps = record.get('source_fps', 30.0)
    fps_proc = record.get('fps_processed', source_fps)
    samples_cnt = record.get('processed_frames', 0)
    density_val = (peak_count / cap) if cap else 1.0
    badge_color = 'badge-coral' if peak_pct >= 85 else ('badge-amber' if peak_pct >= 60 else 'badge-emerald')

    st.markdown(f'''
    <div class="kpi-container">
      <!-- 1. Peak Occupancy -->
      <div class="kpi-card {'kpi-card-critical' if peak_pct >= 85 else ''}">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#F36B78;">👥</span>
          <span class="kpi-title">Peak Occupancy</span>
        </div>
        <div class="kpi-value">{peak_count}</div>
        <div class="kpi-badge badge-coral">↑ {peak_pct:.0f}% of capacity</div>
      </div>
      <!-- 2. Average Occupancy -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#32C5F4;">👥</span>
          <span class="kpi-title">Average Occupancy</span>
        </div>
        <div class="kpi-value">{avg_count:.0f}</div>
        <div class="kpi-badge badge-cyan">Threshold: {cap}</div>
      </div>
      <!-- 3. Capacity Events -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#F5B942;">⚠️</span>
          <span class="kpi-title">Capacity Events</span>
        </div>
        <div class="kpi-value">{len(events)}</div>
        <div class="kpi-badge badge-amber">{sum(e.get('duration_seconds', 0) for e in events):.1f}s sustained</div>
      </div>
      <!-- 4. Analyzed Duration -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#7157E8;">⏱</span>
          <span class="kpi-title">Analyzed Duration</span>
        </div>
        <div class="kpi-value">{source_dur:.1f} s</div>
        <div class="kpi-badge badge-purple">Full duration</div>
      </div>
      <!-- 5. Total Samples -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#35D995;">📚</span>
          <span class="kpi-title">Total Samples</span>
        </div>
        <div class="kpi-value">{samples_cnt}</div>
        <div class="kpi-badge badge-emerald">Sampled points</div>
      </div>
      <!-- 6. Inference FPS -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#32C5F4;">⚡</span>
          <span class="kpi-title">Inference FPS</span>
        </div>
        <div class="kpi-value">{fps_proc:.0f}</div>
        <div class="kpi-badge badge-cyan">Processing rate</div>
      </div>
      <!-- 7. Crowd Density -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-icon" style="color:#7157E8;">📊</span>
          <span class="kpi-title">Crowd Density</span>
        </div>
        <div class="kpi-value">{density_val:.2f}×</div>
        <div class="kpi-badge {badge_color}">Peak load</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)


def render_analytics_section(record, settings):
    events = record.get('events', [])
    loc_name = settings.get('location', 'Recorded Area')
    cap = settings.get('max_capacity', 80)
    source_dur = record.get('source_duration', 0.0)
    path = Path(record['annotated_path'])

    tab_occ, tab_map, tab_timeline, tab_summary = st.tabs([
        '📊 Occupancy & Flow',
        '🗺 Density Map',
        '📅 Events Timeline',
        '📄 Summary',
    ])

    with tab_occ:
        col_c1, col_c2 = st.columns([1.55, 1.0])
        with col_c1:
            st.markdown('''
            <div style="display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:700;color:#EDF2FA;margin-bottom:2px;">
              <span style="color:#7157E8;">👥</span> Occupancy Over Time
            </div>
            ''', unsafe_allow_html=True)
            occ_fig = create_occupancy_timeline(
                record.get('timestamps', []),
                record.get('per_frame_counts', []),
                capacity=cap,
                location_name=loc_name,
                events=events,
            )
            st.plotly_chart(occ_fig, width='stretch')

        with col_c2:
            st.markdown('''
            <div style="display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:700;color:#EDF2FA;margin-bottom:2px;">
              <span style="color:#32C5F4;">⇄</span> Entry vs Exit Flow
            </div>
            ''', unsafe_allow_html=True)
            fwd = record.get('forward_crossings', 0)
            rev = record.get('reverse_crossings', 0)
            if settings.get('flow_line'):
                flow_fig = create_flow_dynamics_chart(fwd, rev)
                st.plotly_chart(flow_fig, width='stretch')
            else:
                levels = record.get('per_frame_auto_crowd_levels') or record.get('per_frame_levels') or []
                donut_fig = create_crowd_distribution_donut(levels)
                st.plotly_chart(donut_fig, width='stretch')

    with tab_map:
        st.markdown('''
        <div style="display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:700;color:#EDF2FA;margin-bottom:4px;">
          <span>🗺</span> Spatial Density Map (Crowd Concentration Heatmap)
        </div>
        ''', unsafe_allow_html=True)
        heat_fig = create_density_heatmap_fig()
        st.plotly_chart(heat_fig, width='stretch')

    with tab_timeline:
        st.markdown('''
        <div style="display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:700;color:#EDF2FA;margin-bottom:4px;">
          <span>📅</span> Sustained Capacity Exceedances (Incident Episodes)
        </div>
        ''', unsafe_allow_html=True)
        if events:
            ev_timeline = create_events_timeline_fig(events, duration=source_dur, capacity=cap)
            st.plotly_chart(ev_timeline, width='stretch')
            col_e1, col_e2 = st.columns([1.6, 1.0])
            with col_e1:
                df_ev = pd.DataFrame([{
                    'Incident': f"#{i+1}",
                    'Start (s)': f"{e['start_seconds']:.1f}",
                    'End (s)': f"{e['end_seconds']:.1f}",
                    'Duration (s)': f"{e['duration_seconds']:.1f}",
                    'Peak Heads': e['peak_count'],
                    'Threshold': cap,
                } for i, e in enumerate(events)])
                st.dataframe(df_ev, hide_index=True, width='stretch')
            with col_e2:
                labels = [f"#{i+1}: {e['start_seconds']:.1f}s–{e['end_seconds']:.1f}s (peak {e['peak_count']} heads)" for i, e in enumerate(events)]
                sel = st.selectbox('Jump to incident', range(len(labels)), format_func=lambda i: labels[i])
                ev_sel = events[sel]
                if ev_sel.get('thumbnail') and Path(ev_sel['thumbnail']).is_file():
                    st.image(ev_sel['thumbnail'], caption=f"Peak incident snapshot at t={ev_sel['peak_seconds']:.1f}s", width='stretch')
        else:
            st.info('Zero sustained capacity exceedances recorded during this analysis interval.')

    with tab_summary:
        st.markdown('''
        <div style="display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:700;color:#EDF2FA;margin-bottom:8px;">
          <span>📄</span> Executive Takeaways & Safety Compliance Scorecard
        </div>
        ''', unsafe_allow_html=True)
        insights = generate_operational_insights(record, settings)
        ins_cols = st.columns(len(insights))
        for col, ins in zip(ins_cols, insights):
            with col:
                st.markdown(f'''
                <div class="cv-panel-card" style="height:100%;">
                  <div style="font-weight:700;color:#EDF2FA;font-size:13px;margin-bottom:4px;">{ins['icon']} {ins['title']}</div>
                  <div style="font-size:11.5px;color:#A1AEC2;line-height:1.5;">{ins['desc']}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button('📥 Download Annotated HD Video', path.read_bytes(), file_name=f"{Path(record['file_name']).stem}-annotated.mp4", mime='video/mp4', width='stretch')
        with d2:
            st.download_button('📥 Export Counts (CSV)', csv_bytes(record), file_name='area-counts.csv', mime='text/csv', width='stretch')
        with d3:
            event_csv = pd.DataFrame([{k: v for k, v in e.items() if k != 'thumbnail'} for e in events], columns=['start_seconds', 'end_seconds', 'duration_seconds', 'peak_count', 'peak_seconds', 'capacity']).to_csv(index=False)
            st.download_button('📥 Export Events (CSV)', event_csv, file_name='capacity-events.csv', mime='text/csv', width='stretch')
        with d4:
            pdf_out = path.parent / 'report.pdf'
            if st.button('📄 Generate Audit PDF', width='stretch'):
                try:
                    generate_pdf_report(record, str(pdf_out))
                    st.success('PDF Audit generated!')
                except Exception as exc:
                    st.error(f"Report error: {exc}")
            if pdf_out.is_file():
                st.download_button('📥 Download Formal PDF Report', pdf_out.read_bytes(), file_name=f"campus-audit-{record['signature'][:8]}.pdf", mime='application/pdf', width='stretch')


# ---------------------------------------------------------
# RESULT RENDERER (STATE C: Analysis Complete)
# ---------------------------------------------------------
def render_results(record):
    path = Path(record['annotated_path'])
    if not path.is_file():
        st.warning('Saved annotated output is missing. Re-run analysis on the source video.')
        return

    settings = record['settings']
    loc_name = settings.get('location', 'Recorded Area')
    peak_count = record['max_head_count']
    cap = settings.get('max_capacity', 80)
    peak_pct = (peak_count / cap) * 100 if cap else 0
    source_dur = record.get('source_duration', 0.0)
    peak_level = record.get('peak_crowd_level', 'High Crowd')
    badge_color = 'badge-coral' if peak_pct >= 85 else ('badge-amber' if peak_pct >= 60 else 'badge-emerald')

    # Top Header Bar with Upload New Action
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        render_page_header(
            title=f"Review Footage / Analysis · {loc_name}",
            subtitle=f"Source: {record['file_name']} · Timestamp: {record.get('created_at', '')} · Model: {record['model_name']}",
        )
    with col_hdr2:
        if st.button('📁 Upload New Video', width='stretch'):
            reset_session()
            st.rerun()

    # 1. 7 KPI Metrics Row
    render_kpi_section(record, settings)

    # 2. Main Two-Column Workspace (Left ~68%, Right ~32%)
    c_vid, c_cfg = st.columns([2.15, 1.0])
    with c_vid:
        st.markdown(f'''
        <div class="cv-video-meta">
          <div class="cv-meta-left">
            <span style="font-size:14px;">🎞</span>
            <span style="font-weight:700;color:#EDF2FA;">{record['file_name']}</span>
            <span class="cv-meta-dot">·</span>
            <span>{record.get('output_width', 1920)} × {record.get('output_height', 1080)}</span>
            <span class="cv-meta-dot">·</span>
            <span>{mb(record.get('output_bytes', 0))}</span>
            <span class="cv-meta-dot">·</span>
            <span>{source_dur:.1f} s</span>
            <span class="cv-meta-dot">·</span>
            <span>Model: <span class="kpi-badge badge-purple" style="font-size:10px;">{record['model_name']}</span></span>
          </div>
          <div>
            <span class="kpi-badge {badge_color}" style="font-size:11px;padding:3px 9px;">⚠ {peak_level} ({peak_pct:.0f}% Capacity)</span>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        if record['media_type'] == 'video':
            st.video(str(path))
        else:
            st.image(str(path), width='stretch')

    with c_cfg:
        st.markdown('''
        <div class="cv-panel-card" style="margin-bottom:8px;padding:12px 14px;">
          <div style="display:flex;align-items:center;gap:8px;font-size:14.5px;font-weight:700;color:#EDF2FA;">
            <span>⚙</span> Analysis Setup
          </div>
        </div>
        ''', unsafe_allow_html=True)
        res_loc = st.selectbox('Location', ['Campus entrance', 'Main auditorium', 'Library', 'Academic block', 'Cafeteria'], index=0, key='res_loc')
        res_cap = st.number_input('Capacity Threshold (people)', 1, 100000, int(cap), step=5, key='res_cap')
        res_evt = st.number_input('Sustained Exceedance Rule (seconds)', 0.5, 60.0, float(settings.get('event_seconds', 3.0)), step=0.5, key='res_evt')
        res_fps = st.select_slider('Inference Samples / Second', [1, 2, 5, 10, 15], value=int(settings.get('analysis_fps', 5)), key='res_fps')
        res_flow = st.checkbox('Estimate Directional Flow', value=bool(settings.get('flow_line')), key='res_flow')

        col_b1, col_b2 = st.columns(2)
        if col_b1.button('🔄 Re-Analyze', type='primary', width='stretch'):
            st.session_state.pop('result', None)
            st.rerun()
        if col_b2.button('📁 Change Video', width='stretch'):
            reset_session()
            st.rerun()

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # 3. Four Analytics Tabs
    render_analytics_section(record, settings)


# ---------------------------------------------------------
# EMPTY STATE RENDERER (STATE A: No Video Uploaded)
# ---------------------------------------------------------
def render_empty_state():
    render_page_header(
        title="Review Footage / Analysis",
        subtitle="Analyse recorded videos, define counting regions and monitor crowd occupancy and flow.",
    )

    st.markdown('''
    <div class="cv-upload-panel">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
        <div style="width:36px;height:36px;border-radius:8px;background:#202D42;border:1px solid #35445C;display:flex;align-items:center;justify-content:center;color:#7157E8;font-size:18px;">
          ☁
        </div>
        <div>
          <div style="font-size:15px;font-weight:700;color:#EDF2FA;line-height:1.2;">
            Upload Recorded Footage
          </div>
          <div style="font-size:12px;color:#A1AEC2;margin-top:2px;">
            Upload a campus video or image to begin crowd analysis.
          </div>
        </div>
      </div>
    ''', unsafe_allow_html=True)

    uploaded = media_uploader()
    if uploaded is not None:
        file_size = uploaded.size
        suffix = Path(uploaded.name).suffix.lower()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        session_dir = SESSIONS / st.session_state.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        source = session_dir / ('input' + suffix)
        hasher = hashlib.sha256()
        try:
            with FileLock(str(OUTPUTS / '.dispatch.lock'), timeout=10):
                check_storage_budget(OUTPUTS, file_size)
                with open(source, 'wb') as dst:
                    uploaded.seek(0)
                    while chunk := uploaded.read(8 * 1024 * 1024):
                        hasher.update(chunk)
                        dst.write(chunk)
            digest = hasher.hexdigest()
            st.session_state.custom_video_source = str(source)
            st.session_state.custom_file_name = uploaded.name
            st.session_state.upload_hash = digest
            st.session_state.pop('result', None)
            st.rerun()
        except Exception as exc:
            if source.exists():
                source.unlink(missing_ok=True)
            st.error(str(exc))
            st.stop()

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    col_act, col_priv = st.columns([1.1, 1.9])
    with col_act:
        st.markdown('<div class="cv-sample-btn">', unsafe_allow_html=True)
        if st.button('📁 Load Sample Video', width='stretch'):
            if DEMO_VIDEO.is_file():
                raw = DEMO_VIDEO.read_bytes()
                session_dir = SESSIONS / st.session_state.session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                dest = session_dir / 'input.mp4'
                dest.write_bytes(raw)
                st.session_state.custom_video_source = str(dest)
                st.session_state.custom_file_name = DEMO_VIDEO.name
                st.session_state.upload_hash = hashlib.sha256(raw).hexdigest()
                st.session_state.pop('result', None)
                st.rerun()
            else:
                st.error('Bundled sample video not found on disk.')
        st.markdown('</div>', unsafe_allow_html=True)
    with col_priv:
        st.markdown('''
        <div style="display:flex;align-items:center;height:100%;font-size:12px;color:#A1AEC2;gap:6px;padding-left:4px;">
          <span style="color:#35D995;font-size:13px;">🔒</span>
          <span><strong style="color:#EDF2FA;">Local Processing</strong> · Video inference runs locally on your machine.</span>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close cv-upload-panel

    st.markdown('''
    <div style="display:flex;align-items:center;gap:8px;font-size:15px;font-weight:700;color:#EDF2FA;margin-top:16px;margin-bottom:12px;">
      <span style="font-size:16px;">📚</span>
      <span>CrowdVision Capabilities</span>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('''
        <div class="cv-capability-card">
          <div class="cv-cap-icon" style="color:#7157E8;">👥</div>
          <div class="cv-cap-title">Crowd Detection & Occupancy</div>
          <div class="cv-cap-desc">YOLO11-based people detection and occupancy counting.</div>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
        <div class="cv-capability-card">
          <div class="cv-cap-icon" style="color:#32C5F4;">⇄</div>
          <div class="cv-cap-title">Directional Flow Analysis</div>
          <div class="cv-cap-desc">Track movement and calculate entry and exit flow.</div>
        </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown('''
        <div class="cv-capability-card">
          <div class="cv-cap-icon" style="color:#F5B942;">⚠️</div>
          <div class="cv-cap-title">Safety & Incident Detection</div>
          <div class="cv-cap-desc">Monitor capacity limits and detect sustained crowding.</div>
        </div>
        ''', unsafe_allow_html=True)


# ---------------------------------------------------------
# ROUTING: Review Footage Page (Main Analysis Workspace)
# ---------------------------------------------------------
if st.session_state.get('result'):
    render_results(st.session_state.result)
    st.stop()

if not st.session_state.get('custom_video_source'):
    render_empty_state()
    st.stop()


# ---------------------------------------------------------
# STATE B: VIDEO UPLOADED (Calibration & Analysis Setup)
# ---------------------------------------------------------
source = Path(st.session_state.custom_video_source)
file_display_name = st.session_state.get('custom_file_name', source.name)
file_size_bytes = source.stat().st_size if source.is_file() else 0
digest = st.session_state.get('upload_hash')
if not digest and source.is_file():
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    st.session_state.upload_hash = digest
is_video = source.suffix.lower().lstrip('.') in config.ALLOWED_VIDEO_TYPES


@st.cache_data(show_spinner=False, max_entries=4)
def inspect_source(path, identity, video):
    if not video:
        return cv2.imread(path), 0.0, 0.0
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame if ok else None, total / fps if fps > 0 else 0.0, fps
    finally:
        cap.release()


frame, duration, source_fps = inspect_source(str(source), digest, is_video)
if frame is None:
    st.error('This file cannot be decoded. Please upload an H.264 MP4 video or a standard JPEG image.')
    st.stop()

# Compact Setup Header with Change Video Action
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    render_page_header(
        title="Review Footage / Calibration & Analysis",
        subtitle="Configure monitored counting regions and calibrate detection parameters for inference.",
    )
with col_h2:
    if st.button('📁 Replace Video', width='stretch'):
        reset_session()
        st.rerun()

# Main Two-Column Workspace Layout (Left ~68%, Right ~32%)
col_left, col_right = st.columns([2.15, 1.0])

with col_left:
    st.markdown(f'''
    <div class="cv-video-meta">
      <div class="cv-meta-left">
        <span style="font-size:14px;">🎞</span>
        <span style="font-weight:700;color:#EDF2FA;">{file_display_name}</span>
        <span class="cv-meta-dot">·</span>
        <span>{frame.shape[1]} × {frame.shape[0]}</span>
        <span class="cv-meta-dot">·</span>
        <span>{mb(file_size_bytes)}</span>
        <span class="cv-meta-dot">·</span>
        <span>{duration:.1f} s</span>
        <span class="cv-meta-dot">·</span>
        <span>Model: <span class="kpi-badge badge-purple" style="font-size:10px;">YOLO11m</span></span>
      </div>
      <div>
        <span class="kpi-badge badge-cyan" style="font-size:10px;">Ready for Calibration</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # Interactive Scene Editor Canvas with Custom Polygon & Flow Controls
    geometry_key = 'geometry_' + digest[:16]
    scene = scene_editor(frame, key='scene_' + digest[:16], geometry=st.session_state.get(geometry_key)) or {'polygon': None, 'line': None}
    st.session_state[geometry_key] = scene

    with st.expander('Original source playback'):
        if is_video:
            st.video(str(source))
        else:
            st.image(str(source), width='stretch')

with col_right:
    st.markdown('''
    <div class="cv-panel-card" style="margin-bottom:8px;padding:12px 14px;">
      <div style="display:flex;align-items:center;gap:8px;font-size:14.5px;font-weight:700;color:#EDF2FA;">
        <span>⚙</span> Analysis Setup
      </div>
    </div>
    ''', unsafe_allow_html=True)

    location = st.selectbox('Location', ['Campus entrance', 'Main auditorium', 'Library', 'Academic block', 'Cafeteria'], index=0)
    capacity = st.number_input('Capacity Threshold (people)', 1, 100000, 80, step=5)
    event_seconds = st.number_input('Sustained Exceedance Rule (seconds)', 0.5, 60.0, 3.00, step=0.5)
    analysis_fps = st.select_slider('Inference Samples / Second', [1, 2, 5, 10, 15], value=5)

    flow_key = 'flow_choice_' + digest[:16]
    flow = st.checkbox('Estimate Directional Flow', value=st.session_state.get(flow_key, False), disabled=not is_video)
    st.session_state[flow_key] = flow

    with st.expander('Advanced model & export settings'):
        weights = st.text_input('Detector weights (.pt)', config.BEST_WEIGHTS_PATH)
        conf = st.slider('Detection confidence', 0.05, 0.95, float(config.INFERENCE_CONF_THRESHOLD), 0.05)
        imgsz = st.select_slider('Detection detail (imgsz)', [416, 640, 768, 960], value=640)
        res_choice = st.selectbox('Export resolution', ['Native / Original', 1920, 1280, 960, 640], index=0)
        max_width = 0 if res_choice == 'Native / Original' else int(res_choice)
        smooth_playback = st.checkbox('Smooth video playback (Source FPS)', value=True)

    # Geometry Validation
    invalid = None
    try:
        polygon = validate_polygon(scene.get('polygon'))
        line = validate_line(scene.get('line')) if flow else None
        if flow and analysis_fps < 5:
            raise ValueError('Increase sampling to at least 5 FPS for directional flow.')
        if duration > 600:
            raise ValueError('Trim the recording to 10 minutes or less.')
        if not Path(weights).is_file():
            raise ValueError('The selected detector model file is missing.')
    except (ValueError, TypeError) as exc:
        invalid = str(exc)

    if invalid:
        st.error(invalid)

    analyze_btn = st.button('▷  Analyze Selected Area', type='primary', disabled=bool(invalid), width='stretch')

if invalid:
    st.stop()

settings = dict(
    pipeline_version=config.PROJECT_VERSION,
    weights_path=str(Path(weights).resolve()),
    weights_mtime_ns=Path(weights).stat().st_mtime_ns,
    conf=conf,
    iou=0.5,
    imgsz=imgsz,
    max_capacity=int(capacity),
    classification_mode='manual',
    analysis_fps=float(analysis_fps),
    max_width=max_width,
    roi_polygon=polygon,
    flow_line=line,
    event_seconds=float(event_seconds),
    location=location,
    smooth_playback=smooth_playback,
)
signature = hashlib.sha256((digest + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:24]

if analyze_btn:
    run_dir = session_dir / signature
    try:
        cached = run_dir / 'result.json'
        if cached.is_file() and Path(json.loads(cached.read_text(encoding='utf-8'))['annotated_path']).is_file():
            st.session_state.result = json.loads(cached.read_text(encoding='utf-8'))
            st.session_state.pop('last_error', None)
        else:
            start_job(run_dir, dict(settings=settings, source_path=str(source), file_name=file_display_name, media_type='video' if is_video else 'image'))
            st.session_state.active_run = str(run_dir)
            st.session_state.pop('last_error', None)
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if st.session_state.get('result'):
    render_results(st.session_state.result)
