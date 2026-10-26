"""CrowdVision: compact local crowd analysis with immutable run settings."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid
import cv2
import pandas as pd
import streamlit as st
import config
from src.services.detection_service import get_model, run_image_detection, run_video_detection
from src.services.report_generator import generate_pdf_report
from src.services.exports import csv_bytes
from src.services.storage import atomic_json, storage_bytes, clear_session, retain_run, LEGACY_FOLDERS, clear_legacy_exports
from src.utils.helpers import get_crowd_level

st.set_page_config(page_title='CrowdVision | Video analytics', page_icon='◉', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0c111b;color:#e8edf6}
.block-container{max-width:1380px;padding-top:3.5rem;padding-bottom:3rem}
h1{font-size:2.4rem!important;letter-spacing:-.07rem;font-weight:700!important}
h2,h3{letter-spacing:-.025rem}
[data-testid="stMetric"]{background:#151e2d;border:1px solid #26344b;border-radius:12px;padding:18px}
[data-testid="stMetricLabel"]{color:#abb8cc}
[data-testid="stMetricValue"]{font-size:1.8rem}
[data-testid="stFileUploader"]{border:1px dashed #40516e;border-radius:12px;padding:12px;background:#111a28}
[data-testid="stSidebar"]{background:#111926}
.stButton>button[kind="primary"]{background:#b7e27c;color:#15200d;border:0;font-weight:700}
.eyebrow{color:#b7e27c;font-size:.75rem;letter-spacing:.15rem;font-weight:700;margin-bottom:8px}
.hero-copy{color:#9eafc8;max-width:750px;line-height:1.6;margin-bottom:24px}
.privacy{color:#889ab6;font-size:.8rem;padding-top:8px}
@media(max-width:700px){.block-container{padding:3.5rem 1rem 1rem}h1{font-size:1.9rem!important}}
</style>''', unsafe_allow_html=True)

ROOT = Path(config.PROJECT_ROOT)
SESSIONS = Path(config.OUTPUTS_ROOT) / 'sessions'
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
session_dir = SESSIONS / st.session_state.session_id


def mb(size):
    return f'{size / 1024**2:.2f} MB'


with st.sidebar:
    st.markdown('### Analysis settings')
    weights = st.text_input('Model checkpoint', value=config.BEST_WEIGHTS_PATH,
                           help='Use the supplied single-class head model. Other checkpoints must contain head/person classes.')
    conf = st.slider('Minimum confidence', 0.05, 0.95, 0.35, 0.05)
    imgsz = st.select_slider('Model input size', options=[320, 416, 512, 640, 768], value=640)
    mode_label = st.selectbox('Crowd classification', ['Occupancy against capacity', 'Image coverage (experimental)'])
    classification_mode = 'manual' if mode_label.startswith('Occupancy') else 'auto'
    capacity = st.number_input('Configured space capacity', min_value=1, max_value=100000, value=80,
                               help='Set this for your monitored area. Low: up to 60%; moderate: up to 85%; high: above 85%. These are configurable occupancy rules, not safety limits.')
    analysis_fps = st.select_slider('Video samples per second', options=[1, 2, 5, 10, 15], value=5,
                                     help='Lower values save processing time and output size. Events between samples can be missed.')
    max_width = st.selectbox('Maximum output width', options=[640, 960, 1280], index=1)
    st.caption('H.264 export · source-based bitrate · silent video. Preview is sampled at the selected rate.')
    st.divider()
    st.markdown('### Local storage')
    st.caption('This session: ' + mb(storage_bytes(session_dir)))
    if st.button('Clear this session', use_container_width=True):
        clear_session(SESSIONS, st.session_state.session_id)
        st.session_state.pop('result', None)
        st.session_state.pop('upload_hash', None)
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()
    with st.expander('Existing files on this computer'):
        st.write('All generated output: ' + mb(storage_bytes(Path(config.OUTPUTS_ROOT))))
        st.caption('Earlier generated exports are preserved until you remove them. Download anything you need first; deletion is permanent.')
        folders = st.multiselect('Old generated folders to clear', options=LEGACY_FOLDERS,
                                 format_func=lambda name: name + ' · ' + mb(storage_bytes(Path(config.OUTPUTS_ROOT)/name)))
        confirm_cleanup = st.checkbox('I have saved the selected exports and want to delete them')
        if st.button('Delete selected old exports', disabled=not (folders and confirm_cleanup), use_container_width=True):
            clear_legacy_exports(config.OUTPUTS_ROOT, folders)
            st.rerun()

st.markdown('<div class="eyebrow">CROWDVISION / LOCAL ANALYTICS</div>', unsafe_allow_html=True)
st.title('Understand the crowd. Keep the detail.')
st.markdown('<div class="hero-copy">Count visible heads in images and recorded video. Review occupancy, inspect changes over time, and export a compact annotated result.</div>', unsafe_allow_html=True)
if classification_mode == 'auto':
    st.info('Image coverage is an experimental visual heuristic. Zoom and perspective affect it; it does not measure people per square metre or predict safety.')

uploaded = st.file_uploader('Choose an image or video', type=config.ALLOWED_IMAGE_TYPES + config.ALLOWED_VIDEO_TYPES,
                             help='Up to 200 MB. Processing stays on this computer.')
if uploaded is None:
    st.markdown('### One recording, a complete review')
    a,b,c = st.columns(3)
    a.markdown('**01 / Add media**\n\nUpload an image or recorded clip, then choose the capacity and processing detail.')
    b.markdown('**02 / Inspect counts**\n\nReview sampled detections, peak occupancy and the source-video timeline.')
    c.markdown('**03 / Export compactly**\n\nDownload H.264 video, frame data and a report. Repeated runs reuse identical results.')
    st.caption('A local prototype for video analytics. Counts require validation for your camera and scene. Live RTSP and unique-person counting are not enabled.')
    st.stop()

raw = uploaded.getbuffer()
upload_hash = hashlib.sha256(raw).hexdigest()
ext = Path(uploaded.name).suffix.lower()
is_video = ext.lstrip('.') in config.ALLOWED_VIDEO_TYPES
settings = dict(pipeline_version=config.PROJECT_VERSION, weights_path=str(Path(weights).resolve()), conf=conf, iou=0.5, imgsz=imgsz,
                max_capacity=int(capacity), classification_mode=classification_mode,
                analysis_fps=float(analysis_fps), max_width=max_width)
weight_file = Path(weights)
if not weight_file.is_file():
    st.error('Model checkpoint was not found. Choose a local .pt checkpoint in Analysis settings.')
    st.stop()
settings['weights_mtime_ns'] = weight_file.stat().st_mtime_ns
signature = hashlib.sha256((upload_hash + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:24]
session_dir.mkdir(parents=True, exist_ok=True)
source = session_dir / ('input' + ext)
if st.session_state.get('upload_hash') != upload_hash:
    # Only replace inputs owned by this app session. Never touch the original media.
    for old_input in session_dir.glob('input.*'):
        old_input.unlink()
    source.write_bytes(raw)
    st.session_state.upload_hash = upload_hash
    st.session_state.pop('result', None)
elif not source.exists():
    source.write_bytes(raw)
del raw

left, right = st.columns([1.4, 1])
with left:
    st.subheader(Path(uploaded.name).name)
    st.caption(f'{mb(uploaded.size)} · {"Recorded video" if is_video else "Image"} · Output limited to {max_width}px wide for video')
with right:
    analyze = st.button('Analyze video' if is_video else 'Analyze image', type='primary', use_container_width=True)

current = st.session_state.get('result')
if current and current.get('signature') != signature:
    st.info('Settings changed. Analyze again to generate a result with these settings. The previous result below retains its original settings.')

if analyze:
    run_dir = session_dir / signature
    run_dir.mkdir(exist_ok=True)
    cached = run_dir / 'result.json'
    st.session_state.pop('last_error', None)
    try:
        if cached.exists():
            record = json.loads(cached.read_text(encoding='utf-8'))
            if not Path(record['annotated_path']).exists():
                raise RuntimeError('The cached output was removed. Clear this session and analyze again.')
        else:
            # Identical reruns use the cache; old run artifacts are pruned after success.
            if current and current.get('signature') != signature:
                st.session_state.pop('result', None)
            model = get_model(weights)
            model_classes = ', '.join(model.names.values())
            model_name = str(model.model.yaml.get('yaml_file', 'YOLO'))
            progress = st.progress(0.0, text='Preparing media…')
            preview = st.empty()
            last_preview = [0.0]
            def on_frame(index, total, frame, count, level, auto):
                now = time.perf_counter()
                if now - last_preview[0] >= 0.5:
                    last_preview[0] = now
                    progress.progress(min(index / total, 1.0) if total else 0.0,
                                      text=f'Frame {index:,} / {total:,} · {count} visible heads')
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            started = time.perf_counter()
            with st.spinner('Analyzing on CPU. Long clips may take several minutes…'):
                common = dict(weights_path=weights, source_path=str(source), conf=conf, iou=0.5,
                              imgsz=imgsz, classification_mode=classification_mode, max_capacity=int(capacity))
                if is_video:
                    output = run_dir / 'annotated.mp4'
                    result = run_video_detection(**common, output_path=str(output), frame_callback=on_frame,
                                                 analysis_fps=float(analysis_fps), max_width=max_width)
                    data = asdict(result)
                    levels = result.per_frame_auto_crowd_levels if classification_mode == 'auto' else [get_crowd_level(n, int(capacity)) for n in result.per_frame_counts]
                    data['per_frame_levels'] = levels
                else:
                    output = run_dir / 'annotated.jpg'
                    annotated, count, manual, auto, _ = run_image_detection(**common)
                    if not cv2.imwrite(str(output), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        raise IOError('Cannot save annotated image. Check available disk space.')
                    level = auto if classification_mode == 'auto' else manual
                    data = dict(max_head_count=count, avg_head_count=float(count), final_head_count=count,
                                final_crowd_level=level, peak_crowd_level=level, per_frame_counts=[count],
                                per_frame_levels=[level], timestamps=[0.0], processed_frames=1,
                                processing_time_seconds=time.perf_counter()-started, input_bytes=uploaded.size,
                                output_bytes=output.stat().st_size, source_fps=None, sample_fps=None)
                record = dict(data, signature=signature, id=signature, file_name=Path(uploaded.name).name,
                              media_type='video' if is_video else 'image', annotated_path=str(output),
                              settings=settings, model_name=model_name, model_classes=model_classes,
                              crowd_level=data['peak_crowd_level'])
                atomic_json(cached, record)
            progress.empty()
            preview.empty()
        retain_run(session_dir, signature)
        st.session_state.result = record
        st.rerun()
    except Exception as exc:
        st.session_state.pop('result', None)
        st.error(str(exc))
        st.caption('Analysis failed. No crowd count or occupancy status is reported for this run.')

record = st.session_state.get('result')
if record:
    st.divider()
    st.subheader('Analysis results')
    for warning in record.get('warnings', []):
        st.warning(warning)
    a,b,c,d = st.columns(4)
    a.metric('Peak visible count', record['max_head_count'])
    b.metric('Average visible count', f"{record['avg_head_count']:.1f}")
    c.metric('Peak crowd level', record['peak_crowd_level'].replace(' Crowd',''))
    d.metric('Annotated file', mb(record['output_bytes']))
    preview_tab, analytics_tab, export_tab = st.tabs(['Annotated preview', 'Timeline & details', 'Downloads'])
    with preview_tab:
        if record['media_type'] == 'video':
            st.video(record['annotated_path'])
            st.caption(f"{record['output_width']} × {record['output_height']} · {record['sample_fps']:.2f} sampled FPS · H.264 · audio omitted")
        else:
            st.image(record['annotated_path'], use_container_width=True)
    with analytics_tab:
        if record['media_type'] == 'video':
            chart = pd.DataFrame({'Source time (s)': record['timestamps'], 'Visible count': record['per_frame_counts']})
            st.line_chart(chart.set_index('Source time (s)'))
            st.caption(f"{record['processed_frames']:,} sampled frames · {record['source_duration']:.2f}s of source video · {record['fps_processed']:.2f} samples processed / wall-clock second")
        st.write('Final crowd level: ' + record['final_crowd_level'])
        st.write('Actual model: ' + record['model_name'] + ' · classes: ' + record['model_classes'])
        st.write('Processing time: ' + f"{record['processing_time_seconds']:.2f}s")
        st.write('Input ' + mb(record['input_bytes']) + ' → output ' + mb(record['output_bytes']))
        with st.expander('Settings used for this result'):
            st.json({k:v for k,v in record['settings'].items() if k not in ('weights_path','weights_mtime_ns')})
        st.caption('Peak and final status are different summaries. Counts are sampled detections, not unique people. Occupancy levels are configured rules, not validated safety predictions.')
    with export_tab:
        path = Path(record['annotated_path'])
        st.download_button('Download annotated video' if record['media_type'] == 'video' else 'Download annotated image',
                           data=path.read_bytes(), file_name='crowdvision' + path.suffix,
                           mime='video/mp4' if record['media_type'] == 'video' else 'image/jpeg', use_container_width=True)
        st.download_button('Download frame data (CSV)', csv_bytes(record), file_name='crowdvision-counts.csv', mime='text/csv', use_container_width=True)
        if st.button('Prepare PDF report', use_container_width=True):
            try:
                report_path = path.parent / 'report.pdf'
                generate_pdf_report(record, str(report_path))
                st.session_state.pdf_for = record['signature']
            except Exception as exc:
                st.error('Report could not be generated: ' + str(exc))
        if st.session_state.get('pdf_for') == record['signature'] and (path.parent / 'report.pdf').exists():
            st.download_button('Download PDF report', (path.parent/'report.pdf').read_bytes(), file_name='crowdvision-report.pdf', mime='application/pdf', use_container_width=True)
    st.markdown('<div class="privacy">Local processing. Your source file is never modified. Clear session storage after downloading your results.</div>', unsafe_allow_html=True)
