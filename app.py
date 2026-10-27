"""CrowdVision Campus: recorded-footage occupancy and flow review."""
from pathlib import Path
import hashlib
import json
import uuid
from datetime import datetime
import cv2
import pandas as pd
import streamlit as st
from filelock import FileLock
import config
from src.core.scene import validate_polygon, validate_line
from src.services.storage import storage_bytes, clear_session, session_inventory, check_storage_budget
from src.services.jobs import start_job, read_status, cancel_job, ACTIVE_STATES
from src.services.exports import csv_bytes
from src.services.report_generator import generate_pdf_report
from src.ui.scene_editor import scene_editor

st.set_page_config(page_title='CrowdVision Campus',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp{background:#0d1420;color:#e5ecf5}.block-container{max-width:1500px;padding:2.8rem 2rem 3rem}
h1{font-size:1.85rem!important;letter-spacing:-.04rem}h2{font-size:1.4rem!important}h3{font-size:1.05rem!important}
[data-testid="stSidebar"]{background:#111c2b;border-right:1px solid #243247}
[data-testid="stMetric"]{background:#152234;border:1px solid #293b51;border-radius:8px;padding:14px}
[data-testid="stMetricLabel"]{color:#a7b7ca}[data-testid="stMetricValue"]{font-size:1.65rem}
[data-testid="stFileUploader"]{border:1px dashed #3e596e;border-radius:8px;padding:8px}
.eyebrow{font:600 11px system-ui;letter-spacing:2px;color:#6dc6df;margin-bottom:6px}
.purpose{color:#a8b9cd;line-height:1.6;font-size:14px;max-width:850px}
.stButton>button[kind="primary"]{background:#6dc6df;color:#0b2028;border:0;font-weight:600}
@media(max-width:700px){.block-container{padding:3rem 1rem 1rem}h1{font-size:1.5rem!important}}
</style>''',unsafe_allow_html=True)
ROOT=Path(config.PROJECT_ROOT); OUTPUTS=Path(config.OUTPUTS_ROOT); SESSIONS=OUTPUTS/'sessions'
for key,value in [('session_id',uuid.uuid4().hex),('upload_generation',0)]:
    if key not in st.session_state: st.session_state[key]=value
session_dir=SESSIONS/st.session_state.session_id

def mb(n): return f'{n/1024**2:.1f} MB'

def reset_session():
    clear_session(SESSIONS,st.session_state.session_id)
    for key in ('result','upload_hash','last_error','active_run','pdf_for'): st.session_state.pop(key,None)
    st.session_state.session_id=uuid.uuid4().hex
    st.session_state.upload_generation+=1


def media_uploader(disabled=False):
    return st.file_uploader('Add recorded footage or a still image',type=config.ALLOWED_IMAGE_TYPES+config.ALLOWED_VIDEO_TYPES,key=f'media_{st.session_state.upload_generation}',disabled=disabled,help='Up to 200 MB and 10 minutes. Your original file is not changed.')

with st.sidebar:
    st.markdown('## ◈ CrowdVision')
    st.caption('CAMPUS OCCUPANCY & FLOW')
    page=st.radio('Workspace',['Review footage','Saved sessions','Model lab'],label_visibility='collapsed')
    st.divider()
    location=st.text_input('Monitored location',value='Campus entrance',max_chars=100)
    capacity=st.number_input('Capacity of selected area',1,100000,80,help='Set capacity for the area you draw, not the whole venue.')
    event_seconds=st.number_input('Flag when over capacity for (seconds)',1.,60.,3.,1.)
    analysis_fps=st.select_slider('Analysis samples / second',[1,2,5,10,15],value=5)
    with st.expander('Model & export settings'):
        weights=st.text_input('Trusted local model',config.BEST_WEIGHTS_PATH)
        conf=st.slider('Detection confidence',.05,.95,float(config.INFERENCE_CONF_THRESHOLD),.05)
        imgsz=st.select_slider('Detection detail',[416,640,768,960],value=640)
        max_width=st.selectbox('Export width',[640,960,1280],index=1)
        st.caption('Higher detail can help small heads but costs processing time. Export width does not change detector input.')
    st.divider()
    st.caption('Recorded footage · local CPU processing')
    st.caption('10-minute clips · 2 GB workspace limit')
    if st.button('Clear this session',disabled=bool(st.session_state.get('active_run')),width='stretch'):
        reset_session(); st.rerun()

st.markdown('<div class="eyebrow">CAMPUS OPERATIONS / RECORDED FOOTAGE</div>',unsafe_allow_html=True)
st.title('Crowd occupancy & flow review')
st.markdown('<div class="purpose">Locate sustained crowd buildup at entrances and event spaces. Define the monitored area, inspect capacity exceedances, and review movement across a chosen line.</div>',unsafe_allow_html=True)


def render_results(record):
    path=Path(record['annotated_path'])
    if not path.is_file():
        st.warning('Saved output is missing. Upload the source and run analysis again.'); return
    settings=record['settings']; events=record.get('events',[])
    st.subheader(settings.get('location','Recorded area')+' · Analysis results')
    created=record.get('created_at')
    try: created=datetime.fromisoformat(created).astimezone().strftime('%d %b %Y, %I:%M %p') if created else 'Earlier analysis'
    except ValueError: created='Earlier analysis'
    st.caption(record['file_name']+' · '+created)
    for warning in record.get('warnings',[]): st.warning(warning)
    a,b,c,d=st.columns(4)
    a.metric('Peak in selected area',record['max_head_count'])
    b.metric('Peak / capacity',f"{record['max_head_count']/settings['max_capacity']*100:.0f}%")
    c.metric('Sustained exceedances',len(events) if record['media_type']=='video' else '—')
    d.metric('Analyzed duration',f"{record.get('source_duration',0):.1f}s" if record['media_type']=='video' else 'Image')
    panel=st.radio('Results view',['Footage & events','Timeline & flow','Export'],horizontal=True,key='review_'+record['signature'])
    if panel=='Footage & events':
        footage,incidents=st.columns([2.2,1])
        with incidents:
            st.markdown('### Capacity events')
            labels=['Start of recording']+[f"{e['start_seconds']:.1f}s–{e['end_seconds']:.1f}s · peak {e['peak_count']}" for e in events]
            selected=st.selectbox('Jump to event',range(len(labels)),format_func=lambda i:labels[i],key='event_'+record['signature'])
            seek=events[selected-1]['peak_seconds'] if selected else 0
            if not events: st.caption('No sustained sampled exceedance met the configured duration. This is not a safety clearance.')
            else:
                event=events[selected-1] if selected else max(events,key=lambda e:e['peak_count'])
                st.write(f"**Peak:** {event['peak_count']} / {settings['max_capacity']}")
                st.write(f"**Observed duration:** {event['duration_seconds']:.1f}s")
                if event.get('thumbnail') and Path(event['thumbnail']).is_file(): st.image(event['thumbnail'],width='stretch')
                st.caption('Review this interval for queue buildup or an obstructed passage. Confirm detections before acting.')
            st.caption(f"Rule: more than {settings['max_capacity']} heads for at least {settings.get('event_seconds',3):g}s of consecutive sampled observations.")
        with footage:
            if record['media_type']=='video': st.video(str(path),start_time=float(seek))
            else: st.image(str(path),width='stretch')
            st.caption('Cyan: counting area · orange: optional flow line · IDs: temporary tracking estimates')
    elif panel=='Timeline & flow':
        chart=pd.DataFrame({'Source time (s)':record['timestamps'],'Heads in area':record['per_frame_counts']})
        chart['Capacity']=settings['max_capacity']
        st.line_chart(chart.set_index('Source time (s)'),color=['#6dc6df','#ffb568'])
        peak=record['per_frame_counts'].index(record['max_head_count'])
        st.caption(f"First sampled peak: {record['timestamps'][peak]:.2f}s · Average sampled count: {record['avg_head_count']:.1f} · {record['processed_frames']} analyzed samples")
        if settings.get('flow_line'):
            a,b=st.columns(2)
            a.metric('Along arrow',record.get('forward_crossings',0)); b.metric('Against arrow',record.get('reverse_crossings',0))
            st.warning('Experimental flow estimates. Occlusion, low sampling rates and identity switches can miss or duplicate crossings. These are not unique visitors.')
            if record.get('crossings'): st.dataframe(pd.DataFrame(record['crossings']),hide_index=True,width='stretch')
        else: st.caption('Draw a line and enable flow before analysis to estimate directional crossings.')
        if events: st.dataframe(pd.DataFrame([{k:v for k,v in e.items() if k!='thumbnail'} for e in events]),hide_index=True,width='stretch')
    else:
        st.caption(f"Source {mb(record['input_bytes'])} → annotated output {mb(record['output_bytes'])}. Video is sampled H.264 without audio.")
        st.download_button('Annotated footage',path.read_bytes(),file_name=Path(record['file_name']).stem+'-annotated'+path.suffix,mime='video/mp4' if record['media_type']=='video' else 'image/jpeg',width='stretch')
        st.download_button('Timestamped counts · CSV',csv_bytes(record),file_name='area-counts.csv',mime='text/csv',width='stretch')
        event_csv=pd.DataFrame([{k:v for k,v in e.items() if k!='thumbnail'} for e in events],columns=['start_seconds','end_seconds','duration_seconds','peak_count','peak_seconds','capacity']).to_csv(index=False)
        st.download_button('Capacity events · CSV',event_csv,file_name='capacity-events.csv',mime='text/csv',width='stretch')
        if st.button('Prepare analysis report',key='pdf_'+record['signature']):
            try: generate_pdf_report(record,str(path.parent/'report.pdf'))
            except Exception as exc: st.error('Report failed: '+str(exc))
        if (path.parent/'report.pdf').is_file(): st.download_button('Download PDF report',(path.parent/'report.pdf').read_bytes(),file_name='campus-occupancy-report.pdf',mime='application/pdf')
    with st.expander('Run details & limitations'):
        st.write('Detector: '+record['model_name']+' · '+record['model_classes'])
        st.write(f"Processing time: {record['processing_time_seconds']:.1f}s")
        st.json({k:v for k,v in settings.items() if k not in ('weights_path','weights_mtime_ns')})
        st.caption('Counts represent detected head centres in the selected area. Missed heads, false detections and events between samples remain possible. Capacity is a user-defined operating threshold, not a validated safety limit.')


@st.fragment(run_every=2)
def job_progress(run_dir):
    status=read_status(run_dir); state=status.get('state','queued')
    if state=='complete':
        st.session_state.result=json.loads((Path(run_dir)/'result.json').read_text(encoding='utf-8'))
        st.session_state.pop('active_run',None); st.session_state.pop('last_error',None); st.rerun()
    if state in ('failed','cancelled'):
        st.session_state.last_error=status.get('error',state)
        st.session_state.pop('active_run',None); st.rerun()
    st.subheader('Analyzing selected area')
    st.progress(float(status.get('progress',0)),text=status.get('message','Starting worker'))
    st.caption('You can leave this page and recover the job from Saved sessions. Only one analysis runs at a time.')
    if (Path(run_dir)/'preview.jpg').exists(): st.image(str(Path(run_dir)/'preview.jpg'),width=640)
    if st.button('Cancel analysis',key='cancel_job'):
        cancel_job(run_dir); st.info('Cancellation requested. The current inference call will finish before the worker stops.')

if st.session_state.get('active_run'):
    # Keep the uploader mounted while polling: otherwise Streamlit removes its
    # widget state and the selected source disappears after the job finishes.
    media_uploader(disabled=True)
    job_progress(st.session_state.active_run); st.stop()
if st.session_state.get('last_error'):
    st.error(st.session_state.last_error); st.caption('The last successful result, if shown, is unchanged.')

if page=='Model lab':
    st.subheader('Model comparison')
    evaluation=ROOT/'docs'/'model_evaluation.json'
    if evaluation.is_file():
        data=json.loads(evaluation.read_text(encoding='utf-8')); st.info(data.get('decision','Comparison available'))
        if 'baseline' in data and 'candidate' in data:
            labels={'mae':'Average count error (heads)','rmse':'Count RMSE (heads)','bias':'Count bias (heads)','precision':'Detection precision','recall':'Detection recall'}
            metrics=pd.DataFrame({'Current model':data['baseline'],'Trained candidate':data['candidate']}).reindex(labels).rename(index=labels)
            st.dataframe(metrics.style.format('{:.3f}'),width='stretch')
            st.caption(f"Compared on {int(data['baseline']['images'])} test images. Lower count error is better. Precision and recall use IoU 0.5; these are not a campus accuracy percentage.")
        if data.get('completed_epochs'): st.caption(f"{data['hardware']} · {data['completed_epochs']} epochs · {data['split_images']['train']} training images · {data['split_images']['tune']} validation images")
        st.write(data.get('limitation',''))
        with st.expander('Experiment details'): st.json(data)
    else: st.info('No completed local model comparison is available yet. No accuracy percentage is claimed.')
    st.markdown('### Training data integrity')
    st.warning('The original local dataset has corrupted repeated boxes and is blocked from training. Use the new audited SCUT-HEAD conversion instead.')
    st.markdown('[SCUT-HEAD research dataset](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) · Academic research use only.')
    st.caption('A benchmark improvement does not establish accuracy at your campus. Validate representative camera footage before operational use.')
    st.stop()

if page=='Saved sessions':
    st.subheader('Saved sessions & storage')
    used=storage_bytes(OUTPUTS); st.progress(min(used/(2*1024**3),1.),text=f'{mb(used)} of 2 GB workspace budget')
    inventory=session_inventory(SESSIONS)
    for item in inventory:
        directory=SESSIONS/item['id']
        with st.expander(f"Session {item['id'][:8]} · {mb(item['bytes'])} · {len(item['runs'])} saved runs"):
            active=False
            for p in directory.glob('*/status.json'):
                status=read_status(p.parent)
                if status.get('state') in ACTIVE_STATES:
                    active=True; st.write(status.get('message','Running analysis'))
                    if st.button('Open job progress',key='job_'+str(p.parent)):
                        st.session_state.active_run=str(p.parent); st.rerun()
            for record in item['runs']:
                if st.button(record['file_name']+' · '+record.get('created_at',''),key='open_'+item['id']+record['signature']): st.session_state.saved_review=record
            confirm=st.checkbox('I have downloaded the outputs I need',key='confirm_'+item['id'],disabled=active)
            if st.button('Delete saved session',key='delete_'+item['id'],disabled=active or not confirm):
                with FileLock(str(OUTPUTS/'.dispatch.lock'),timeout=5):
                    if any(read_status(p.parent).get('state') in ACTIVE_STATES for p in directory.glob('*/status.json')): st.error('This session has an active job.')
                    else:
                        if item['id']==st.session_state.session_id: reset_session()
                        else: clear_session(SESSIONS,item['id'])
                        st.session_state.pop('saved_review',None); st.rerun()
    if not inventory: st.caption('Completed and interrupted analyses will appear here.')
    if st.session_state.get('saved_review'): render_results(st.session_state.saved_review)
    st.stop()

uploaded=media_uploader()
if uploaded is None:
    if st.session_state.get('result'): render_results(st.session_state.result)
    else:
        with st.container(border=True):
            st.subheader('Review an entrance, corridor or event area')
            st.write('Add footage, draw the area you want to monitor, and set its capacity. The analysis flags sustained exceedances and lets you jump to the relevant moments.')
            st.caption('Select Saved sessions to reopen previous results. Source footage and analysis stay on this computer.')
    st.stop()

raw=uploaded.getbuffer(); digest=hashlib.sha256(raw).hexdigest(); suffix=Path(uploaded.name).suffix.lower()
is_video=suffix.lstrip('.') in config.ALLOWED_VIDEO_TYPES
source=session_dir/('input'+suffix)
if st.session_state.get('upload_hash')!=digest or not source.exists():
    try:
        OUTPUTS.mkdir(parents=True,exist_ok=True)
        with FileLock(str(OUTPUTS/'.dispatch.lock'),timeout=5):
            check_storage_budget(OUTPUTS,len(raw))
            if st.session_state.get('upload_hash') and st.session_state.upload_hash!=digest:
                st.session_state.session_id=uuid.uuid4().hex; session_dir=SESSIONS/st.session_state.session_id; source=session_dir/('input'+suffix)
            session_dir.mkdir(parents=True,exist_ok=True); source.write_bytes(raw)
        st.session_state.upload_hash=digest; st.session_state.pop('result',None)
    except Exception as exc: st.error(str(exc)); st.stop()
del raw

@st.cache_data(show_spinner=False,max_entries=4)
def inspect_source(path,identity,video):
    if not video: return cv2.imread(path),0.,0.
    cap=cv2.VideoCapture(path)
    try:
        ok,frame=cap.read(); fps=cap.get(cv2.CAP_PROP_FPS); total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame if ok else None,total/fps if fps>0 else 0.,fps
    finally: cap.release()

frame,duration,source_fps=inspect_source(str(source),digest,is_video)
if frame is None: st.error('This file cannot be decoded. Try an H.264 MP4 or a JPEG image.'); st.stop()
st.subheader(Path(uploaded.name).name)
st.caption(f"{frame.shape[1]} × {frame.shape[0]} · {mb(uploaded.size)}"+(f' · {duration:.1f}s source duration' if is_video else ' · Still image'))
left,right=st.columns([2.3,1])
with left:
    geometry_key='geometry_'+digest[:16]
    scene=scene_editor(frame,key='scene_'+digest[:16],geometry=st.session_state.get(geometry_key)) or {'polygon':None,'line':None}
    st.session_state[geometry_key]=scene
    with st.expander('Original source playback'):
        if is_video:
            st.video(str(source)); st.caption('If the browser cannot play this codec, use the locally decoded editor frame above.')
        else: st.image(str(source),width='stretch')
with right:
    st.markdown('### Analysis setup'); st.write('**Location:** '+location)
    st.write(f"**Capacity:** {capacity} {'person' if capacity==1 else 'people'} in selected area"); st.write(f'**Event rule:** exceed capacity for {event_seconds:g}s')
    flow_key='flow_choice_'+digest[:16]
    flow=st.checkbox('Estimate directional flow',value=st.session_state.get(flow_key,False),disabled=not is_video)
    st.session_state[flow_key]=flow
    st.caption('Draw an orange line first. The arrow indicates positive crossing direction. At least 5 samples/second required.')
    invalid=None
    try:
        polygon=validate_polygon(scene.get('polygon')); line=validate_line(scene.get('line')) if flow else None
        if flow and analysis_fps<5: raise ValueError('Increase sampling to at least 5 FPS for flow.')
        if duration>600: raise ValueError('Trim the recording to 10 minutes or less.')
        if not Path(weights).is_file(): raise ValueError('The selected model file is missing.')
    except (ValueError,TypeError) as exc: invalid=str(exc)
    if invalid: st.error(invalid)
    analyze=st.button('Analyze selected area',type='primary',disabled=bool(invalid),width='stretch')
    st.caption('Head centres inside the cyan area are counted. Draw and apply the area before starting.')
if invalid: st.stop()
settings=dict(pipeline_version=config.PROJECT_VERSION,weights_path=str(Path(weights).resolve()),weights_mtime_ns=Path(weights).stat().st_mtime_ns,conf=conf,iou=.5,imgsz=imgsz,max_capacity=int(capacity),classification_mode='manual',analysis_fps=float(analysis_fps),max_width=max_width,roi_polygon=polygon,flow_line=line,event_seconds=float(event_seconds),location=location)
signature=hashlib.sha256((digest+json.dumps(settings,sort_keys=True)).encode()).hexdigest()[:24]
if analyze:
    run_dir=session_dir/signature
    try:
        cached=run_dir/'result.json'
        if cached.is_file() and Path(json.loads(cached.read_text(encoding='utf-8'))['annotated_path']).is_file():
            st.session_state.result=json.loads(cached.read_text(encoding='utf-8')); st.session_state.pop('last_error',None)
        else:
            start_job(run_dir,dict(settings=settings,source_path=str(source),file_name=Path(uploaded.name).name,media_type='video' if is_video else 'image'))
            st.session_state.active_run=str(run_dir); st.session_state.pop('last_error',None)
        st.rerun()
    except Exception as exc: st.error(str(exc))
if st.session_state.get('result'):
    if st.session_state.result['signature']!=signature: st.info('Setup changed. Results below retain their original area and settings. Analyze again to apply changes.')
    render_results(st.session_state.result)
