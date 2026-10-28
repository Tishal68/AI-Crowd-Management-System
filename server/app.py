"""Single-team, single-worker recorded-footage service. Media never bypasses authentication."""
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import hashlib
import hmac
import json
import os
import re
import shutil
import time
import uuid
import cv2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from filelock import FileLock, Timeout
import config
from src.services.storage import atomic_json, check_storage_budget, clear_session
from src.services.worker import run
from src.core.zones import ZoneAnalytics
from src.core.detector import load_model
from src.services.report_generator import generate_pdf_report

ROOT = Path(config.OUTPUTS_ROOT)
OWNER = 'c' * 32  # One explicitly shared campus-team workspace, not multi-tenant accounts.
SESSIONS = ROOT / 'sessions' / OWNER
ACTIVE = ('queued', 'running', 'cancelling')
MAX_UPLOAD = 200 * 1024**2
pool = ThreadPoolExecutor(max_workers=1)
model = None
attempts = {}
upload_lock = asyncio.Lock()


def read(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback if fallback is not None else {}


def directory(key):
    if not re.fullmatch('[a-f0-9]{32}', key):
        raise HTTPException(404, 'Session not found.')
    path = SESSIONS / key
    if not path.is_dir() or path.is_symlink() or path.resolve().parent != SESSIONS.resolve():
        raise HTTPException(404, 'Session not found.')
    return path


def cleanup():
    with FileLock(str(ROOT/'.dispatch.lock'), timeout=5):
        _cleanup_expired()


def _cleanup_expired():
    cutoff = time.time() - int(os.getenv('CROWD_RETENTION_DAYS', '7')) * 86400
    for p in SESSIONS.glob('*/metadata.json'):
        if read(p).get('created', time.time()) < cutoff and read(p.parent/'status.json').get('state') not in ACTIVE:
            clear_session(SESSIONS, p.parent.name)


def recover_jobs():
    for path in SESSIONS.iterdir():
        if not path.is_dir() or path.is_symlink() or not re.fullmatch('[a-f0-9]{32}', path.name):
            continue
        if not (path/'metadata.json').exists():
            # A process interrupted during upload cannot leave unaccounted orphan media.
            clear_session(SESSIONS, path.name)
        elif read(path/'status.json').get('state') in ACTIVE:
            if read(path/'result.json').get('processed_frames', 0) > 0 and (path/'annotated.mp4').is_file():
                atomic_json(path/'status.json', dict(state='complete', progress=1, message='Completed result recovered after restart.', updated=time.time()))
            else:
                atomic_json(path/'status.json', dict(state='failed', error='Service restarted during processing. Start analysis again using the retained recording.', updated=time.time()))
                (path/'annotated.mp4').unlink(missing_ok=True)
                (path/'result.json').unlink(missing_ok=True)


async def retention_loop():
    while True:
        await asyncio.sleep(3600)
        await asyncio.to_thread(cleanup)


@asynccontextmanager
async def lifespan(app):
    production = os.getenv('CROWD_ENV') == 'production'
    if production and (len(os.getenv('CROWD_ACCESS_PASSWORD', '')) < 16 or len(os.getenv('CROWD_SESSION_SECRET', '')) < 32):
        raise RuntimeError('Production requires a 16-character team password and 32-character session secret.')
    if not 1 <= int(os.getenv('CROWD_RETENTION_DAYS', '7')) <= 365:
        raise RuntimeError('Retention must be between 1 and 365 days.')
    if not Path(config.BEST_WEIGHTS_PATH).is_file():
        raise RuntimeError('Configured detector checkpoint is missing.')
    SESSIONS.mkdir(parents=True, exist_ok=True)
    # A single process owns this workspace. Never launch uvicorn with multiple workers.
    service_lock = FileLock(str(ROOT / '.web-service.lock'), timeout=0)
    service_lock.acquire()
    recover_jobs()
    cleanup()
    task = asyncio.create_task(retention_loop())
    yield
    task.cancel()
    for p in SESSIONS.glob('*/status.json'):
        if read(p).get('state') in ACTIVE:
            (p.parent/'cancel').touch()
    pool.shutdown(wait=True)
    service_lock.release()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(Timeout)
async def workspace_busy(request, exc):
    return JSONResponse({'detail': 'The workspace is handling another operation. Try again shortly.'}, status_code=409)


def token():
    expiry = str(int(time.time()) + 12*3600)
    digest = hmac.new(os.environ['CROWD_SESSION_SECRET'].encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return expiry + '.' + digest


def authorized(request):
    # Development bypass is explicitly opt-in and restricted to loopback connections.
    if os.getenv('CROWD_DEV_AUTH') == '1' and os.getenv('CROWD_ENV') != 'production' and request.client.host in ('127.0.0.1', '::1', 'testclient'):
        return True
    try:
        expiry, signature = request.cookies.get('campus_session', '').split('.')
        expected = hmac.new(os.environ['CROWD_SESSION_SECRET'].encode(), expiry.encode(), hashlib.sha256).hexdigest()
        return int(expiry) > time.time() and hmac.compare_digest(signature, expected)
    except (ValueError, KeyError):
        return False


@app.middleware('http')
async def guard(request, call_next):
    if request.url.path.startswith('/api/'):
        if request.url.path != '/api/upload' and int(request.headers.get('content-length', '0') or '0') > 65536:
            return JSONResponse({'detail': 'Request settings exceed the allowed size.'}, status_code=413)
        if request.method not in ('GET', 'HEAD') and request.headers.get('x-campus-request') != '1':
            return JSONResponse({'detail': 'Missing same-origin request header.'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin requests are disabled.'}, status_code=403)
        if request.url.path not in ('/api/login', '/api/auth') and not authorized(request):
            return JSONResponse({'detail': 'Sign in to access the team workspace.'}, status_code=401)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Cache-Control'] = 'no-store' if request.url.path.startswith('/api/') else 'no-cache'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
    return response


@app.get('/healthz')
def health():
    return {'status': 'ok'}


@app.get('/api/auth')
def auth(request: Request):
    return {'authenticated': authorized(request), 'retention_days': int(os.getenv('CROWD_RETENTION_DAYS', '7'))}


class Login(BaseModel):
    password: str = Field(max_length=256)


@app.post('/api/login')
def login(body: Login, request: Request):
    now = time.time()
    # Bounded global throttle: cannot evade by spoofing forwarded IP headers.
    history = [v for v in attempts.get('login', []) if v > now-60]
    if len(history) >= 10:
        raise HTTPException(429, 'Too many attempts. Wait one minute.')
    attempts['login'] = history + [now]
    configured = os.getenv('CROWD_ACCESS_PASSWORD', '')
    if not configured or not hmac.compare_digest(body.password, configured) or not os.getenv('CROWD_SESSION_SECRET'):
        raise HTTPException(401, 'Incorrect team password or access is not configured.')
    response = JSONResponse({'authenticated': True})
    response.set_cookie('campus_session', token(), httponly=True, secure=os.getenv('CROWD_ENV') == 'production', samesite='strict', max_age=43200)
    return response


@app.post('/api/logout')
def logout():
    response = JSONResponse({'ok': True})
    response.delete_cookie('campus_session')
    return response


def summary(path):
    return dict(read(path/'metadata.json'), id=path.name, status=read(path/'status.json', {'state': 'ready'}))


@app.get('/api/sessions')
def sessions():
    return sorted([summary(p.parent) for p in SESSIONS.glob('*/metadata.json')], key=lambda s: s['created'], reverse=True)


@app.post('/api/upload')
async def upload(request: Request):
    name = request.headers.get('x-file-name', 'recording.mp4')
    from urllib.parse import unquote
    name = Path(unquote(name).replace('\\', '/')).name[:120]
    suffix = Path(name).suffix.lower()
    if suffix not in ('.mp4', '.mov', '.avi', '.mkv'):
        raise HTTPException(415, 'Choose an MP4, MOV, AVI or MKV recording. Images remain supported in the legacy app.')
    if int(request.headers.get('content-length', '0') or '0') > MAX_UPLOAD:
        raise HTTPException(413, 'Recording exceeds the 200 MB upload limit.')
    if upload_lock.locked():
        raise HTTPException(409, 'Another upload is in progress. Try again shortly.')
    async with upload_lock:
        dispatch_lock = FileLock(str(ROOT/'.dispatch.lock'), timeout=5, thread_local=False)
        await asyncio.to_thread(dispatch_lock.acquire)
        path = SESSIONS / uuid.uuid4().hex
        try:
            check_storage_budget(ROOT, MAX_UPLOAD)
            path.mkdir(parents=True)
            size = 0
            with (path/('source'+suffix)).open('wb') as target:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'Recording exceeds the 200 MB upload limit.')
                    target.write(chunk)
            cap = cv2.VideoCapture(str(path/('source'+suffix)))
            try:
                width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not 1 <= width <= 4096 or not 1 <= height <= 2160 or not 0 < fps <= 240 or not 0 < frames/fps <= 600:
                    raise HTTPException(422, 'Use a decodable recording up to 4096 × 2160, 240 FPS and 10 minutes.')
                ok, frame = cap.read()
                if not ok:
                    raise HTTPException(422, 'No readable video frame. Re-export as H.264 MP4.')
                scale = min(1, 1280/width)
                cv2.imwrite(str(path/'poster.jpg'), cv2.resize(frame, (int(width*scale), int(height*scale))))
            finally:
                cap.release()
            atomic_json(path/'metadata.json', dict(file_name=name, source='source'+suffix, created=time.time(), bytes=size, duration=frames/fps, width=width, height=height))
            return summary(path)
        except ValueError as exc:
            shutil.rmtree(path, ignore_errors=True)
            raise HTTPException(507, str(exc))
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        finally:
            await asyncio.to_thread(dispatch_lock.release)


class Settings(BaseModel):
    location: str = Field(default='Campus event', max_length=80)
    zones: list[dict] = Field(min_length=1, max_length=5)
    conf: float = Field(default=.35, ge=.05, le=.9)
    analysis_fps: float = Field(default=5, ge=1, le=10)
    event_seconds: float = Field(default=3, ge=1, le=60)
    imgsz: int = Field(default=640, ge=320, le=640)


def execute(path):
    global model
    try:
        if model is None:
            model = load_model(config.BEST_WEIGHTS_PATH)
        run(path, preloaded_model=model)
    except Exception:
        atomic_json(path/'status.json', dict(state='failed', error='Detector could not be loaded. Check the server model configuration.', updated=time.time()))


@app.post('/api/sessions/{key}/start')
def start(key: str, body: Settings):
    path = directory(key)
    try:
        spatial = ZoneAnalytics(body.zones)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    with FileLock(str(ROOT/'.dispatch.lock'), timeout=5):
        if (path/'result.json').exists():
            raise HTTPException(409, 'Completed results are immutable. Upload the recording to create another analysis.')
        if any(read(p).get('state') in ACTIVE for p in (ROOT/'sessions').glob('*/*/status.json')):
            raise HTTPException(409, 'The CPU worker is occupied. Wait for the current analysis or cancel it.')
        try:
            check_storage_budget(ROOT, 150*1024**2)
        except ValueError as exc:
            raise HTTPException(507, str(exc))
        settings = body.model_dump()
        settings.update(weights_path=config.BEST_WEIGHTS_PATH, max_capacity=sum(z['capacity'] for z in spatial.zones), roi_polygon=None, flow_line=None, max_width=960)
        meta = read(path/'metadata.json')
        atomic_json(path/'job.json', dict(settings=settings, source_path=str(path/meta['source']), file_name=meta['file_name'], media_type='video'))
        (path/'cancel').unlink(missing_ok=True)
        atomic_json(path/'status.json', dict(state='queued', progress=0, updated=time.time(), message='Waiting for detector'))
        pool.submit(execute, path)
    return {'ok': True}


@app.get('/api/sessions/{key}')
def session(key: str):
    path = directory(key)
    record = read(path/'result.json')
    # Never expose server paths, model locations, or worker logs over HTTP.
    for field in ('annotated_path', 'output_path', 'source_path'):
        record.pop(field, None)
    if record:
        record['settings'].pop('weights_path', None)
        for e in record.get('events', []):
            e.pop('thumbnail', None)
    settings = read(path/'job.json').get('settings', {})
    settings.pop('weights_path', None)
    return dict(summary(path), result=record or None, settings=settings)


@app.post('/api/sessions/{key}/cancel')
def cancel(key: str):
    path = directory(key)
    if read(path/'status.json').get('state') in ACTIVE:
        (path/'cancel').touch()
    return {'ok': True}


@app.delete('/api/sessions/{key}')
def delete(key: str):
    path = directory(key)
    with FileLock(str(ROOT/'.dispatch.lock'), timeout=5):
        if read(path/'status.json').get('state') in ACTIVE:
            raise HTTPException(409, 'Cancel processing before deleting this recording.')
        clear_session(SESSIONS, key)
    return {'ok': True}


@app.get('/api/sessions/{key}/media/{kind}')
def media(key: str, kind: str):
    path = directory(key)
    names = {'poster': 'poster.jpg', 'preview': 'preview.jpg', 'video': 'annotated.mp4'}
    if kind not in names or not (path/names[kind]).is_file():
        raise HTTPException(404, 'Media is not ready.')
    return FileResponse(path/names[kind])


@app.get('/api/sessions/{key}/export/{kind}')
def export(key: str, kind: str):
    path = directory(key)
    result = read(path/'result.json')
    if not result:
        raise HTTPException(409, 'Complete an analysis before exporting.')
    headers = {'Content-Disposition': f'attachment; filename="campus-{key[:8]}.{kind}"'}
    if kind == 'csv':
        import csv
        import io
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer)
        zones = result.get('zones', [])
        writer.writerow(['source_time_seconds', 'union_head_count'] + ['zone: '+z['name'] for z in zones])
        for i, timestamp in enumerate(result['timestamps']):
            writer.writerow([timestamp, result['per_frame_counts'][i]] + [z['counts'][i] for z in zones])
        return Response(buffer.getvalue(), media_type='text/csv', headers=headers)
    if kind == 'json':
        return JSONResponse(session(key)['result'], headers=headers)
    if kind == 'mp4':
        return FileResponse(path/'annotated.mp4', filename=f'campus-{key[:8]}.mp4')
    if kind == 'pdf':
        with FileLock(str(path/'.report.lock'), timeout=5):
            if not (path/'report.pdf').exists():
                generate_pdf_report(result, str(path/'report.pdf'))
        return FileResponse(path/'report.pdf', filename=f'campus-{key[:8]}.pdf')
    if kind == 'events.csv':
        import csv
        import io
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=['zone','start_seconds','end_seconds','duration_seconds','peak_count','capacity'], extrasaction='ignore')
        writer.writeheader()
        for event in result['events']:
            safe = dict(event)
            if str(safe.get('zone', '')).startswith(('=', '+', '-', '@', '\t', '\r')):
                safe['zone'] = "'" + safe['zone']
            writer.writerow(safe)
        return Response(buffer.getvalue(), media_type='text/csv', headers=headers)
    raise HTTPException(404, 'Export format not found.')


@app.get('/api/evaluation')
def evaluation():
    return read(Path(config.PROJECT_ROOT)/'docs/model_evaluation.json')


build = Path(config.PROJECT_ROOT)/'frontend/dist'
if build.exists():
    app.mount('/', StaticFiles(directory=build, html=True), name='frontend')
