# CrowdVision — AI Crowd Management System

Local image and recorded-video crowd counting with a Streamlit dashboard, compact H.264 exports, source-timeline CSV data and PDF reports. This is an internship prototype; scene-specific accuracy needs evaluation.

## Windows setup

Python 3.12 is recommended.

```powershell
cd AI-Crowd-Management-System
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Stop with Ctrl+C in the terminal. No API keys are required. `requirements-tested.txt` records the direct versions used for Windows validation; use it instead of `requirements.txt` when you need those exact versions. `imageio-ffmpeg` supplies an encoder if FFmpeg is not installed on PATH. Video compression uses libx264; no uncompressed intermediate video is saved.

The included default model is `models/head_detector.pt`, a single-class YOLO11m head detector. Select another trusted checkpoint in the sidebar or set `CROWD_MODEL_PATH` before launch. The model must include a head/person class; unrelated classes are rejected. See [model provenance and historical metric limitations](docs/MODEL_CARD.md).

## Workflow

1. Upload JPG/PNG/BMP or MP4/AVI/MOV/MKV, up to 200 MB.
2. Configure the monitored area's capacity. Default classification is occupancy against that capacity: low up to 60%, moderate up to 85%, high above 85%. These bands are operational rules, not safety thresholds. Image-coverage classification is explicitly experimental.
3. Choose video sampling rate (1–15 FPS) and maximum output width (640/960/1280). Defaults are 5 FPS and 960 pixels wide. Inference runs on CPU and may take substantially longer than clip duration.
4. Analyze; inspect the annotated preview, peak/average/final counts and source timeline.
5. Download the annotated media, CSV or PDF. Clear session storage after downloading when it is no longer needed.

## Output-size fix

The previous version re-encoded every full-resolution frame with unrestricted MPEG-4 output and retained repeated uploads/exports. This version streams resized sampled frames directly to H.264, limits bitrate to the source's average bitrate and a 2 Mbps ceiling, and writes one completed result per session. Identical input bytes + model version + settings reuse the result. Changing settings creates a new result and removes the earlier managed run only after success.

An output budget stops and removes an export if it exceeds the smaller of 128 MB or (1.5 × input bytes + 2 MB). Encoder buffering can briefly exceed that budget before it is detected. Inputs, reports and the single annotated output also consume disk space; old legacy exports are not automatically deleted. A low-bitrate long clip may hit the budget and require lower resolution/a shorter clip. Output size is not guaranteed to be below input size. Audio is intentionally omitted. Sampling reduces temporal detail and can miss brief events.

Source-video timestamps are independent of processing speed. Export playback uses the sampling frame rate, preserving approximate source duration (up to one sampling interval rounding). Constant-frame-rate recordings are recommended; variable-frame-rate footage should be converted before analysis.

## Storage and privacy

New files live under `outputs/sessions/<random-session-id>/`. Original media outside this folder is never changed. Session IDs are per browser session; orphaned sessions after a browser/server restart remain on disk and may be removed after inspection. The app exposes the current session size and total generated storage. Historical camera URL credentials are redacted by the history/report services; recording history and all media are excluded from Git.

Preview earlier output sizes:

```powershell
.\.venv\Scripts\python.exe scripts/storage.py
```

Explicitly remove one old generated folder's files after saving anything you need:

```powershell
.\.venv\Scripts\python.exe scripts/storage.py --delete-folder streamlit_videos
```

This command deletes that folder's direct generated files; it does not delete source videos elsewhere. Other permitted legacy folders are listed in `--help`. Clearing storage is permanent.

## Reliability changes

- Default model points to the packaged head detector, not the old COCO checkpoint.
- Inference/decode/encoder failures stop the run and do not publish zero-count “Safe” results.
- Detection-index smoothing is disabled; untracked boxes are never mixed across people.
- Peak and final occupancy are reported separately; no validated safety prediction is claimed.
- Content hashes distinguish changed uploads with the same filename/size.
- Each result carries its original settings/model metadata; later settings edits do not rewrite old results.
- Atomic JSON results, random session directories, bounded exports and safe cleanup reduce collisions and stale data.
- No live RTSP, unique-person tracking, entry/exit counting or calibrated physical density is claimed in this release. Tracking utilities remain experimental and are not used by the app.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover H.264 output size/duration/sampling, timeline correctness, inference failures and partial-file cleanup, invalid model classes, smoothing reorder, secret redaction, storage containment and PDF generation. A synthetic detector is used in regression tests; passing them does not measure ML accuracy. Real-model smoke results and compression measurements are recorded in `docs/VALIDATION.md`.

## Dataset and training

The dataset is not published: provenance/license and camera privacy have not been established. Place permitted YOLO images/labels under `dataset/images/{train,val}` and `dataset/labels/{train,val}`, then copy `dataset/data.example.yaml` to `dataset/data.yaml` and set absolute paths appropriate for your machine. Alternatively use `--data` to reference a permitted external dataset. Training does not run automatically.

```powershell
.\.venv\Scripts\python.exe train.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --name crowd_head_experiment
```

Training outputs go under `runs/train/`. Evaluate and choose a trained checkpoint explicitly; training does not replace the packaged model automatically. The supplied historical dataset has 1,500 train and 300 validation images, but no independent test set or provenance document. Verify whether its `person` labels represent body boxes or heads before mixing it with the supplied head model.

## Structure

```text
app.py                       dashboard and immutable run flow
config.py                    paths and detector/training defaults
src/core/detector.py         image/frame inference
src/core/video_processor.py  sampling, counts and source timestamps
src/core/encoding.py         bounded H.264 pipe encoder
src/core/tracker.py          experimental tracker utility (not app flow)
src/services/               model loading, storage, reports and legacy history
src/utils/                  drawing, dataset validation and privacy helpers
models/head_detector.pt     packaged single-class head checkpoint
dataset/data.example.yaml   dataset template; data not included
tests/                      regression tests
scripts/storage.py          explicit legacy output cleanup
```

Generated media, the local virtual environment, old checkpoints/training runs and archives are excluded from version control. See the model card for upstream licensing and provenance limitations.
