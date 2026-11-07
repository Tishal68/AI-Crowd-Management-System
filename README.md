# CrowdVision AI — Campus Crowd Intelligence & Monitoring

**Primary & Official Application Entry Point:** CrowdVision is an AI-powered campus crowd monitoring and spatial analytics platform built with Streamlit (`streamlit run app.py`). It features real-time and recorded video/image crowd analysis, YOLO11-based people/head detection, interactive convex counting polygons, directional tripwire flow analysis, sustained capacity exceedance detection, spatial density heatmaps, and formal compliance PDF audit reports.

The interface is engineered with a modern enterprise visual identity:
- **Charcoal backgrounds:** Soft charcoal (`#171A18`), deep charcoal (`#202522`), and stone black cards (`#292E2A`).
- **Deep emerald primary accents:** Active navigation, primary actions, and timeline curves (`#087F5B`).
- **Warm copper secondary accents:** Turnstile flow markers and secondary highlights (`#B87345`).
- **Warm amber highlights:** Exceedance alerts and detection heatmaps (`#D9A441`).

All video decoding, YOLO11 inference, and tracking run 100% locally on your machine with zero external network transmission.

---

## Quick Start (Primary Streamlit Application)

Requires Python 3.12:

```powershell
cd AI-Crowd-Management-System
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Open **http://localhost:8501** in your browser. Stop with `Ctrl+C` in the terminal. No external API keys or cloud services are required.

---

## Optional Web Workspace (React + FastAPI)

The repository also includes an isolated, optional web workspace built with React, TypeScript, and FastAPI for headless or containerized multi-zone deployments. It is completely independent and not required for running the primary Streamlit application.

To launch the optional web workspace:

```powershell
cd frontend
npm ci
npm run build
cd ..
$env:CROWD_DEV_AUTH='1'
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8080
```

Open **http://127.0.0.1:8080**.

The included default model is `models/head_detector.pt`, a single-class YOLO11m head detector. Select another trusted checkpoint in the sidebar or set `CROWD_MODEL_PATH` before launch. The model must include a head/person class; unrelated classes are rejected. See [model provenance and historical metric limitations](docs/MODEL_CARD.md).

## Workflow

1. Upload MP4/MOV/AVI/MKV recording or JPG/PNG image (up to 4K resolution, 240 FPS and 10 minutes), validated dynamically against server disk capacity and configurable storage quotas.
2. Draw 3–12 corners around a convex counting area and click **Apply area & line**. Set the location, capacity of this area and required exceedance duration. Only detected head centres inside the area count.
3. Choose video sampling rate (1–15 FPS) and maximum output width (640/960/1280). Defaults are 5 FPS and 960 pixels wide. Inference runs on CPU and may take substantially longer than clip duration.
4. Optionally draw a two-point flow line and enable directional estimates (at least 5 samples/second). The orange arrow identifies the positive direction. Short-lived tracking IDs are not unique visitors.
5. Analyze in the background. Inspect peak occupancy, sustained exceedances, event thumbnails and the source timeline. Cancel if needed; reopen progress or completed results from **Saved sessions**, including after a browser refresh.
6. Download annotated media, counts, event CSV or a PDF. Explicitly delete saved sessions after downloading anything needed. Clips are limited to 10 minutes; the generated workspace has a 2 GB admission budget, including reservations for active jobs.

## Output-size fix

The previous version re-encoded every full-resolution frame with unrestricted MPEG-4 output. This version streams resized sampled frames directly to H.264 and limits bitrate to the source's average bitrate and a 2 Mbps ceiling. Identical input bytes + model version + settings reuse the result within a session. Changed settings create a separate run; saved runs remain until explicitly deleted. No per-frame image archive is generated.

An output budget stops and removes an export if it exceeds the smaller of 128 MB or (1.5 × input bytes + 2 MB). Encoder buffering can briefly exceed that budget before it is detected. Inputs, reports and the single annotated output also consume disk space; old legacy exports are not automatically deleted. A low-bitrate long clip may hit the budget and require lower resolution/a shorter clip. Output size is not guaranteed to be below input size. Audio is intentionally omitted. Sampling reduces temporal detail and can miss brief events.

Counts and events use decoder timestamps, normalized to the first decoded frame. Variable frame spacing is supported when timestamps are valid; missing or non-monotonic timestamps stop the run with a re-export instruction. Export playback uses a constant sampling rate and repeats the prior annotated frame across timing gaps. Event calculations never fill those gaps with invented observations or extend an event beyond its final sample.

## Storage and privacy

New files live under `outputs/sessions/<random-session-id>/`. Original media outside this folder is never changed. **Saved sessions** lists recoverable completed results and active jobs with storage usage. Active jobs cannot be deleted there. Uploads and outputs stay on the machine hosting Streamlit; this app has no authentication and is intended for local use. Historical camera URL credentials are redacted by the history/report services; recording history and all media are excluded from Git. Legacy generated directories can be inspected using the script below.

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
- Directional line crossings use conservative centroid association, confirmed side changes and finite-line intersection. Occlusion, identity switches and sparse samples can miss or duplicate crossings; this is not validated entry/exit accounting.
- Background workers persist progress and errors, serialize analysis across sessions, support cancellation, and remove partial video exports on failure.
- The original training labels contain widespread duplicate boxes. Training validation rejects duplicate, non-finite, malformed and out-of-range annotations.

## Cloud GPU training

[Open the self-contained Colab notebook](https://colab.research.google.com/github/Tishal68/AI-Crowd-Management-System/blob/main/notebooks/CrowdVision_Campus_GPU_Training.ipynb). Select a T4 GPU runtime and run all cells. CPU training is refused. The notebook downloads the official SCUT-HEAD Part B archive, verifies its checksum, converts VOC head boxes, and excludes exact duplicate images before training. No campus footage or Google Drive mount is needed.

It fine-tunes the supplied model for up to 30 epochs with early stopping. Confidence is selected using validation images; a separate test split compares counting MAE/RMSE and box precision/recall. Promotion requires more than 2% lower MAE, no higher RMSE, and no precision or recall drop larger than two percentage points. It never automatically overwrites the packaged model. Download the results ZIP before the Colab runtime expires.

SCUT-HEAD is for academic research. The supplied checkpoint's original training exposure is unknown, so this test cannot establish that its images were never seen by the baseline. Exact-image deduplication does not remove near-duplicates. A separate labelled sample from the intended camera is still needed for a deployment claim. See [training and validation notes](docs/TRAINING.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover H.264 output size/duration/sampling, timeline correctness, inference failures and partial-file cleanup, invalid model classes, smoothing reorder, secret redaction, storage containment, PDF generation, zone summaries and API access controls. A synthetic detector is used in regression tests; passing them does not measure ML accuracy. Real-model execution checks and deployment validation are recorded in [validation notes](docs/WEB_VALIDATION.md).

## Dataset and training

The dataset is not published: provenance/license and camera privacy have not been established. Place permitted YOLO images/labels under `dataset/images/{train,val}` and `dataset/labels/{train,val}`, then copy `dataset/data.example.yaml` to `dataset/data.yaml` and set absolute paths appropriate for your machine. Alternatively use `--data` to reference a permitted external dataset. Training does not run automatically.

```powershell
.\.venv\Scripts\python.exe train.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --name crowd_head_experiment
```

Training outputs go under `runs/train/`. Evaluate and choose a trained checkpoint explicitly; training does not replace the packaged model automatically. Do not use the supplied historical 1,500/300-image dataset: its annotations failed the duplicate-box integrity audit. Use the audited SCUT conversion for the current research experiment.

## Structure

```text
frontend/                   primary React/TypeScript workspace
server/app.py               authenticated API and single-worker orchestration
app.py                       preserved local Streamlit interface
config.py                    paths and detector/training defaults
src/core/detector.py         image/frame inference
src/core/video_processor.py  sampling, counts and source timestamps
src/core/encoding.py         bounded H.264 pipe encoder
src/core/scene.py            area geometry, capacity events and line crossings
src/core/zones.py            named-zone counts and bounded heatmap summaries
src/services/               storage, jobs, exports and reports
src/services/jobs.py         background job dispatch and recovery
src/services/worker.py       cancellable analysis worker
src/ui/scene_component/      local interactive area/line editor
src/utils/                  drawing, dataset validation and privacy helpers
models/head_detector.pt     packaged single-class head checkpoint
dataset/data.example.yaml   dataset template; data not included
tests/                      regression tests
scripts/storage.py          explicit legacy output cleanup
scripts/model_experiment.py audited training and comparison
notebooks/                  self-contained Colab GPU notebook
```

Generated media, the local virtual environment, old checkpoints/training runs and archives are excluded from version control. See the model card for upstream licensing and provenance limitations.
