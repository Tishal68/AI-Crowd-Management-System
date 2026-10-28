# Campus Vision web workspace

The primary experience is React + FastAPI. It reuses the existing detector, timestamp sampling, compact H.264 writer, storage admission checks and capacity-event logic. Streamlit remains available for existing saved sessions, image analysis and experimental directional flow. Existing files are not migrated or deleted by this release.

## Local setup (Windows)

Use Python 3.12 and Node 22. Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
$env:CROWD_DEV_AUTH='1'
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080. Stop with Ctrl+C. The explicit development bypass works only on loopback and is disabled in production. Do not expose the development server through a tunnel. For password-protected local testing, unset `CROWD_DEV_AUTH` and set `CROWD_ACCESS_PASSWORD` and `CROWD_SESSION_SECRET` in the environment. `.env.example` documents variables; files are not loaded automatically.

## Workflow

1. **Overview** summarizes saved recordings and worker availability.
2. **Analyze footage** accepts a recording up to 200 MB, 10 minutes, 4096 × 2160 pixels and 240 source FPS. Invalid media is rejected and partial uploads removed.
3. Name up to five convex counting zones, set capacities in heads, and choose the minimum sustained duration. Draw corners or edit normalized coordinates with the keyboard. Current settings are stored on the device; completed results retain their original settings.
4. Start analysis. One CPU worker runs at a time; another start gets an explicit busy response. Navigate away and return through **Sessions**. Cancelling removes partial video. Restarted jobs become failed with a retry explanation, rather than claiming completion or silently resuming a corrupt output.
5. Select a zone, click an event or use the timeline slider to seek footage. Export counts, structured metadata, event CSV, PDF or annotated MP4 from one menu.
6. **Model evaluation** presents the actual SCUT-HEAD comparison. The Colab candidate was rejected because counting error increased. No new detector or unsupported accuracy claim is introduced.

## Measurements

- Each zone counts detected head centers within its polygon. A detection in overlapping zones contributes to each zone, but only once to the overall union count. The summed capacities of overlapping zones are not a physical occupancy limit; review per-zone events.
- A buildup event requires observed counts above that zone's threshold for the configured duration. Missing sampling gaps break episodes; observations are not extrapolated past the last sample.
- Heatmap: a fixed 64 × 36 grid accumulates head-center observations across sampled frames. Brightness is relative to the maximum cell. It is not unique visitors, dwell time, people per square meter or a safety score. Camera movement and perspective affect interpretation.
- Use a fixed camera. Crowded occlusion, poor lighting, blur, unusual viewpoints and very small heads require manual review. Neither new feature has been independently validated against campus annotations.

## Storage and privacy

This is one **shared team workspace**, not per-user isolated accounts. Anyone with the team password can access all its sessions. Sessions use a signed, HttpOnly, SameSite=Strict cookie with a 12-hour expiry (Secure in production). Every media/export route requires authentication. Mutation requests require a same-origin custom header; cross-origin requests are rejected. Login attempts are throttled. Server paths and model settings are not accepted from clients.

The 2 GB workspace admission budget and 150 MB active-job reserve remain. Video output is capped at the smaller of 128 MB or 1.5 × input size + 2 MB; no per-frame image archive is written. Only one upload and one analysis are admitted concurrently. A single cached detector is reused; run exactly one Uvicorn worker.

New web sessions expire after seven days (configurable with `CROWD_RETENTION_DAYS`). Cleanup runs on startup and hourly and skips active jobs. This policy applies only to the new web workspace. Legacy sessions remain explicitly managed in Streamlit. Platform snapshots/backups may retain deleted files beyond application retention; account for this in your institution's policy. Avoid personal data in filenames and restrict the shared password to authorized reviewers.

## Render deployment

`Dockerfile` builds the locked frontend, installs CPU PyTorch and serves the UI and API from one origin. `render.yaml` defines a single web service, `/healthz`, environment secrets and a 3 GB persistent disk mounted at `/var/data`. Only that directory survives redeploys. No Redis or separate worker service is needed at this scale.

The prepared plan is `1c-2g` (1 CPU, 2 GB RAM), plus a paid persistent disk. The 512 MB plans are not a credible default for this YOLO11m/PyTorch application. This is a starting deployment size, not a throughput guarantee; measure memory with representative clips and increase resources or reduce limits if needed. No resources have been purchased and no deployment has been performed. Check current pricing before creating the service.

1. Push source and the packaged detector only; do not include recordings, datasets, exports, `.env`, or secrets.
2. Open the Blueprint setup for this repository: https://dashboard.render.com/blueprint/new?repo=https://github.com/Tishal68/AI-Crowd-Management-System
3. Review the paid plan and disk costs before confirming creation. Set a random team password of at least 16 characters. Render generates the session secret; keep it stable across redeploys. Do not enable development auth.
4. Deploy, check `/healthz`, sign in, then upload an authorized non-private test clip. Verify processing, authenticated video seeking, exports and recovery after restart before institutional use.
5. Automatic deployment is off. Updates can interrupt an active analysis; wait until the worker is idle before redeploying. A persistent-disk service is intentionally single-instance and does not support this application's horizontal scaling.

References: [Blueprint specification](https://render.com/docs/blueprint-spec), [compute plans](https://render.com/docs/compute-plans), [persistent disks](https://render.com/docs/disks), [web services](https://render.com/docs/web-services).

## Validation commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm ci
npm run build
npm audit --audit-level=high
```

The web suite covers authentication, cross-origin mutations, malformed uploads, active-job exclusion, immutable results, cancellation requests, retention and overlapping-zone heatmap semantics. Pipeline tests cover source timing, compression limits, inference failure, cancellation cleanup and storage protection. Browser and real-model verification are recorded separately in `docs/WEB_VALIDATION.md`; passing unit tests alone is not proof of hosted operation.
