# Web workspace validation — 2026-10-28

## Completed locally

- 37 Python tests passed: existing 30 pipeline/scene/storage/Streamlit regressions plus 7 spatial/API tests. The final API tests include restart recovery and request-size rejection.
- TypeScript checking and production Vite build passed. `npm audit --audit-level=high` reports zero vulnerabilities after updating Vite to 6.4.5.
- `render.yaml` validated against Render's current JSON Schema. The `off` deployment trigger is quoted to avoid YAML boolean parsing.
- Actual packaged YOLO11m inference, through the browser and FastAPI worker, on a six-second 960 × 540 recording: 30 analyzed samples, two overlapping zones, 20-head union peak, one sustained episode from 0.0 to 5.8 seconds. Input: 10,434,360 bytes; final annotated H.264 output: 583,327 bytes; local processing: 45.72 seconds.
- This fixture repeats an existing already-annotated office frame. It verifies software operation only. It is **not** a campus accuracy, tracking, temporal behavior or counting ground-truth evaluation.
- Cancelled a separate actual-inference job after processing began; status became cancelled and partial video returned 404. Retried the retained recording successfully. The server reused its cached model.
- CSV, JSON, PDF, event CSV and MP4 exports returned valid content. Counts CSV has one row per sample and separate zone columns. Browser CSV download succeeded. Video HTTP range request returned 206 with the requested 100-byte segment.
- Event navigation verified in the browser: keyboard seek moved video to 6.0 seconds; selecting the event moved it back to its 0.0-second start. Selecting the second zone changed counts, threshold and event list. Heatmap toggle and interpretation label were inspected.
- Configuration persisted across navigation and reload. Saved completed sessions reopened after a server restart. Unit tests simulate interrupted jobs, partial-upload cleanup and a completed-result/status-write interruption.
- Desktop and 390 × 844 mobile layouts inspected. No document horizontal overflow on the tested mobile viewport. Mobile navigation and sign-out remain accessible. Keyboard coordinate editing and timeline controls were exercised.
- Unauthenticated media/session access, cross-origin mutation, missing request header, malformed video, unsupported extension, active-worker conflict, deletion of active jobs, immutable completed results, retention boundaries and duplicate-zone validation were tested.

## Deployment validation boundary

The GitHub workflow builds the Docker image, starts it in production mode, checks `/healthz` and verifies unauthenticated API rejection. Consult the workflow result for the delivered commit. A successful image build/startup is not a live Render deployment or a Render memory/throughput measurement.

The first delivery, `cd7ada9`, passed all three GitHub jobs (Python tests, frontend build/audit, and Docker build/production startup): https://github.com/Tishal68/AI-Crowd-Management-System/actions/runs/40688509416. A final navigation refinement resets scroll when changing screens; its delivery is checked by the same workflow.

No Render resources were provisioned. No private footage or generated exports were pushed. Screenshots and test media remain local. CPU processing is slower than playback on the tested machine. The proposed 2 GB service requires measurement on actual Render hardware before operational use.

## Remaining limitations

- Single shared team password, one active CPU job, no per-user roles or audit trail. Appropriate for a controlled academic demonstration; institutional rollout needs individual accounts and institution-specific data governance.
- The new web UI focuses on recorded videos. Image analysis, historical Streamlit sessions and experimental directional flow remain in the preserved Streamlit app. They were not silently deleted or claimed as migrated.
- No independently annotated campus evaluation exists. The current detector's original training exposure is unknown. The Colab candidate was rejected; the packaged detector remains unchanged.
- Zones must be convex and footage should use a fixed camera. Heatmap intensity is relative detection frequency, not physical density. No reliable dwell-time, unique-person count, face recognition or safety prediction is claimed.
- Interrupted processing restarts from the beginning on explicit retry. It does not resume halfway through an encoded video. Retention cleanup is hourly, and provider backups can outlive application deletion.
