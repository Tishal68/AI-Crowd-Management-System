# Project re-evaluation — 26 October 2026

The project is a functional recorded-media crowd-detection prototype and is suitable as an internship engineering project. It is not yet a validated crowd-safety system. This review covers the current code and targeted tests; it is not an independent accuracy evaluation.

## Bugs repaired in this revision

- Export resolution previously changed detector input and could alter counts. Detection now receives the original frame; annotation is resized only for export. A regression test verifies this separation.
- Clear session left the uploader populated, causing the input to be written again immediately. Clearing now rotates the uploader key and resets report/error state.
- Failed re-analysis discarded the last successful result. The last success is retained, with a persistent failure notice and settings-change notice.
- Missing cached video required clearing the entire session. Missing outputs now invalidate the cache and can be rebuilt by running analysis again.
- Hidden download tabs read full exports on every rerun. Review sections now execute only when selected.

## Highest-priority remaining issues

1. **P1 — Session storage accumulates across browser reloads.** Each fresh session gets a new directory. Per-run size limits do not bound total disk use. Add a durable session index, explicit old-session cleanup and a global storage quota before unattended use. Current clear controls target the active session and legacy export folders only.
2. **P1 — No measured accuracy on held-out camera footage.** Model metadata is not independent evidence. Build a labeled test set with different lighting, camera angles and crowd sizes; report count MAE, missed heads and false positives. Validate dataset/model provenance before claiming original training or performance.
3. **P2 — Timeline assumes constant frame rate.** Index/FPS timestamps may be inaccurate for variable-frame-rate phone footage. Use decoder presentation timestamps or normalize the source with FFmpeg and preserve mapping.
4. **P2 — Long jobs have no cancellation or duration limit.** A small, very long compressed file can occupy the CPU and grow in-memory sample arrays. Add duration/sample limits and a cancellable background job with progress state.
5. **P2 — Capacity covers the full camera frame.** People outside the monitored area are included. Add a user-defined counting region and scene calibration before occupancy decisions.

## UI changes

The interface now follows source inspection, detection review and evidence export. Original media is visible before processing, an analysis plan summarizes selected settings, model tuning is collapsed, and the timeline includes configured capacity and the first sampled peak. Metrics explicitly say sampled count. Exports retain the source filename. No fabricated live-feed, tracking or safety features are shown.

## Validation

14 local tests passed, including compression, timestamps, encoder cleanup, inference failures, CSV consistency, privacy, contained cleanup, unchanged detection input across export widths, and Streamlit workspace/session-reset behavior. These tests do not establish detector accuracy or production readiness.
