# Validation

## Campus workflow — 28 October 2026

- 30 regression tests passed on Windows/Python 3.12. These include area geometry, boundary inclusion, confirmed directional crossings, capacity duration/gap rules, variable decoder timestamps, duplicate annotation detection, cancellation cleanup, storage reservations, cross-session job admission, failed-worker recovery, PDF generation and workspace navigation.
- A browser run with the actual packaged detector on a controlled six-second repeated-frame fixture completed 30 samples, produced one 5.8-second sampled capacity event and an evidence thumbnail, and generated a downloadable PDF. This fixture checks the workflow, not detector accuracy or movement tracking.
- Fixture input: 10,434,360 bytes; annotated H.264 output: 575,989 bytes (about 5.5%). Browser output playback reported six seconds, readyState 4 and no media error.
- A real short recording also completed with a drawn area: the selected area peaked at eight heads. No ground truth exists for that clip; this is execution evidence only.
- A second browser run confirmed that the uploaded file, selected polygon and enabled flow setting survived the background job. PDF download and reopening a completed result from Saved sessions succeeded.
- Desktop and 390-pixel mobile layouts were visually inspected; the mobile page and main content had no horizontal overflow.
- Colab Tesla T4 training completed 21 epochs with early stopping. The comparison used 494 test images after exact-image deduplication. Existing/candidate counting MAE was 1.279/1.441 and RMSE 3.103/4.675. Candidate promotion was rejected. Details, hashes and limitations are in `TRAINING.md` and `model_evaluation.json`.
- Installed dependency consistency passed. GitHub CI is tracked separately; local passing results do not establish remote CI completion.

## Earlier checks — 26 October 2026

## Automated and execution checks

- 12 local regression tests passed on Windows/Python 3.12.6. These cover H.264 sampling/output duration, size-budget enforcement, inference failure and partial cleanup, CSV source timestamps, smoothing reorder, unsupported classes, occupancy boundaries, redaction, storage containment/pruning and PDF generation.
- Python source syntax checks passed. Installed dependency consistency check passed.
- Streamlit AppTest startup and switching classification settings completed without exceptions.
- Actual packaged head checkpoint loaded as YOLO11m with one `head` class. The packaging process removes absolute training paths/optimizer metadata; the learned tensor values were compared with the original supplied model.
- Real-model inference on an existing local test recording produced sampled visible counts 16 and 19 at source timestamps 0.0s and 0.2s. The 4,689,181-byte test input produced a 199,664-byte annotated H.264 file at 960 × 540, 5 FPS. This is execution evidence only: no manual ground truth was supplied.
- Browser CSV download completed and contained source times 0.0 and 0.2 seconds; the PDF preparation control generated an available PDF download.
- Mobile welcome layout at 390px had no horizontal overflow; desktop layout was also visually inspected.
- End-to-end browser video analysis completed. The browser decoded the output at 960 × 540 with readyState 4, a 0.4-second duration and no media error.

## Compression benchmark on an actual larger recording

Input: 41,477,178 bytes (39.56 MiB). Output: 2,487,418 bytes (2.37 MiB), approximately 6.0% of the input size. Settings: width 960, output size 960 × 446, 5 sampled FPS, H.264, source-based bitrate capped at 2 Mbps, audio omitted. Decoded duration approximately 11.03 seconds, 56 analyzed samples. Export-only runtime approximately 5.92 seconds on this machine.

This benchmark deliberately used an empty synthetic detector to isolate decoding/compression and file growth; it is not a full-model speed benchmark. The container over-reported its frame count by five, which is surfaced as a warning rather than rejecting an otherwise readable clip. Small frame-count discrepancies are tolerated; substantial early decode termination fails the run.

## Scope and limitations

These execution results do not establish counting accuracy, robustness to a new camera, physical crowd density, unique-person identity or safe crowd conditions. Sampling trades temporal detail for CPU and storage use. Very low source bitrate can produce artifacts or hit the hard export budget; use a shorter clip or lower resolution. The later Colab comparison is documented above and has separate limitations.

Supported dependency ranges are in requirements.txt. Direct versions used for local validation are recorded in requirements-tested.txt. GitHub CI runs regression checks separately; local passing results do not imply remote CI has completed.
