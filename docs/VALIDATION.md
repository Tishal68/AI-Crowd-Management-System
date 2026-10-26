# Validation — 26 October 2026

## Automated and execution checks

- 11 local regression tests passed on Windows/Python 3.12.6. These cover H.264 sampling/output duration, size-budget enforcement, inference failure and partial cleanup, CSV source timestamps, smoothing reorder, unsupported classes, occupancy boundaries, redaction, storage containment/pruning and PDF generation.
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

These results do not establish counting accuracy, robustness to a new camera, physical crowd density, unique-person identity or safe crowd conditions. Sampling and resizing trade temporal/spatial detail for CPU and storage use. Very low source bitrate can produce artifacts or hit the hard export budget; use a shorter clip or lower resolution. Model training and a complete held-out evaluation were not performed during this repair.

Supported dependency ranges are in requirements.txt. Direct versions used for local validation are recorded in requirements-tested.txt. GitHub CI runs regression checks separately; local passing results do not imply remote CI has completed.
