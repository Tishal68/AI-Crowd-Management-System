# Changes in 4.0

- Reframed the project around campus entrance and event-space occupancy review, with an interactive counting area, sustained capacity events, event navigation and experimental directional line crossings.
- Added cancellable background analysis, saved-session recovery, cross-session admission locking and a 2 GB workspace budget with active-job reservations.
- Used decoded timestamps for sampling and events; rejected missing/non-monotonic times and prevented events from bridging observation gaps.
- Blocked corrupt repeated training labels, converted the official SCUT-HEAD Part B data and trained on a Colab Tesla T4. Published the negative model comparison; retained the stronger existing checkpoint.
- Added Model lab, event CSV/PDF details, compact evidence thumbnails and area/flow settings recovery after background jobs.
- Expanded regression coverage to 30 tests. Historical 3.1 notes below describe the earlier release; 4.0 retains separate saved runs until explicitly deleted.

# Changes in 3.1

- Compact H.264 export replaces unrestricted full-resolution MPEG-4. Sample rate, output width, source-aware bitrate and an explicit size budget control disk use.
- One cached completed result per browser session. Stable content hashes detect changed same-size uploads; successful new runs prune previous managed outputs.
- Correct head-model default with provenance documentation. Image/video errors stop analysis rather than publishing false empty-scene results.
- Disabled unassociated smoothing; fixed tracker utility reset and error propagation, while keeping tracking outside the supported UI.
- Correct source-timeline CSV and immutable analysis settings, separate peak/final occupancy, and accurate reports without safety predictions.
- Updated responsive UI, browser-compatible annotated video, direct downloads, local storage display and explicit cleanup.
- Redacted saved camera credentials; ignored generated files, dataset media, old checkpoints, archives, secrets and local environments in Git.
- Added regression tests, GitHub Actions, dataset template, small-model training defaults, safe resume/overwrite behavior and current setup instructions.
