# Changes in 3.1

- Compact H.264 export replaces unrestricted full-resolution MPEG-4. Sample rate, output width, source-aware bitrate and an explicit size budget control disk use.
- One cached completed result per browser session. Stable content hashes detect changed same-size uploads; successful new runs prune previous managed outputs.
- Correct head-model default with provenance documentation. Image/video errors stop analysis rather than publishing false empty-scene results.
- Disabled unassociated smoothing; fixed tracker utility reset and error propagation, while keeping tracking outside the supported UI.
- Correct source-timeline CSV and immutable analysis settings, separate peak/final occupancy, and accurate reports without safety predictions.
- Updated responsive UI, browser-compatible annotated video, direct downloads, local storage display and explicit cleanup.
- Redacted saved camera credentials; ignored generated files, dataset media, old checkpoints, archives, secrets and local environments in Git.
- Added regression tests, GitHub Actions, dataset template, small-model training defaults, safe resume/overwrite behavior and current setup instructions.
