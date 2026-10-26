# Model details

The default checkpoint is `models/head_detector.pt`: a YOLO11m detector with one class, `head`, approximately 20 million parameters. It is packaged from the supplied local root `best.pt`; optimizer state and nonessential training paths were removed. The original local checkpoint is preserved and excluded from Git.

The supplied checkpoint recorded precision 0.921, recall 0.91578, mAP50 0.95289 and mAP50-95 0.5327. These are historical checkpoint metrics, not reproduced metrics for this repository's dataset. Training metadata referenced an external training environment. Ownership, dataset provenance and redistribution permission must be established by the project author; no claim of novel architecture or sole authorship is made here.

The old default under `runs/train/.../weights/best.pt` is an 80-class COCO YOLO11x model. It is not the head model used by this release. Old local training plots/configuration do not describe one coherent run of the packaged head detector.

Use only trusted checkpoint files. Ultralytics checkpoints use PyTorch serialization. Follow [Ultralytics licensing](https://www.ultralytics.com/license) for AGPL-3.0 or enterprise terms; this repository does not grant additional rights to upstream code, model weights or datasets.

For reproducible local evaluation, record the checkpoint SHA-256, dataset source/license, held-out split, hardware, confidence threshold, model input size and output sampling settings. Report counting MAE and detector precision/recall/mAP where box ground truth exists. Do not equate checkpoint mAP with end-user counting accuracy.

Packaged checkpoint SHA-256: `667b4a6da2b8739b4a11e2070e281506e6303de76589cb2832d80c261b99004c`.
