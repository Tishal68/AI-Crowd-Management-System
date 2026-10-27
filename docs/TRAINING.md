# Training and evaluation record

## Data decision

The supplied local labels were unsuitable for training: 1,800 files contained 655,742 rows, of which 652,751 were repeated rows within their files. 854 files consisted entirely of one repeated box. The associated CPU pilot was stopped, and no model from it was promoted.

Replacement source: [official SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release), Part B, academic research only. The archive SHA-256 is `02afedc58065aca9cb5f28333963a818e37564d75d8420f9bcb94d1e6c340a98`.

Conversion preserves the official train/validation/test assignments. Before deduplication: train 1,443 images / 26,272 heads; validation 462 / 8,212; test 500 / 9,446. 62 XML files had zero dimensions; decoded image dimensions were used. The converter records this correction. Image-content SHA-256 removes exact duplicates with train, validation, test precedence. The run manifest records the resulting split membership and exclusions. Near-duplicates are not detected.

## Reproducibility

`scripts/prepare_scut.py` converts the checked archive. `scripts/model_experiment.py` audits annotations and splits, trains, tunes confidence on validation data, and evaluates both models on the same separate test split. `scripts/build_colab.py` embeds these sources in a notebook, avoiding dependence on uncommitted local code at runtime.

Colab configuration: YOLO11m head detector, 768-pixel training input, batch 8, AdamW at 0.0001, all layers trainable, 30 maximum epochs, patience 8, seed 20261027. Comparison uses the application's 640-pixel inference and aspect-ratio filter. Confidence candidates are 0.15, 0.25, 0.35, 0.45. Detection matching uses greedy confidence-ordered one-to-one matches at IoU 0.5. Reported precision/recall are fixed-threshold metrics, not mAP.

Promotion requires candidate MAE < 0.98 × baseline MAE, candidate RMSE ≤ baseline RMSE, and precision/recall no more than 0.02 below baseline. Passing these gates supports this benchmark comparison only. The original checkpoint's training exposure is unknown; contamination of its historical training cannot be ruled out.

The results ZIP contains best/last checkpoints, training CSV and arguments, full comparison, split manifest and conversion provenance. Keep it before disconnecting Colab. The application reads a published compact `docs/model_evaluation.json` when one exists. An absent report means no completed comparison has been published.

## Campus validation still required

Manually annotate a representative, permissioned sample across camera angles, lighting, crowd sizes and occlusion. Split by recording or camera rather than adjacent frames. Report counting MAE, undercount bias, detection precision/recall and separate flow-crossing errors. Test complete sessions, not just favourable screenshots. Do not call a count/capacity percentage model accuracy or a safety prediction.

## Completed Colab run — 27 October 2026

Experiment `scut_gpu_20261027_132838` completed on a Tesla T4. Early stopping ended training after 21 epochs (1,860.82 training seconds). After exact-image deduplication, the splits contained 1,440 training, 456 validation and 494 test images.

Validation selected confidence 0.35 for the existing model and 0.25 for the candidate. On the test split, existing/candidate counting MAE was 1.279/1.441 heads, RMSE 3.103/4.675, precision 0.937/0.918 and recall 0.939/0.957. Higher candidate recall came with more false detections and worse count error. The candidate failed the promotion gate; the packaged checkpoint and its 0.35 default confidence were retained. Full numeric results and hashes are in `model_evaluation.json`. This is a measured negative training result, not evidence of improved campus accuracy.
