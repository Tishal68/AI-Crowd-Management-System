from __future__ import annotations
import argparse
import os
from ultralytics import YOLO
import config
from src.utils.helpers import (
    clear_dataset_cache,
    log_resolved_dataset_paths,
    normalize_data_yaml,
    setup_logger,
    validate_dataset,
)

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO head/person detection (CPU).")
    parser.add_argument("--data", type=str, default=config.DATA_YAML_PATH, help="Path to existing data.yaml")
    parser.add_argument("--model", type=str, default=config.TRAIN_BASE_MODEL, help="Base model / weights to start from")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=config.TRAIN_IMGSZ, help="Training image size (CPU: keep <= 640)")
    parser.add_argument("--batch", type=int, default=config.TRAIN_BATCH, help="Batch size (CPU: keep small, e.g. 4-16)")
    parser.add_argument("--workers", type=int, default=config.TRAIN_WORKERS, help="Dataloader worker processes")
    parser.add_argument("--device", type=str, default=config.TRAIN_DEVICE, help="Training device, keep 'cpu'")
    parser.add_argument("--project", type=str, default=config.TRAIN_PROJECT_DIR, help="Output project directory")
    parser.add_argument("--name", type=str, default=config.TRAIN_RUN_NAME, help="Run name")
    parser.add_argument("--patience", type=int, default=config.TRAIN_PATIENCE, help="Early stopping patience (epochs)")
    parser.add_argument("--cache", type=str, default=None, choices=[None, "ram", "disk"], help="Dataset caching")
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted training run")
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Do NOT delete existing train.cache/val.cache before training (default: cache is cleared).",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the pre-training image/label pairing check (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        last = os.path.join(args.project, args.name, "weights", "last.pt")
        if not os.path.isfile(last):
            raise FileNotFoundError(f"Cannot resume: checkpoint missing at {last}")
        YOLO(last).train(resume=True)
        return

    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"data.yaml not found at '{args.data}'. This script never creates or "
            f"modifies your dataset -- please verify the path."
        )

    # --- Fix for "All labels empty in .../train.cache" ------------------------
    # Ultralytics trusts an existing .cache file blindly and will NOT re-scan
    # labels on disk if a stale cache is present. Clear it so training always
    # re-scans your actual, current label files.
    if not args.resume and not args.keep_cache:
        logger.info("Clearing any stale train/val .cache files before scanning labels...")
        clear_dataset_cache(args.data, logger=logger)

    # Validate image<->label pairing BEFORE calling model.train(), so a
    # broken dataset produces a clear, actionable error instead of the
    # opaque "All labels empty in ...cache" ValueError from Ultralytics.
    if not args.resume and not args.skip_validation:
        logger.info("Validating image/label pairing for '%s'...", args.data)
        validate_dataset(args.data, logger=logger)
        logger.info("Dataset validation passed.")

    # --- Debug: print exactly what Ultralytics will resolve, before training ---
    logger.info("Resolved training paths for '%s':", args.data)
    log_resolved_dataset_paths(args.data, logger=logger)

    # --- Fix for "All labels empty in .../train.cache" (real root cause) -----
    # Ultralytics maps an image path to its label path with a hardcoded,
    # OS-specific substitution of f"{os.sep}images{os.sep}" -> f"{os.sep}labels{os.sep}".
    # If data.yaml uses forward slashes on Windows, that substitution silently
    # fails to match and Ultralytics looks for labels in the wrong directory,
    # reporting every image as having an empty label set -- even when the
    # dataset on disk is completely correct. Normalize the paths Ultralytics
    # actually receives (in a temp copy; the user's data.yaml is never touched).
    resolved_data = normalize_data_yaml(args.data, logger=logger)
    logger.info("Path Ultralytics will train on: %s", resolved_data)

    logger.info("Loading base model: %s", args.model)
    if os.path.abspath(args.model) == os.path.abspath(config.PRETRAINED_BASE_CHECKPOINT):
        logger.warning(
            "Base model is the STOCK pretrained checkpoint (full 80-class COCO, never "
            "fine-tuned on your dataset). This training run will fine-tune it toward the "
            "classes in '%s' -- that's expected and is exactly what turns it into a real "
            "head-detection model.",
            args.data,
        )
    model = YOLO(args.model)

    run_dir = os.path.join(args.project, args.name)
    existing_best = os.path.join(run_dir, "weights", "best.pt")
    if not args.resume and os.path.isfile(existing_best):
        raise FileExistsError(f"Run already exists at {run_dir}. Use a new --name or --resume.")

    logger.info(
        "Starting CPU training | epochs=%d imgsz=%d batch=%d workers=%d",
        args.epochs, args.imgsz, args.batch, args.workers,
    )

    train_kwargs = dict(
        data=resolved_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        verbose=True,
        resume=args.resume,
    )
    if args.cache:
        train_kwargs["cache"] = args.cache

    try:
        results = model.train(**train_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        raise

    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    logger.info("Training complete.")
    if os.path.isfile(best_weights):
        logger.info("Best weights saved at: %s", best_weights)
    else:
        logger.warning("Expected best.pt not found at %s -- check the run directory.", best_weights)

    # Run a lightweight validation pass on CPU to report final metrics.
    try:
        logger.info("Running validation on best weights...")
        val_model = YOLO(best_weights) if os.path.isfile(best_weights) else model
        val_model.val(data=resolved_data, imgsz=args.imgsz, device=args.device, batch=args.batch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Validation step skipped due to error: %s", exc)


if __name__ == "__main__":
    main()
