"""
finetune_india.py — Fine-Tune YOLOv8 for India/Tamil Nadu Garbage Detection
=============================================================================
Fine-tunes a YOLOv8n model on the India-specific 2-class dataset:
    Class 0: not_overflow  — normal / properly managed garbage bin
    Class 1: overflow      — overflowing / spilling garbage bin

The model learns to simultaneously detect both classes so that in any
given frame, bins are classified as either overflowing or not.

Usage:
    python finetune_india.py                     # Default settings
    python finetune_india.py --epochs 50         # More epochs
    python finetune_india.py --device cuda:0     # Use GPU
"""

import os
import sys
import shutil
import yaml
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils import setup_logger, ensure_directories

logger = setup_logger("finetune")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "base_model": "yolov8n.pt",           # Pretrained base model
    "data_yaml": None,                      # Auto-detected
    "epochs": 30,                           # Training epochs
    "imgsz": 640,                           # Image size
    "batch": 8,                             # Batch size (CPU-friendly)
    "patience": 10,                         # Early stopping patience
    "lr0": 0.01,                            # Initial learning rate
    "lrf": 0.01,                            # Final learning rate factor
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,                             # Box loss gain
    "cls": 0.5,                             # Classification loss gain
    "dfl": 1.5,                             # DFL loss gain
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.1,
    "degrees": 10.0,
    "flipud": 0.3,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "device": "",                           # Auto (CPU or GPU)
}


# ---------------------------------------------------------------------------
# Dataset Verification
# ---------------------------------------------------------------------------

def verify_dataset(data_yaml_path: str) -> dict:
    """
    Verify the dataset exists and count images per split.

    Returns:
        Dict with counts and the loaded config.
    """
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    base_path = config.get("path", "")
    counts = {}

    for split in ["train", "val", "test"]:
        split_key = split
        if split_key not in config:
            continue

        img_dir = os.path.join(base_path, config[split_key])
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir)
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
            counts[split] = img_count
        else:
            counts[split] = 0

    logger.info("Dataset verification:")
    logger.info("  Path: %s", base_path)
    logger.info("  Classes: %s", config.get("names", []))
    for split, count in counts.items():
        logger.info("  %s: %d images", split, count)

    total = sum(counts.values())
    if total == 0:
        raise ValueError("No images found in dataset! Run prepare_india_dataset.py first.")

    return {"config": config, "counts": counts, "total": total}


# ---------------------------------------------------------------------------
# Fine-Tuning
# ---------------------------------------------------------------------------

def finetune(data_yaml: str, config: dict) -> tuple:
    """
    Fine-tune YOLOv8n on the India garbage detection dataset.

    Args:
        data_yaml:  Path to data.yaml
        config:     Training configuration dict

    Returns:
        (model, results) tuple
    """
    from ultralytics import YOLO

    # Load base model
    base_model = config.get("base_model", "yolov8n.pt")
    base_path = os.path.join(PROJECT_ROOT, base_model)
    if os.path.exists(base_path):
        base_model = base_path

    logger.info("Loading base model: %s", base_model)
    model = YOLO(base_model)

    # Training arguments
    run_name = f"india_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_dir = os.path.join(PROJECT_ROOT, "runs")

    train_args = {
        "data": data_yaml,
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "patience": config["patience"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": config["warmup_epochs"],
        "warmup_momentum": config["warmup_momentum"],
        "box": config["box"],
        "cls": config["cls"],
        "dfl": config["dfl"],
        "augment": config["augment"],
        "mosaic": config["mosaic"],
        "mixup": config["mixup"],
        "degrees": config["degrees"],
        "flipud": config["flipud"],
        "fliplr": config["fliplr"],
        "hsv_h": config["hsv_h"],
        "hsv_s": config["hsv_s"],
        "hsv_v": config["hsv_v"],
        "name": run_name,
        "project": runs_dir,
        "exist_ok": True,
        "verbose": True,
    }

    # Device selection
    device = config.get("device", "")
    if device:
        train_args["device"] = device

    logger.info("\n🚀 Starting fine-tuning...")
    logger.info("   Epochs: %d | Image Size: %d | Batch: %d",
                config["epochs"], config["imgsz"], config["batch"])
    logger.info("   Run name: %s", run_name)

    # Train
    results = model.train(**train_args)

    return model, results, run_name


# ---------------------------------------------------------------------------
# Validation & Metrics
# ---------------------------------------------------------------------------

def validate_model(model, data_yaml: str) -> dict:
    """
    Run validation and extract per-class and overall metrics.

    Returns:
        Dict with mAP50, mAP50-95, precision, recall, and per-class metrics.
    """
    logger.info("\n📊 Running validation...")
    metrics = model.val(data=data_yaml, verbose=True)

    results = {
        "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
        "mAP50_95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
        "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
        "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0,
    }

    # Per-class metrics
    try:
        class_names = ["not_overflow", "overflow"]
        if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
            ap50_per_class = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, 'tolist') else []
            for i, ap in enumerate(ap50_per_class):
                if i < len(class_names):
                    results[f"ap50_{class_names[i]}"] = float(ap)
    except Exception:
        pass

    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 50)
    logger.info("  mAP@50:      %.4f", results["mAP50"])
    logger.info("  mAP@50-95:   %.4f", results["mAP50_95"])
    logger.info("  Precision:   %.4f", results["precision"])
    logger.info("  Recall:      %.4f", results["recall"])
    if "ap50_not_overflow" in results:
        logger.info("  AP50 (not_overflow): %.4f", results["ap50_not_overflow"])
    if "ap50_overflow" in results:
        logger.info("  AP50 (overflow):     %.4f", results["ap50_overflow"])
    logger.info("=" * 50)

    return results


# ---------------------------------------------------------------------------
# Save Best Model
# ---------------------------------------------------------------------------

def save_best_model(model, run_name: str) -> str:
    """
    Copy the best model weights to models/best_india.pt.

    Returns:
        Path to the saved model.
    """
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, "best_india.pt")

    # Find best.pt from the training run
    best_path = None
    runs_dir = os.path.join(PROJECT_ROOT, "runs")

    # Look specifically in the run directory
    run_dir = os.path.join(runs_dir, run_name, "weights", "best.pt")
    if os.path.exists(run_dir):
        best_path = run_dir
    else:
        # Fallback: search all runs
        for root, dirs, files in os.walk(runs_dir):
            if "best.pt" in files:
                candidate = os.path.join(root, "best.pt")
                if best_path is None or os.path.getmtime(candidate) > os.path.getmtime(best_path):
                    best_path = candidate

    if best_path and os.path.exists(best_path):
        shutil.copy2(best_path, dest)
        logger.info("✅ Best model saved to %s", dest)
    else:
        logger.warning("⚠ best.pt not found, saving current model state")
        model.save(dest)
        logger.info("✅ Model saved to %s", dest)

    # Also copy as the default best.pt
    default_dest = os.path.join(models_dir, "best.pt")
    shutil.copy2(dest, default_dest)
    logger.info("   Also copied to %s", default_dest)

    return dest


# ---------------------------------------------------------------------------
# Training Report
# ---------------------------------------------------------------------------

def generate_report(metrics: dict, dataset_info: dict, run_name: str,
                    model_path: str, data_yaml: str):
    """Generate a training report saved to the models directory."""
    report_path = os.path.join(PROJECT_ROOT, "models", "training_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  INDIA GARBAGE OVERFLOW DETECTION — TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {data_yaml}\n\n")

        f.write("DATASET\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Classes: not_overflow (0), overflow (1)\n")
        for split, count in dataset_info.get("counts", {}).items():
            f.write(f"  {split}: {count} images\n")
        f.write(f"  Total: {dataset_info.get('total', 0)} images\n\n")

        f.write("METRICS\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")

        f.write("SIMULTANEOUS DETECTION\n")
        f.write("-" * 40 + "\n")
        f.write("  The model detects BOTH classes in every frame:\n")
        f.write("  - not_overflow (GREEN box) — normal bins\n")
        f.write("  - overflow (RED box) — overflowing bins\n")
        f.write("  Both appear simultaneously on screen.\n\n")

        f.write("=" * 60 + "\n")

    # Also save as YAML
    metrics_yaml = os.path.join(PROJECT_ROOT, "models", "metrics.yaml")
    with open(metrics_yaml, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)

    logger.info("📝 Training report saved to %s", report_path)
    logger.info("📝 Metrics saved to %s", metrics_yaml)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for India Garbage Overflow Detection"
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: 'cpu', 'cuda:0', etc. (default: auto)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml (default: auto-detect)")
    parser.add_argument("--prepare-dataset", action="store_true",
                        help="Run dataset preparation before training")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  INDIA GARBAGE OVERFLOW DETECTION — FINE-TUNING")
    print("=" * 60 + "\n")

    ensure_directories()

    # --- Step 0: Optionally prepare dataset ---
    if args.prepare_dataset:
        logger.info("📦 Running dataset preparation first...")
        from prepare_india_dataset import main as prepare_main
        prepare_main()

    # --- Step 1: Find data.yaml ---
    if args.data:
        data_yaml = args.data
    else:
        # Auto-detect: prefer india_bins data.yaml
        india_yaml = os.path.join(PROJECT_ROOT, "data", "india_bins", "data.yaml")
        project_yaml = os.path.join(PROJECT_ROOT, "data.yaml")

        if os.path.exists(india_yaml):
            data_yaml = india_yaml
        elif os.path.exists(project_yaml):
            data_yaml = project_yaml
        else:
            logger.error("❌ No data.yaml found! Run prepare_india_dataset.py first.")
            logger.info("   Or use: python finetune_india.py --prepare-dataset")
            return

    logger.info("Using dataset config: %s", data_yaml)

    # --- Step 2: Verify dataset ---
    try:
        dataset_info = verify_dataset(data_yaml)
    except (FileNotFoundError, ValueError) as e:
        logger.error("❌ Dataset error: %s", str(e))
        logger.info("Run: python prepare_india_dataset.py")
        return

    # --- Step 3: Configure training ---
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch"] = args.batch
    config["imgsz"] = args.imgsz
    if args.device:
        config["device"] = args.device

    # --- Step 4: Fine-tune ---
    logger.info("\n🏋️ Step 4 — Fine-tuning YOLOv8n on India dataset...")
    model, results, run_name = finetune(data_yaml, config)

    # --- Step 5: Validate ---
    logger.info("\n📊 Step 5 — Validating model...")
    metrics = validate_model(model, data_yaml)

    # --- Step 6: Auto-retrain if needed ---
    if metrics["mAP50"] < 0.3:
        logger.warning("⚠ mAP50 (%.4f) too low — retraining with tuned hyperparameters...",
                        metrics["mAP50"])
        config["epochs"] = args.epochs + 20
        config["lr0"] = 0.005
        config["cls"] = 1.0
        config["mosaic"] = 0.8
        config["mixup"] = 0.15

        model, results, run_name = finetune(data_yaml, config)
        metrics = validate_model(model, data_yaml)

    # --- Step 7: Save best model ---
    saved_path = save_best_model(model, run_name)

    # --- Step 8: Generate report ---
    generate_report(metrics, dataset_info, run_name, saved_path, data_yaml)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Model:       {saved_path}")
    print(f"  mAP@50:      {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:   {metrics['mAP50_95']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Classes:     not_overflow (0), overflow (1)")
    print("=" * 60)
    print("\n  Next steps:")
    print("    python detect.py --source sample_india_video.mp4")
    print("    python app.py --with-detection --source sample_india_video.mp4")
    print()

    return metrics


if __name__ == "__main__":
    main()
