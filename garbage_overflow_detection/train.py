"""
train.py — YOLOv8 Model Training for Garbage Overflow Detection
================================================================
This script handles the full training pipeline:
  1. Attempt to download the dataset from Roboflow
  2. If that fails, generate a synthetic dataset
  3. Create data.yaml configuration
  4. Fine-tune YOLOv8n with specified hyperparameters
  5. Validate and report metrics (mAP50, mAP50-95, precision, recall)
  6. Auto-retrain with tuned hyperparameters if mAP50 < 0.5
  7. Save best weights to models/best.pt

Usage:
    python train.py
"""

import os
import sys
import shutil
import random
import yaml
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils import ensure_directories, setup_logger, get_project_root

logger = setup_logger("train")


# ---------------------------------------------------------------------------
# STEP 2A — Attempt Roboflow Dataset Download
# ---------------------------------------------------------------------------

def download_roboflow_dataset(dest_dir: str) -> bool:
    """
    Try to download the garbage classification dataset from Roboflow.
    Returns True on success, False on failure.
    """
    try:
        from roboflow import Roboflow
        logger.info("Attempting Roboflow dataset download...")

        # Try with a public/free key — this may or may not work
        rf = Roboflow(api_key="YOUR_FREE_KEY")
        project = rf.workspace("public").project("garbage-classification-uwjdk")
        dataset = project.version(2).download("yolov8", location=dest_dir)
        logger.info("✅ Roboflow dataset downloaded to %s", dest_dir)
        return True
    except Exception as e:
        logger.warning("⚠ Roboflow download failed: %s", str(e))
        logger.info("Falling back to synthetic dataset generation...")
        return False


# ---------------------------------------------------------------------------
# STEP 2B — Synthetic Dataset Generation (Fallback)
# ---------------------------------------------------------------------------

def random_color(base_h: int, s_range=(100, 255), v_range=(100, 255)):
    """Generate a random BGR color from an HSV base hue."""
    h = base_h + random.randint(-10, 10)
    s = random.randint(*s_range)
    v = random.randint(*v_range)
    hsv = np.uint8([[[h % 180, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


def draw_synthetic_bin(img: np.ndarray, x: int, y: int,
                       w: int, h: int, fill_level: float,
                       class_idx: int) -> list:
    """
    Draw a synthetic garbage bin on the image and return its YOLO label.

    Args:
        img:         Image array to draw on (modified in-place)
        x, y:        Top-left corner of the bin
        w, h:        Width and height of the bin body
        fill_level:  0.0 to 1.0, how full the bin is
        class_idx:   Class index (0=trash_bin, 1=garbage, 2=overflow, 3=lid_open)

    Returns:
        YOLO format label: [class_idx, cx, cy, nw, nh]
    """
    img_h, img_w = img.shape[:2]

    # Bin body (dark grey rectangle)
    bin_color = random_color(0, s_range=(0, 30), v_range=(50, 100))
    cv2.rectangle(img, (x, y), (x + w, y + h), bin_color, -1)
    # Bin outline
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), 2)

    # Rim at top
    rim_h = max(3, h // 15)
    rim_color = random_color(0, s_range=(0, 20), v_range=(80, 130))
    cv2.rectangle(img, (x - 3, y), (x + w + 3, y + rim_h), rim_color, -1)

    # Fill contents (garbage inside)
    fill_px = int(h * fill_level)
    if fill_px > 0:
        garbage_y = y + h - fill_px
        # Random garbage colors
        for stripe in range(0, fill_px, max(4, fill_px // 8)):
            stripe_color = random_color(random.randint(0, 179))
            sy = garbage_y + stripe
            sh = min(max(4, fill_px // 8), y + h - sy)
            cv2.rectangle(img, (x + 2, sy), (x + w - 2, sy + sh), stripe_color, -1)

    # For "overflow" class — draw garbage spilling above the bin
    total_h = h
    bbox_y = y
    if class_idx == 2:  # overflow
        overflow_h = random.randint(h // 6, h // 3)
        for oy in range(0, overflow_h, 5):
            oc = random_color(random.randint(0, 179))
            ow = random.randint(w // 3, w + 10)
            ox = x + (w - ow) // 2
            cv2.rectangle(img, (ox, y - oy - 5), (ox + ow, y - oy), oc, -1)
        bbox_y = y - overflow_h
        total_h = h + overflow_h

    # For "lid_open" class — draw a lid tilted above
    if class_idx == 3:  # lid_open
        lid_pts = np.array([
            [x, y],
            [x + w, y],
            [x + w + 10, y - 15],
            [x - 5, y - 20],
        ], dtype=np.int32)
        cv2.fillPoly(img, [lid_pts], rim_color)
        bbox_y = y - 20
        total_h = h + 20

    # Compute YOLO-format normalised coordinates
    cx = (x + w / 2) / img_w
    cy = (bbox_y + total_h / 2) / img_h
    nw = w / img_w
    nh = total_h / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return [class_idx, cx, cy, nw, nh]


def generate_background(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a random urban-ish background."""
    # Random gradient sky
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    top_color = np.array(random.choice([
        [180, 130, 70], [200, 180, 150], [100, 100, 100],
        [50, 50, 50], [150, 160, 170], [80, 70, 60],
    ]))
    bot_color = np.array(random.choice([
        [60, 60, 60], [80, 90, 80], [100, 80, 60],
        [50, 45, 40], [70, 75, 80], [90, 85, 75],
    ]))

    for row in range(height):
        ratio = row / height
        bg[row] = (top_color * (1 - ratio) + bot_color * ratio).astype(np.uint8)

    # Random rectangles to simulate walls / ground
    for _ in range(random.randint(2, 6)):
        rx = random.randint(0, width)
        ry = random.randint(height // 3, height)
        rw = random.randint(50, width // 2)
        rh = random.randint(30, height // 2)
        rc = random_color(random.randint(0, 179), s_range=(10, 60), v_range=(40, 120))
        cv2.rectangle(bg, (rx, ry), (rx + rw, ry + rh), rc, -1)

    # Add some noise for texture
    noise = np.random.randint(0, 15, bg.shape, dtype=np.uint8)
    bg = cv2.add(bg, noise)

    return bg


def generate_synthetic_dataset(base_dir: str, num_train: int = 200,
                                num_val: int = 25, num_test: int = 25):
    """
    Generate a synthetic dataset of garbage bin images with YOLO-format labels.

    Classes:
        0: trash_bin   — normal bin
        1: garbage     — loose garbage (no bin)
        2: overflow    — overflowing bin
        3: lid_open    — bin with lid open

    Images are saved in YOLOv8 directory structure:
        data/train/images/ + data/train/labels/
        data/valid/images/ + data/valid/labels/
        data/test/images/  + data/test/labels/
    """
    CLASS_NAMES = ["trash_bin", "garbage", "overflow", "lid_open"]

    splits = {
        "train": num_train,
        "valid": num_val,
        "test":  num_test,
    }

    for split, count in splits.items():
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(count):
            width = random.choice([480, 640, 800])
            height = random.choice([360, 480, 600])
            img = generate_background(width, height)
            labels = []

            # Place 1-4 bins per image
            num_bins = random.randint(1, 4)
            for b in range(num_bins):
                # Random class: weighted toward trash_bin and overflow
                class_idx = random.choices([0, 1, 2, 3],
                                           weights=[0.35, 0.15, 0.35, 0.15])[0]

                # Random fill level based on class
                if class_idx == 0:  # trash_bin → low to medium fill
                    fill = random.uniform(0.1, 0.6)
                elif class_idx == 1:  # garbage (loose) → high fill appearance
                    fill = random.uniform(0.3, 0.7)
                elif class_idx == 2:  # overflow → high fill
                    fill = random.uniform(0.8, 1.0)
                else:  # lid_open
                    fill = random.uniform(0.2, 0.7)

                # Random position and size
                bw = random.randint(width // 8, width // 3)
                bh = random.randint(height // 4, int(height * 0.7))
                bx = random.randint(10, max(11, width - bw - 10))
                by = random.randint(height // 6, max(height // 6 + 1, height - bh - 10))

                label = draw_synthetic_bin(img, bx, by, bw, bh, fill, class_idx)
                labels.append(label)

            # Apply random augmentations
            # Random brightness
            if random.random() > 0.5:
                factor = random.uniform(0.6, 1.4)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)

            # Random blur
            if random.random() > 0.7:
                ksize = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            # Random noise
            if random.random() > 0.6:
                noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
                img = cv2.add(img, noise)

            # Save image and label
            fname = f"{split}_{i:04d}"
            cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), img)

            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        logger.info("  Generated %d %s images", count, split)

    logger.info("✅ Synthetic dataset generated: %d train, %d val, %d test",
                num_train, num_val, num_test)
    return CLASS_NAMES


# ---------------------------------------------------------------------------
# STEP 2C — Create data.yaml
# ---------------------------------------------------------------------------

def create_data_yaml(data_dir: str, class_names: list) -> str:
    """
    Create the YOLOv8 data.yaml configuration file.

    Returns:
        Path to the created data.yaml file.
    """
    yaml_path = os.path.join(PROJECT_ROOT, "data.yaml")

    data_config = {
        "path": data_dir.replace("\\", "/"),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info("✅ data.yaml created at %s", yaml_path)
    logger.info("   Classes (%d): %s", len(class_names), class_names)
    return yaml_path


# ---------------------------------------------------------------------------
# STEP 3 — Model Training
# ---------------------------------------------------------------------------

def train_model(data_yaml: str, epochs: int = 50, retrain: bool = False) -> dict:
    """
    Fine-tune YOLOv8n on the garbage detection dataset.

    Args:
        data_yaml:  Path to data.yaml
        epochs:     Number of training epochs
        retrain:    If True, use tuned hyperparameters (second attempt)

    Returns:
        Dictionary with training metrics.
    """
    from ultralytics import YOLO

    logger.info("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    # Training hyperparameters
    train_args = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=10,
        name="garbage_detector" if not retrain else "garbage_detector_v2",
        augment=True,
        degrees=10.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=os.path.join(PROJECT_ROOT, "runs"),
        exist_ok=True,
        verbose=True,
    )

    # If retrain — adjust hyperparameters for better convergence
    if retrain:
        logger.info("🔧 Retraining with tuned hyperparameters...")
        train_args.update(dict(
            epochs=epochs + 30,
            lr0=0.005,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5.0,
            warmup_momentum=0.8,
            box=7.5,
            cls=1.0,
            dfl=1.5,
            mosaic=0.8,
            mixup=0.15,
            copy_paste=0.1,
        ))

    logger.info("🚀 Starting training — %d epochs...", train_args["epochs"])
    results = model.train(**train_args)

    return model, results


def validate_model(model, data_yaml: str) -> dict:
    """
    Run validation and extract key metrics.

    Returns:
        Dict with mAP50, mAP50-95, precision, recall.
    """
    logger.info("📊 Running validation...")
    metrics = model.val(data=data_yaml, verbose=True)

    results = {
        "mAP50":     float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
        "mAP50_95":  float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
        "precision":  float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
        "recall":     float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0,
    }

    logger.info("=" * 50)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 50)
    logger.info("  mAP@50:      %.4f", results["mAP50"])
    logger.info("  mAP@50-95:   %.4f", results["mAP50_95"])
    logger.info("  Precision:   %.4f", results["precision"])
    logger.info("  Recall:      %.4f", results["recall"])
    logger.info("=" * 50)

    return results


def save_best_model(model):
    """Copy the best model weights to models/best.pt."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, "best.pt")

    # Find the best.pt from the training run
    best_path = None
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    for root, dirs, files in os.walk(runs_dir):
        if "best.pt" in files:
            candidate = os.path.join(root, "best.pt")
            # Pick the most recently modified one
            if best_path is None or os.path.getmtime(candidate) > os.path.getmtime(best_path):
                best_path = candidate

    if best_path and os.path.exists(best_path):
        shutil.copy2(best_path, dest)
        logger.info("✅ Best model saved to %s", dest)
    else:
        # Fallback — export current model
        logger.warning("⚠ best.pt not found in runs/, saving current model state")
        model.save(dest)
        logger.info("✅ Model saved to %s", dest)

    return dest


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full training pipeline."""
    print("\n" + "=" * 60)
    print("  GARBAGE OVERFLOW DETECTION — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # --- Step 1: Ensure directories ---
    base = ensure_directories()
    data_dir = os.path.join(base, "data")
    logger.info("✅ Step 1 complete — Directories created")

    # --- Step 2: Dataset ---
    logger.info("\n📦 Step 2 — Acquiring Dataset...")
    class_names = ["trash_bin", "garbage", "overflow", "lid_open"]

    # Try Roboflow first
    roboflow_ok = download_roboflow_dataset(data_dir)

    if not roboflow_ok:
        # Generate synthetic dataset
        logger.info("🎨 Generating synthetic training dataset...")
        class_names = generate_synthetic_dataset(
            data_dir, num_train=200, num_val=25, num_test=25
        )

    # Create data.yaml
    data_yaml = create_data_yaml(data_dir, class_names)
    logger.info("✅ Step 2 complete — Dataset ready")

    # --- Step 3: Training ---
    logger.info("\n🏋️ Step 3 — Training YOLOv8 Model...")
    model, results = train_model(data_yaml, epochs=50)

    # Validate
    metrics = validate_model(model, data_yaml)

    # Auto-retrain if mAP50 is too low
    if metrics["mAP50"] < 0.5:
        logger.warning("⚠ mAP50 (%.4f) < 0.5 — triggering auto-retrain with tuned hyperparams",
                        metrics["mAP50"])
        model, results = train_model(data_yaml, epochs=50, retrain=True)
        metrics = validate_model(model, data_yaml)

    # Save best model
    model_dest = save_best_model(model)
    logger.info("✅ Step 3 complete — Model trained and saved")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:       {model_dest}")
    print(f"  mAP@50:      {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:   {metrics['mAP50_95']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print("=" * 60 + "\n")

    # Save metrics to a file for the README
    metrics_path = os.path.join(base, "models", "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info("📝 Metrics saved to %s", metrics_path)

    return metrics


if __name__ == "__main__":
    main()
