"""
prepare_india_dataset.py — India/Tamil Nadu Garbage Bin Dataset Preparation
============================================================================
Downloads real-world reference images of Indian garbage bins (overflowing
and normal), auto-annotates them with YOLO-World as a teacher model, applies
India-specific augmentations, and prepares a YOLO-format dataset for
fine-tuning.

Classes:
    0: not_overflow  — normal / properly managed garbage bin
    1: overflow      — overflowing / spilling garbage bin

Usage:
    python prepare_india_dataset.py
"""

import os
import sys
import random
import shutil
import urllib.request
import ssl
import yaml
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils import setup_logger

logger = setup_logger("dataset_prep")

# ---------------------------------------------------------------------------
# India/Tamil Nadu Reference Image URLs (Public domain / Creative Commons)
# ---------------------------------------------------------------------------
# These are publicly accessible images depicting Indian garbage bins and
# street waste scenarios. Each tuple: (url, class_id, description)
#   class 0 = not_overflow (normal bin)
#   class 1 = overflow (overflowing / spilled)

INDIA_IMAGE_SOURCES = [
    # --- NOT OVERFLOW (Class 0): Clean / Normal Indian bins ---
    ("https://images.unsplash.com/photo-1610972449466-8b56db0be95d?w=640", 0, "Indian green municipal bin clean"),
    ("https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=640", 0, "Clean waste bin on street"),
    ("https://images.unsplash.com/photo-1595278474246-17b189ff70a5?w=640", 0, "Municipal garbage bin normal"),
    ("https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?w=640", 0, "Recycle bin normal state"),
    ("https://images.pexels.com/photos/2547565/pexels-photo-2547565.jpeg?w=640", 0, "Street garbage bin India"),
    ("https://images.pexels.com/photos/3174350/pexels-photo-3174350.jpeg?w=640", 0, "Public waste bin clean"),
    ("https://images.pexels.com/photos/6964129/pexels-photo-6964129.jpeg?w=640", 0, "Garbage collection bin"),
    ("https://images.unsplash.com/photo-1604187351574-c75ca79f5807?w=640", 0, "Dustbin on sidewalk"),

    # --- OVERFLOW (Class 1): Overflowing / messy Indian waste ---
    ("https://images.unsplash.com/photo-1605600659873-d808a13e4d2a?w=640", 1, "Overflowing garbage pile"),
    ("https://images.unsplash.com/photo-1530587191325-3db32d826c18?w=640", 1, "Street waste overflow India"),
    ("https://images.unsplash.com/photo-1567963989917-93a1a5df9754?w=640", 1, "Garbage overflow on road"),
    ("https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=640", 1, "Trash pile overflowing"),
    ("https://images.pexels.com/photos/2768961/pexels-photo-2768961.jpeg?w=640", 1, "Overflowing waste bin"),
    ("https://images.pexels.com/photos/3962294/pexels-photo-3962294.jpeg?w=640", 1, "Garbage dump pile overflow"),
    ("https://images.pexels.com/photos/2409022/pexels-photo-2409022.jpeg?w=640", 1, "Overflowing municipal bin India"),
    ("https://images.unsplash.com/photo-1550989460-0adf9ea622e2?w=640", 1, "Garbage pile street India"),
]


# ---------------------------------------------------------------------------
# Download Images
# ---------------------------------------------------------------------------

def download_images(dest_dir: str) -> list:
    """
    Download reference images from public URLs.

    Returns:
        List of (filepath, class_id, description) for successfully downloaded images.
    """
    os.makedirs(dest_dir, exist_ok=True)
    downloaded = []

    # Create SSL context that doesn't verify (for corporate proxies)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for i, (url, cls_id, desc) in enumerate(INDIA_IMAGE_SOURCES):
        fname = f"india_{cls_id}_{i:03d}.jpg"
        fpath = os.path.join(dest_dir, fname)

        if os.path.exists(fpath):
            logger.info("  [%d/%d] Already exists: %s", i + 1, len(INDIA_IMAGE_SOURCES), fname)
            downloaded.append((fpath, cls_id, desc))
            continue

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                img_data = resp.read()

            with open(fpath, 'wb') as f:
                f.write(img_data)

            # Verify it's a valid image
            img = cv2.imread(fpath)
            if img is None:
                os.remove(fpath)
                logger.warning("  [%d/%d] Invalid image, skipped: %s", i + 1, len(INDIA_IMAGE_SOURCES), url)
                continue

            downloaded.append((fpath, cls_id, desc))
            logger.info("  [%d/%d] Downloaded: %s (%s)", i + 1, len(INDIA_IMAGE_SOURCES), fname, desc)

        except Exception as e:
            logger.warning("  [%d/%d] Failed: %s — %s", i + 1, len(INDIA_IMAGE_SOURCES), url, str(e))

    logger.info("Downloaded %d / %d images", len(downloaded), len(INDIA_IMAGE_SOURCES))
    return downloaded


# ---------------------------------------------------------------------------
# Auto-Annotation with YOLO-World (Teacher Model)
# ---------------------------------------------------------------------------

def auto_annotate_with_yolo_world(image_path: str, model, class_id: int) -> list:
    """
    Use YOLO-World as a teacher model to find garbage bin bounding boxes.
    Falls back to center-crop annotation if no detections found.

    Returns:
        List of YOLO-format labels: [class_id, cx, cy, w, h] (normalised)
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    labels = []

    try:
        results = model(img, conf=0.15, imgsz=640, verbose=False)

        if results and len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())

                # Convert to YOLO format (normalised center x, y, width, height)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                # Clamp
                cx = max(0.001, min(0.999, cx))
                cy = max(0.001, min(0.999, cy))
                bw = max(0.01, min(0.999, bw))
                bh = max(0.01, min(0.999, bh))

                labels.append([class_id, cx, cy, bw, bh])

            return labels
    except Exception as e:
        logger.debug("YOLO-World annotation failed for %s: %s", image_path, str(e))

    # Fallback: use the central region as the primary object
    # Randomize slightly for variety
    cx = 0.5 + random.uniform(-0.05, 0.05)
    cy = 0.5 + random.uniform(-0.05, 0.05)
    bw = random.uniform(0.5, 0.8)
    bh = random.uniform(0.5, 0.85)
    labels.append([class_id, cx, cy, bw, bh])

    return labels


# ---------------------------------------------------------------------------
# India-Specific Augmentations
# ---------------------------------------------------------------------------

def apply_india_augmentation(img: np.ndarray, aug_type: str) -> np.ndarray:
    """Apply India-specific augmentations to simulate real conditions."""
    h, w = img.shape[:2]
    result = img.copy()

    if aug_type == "warm_sunlight":
        # Warm color shift — Indian afternoon sun
        warm = np.full_like(result, (0, 30, 60), dtype=np.uint8)
        result = cv2.add(result, warm)

    elif aug_type == "dust_haze":
        # Dust/haze overlay common in Indian streets
        haze = np.full_like(result, (180, 190, 200), dtype=np.uint8)
        alpha = random.uniform(0.1, 0.3)
        result = cv2.addWeighted(result, 1 - alpha, haze, alpha, 0)

    elif aug_type == "harsh_shadow":
        # Harsh midday shadows
        mask = np.zeros((h, w), dtype=np.uint8)
        # Random diagonal shadow
        pts = np.array([
            [random.randint(0, w // 2), 0],
            [w, 0],
            [w, h],
            [random.randint(w // 3, w), h]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        shadow = result.copy()
        shadow = (shadow * 0.5).astype(np.uint8)
        result = np.where(mask[:, :, np.newaxis] > 0, shadow, result)

    elif aug_type == "evening_light":
        # Golden hour / evening lighting
        warm = np.full_like(result, (0, 20, 50), dtype=np.uint8)
        result = cv2.add(result, warm)
        result = (result * 0.75).astype(np.uint8)

    elif aug_type == "rain_effect":
        # Light rain streaks
        rain = np.zeros_like(result)
        for _ in range(100):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(10, 30)
            cv2.line(rain, (x, y), (x + 1, y + length), (200, 200, 200), 1)
        result = cv2.add(result, rain)

    elif aug_type == "noise":
        # Sensor noise (low-quality phone cameras)
        noise = np.random.randint(0, 30, result.shape, dtype=np.uint8)
        result = cv2.add(result, noise)

    elif aug_type == "blur":
        # Slight motion blur
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    elif aug_type == "brightness_up":
        factor = random.uniform(1.1, 1.5)
        result = np.clip(result * factor, 0, 255).astype(np.uint8)

    elif aug_type == "brightness_down":
        factor = random.uniform(0.5, 0.8)
        result = np.clip(result * factor, 0, 255).astype(np.uint8)

    return result


def augment_image(img: np.ndarray, num_augmentations: int = 1) -> list:
    """
    Generate augmented versions of an image.

    Returns:
        List of augmented images.
    """
    aug_types = [
        "warm_sunlight", "dust_haze", "harsh_shadow", "evening_light",
        "rain_effect", "noise", "blur", "brightness_up", "brightness_down",
    ]

    augmented = []
    chosen = random.sample(aug_types, min(num_augmentations, len(aug_types)))

    for aug_type in chosen:
        aug_img = apply_india_augmentation(img, aug_type)

        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 1)

        # Random resize within reasonable range
        scale = random.uniform(0.8, 1.2)
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        aug_img = cv2.resize(aug_img, (new_w, new_h))

        augmented.append((aug_img, aug_type))

    return augmented


# ---------------------------------------------------------------------------
# Synthetic India Bin Generator (for filling out the dataset)
# ---------------------------------------------------------------------------

def generate_india_synthetic_bin(width: int = 640, height: int = 480,
                                  class_id: int = 0) -> tuple:
    """
    Generate a synthetic image of an Indian-style garbage bin.
    Uses realistic colors and shapes common in Indian municipalities.

    Returns:
        (image, labels_list) where labels_list is YOLO format
    """
    # Indian urban background
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Warm earth tones typical of Indian streets
    bg_colors = [
        (60, 80, 120),   # Warm brown road
        (70, 90, 110),   # Dusty sidewalk
        (80, 100, 90),   # Greenish concrete
        (50, 60, 80),    # Dark road
        (90, 100, 120),  # Sandy
    ]
    bg[:] = random.choice(bg_colors)

    # Add texture
    noise = np.random.randint(0, 20, bg.shape, dtype=np.uint8)
    bg = cv2.add(bg, noise)

    # Draw road/sidewalk
    road_y = int(height * random.uniform(0.5, 0.7))
    cv2.rectangle(bg, (0, road_y), (width, height), (50, 50, 55), -1)

    labels = []
    num_bins = random.randint(1, 3)

    for _ in range(num_bins):
        # Bin dimensions
        bw = random.randint(width // 6, width // 3)
        bh = random.randint(height // 3, int(height * 0.65))
        bx = random.randint(10, max(11, width - bw - 10))
        by = random.randint(height // 5, max(height // 5 + 1, height - bh - 10))

        # Indian municipal bin colors
        bin_colors = [
            (0, 128, 0),     # Green (wet waste)
            (180, 80, 0),    # Blue (dry waste)
            (40, 40, 40),    # Black bin
            (50, 50, 120),   # Brown/maroon
            (80, 80, 80),    # Grey concrete
        ]
        bin_color = random.choice(bin_colors)

        if class_id == 0:  # Normal bin
            # Draw bin body
            cv2.rectangle(bg, (bx, by), (bx + bw, by + bh), bin_color, -1)
            cv2.rectangle(bg, (bx, by), (bx + bw, by + bh), (30, 30, 30), 2)

            # Lid
            lid_h = max(5, bh // 10)
            cv2.rectangle(bg, (bx - 3, by - lid_h), (bx + bw + 3, by), (60, 60, 60), -1)

            # Fill inside (partial)
            fill_level = random.uniform(0.1, 0.5)
            fill_h = int(bh * fill_level)
            fill_color = random.choice([(80, 100, 60), (100, 80, 50), (60, 80, 70)])
            cv2.rectangle(bg, (bx + 3, by + bh - fill_h), (bx + bw - 3, by + bh), fill_color, -1)

            # YOLO label
            cx = (bx + bw / 2) / width
            cy = (by - lid_h + (bh + lid_h) / 2) / height
            nw = (bw + 6) / width
            nh = (bh + lid_h) / height

        else:  # Overflow
            # Draw bin body
            cv2.rectangle(bg, (bx, by), (bx + bw, by + bh), bin_color, -1)
            cv2.rectangle(bg, (bx, by), (bx + bw, by + bh), (30, 30, 30), 2)

            # Overflowing garbage above rim
            overflow_h = random.randint(bh // 5, bh // 2)
            for oy in range(0, overflow_h, 4):
                oc = (
                    random.randint(40, 180),
                    random.randint(40, 150),
                    random.randint(30, 120)
                )
                ow = random.randint(bw // 2, bw + 15)
                ox = bx + (bw - ow) // 2
                cv2.rectangle(bg, (ox, by - oy - 4), (ox + ow, by - oy), oc, -1)

            # Garbage spilling on ground
            for _ in range(random.randint(3, 8)):
                sx = bx + random.randint(-bw // 2, bw)
                sy = by + bh + random.randint(5, 40)
                sw = random.randint(10, 40)
                sh = random.randint(8, 25)
                sc = (
                    random.randint(40, 160),
                    random.randint(40, 130),
                    random.randint(30, 100)
                )
                cv2.rectangle(bg, (sx, sy), (sx + sw, sy + sh), sc, -1)

            # Fill inside (full)
            fill_color = random.choice([(80, 100, 60), (100, 80, 50), (60, 80, 70)])
            cv2.rectangle(bg, (bx + 3, by + 5), (bx + bw - 3, by + bh), fill_color, -1)

            # YOLO label covers bin + overflow + spill
            total_top = by - overflow_h
            total_bottom = by + bh + 45
            cx = (bx + bw / 2) / width
            cy = ((total_top + total_bottom) / 2) / height
            nw = (bw + 30) / width
            nh = (total_bottom - total_top) / height

        # Clamp
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        nw = max(0.01, min(0.999, nw))
        nh = max(0.01, min(0.999, nh))

        labels.append([class_id, cx, cy, nw, nh])

    # Apply random India augmentation
    aug_type = random.choice(["warm_sunlight", "dust_haze", "noise", "brightness_up", "brightness_down"])
    bg = apply_india_augmentation(bg, aug_type)

    return bg, labels


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

def build_dataset(download_dir: str, output_dir: str,
                  target_train: int = 150, target_val: int = 30, target_test: int = 20):
    """
    Build the full India-specific YOLO dataset.

    Steps:
        1. Download real images
        2. Auto-annotate with YOLO-World
        3. Augment to reach target counts
        4. Fill remaining with synthetic images
        5. Split into train/val/test
    """
    # --- Step 1: Download / Find local reference images ---
    logger.info("\n📥 Step 1: Finding India/Tamil Nadu reference images...")
    downloaded = download_images(download_dir)

    # Also scan for any local images already in the raw directory
    # (e.g., AI-generated reference images copied there)
    if os.path.exists(download_dir):
        existing_paths = {d[0] for d in downloaded}  # avoid duplicates
        for f in sorted(os.listdir(download_dir)):
            if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            fpath = os.path.join(download_dir, f)
            if fpath in existing_paths:
                continue

            # Determine class from filename: _0_ = not_overflow, _1_ = overflow
            if "_1_" in f or "overflow" in f.lower():
                cls_id = 1
                desc = f"Local overflow image: {f}"
            else:
                cls_id = 0
                desc = f"Local normal bin image: {f}"

            # Verify it's a valid image
            img_check = cv2.imread(fpath)
            if img_check is not None:
                downloaded.append((fpath, cls_id, desc))
                logger.info("  Found local image: %s (class %d)", f, cls_id)

    logger.info("Total reference images available: %d", len(downloaded))

    if len(downloaded) == 0:
        logger.warning("No images found. Will generate entirely synthetic dataset.")

    # --- Step 2: Auto-annotate ---
    logger.info("\n🏷 Step 2: Auto-annotating with YOLO-World teacher model...")
    annotated_images = []  # (image_array, labels_list, source_desc)

    try:
        from ultralytics import YOLO
        teacher_model = YOLO("yolov8s-world.pt")
        teacher_model.set_classes([
            "garbage bin", "trash can", "dustbin", "waste bin",
            "overflowing garbage", "garbage pile", "trash pile"
        ])
        logger.info("  YOLO-World teacher model loaded")
        has_teacher = True
    except Exception as e:
        logger.warning("  Could not load YOLO-World teacher: %s", str(e))
        logger.info("  Will use fallback center-crop annotations")
        has_teacher = False
        teacher_model = None

    for fpath, cls_id, desc in downloaded:
        img = cv2.imread(fpath)
        if img is None:
            continue

        if has_teacher:
            labels = auto_annotate_with_yolo_world(fpath, teacher_model, cls_id)
        else:
            # Fallback center annotation
            cx = 0.5 + random.uniform(-0.05, 0.05)
            cy = 0.5 + random.uniform(-0.05, 0.05)
            bw = random.uniform(0.5, 0.8)
            bh = random.uniform(0.5, 0.85)
            labels = [[cls_id, cx, cy, bw, bh]]

        annotated_images.append((img, labels, desc))
        logger.info("  Annotated: %s — %d boxes", desc, len(labels))

    # --- Step 3: Augment real images ---
    logger.info("\n🎨 Step 3: Augmenting real images with India-specific effects...")
    augmented_images = []

    for img, labels, desc in annotated_images:
        # Generate 5-8 augmented versions of each real image
        num_augs = random.randint(5, 8)
        aug_results = augment_image(img, num_augs)

        for aug_img, aug_type in aug_results:
            # Labels stay the same (normalised coords) — but flip cx if horizontally flipped
            # Note: flip is handled inside augment_image, labels use normalised coords
            augmented_images.append((aug_img, labels, f"{desc}_{aug_type}"))

    logger.info("  Generated %d augmented images from %d originals",
                len(augmented_images), len(annotated_images))

    # Combine original + augmented
    all_images = annotated_images + augmented_images

    # --- Step 4: Fill with synthetic images ---
    total_target = target_train + target_val + target_test
    num_synthetic_needed = max(0, total_target - len(all_images))

    if num_synthetic_needed > 0:
        logger.info("\n🏗 Step 4: Generating %d synthetic India-style bin images...", num_synthetic_needed)
        for i in range(num_synthetic_needed):
            cls_id = 0 if i % 2 == 0 else 1  # Alternate classes
            img, labels = generate_india_synthetic_bin(
                width=random.choice([480, 640, 800]),
                height=random.choice([360, 480, 600]),
                class_id=cls_id
            )
            all_images.append((img, labels, f"synthetic_{cls_id}_{i}"))

        logger.info("  Total dataset size: %d images", len(all_images))

    # --- Step 5: Shuffle and split ---
    logger.info("\n📂 Step 5: Splitting dataset into train/val/test...")
    random.shuffle(all_images)

    # Ensure balanced classes in each split
    class_0 = [x for x in all_images if x[1][0][0] == 0]
    class_1 = [x for x in all_images if x[1][0][0] == 1]

    def split_list(lst, train_ratio=0.7, val_ratio=0.2):
        n = len(lst)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    c0_train, c0_val, c0_test = split_list(class_0)
    c1_train, c1_val, c1_test = split_list(class_1)

    splits = {
        "train": c0_train + c1_train,
        "valid": c0_val + c1_val,
        "test": c0_test + c1_test,
    }

    for split_name, split_data in splits.items():
        random.shuffle(split_data)

    # --- Save to YOLO format ---
    for split_name, split_data in splits.items():
        img_dir = os.path.join(output_dir, split_name, "images")
        lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for idx, (img, labels, desc) in enumerate(split_data):
            fname = f"india_{split_name}_{idx:04d}"

            # Resize to consistent size for training
            img_resized = cv2.resize(img, (640, 480))
            cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), img_resized)

            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        logger.info("  %s: %d images saved", split_name, len(split_data))

    # --- Create data.yaml ---
    yaml_path = os.path.join(output_dir, "data.yaml")
    data_config = {
        "path": output_dir.replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["not_overflow", "overflow"],
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    # Also update the project-level data.yaml
    project_yaml = os.path.join(PROJECT_ROOT, "data.yaml")
    with open(project_yaml, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info("\n✅ Dataset preparation complete!")
    logger.info("   data.yaml: %s", yaml_path)
    logger.info("   Train: %d | Val: %d | Test: %d",
                len(splits["train"]), len(splits["valid"]), len(splits["test"]))

    return yaml_path, splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  INDIA/TAMIL NADU GARBAGE BIN DATASET PREPARATION")
    print("=" * 60 + "\n")

    download_dir = os.path.join(PROJECT_ROOT, "data", "india_raw")
    output_dir = os.path.join(PROJECT_ROOT, "data", "india_bins")

    yaml_path, splits = build_dataset(
        download_dir=download_dir,
        output_dir=output_dir,
        target_train=150,
        target_val=30,
        target_test=20,
    )

    print("\n" + "=" * 60)
    print("  DATASET READY FOR TRAINING")
    print("=" * 60)
    print(f"  Config:  {yaml_path}")
    print(f"  Train:   {len(splits['train'])} images")
    print(f"  Val:     {len(splits['valid'])} images")
    print(f"  Test:    {len(splits['test'])} images")
    print(f"  Classes: not_overflow (0), overflow (1)")
    print("=" * 60 + "\n")

    return yaml_path


if __name__ == "__main__":
    main()
