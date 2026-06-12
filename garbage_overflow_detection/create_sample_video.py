"""
create_sample_video.py — Sample Demo Video Generator
=====================================================
Creates a realistic demo video from India/Tamil Nadu garbage bin reference
images by applying Ken Burns effects (slow pan/zoom), transitions, and
labels to simulate real surveillance footage.

The video alternates between overflow and non-overflow scenes so the
detection system can demonstrate simultaneous classification.

Output: sample_india_video.mp4 (~30-60 seconds, 640x480, 24fps)

Usage:
    python create_sample_video.py
"""

import os
import sys
import random
import glob
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils import setup_logger

logger = setup_logger("video_gen")

# Video settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS = 24
SCENE_DURATION_SEC = 3  # Seconds per scene
TRANSITION_FRAMES = 12  # Frames for cross-fade transition


def ken_burns_effect(img: np.ndarray, frame_idx: int, total_frames: int,
                     direction: str = "zoom_in") -> np.ndarray:
    """
    Apply Ken Burns effect (slow pan/zoom) to a static image.

    Args:
        img:           Source image (should be larger than output resolution)
        frame_idx:     Current frame index within this scene
        total_frames:  Total frames for this scene
        direction:     "zoom_in", "zoom_out", "pan_left", "pan_right"

    Returns:
        Cropped/transformed frame at VIDEO_WIDTH x VIDEO_HEIGHT
    """
    h, w = img.shape[:2]
    progress = frame_idx / max(total_frames - 1, 1)

    if direction == "zoom_in":
        # Start wide, end tight
        start_scale = 0.7
        end_scale = 1.0
        scale = start_scale + (end_scale - start_scale) * progress

        crop_w = int(w / scale)
        crop_h = int(h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    elif direction == "zoom_out":
        start_scale = 1.0
        end_scale = 0.75
        scale = start_scale + (end_scale - start_scale) * progress

        crop_w = int(w / scale)
        crop_h = int(h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    elif direction == "pan_left":
        crop_w = int(w * 0.8)
        crop_h = h
        x = int((w - crop_w) * (1 - progress))
        y = 0

    elif direction == "pan_right":
        crop_w = int(w * 0.8)
        crop_h = h
        x = int((w - crop_w) * progress)
        y = 0

    else:
        crop_w, crop_h = w, h
        x, y = 0, 0

    # Clamp
    x = max(0, min(x, w - crop_w))
    y = max(0, min(y, h - crop_h))
    crop_w = min(crop_w, w - x)
    crop_h = min(crop_h, h - y)

    if crop_w <= 0 or crop_h <= 0:
        return cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))

    cropped = img[y:y + crop_h, x:x + crop_w]
    return cv2.resize(cropped, (VIDEO_WIDTH, VIDEO_HEIGHT))


def cross_fade(frame_a: np.ndarray, frame_b: np.ndarray,
               progress: float) -> np.ndarray:
    """Cross-fade between two frames."""
    alpha = min(1.0, max(0.0, progress))
    return cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)


def add_scene_label(frame: np.ndarray, label: str, class_id: int) -> np.ndarray:
    """Add a label overlay to identify the scene type."""
    h, w = frame.shape[:2]

    if class_id == 0:
        color = (0, 200, 0)  # Green for normal
        status_text = "NOT OVERFLOWING"
    else:
        color = (0, 0, 255)  # Red for overflow
        status_text = "OVERFLOWING"

    # Background bar
    bar_h = 35
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status text
    cv2.putText(frame, f"Scene: {status_text}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Location label
    locations = [
        "Chennai, Tamil Nadu", "Coimbatore, Tamil Nadu",
        "Madurai, Tamil Nadu", "Trichy, Tamil Nadu",
        "Salem, Tamil Nadu", "Bangalore, Karnataka",
        "Mumbai, Maharashtra", "Delhi NCR",
    ]
    loc = random.choice(locations)
    cv2.putText(frame, loc, (w - 250, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def generate_synthetic_scene(class_id: int) -> np.ndarray:
    """
    Generate a synthetic India-style garbage bin scene as a fallback
    when no downloaded images are available.
    """
    # Import from prepare_india_dataset
    try:
        from prepare_india_dataset import generate_india_synthetic_bin
        img, _ = generate_india_synthetic_bin(
            width=VIDEO_WIDTH + 100,
            height=VIDEO_HEIGHT + 80,
            class_id=class_id
        )
        return img
    except ImportError:
        # Manual fallback
        img = np.zeros((VIDEO_HEIGHT + 80, VIDEO_WIDTH + 100, 3), dtype=np.uint8)
        img[:] = (60, 80, 100) if class_id == 0 else (40, 40, 60)

        # Draw a simple bin
        bx, by = 200, 100
        bw, bh = 200, 280
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 100, 0), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 30, 30), 3)

        if class_id == 1:  # Overflow
            for i in range(15):
                ox = bx + random.randint(-30, bw)
                oy = by - random.randint(5, 80)
                cv2.rectangle(img, (ox, oy), (ox + 30, oy + 15),
                              (random.randint(50, 180), random.randint(40, 150), random.randint(30, 120)), -1)

        return img


def create_sample_video(output_path: str = None):
    """
    Create a sample demo video showing India/TN garbage bin scenes.

    The video alternates between normal and overflowing bins to demonstrate
    simultaneous detection of both classes.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "sample_india_video.mp4")

    logger.info("🎬 Generating sample India demo video...")

    # Collect source images
    raw_dir = os.path.join(PROJECT_ROOT, "data", "india_raw")
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "india_bins")

    # Find all available images
    image_sources = []

    # Check downloaded raw images
    if os.path.exists(raw_dir):
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                fpath = os.path.join(raw_dir, f)
                # Determine class from filename (india_0_xxx = not_overflow, india_1_xxx = overflow)
                cls_id = 1 if "_1_" in f else 0
                image_sources.append((fpath, cls_id))

    # Check dataset train images
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    if os.path.exists(train_img_dir):
        for f in sorted(os.listdir(train_img_dir))[:20]:  # Limit to 20
            if f.endswith(('.jpg', '.png', '.jpeg')):
                fpath = os.path.join(train_img_dir, f)
                # Read label to determine class
                lbl_file = os.path.join(dataset_dir, "train", "labels",
                                        f.replace('.jpg', '.txt').replace('.png', '.txt'))
                cls_id = 0
                if os.path.exists(lbl_file):
                    with open(lbl_file, 'r') as lf:
                        first_line = lf.readline().strip()
                        if first_line:
                            cls_id = int(first_line.split()[0])
                image_sources.append((fpath, cls_id))

    # If no images found, generate synthetic ones
    if len(image_sources) < 6:
        logger.info("  Generating synthetic scenes for video...")
        for i in range(10):
            cls_id = i % 2
            img = generate_synthetic_scene(cls_id)
            fpath = os.path.join(PROJECT_ROOT, "data", "india_raw", f"synthetic_video_{cls_id}_{i}.jpg")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            cv2.imwrite(fpath, img)
            image_sources.append((fpath, cls_id))

    # Ensure we have both classes represented
    class_0_imgs = [x for x in image_sources if x[1] == 0]
    class_1_imgs = [x for x in image_sources if x[1] == 1]

    if not class_0_imgs:
        for i in range(5):
            img = generate_synthetic_scene(0)
            fpath = os.path.join(PROJECT_ROOT, "data", "india_raw", f"synth_normal_{i}.jpg")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            cv2.imwrite(fpath, img)
            class_0_imgs.append((fpath, 0))

    if not class_1_imgs:
        for i in range(5):
            img = generate_synthetic_scene(1)
            fpath = os.path.join(PROJECT_ROOT, "data", "india_raw", f"synth_overflow_{i}.jpg")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            cv2.imwrite(fpath, img)
            class_1_imgs.append((fpath, 1))

    # Interleave classes for the video
    scenes = []
    max_scenes = min(len(class_0_imgs), len(class_1_imgs), 8)
    for i in range(max_scenes):
        scenes.append(class_0_imgs[i % len(class_0_imgs)])
        scenes.append(class_1_imgs[i % len(class_1_imgs)])

    # Add a few extra scenes
    remaining = class_0_imgs[max_scenes:] + class_1_imgs[max_scenes:]
    random.shuffle(remaining)
    scenes.extend(remaining[:4])

    logger.info("  Video will have %d scenes (%d normal + %d overflow)",
                len(scenes),
                sum(1 for _, c in scenes if c == 0),
                sum(1 for _, c in scenes if c == 1))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    if not out.isOpened():
        logger.error("❌ Could not create video writer for %s", output_path)
        return None

    frames_per_scene = SCENE_DURATION_SEC * FPS
    directions = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
    total_frames_written = 0

    for scene_idx, (img_path, cls_id) in enumerate(scenes):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = generate_synthetic_scene(cls_id)

        # Ensure image is large enough for Ken Burns
        min_dim = max(VIDEO_WIDTH, VIDEO_HEIGHT) + 100
        if img.shape[0] < min_dim or img.shape[1] < min_dim:
            scale = max(min_dim / img.shape[0], min_dim / img.shape[1]) * 1.2
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        direction = directions[scene_idx % len(directions)]

        # Write scene frames with Ken Burns effect
        for frame_idx in range(frames_per_scene):
            frame = ken_burns_effect(img, frame_idx, frames_per_scene, direction)
            frame = add_scene_label(frame, f"Scene {scene_idx + 1}", cls_id)

            # Cross-fade transition at the end of each scene (except last)
            if scene_idx < len(scenes) - 1 and frame_idx >= frames_per_scene - TRANSITION_FRAMES:
                # Pre-load next scene's first frame for transition
                next_img_path, next_cls_id = scenes[scene_idx + 1]
                next_img = cv2.imread(next_img_path)
                if next_img is None:
                    next_img = generate_synthetic_scene(next_cls_id)

                if next_img.shape[0] < min_dim or next_img.shape[1] < min_dim:
                    scale = max(min_dim / next_img.shape[0], min_dim / next_img.shape[1]) * 1.2
                    next_img = cv2.resize(next_img, (int(next_img.shape[1] * scale), int(next_img.shape[0] * scale)))

                next_direction = directions[(scene_idx + 1) % len(directions)]
                next_frame = ken_burns_effect(next_img, 0, frames_per_scene, next_direction)
                next_frame = add_scene_label(next_frame, f"Scene {scene_idx + 2}", next_cls_id)

                fade_progress = (frame_idx - (frames_per_scene - TRANSITION_FRAMES)) / TRANSITION_FRAMES
                frame = cross_fade(frame, next_frame, fade_progress)

            out.write(frame)
            total_frames_written += 1

    # Add title card at the beginning
    # We'll prepend it by recreating the video (simpler approach: just note it)
    out.release()

    duration_sec = total_frames_written / FPS
    logger.info("✅ Sample video created: %s", output_path)
    logger.info("   Duration: %.1f seconds | Frames: %d | FPS: %d",
                duration_sec, total_frames_written, FPS)
    logger.info("   Scenes: %d (alternating overflow / not overflow)", len(scenes))

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SAMPLE INDIA DEMO VIDEO GENERATOR")
    print("=" * 60 + "\n")

    video_path = create_sample_video()

    if video_path:
        print(f"\n✅ Video saved to: {video_path}")
        print("   Run detection with: python detect.py --source sample_india_video.mp4")
    else:
        print("\n❌ Video generation failed")
