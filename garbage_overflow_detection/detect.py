"""
detect.py — Real-Time Garbage Overflow Detection (India/Tamil Nadu)
====================================================================
Main detection script that processes webcam or video input, runs YOLOv8
inference on each frame, and simultaneously classifies garbage bins as
'overflow' or 'not_overflow' with distinct visual overlays.

The fine-tuned model (best_india.pt) detects both classes in every frame:
  - not_overflow (GREEN) — normal bins, properly managed waste
  - overflow (RED) — overflowing bins, spilled garbage

Falls back to YOLO-World zero-shot mode if fine-tuned model is not found.

Usage:
    python detect.py --source 0                     # webcam
    python detect.py --source path/to/video.mp4     # video file
    python detect.py --source 0 --headless           # no GUI window (for Flask)
    python detect.py --mode finetune                 # use fine-tuned model
    python detect.py --mode zeroshot                 # use YOLO-World zero-shot

Controls:
    q — Quit
    s — Save screenshot to screenshots/
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fill_estimator import (
    estimate_fill_level, classify_bin, status_color,
    draw_detection, draw_overflow_alert, draw_fill_bar,
    draw_simultaneous_status,
)
from utils import (
    ensure_directories, setup_logger, timestamp_filename,
    init_event_csv, log_event, model_path,
)

logger = setup_logger("detect")


# ---------------------------------------------------------------------------
# Global state for Flask streaming
# ---------------------------------------------------------------------------
class DetectionState:
    """Shared state between the detection loop and Flask dashboard."""
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.total_bins = 0
        self.overflow_count = 0
        self.near_full_count = 0
        self.ok_count = 0
        self.fps = 0.0
        self.running = False

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_stats(self):
        return {
            "total_bins": self.total_bins,
            "overflow_count": self.overflow_count,
            "near_full_count": self.near_full_count,
            "ok_count": self.ok_count,
            "fps": round(self.fps, 1),
        }


# Singleton instance shared with app.py
detection_state = DetectionState()


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_finetune_model(model_file: str = None):
    """
    Load the fine-tuned India model (2-class: not_overflow / overflow).

    Falls back to YOLO-World zero-shot if fine-tuned model not found.

    Returns:
        (model, mode) where mode is 'finetune' or 'zeroshot'
    """
    from ultralytics import YOLO

    # Priority order for fine-tuned model
    finetune_paths = [
        model_file,
        os.path.join(PROJECT_ROOT, "models", "best_india.pt"),
        os.path.join(PROJECT_ROOT, "models", "best.pt"),
    ]

    for path in finetune_paths:
        if path and os.path.exists(path):
            try:
                model = YOLO(path)
                # Verify it's a 2-class model (not_overflow / overflow)
                names = model.names
                if names and len(names) == 2:
                    logger.info("✅ Loaded FINE-TUNED model: %s", path)
                    logger.info("   Classes: %s", names)
                    return model, "finetune"
                else:
                    logger.info("   Model at %s has %d classes (expected 2), trying next...",
                                path, len(names) if names else 0)
            except Exception as e:
                logger.warning("   Failed to load %s: %s", path, str(e))

    # Fallback: YOLO-World zero-shot
    logger.info("⚠ No fine-tuned 2-class model found, falling back to YOLO-World zero-shot")
    return None, "zeroshot"


def load_zeroshot_model():
    """Load YOLO-World zero-shot model with custom classes."""
    from ultralytics import YOLO

    model_file = os.path.join(PROJECT_ROOT, "yolov8s-world.pt")
    if not os.path.exists(model_file):
        model_file = "yolov8s-world.pt"

    logger.info("Loading YOLO-World zero-shot model: %s", model_file)
    model = YOLO(model_file)

    custom_classes = [
        "garbage bin",
        "blue plastic barrel",
        "trash bucket",
        "overflowing garbage bin",
        "garbage spilled on the ground"
    ]
    model.set_classes(custom_classes)
    logger.info("✅ YOLO-World loaded with custom zero-shot classes: %s", custom_classes)

    return model


# ---------------------------------------------------------------------------
# Detection Loop — Fine-Tuned Model (Simultaneous 2-Class)
# ---------------------------------------------------------------------------

def process_frame_finetune(frame, results, model, csv_file, bin_counter):
    """
    Process detections from the fine-tuned 2-class model.

    Both not_overflow (class 0) and overflow (class 1) are detected
    simultaneously in every frame.

    Returns:
        (overflow_bins, ok_bins, near_full_bins, bin_counter)
    """
    overflow_bins = 0
    ok_bins = 0
    near_full_bins = 0
    frame_h, frame_w = frame.shape[:2]

    if results and len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_coords = (x1, y1, x2, y2)

            # Box dimensions for fill estimation
            box_h = y2 - y1
            box_w = x2 - x1

            if cls_id == 1:  # overflow
                status = "OVERFLOW"
                fill_level = 1.0
                overflow_bins += 1

                # Draw with overflow styling (thick red border, pulsing effect)
                draw_detection(frame, box_coords, "OVERFLOW ⚠",
                              confidence, fill_level, is_overflow=True)

            elif cls_id == 0:  # not_overflow
                # Estimate fill level from bounding box geometry
                fill_level = estimate_fill_level(
                    bbox_height=box_h, frame_height=frame_h,
                    bbox_width=box_w, bbox_y=y1,
                    frame_width=frame_w
                )

                status = classify_bin(fill_level)

                if status == "OVERFLOW":
                    # Geometry says overflow but model says not — trust model, cap at NEAR_FULL
                    fill_level = min(fill_level, 0.79)
                    status = "NEAR_FULL"
                    near_full_bins += 1
                elif status == "NEAR_FULL":
                    near_full_bins += 1
                else:
                    ok_bins += 1

                draw_detection(frame, box_coords, "Normal Bin ✓",
                              confidence, fill_level, is_overflow=False)

            # Log event
            bin_counter += 1
            location = f"({int((x1 + x2) / 2)},{int((y1 + y2) / 2)})"
            log_event(csv_file, bin_id=bin_counter, location=location,
                      fill_percent=fill_level * 100, status=status,
                      confidence=confidence)

    return overflow_bins, ok_bins, near_full_bins, bin_counter


# ---------------------------------------------------------------------------
# Detection Loop — Zero-Shot Model (Fallback)
# ---------------------------------------------------------------------------

def process_frame_zeroshot(frame, results, model, csv_file, bin_counter):
    """
    Process detections from YOLO-World zero-shot model.
    Maps semantic classes to overflow/not_overflow status.

    Returns:
        (overflow_bins, ok_bins, near_full_bins, spilled_count, bin_counter)
    """
    overflow_bins = 0
    ok_bins = 0
    near_full_bins = 0
    spilled_count = 0
    frame_h, frame_w = frame.shape[:2]

    if results and len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_coords = (x1, y1, x2, y2)

            if cls_id == 3:  # "overflowing garbage bin"
                status = "OVERFLOW"
                fill_level = 1.0
                overflow_bins += 1
                draw_detection(frame, box_coords, name, confidence,
                              fill_level, is_overflow=True)

            elif cls_id in [0, 1, 2]:  # Normal bins / barrels / buckets
                status = "OK"
                fill_level = 0.5
                ok_bins += 1
                draw_detection(frame, box_coords, name, confidence,
                              fill_level, is_overflow=False)

            elif cls_id == 4:  # "garbage spilled on the ground"
                spilled_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (0, 0, 255), 2)
                cv2.putText(frame, f"Spilled Trash", (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue  # Don't log as a "bin"

            if cls_id != 4:
                bin_counter += 1
                location = f"({int((x1 + x2) / 2)},{int((y1 + y2) / 2)})"
                log_event(csv_file, bin_id=bin_counter, location=location,
                          fill_percent=fill_level * 100, status=status,
                          confidence=confidence)

    return overflow_bins, ok_bins, near_full_bins, spilled_count, bin_counter


# ---------------------------------------------------------------------------
# Main Detection Loop
# ---------------------------------------------------------------------------

def run_detection(source=0, model_file=None, conf_threshold=0.25,
                  headless=False, max_frames=None, csv_path=None,
                  mode="auto"):
    """
    Main detection loop with simultaneous overflow/not_overflow detection.

    Args:
        source:         0 for webcam, or path to video file
        model_file:     Path to YOLO weights (default: auto-detect)
        conf_threshold: Minimum confidence for detections
        headless:       If True, don't show CV2 window (for Flask mode)
        max_frames:     Stop after N frames (for testing), None = infinite
        csv_path:       Path to events CSV, or None for default
        mode:           'auto', 'finetune', or 'zeroshot'
    """
    from ultralytics import YOLO

    # --- Load Model ---
    detection_mode = "zeroshot"  # default

    if mode == "finetune" or mode == "auto":
        model, detection_mode = load_finetune_model(model_file)
        if model is None and mode == "finetune":
            logger.error("❌ Fine-tuned model not found! Run finetune_india.py first.")
            return
        elif model is None:
            detection_mode = "zeroshot"

    if detection_mode == "zeroshot":
        model = load_zeroshot_model()

    logger.info("🔍 Detection mode: %s", detection_mode.upper())

    # --- Open Video Source ---
    source_int = None
    try:
        source_int = int(source)
        if os.name == 'nt':
            cap = cv2.VideoCapture(source_int, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source_int)
        source_name = f"Webcam {source_int}"
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(str(source))
        source_name = str(source)

    if not cap.isOpened():
        logger.error("❌ Cannot open video source: %s", source)
        return

    # Optimize webcam resolution for real-time FPS
    if type(source_int) == int:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Drop buffer to avoid lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info("📹 Video source opened: %s", source_name)

    # --- Setup ---
    csv_file = init_event_csv(csv_path)
    screenshots_dir = os.path.join(PROJECT_ROOT, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    detection_state.running = True
    frame_count = 0
    bin_counter = 0  # Running bin ID counter
    prev_time = time.time()
    fps_smooth = 0.0

    # --- CPU Real-Time Optimizations ---
    frame_skip = 3  # Only run AI on every 3rd frame
    last_results = None

    logger.info("🚀 Detection started — press 'q' to quit, 's' for screenshot")
    logger.info("   Mode: %s | Simultaneous overflow + normal detection",
                detection_mode.upper())

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # If video file, loop back to start for continuous demo
                if source_name != f"Webcam {source}":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1

            # --- FPS Calculation ---
            curr_time = time.time()
            instant_fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            detection_state.fps = fps_smooth

            # --- YOLOv8 Inference (Optimized for Real-Time CPU) ---
            if frame_count % frame_skip == 1 or last_results is None:
                results = model(frame, conf=conf_threshold, imgsz=320, verbose=False)
                last_results = results
            else:
                results = last_results

            # --- Process Detections (mode-specific) ---
            overflow_bins = 0
            near_full_bins = 0
            ok_bins = 0
            spilled_trash_count = 0
            frame_h, frame_w = frame.shape[:2]

            if detection_mode == "finetune":
                overflow_bins, ok_bins, near_full_bins, bin_counter = \
                    process_frame_finetune(frame, results, model, csv_file, bin_counter)
            else:
                overflow_bins, ok_bins, near_full_bins, spilled_trash_count, bin_counter = \
                    process_frame_zeroshot(frame, results, model, csv_file, bin_counter)

            # --- Overflow Alert Banner ---
            if overflow_bins > 0 or spilled_trash_count > 0:
                draw_overflow_alert(frame, count=overflow_bins + spilled_trash_count)

            # --- Simultaneous Status Panel ---
            # Shows both classes side-by-side at the top-right
            total = overflow_bins + near_full_bins + ok_bins
            draw_simultaneous_status(
                frame,
                normal_count=ok_bins + near_full_bins,
                overflow_count=overflow_bins,
                mode=detection_mode
            )

            # --- Update global state ---
            detection_state.total_bins = total
            detection_state.overflow_count = overflow_bins
            detection_state.near_full_count = near_full_bins
            detection_state.ok_count = ok_bins

            # --- FPS Counter ---
            fps_text = f"FPS: {fps_smooth:.1f}"
            cv2.putText(frame, fps_text, (10, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # --- Source info ---
            mode_label = "FINE-TUNED" if detection_mode == "finetune" else "ZERO-SHOT"
            cv2.putText(frame, f"[{mode_label}] Source: {source_name}", (10, frame_h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            # --- Stats bar at bottom ---
            stats_text = f"Total: {total} | OK: {ok_bins} | Near Full: {near_full_bins} | Overflow: {overflow_bins}"
            cv2.putText(frame, stats_text, (frame_w - 500, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # --- Update shared frame for Flask ---
            detection_state.update_frame(frame)

            # --- Display (if not headless) ---
            if not headless:
                cv2.imshow("Garbage Overflow Detection — India", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Quit requested")
                    break
                elif key == ord('s'):
                    fname = f"screenshot_{timestamp_filename()}.png"
                    fpath = os.path.join(screenshots_dir, fname)
                    cv2.imwrite(fpath, frame)
                    logger.info("📸 Screenshot saved: %s", fpath)

            # --- Max frames limit (for testing) ---
            if max_frames is not None and frame_count >= max_frames:
                logger.info("Reached max frames (%d), stopping.", max_frames)
                break

    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    finally:
        detection_state.running = False
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("✅ Detection stopped. Processed %d frames.", frame_count)
        logger.info("📝 Events logged to: %s", csv_file)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Garbage Overflow Detection — Real-time YOLOv8 inference (India/TN)"
    )
    # Auto-detect best default source
    default_source = "test_video.mp4"
    india_video = os.path.join(PROJECT_ROOT, "sample_india_video.mp4")
    if os.path.exists(india_video):
        default_source = "sample_india_video.mp4"

    parser.add_argument("--source", type=str, default=default_source,
                        help="Video source: path to video file or webcam index")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO model weights (default: auto-detect)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confidence threshold (default: 0.10)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window (for Flask streaming)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (for testing)")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "finetune", "zeroshot"],
                        help="Detection mode: auto (try finetune first), finetune, zeroshot")
    return parser.parse_args()


if __name__ == "__main__":
    ensure_directories()
    args = parse_args()

    print("\n" + "=" * 60)
    print("  GARBAGE OVERFLOW DETECTION — INDIA/TAMIL NADU")
    print("  Simultaneous Overflow + Normal Bin Detection")
    print("=" * 60 + "\n")

    run_detection(
        source=args.source,
        model_file=args.model,
        conf_threshold=args.conf,
        headless=args.headless,
        max_frames=args.max_frames,
        mode=args.mode,
    )
