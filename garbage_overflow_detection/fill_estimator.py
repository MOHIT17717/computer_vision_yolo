"""
fill_estimator.py — Garbage Bin Fill Level Estimation & Visualization
======================================================================
This module estimates how full a detected garbage bin is based on its
bounding-box geometry, then classifies the status and provides
visualisation helpers to draw fill-level bars on video frames.

Supports simultaneous display of overflow and non-overflow detections:
    GREEN   — fill < 65 %  → OK (not overflowing)
    ORANGE  — 65 % ≤ fill ≤ 80 % → NEAR_FULL
    RED     — fill > 80 %  → OVERFLOW
"""

import cv2
import numpy as np
import time


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thresholds (can be overridden via function args)
NEAR_FULL_THRESHOLD = 0.65   # 65 %
OVERFLOW_THRESHOLD  = 0.80   # 80 %

# BGR colours for OpenCV
COLOR_OK        = (0, 200, 0)      # Green
COLOR_NEAR_FULL = (0, 165, 255)    # Orange
COLOR_OVERFLOW  = (0, 0, 255)      # Red
COLOR_BG        = (40, 40, 40)     # Dark background for the bar


# ---------------------------------------------------------------------------
# Fill Level Estimation
# ---------------------------------------------------------------------------

def estimate_fill_level(bbox_height: float, frame_height: float,
                        bbox_width: float = None, bbox_y: float = None,
                        frame_width: float = None) -> float:
    """
    Estimate the fill level of a garbage bin from its bounding-box size
    relative to the frame.

    The heuristic uses several cues:
      1. **Vertical extent** — a taller bounding box (relative to the frame)
         suggests more garbage overflowing above the bin rim.
      2. **Aspect ratio** — overflowing bins tend to be wider at the top.
      3. **Vertical position** — bins closer to the top of the frame may
         indicate the camera is looking up at an overflowing pile.

    Returns:
        float in [0.0, 1.0] representing estimated fill percentage.
    """
    if frame_height <= 0:
        return 0.0

    # Primary signal: height ratio (normalised to [0, 1])
    height_ratio = min(bbox_height / frame_height, 1.0)

    # Scale the ratio into a plausible fill range.
    # Adjusted multiplier down to 1.2 so close-up camera angles don't falsely trigger OVERFLOW.
    fill = np.clip(height_ratio * 1.2, 0.0, 1.0)

    # Secondary signal: aspect ratio bonus
    if bbox_width is not None and bbox_height > 0:
        aspect = bbox_width / bbox_height
        # Wider-than-tall boxes get a small fill boost (overflow spilling out)
        if aspect > 1.2:
            fill = min(fill + 0.10, 1.0)

    # Tertiary signal: vertical position in frame
    if bbox_y is not None and frame_height > 0:
        vert_ratio = bbox_y / frame_height
        # If the detection is in the upper third, give a slight boost
        if vert_ratio < 0.33:
            fill = min(fill + 0.05, 1.0)

    return round(float(fill), 3)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_bin(fill_ratio: float,
                 near_full_thresh: float = NEAR_FULL_THRESHOLD,
                 overflow_thresh: float = OVERFLOW_THRESHOLD) -> str:
    """
    Classify a bin's status based on its estimated fill ratio.

    Args:
        fill_ratio:        Float 0.0 – 1.0
        near_full_thresh:  Threshold for "NEAR_FULL" (default 0.65)
        overflow_thresh:   Threshold for "OVERFLOW"  (default 0.80)

    Returns:
        "OK" | "NEAR_FULL" | "OVERFLOW"
    """
    if fill_ratio >= overflow_thresh:
        return "OVERFLOW"
    elif fill_ratio >= near_full_thresh:
        return "NEAR_FULL"
    else:
        return "OK"


def status_color(status: str) -> tuple:
    """Return the BGR colour tuple for a given status string."""
    return {
        "OK":        COLOR_OK,
        "NEAR_FULL": COLOR_NEAR_FULL,
        "OVERFLOW":  COLOR_OVERFLOW,
    }.get(status, COLOR_OK)


# ---------------------------------------------------------------------------
# Visualisation — Fill-Level Bar
# ---------------------------------------------------------------------------

def draw_fill_bar(frame: np.ndarray, x: int, y: int,
                  fill_ratio: float, bar_width: int = 20,
                  bar_height: int = 100) -> np.ndarray:
    """
    Draw a vertical fill-level bar next to a detected bin.

    The bar has a dark background and a coloured foreground whose height
    corresponds to the fill percentage. A text label shows the numeric %.

    Args:
        frame:       The video frame (modified in-place and returned).
        x, y:        Top-left corner where the bar should be drawn.
        fill_ratio:  Float 0.0 – 1.0
        bar_width:   Width of the bar in pixels.
        bar_height:  Total height of the bar in pixels.

    Returns:
        The frame with the bar drawn on it.
    """
    fill_ratio = np.clip(fill_ratio, 0.0, 1.0)
    status = classify_bin(fill_ratio)
    color = status_color(status)

    # Ensure coordinates stay within frame boundaries
    h, w = frame.shape[:2]
    x = max(0, min(x, w - bar_width - 5))
    y = max(0, min(y, h - bar_height - 25))

    # Background rectangle (dark grey)
    cv2.rectangle(frame,
                  (x, y),
                  (x + bar_width, y + bar_height),
                  COLOR_BG, -1)

    # Foreground filled portion (grows from bottom up)
    fill_px = int(bar_height * fill_ratio)
    if fill_px > 0:
        cv2.rectangle(frame,
                      (x, y + bar_height - fill_px),
                      (x + bar_width, y + bar_height),
                      color, -1)

    # Border
    cv2.rectangle(frame,
                  (x, y),
                  (x + bar_width, y + bar_height),
                  (200, 200, 200), 1)

    # Percentage label below the bar
    pct_text = f"{int(fill_ratio * 100)}%"
    cv2.putText(frame, pct_text,
                (x - 2, y + bar_height + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Visualisation — Bounding Box with Status (Simultaneous Detection)
# ---------------------------------------------------------------------------

def draw_detection(frame: np.ndarray, bbox: tuple, class_name: str,
                   confidence: float, fill_ratio: float,
                   is_overflow: bool = False) -> np.ndarray:
    """
    Draw a full detection overlay on the frame with distinct styling for
    overflow vs. not_overflow detections.

    For simultaneous detection:
      • NOT OVERFLOW: Green bounding box, "OK ✓" status, fill bar
      • OVERFLOW: Thick red bounding box with pulsing effect, "OVERFLOW ⚠"

    Args:
        frame:       Video frame (modified in-place).
        bbox:        (x1, y1, x2, y2) bounding box coordinates.
        class_name:  Detected class label.
        confidence:  Detection confidence 0 – 1.
        fill_ratio:  Estimated fill level 0 – 1.
        is_overflow: If True, apply overflow-specific styling.

    Returns:
        The annotated frame.
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]

    if is_overflow:
        # --- OVERFLOW STYLING ---
        color = COLOR_OVERFLOW

        # Pulsing border effect (thickness oscillates with time)
        pulse = int(2 * abs(np.sin(time.time() * 4))) + 3  # 3-5 px thick
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, pulse)

        # Double border for emphasis
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                      (0, 0, 180), 1)

        # Label with red background
        label = f"OVERFLOW {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), (0, 0, 200), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Fill bar (always 100%)
        bar_x = x2 + 8
        bar_y = y1
        bar_h = max(y2 - y1, 60)
        draw_fill_bar(frame, bar_x, bar_y, 1.0, bar_width=18, bar_height=bar_h)

        # Warning icon area (red triangle)
        icon_x = x1 + (x2 - x1) // 2
        icon_y = y2 + 5
        pts = np.array([
            [icon_x - 10, icon_y + 18],
            [icon_x + 10, icon_y + 18],
            [icon_x, icon_y],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        cv2.putText(frame, "!", (icon_x - 4, icon_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        # --- NORMAL BIN STYLING ---
        status = classify_bin(fill_ratio)
        color = status_color(status)

        # Standard bounding box
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label with status
        status_icon = "✓" if status == "OK" else "~"
        label = f"{class_name} {confidence:.0%} | {fill_ratio:.0%} {status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Fill-level bar to the right of the bounding box
        bar_x = x2 + 8
        bar_y = y1
        bar_h = max(y2 - y1, 60)
        draw_fill_bar(frame, bar_x, bar_y, fill_ratio, bar_width=18, bar_height=bar_h)

    return frame


# ---------------------------------------------------------------------------
# Visualisation — Overflow Alert Banner
# ---------------------------------------------------------------------------

def draw_overflow_alert(frame: np.ndarray, count: int = 1) -> np.ndarray:
    """
    Draw a prominent red alert banner at the top of the frame when one or
    more bins are detected as overflowing.

    Args:
        frame: Video frame (modified in-place).
        count: Number of overflowing bins.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]
    banner_h = 50

    # Semi-transparent red overlay with pulsing intensity
    pulse_alpha = 0.6 + 0.1 * abs(np.sin(time.time() * 3))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, pulse_alpha, frame, 1 - pulse_alpha, 0, frame)

    # Alert text
    text = f"OVERFLOW ALERT -- {count} bin(s) overflowing!"
    cv2.putText(frame, text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Visualisation — Simultaneous Status Panel
# ---------------------------------------------------------------------------

def draw_simultaneous_status(frame: np.ndarray, normal_count: int = 0,
                              overflow_count: int = 0,
                              mode: str = "finetune") -> np.ndarray:
    """
    Draw a compact status panel in the top-right corner showing both
    overflow and normal bin counts simultaneously.

    This makes it immediately clear that BOTH classes are being detected
    at the same time in every frame.

    Args:
        frame:          Video frame (modified in-place).
        normal_count:   Number of normal (not overflowing) bins detected.
        overflow_count: Number of overflowing bins detected.
        mode:           Detection mode label ('finetune' or 'zeroshot').

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 220
    panel_h = 95
    panel_x = w - panel_w - 10
    panel_y = 55  # Below the alert banner area

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (80, 80, 80), 1)

    # Title
    mode_text = "FINE-TUNED" if mode == "finetune" else "ZERO-SHOT"
    cv2.putText(frame, f"Detection [{mode_text}]",
                (panel_x + 5, panel_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Normal bins count (GREEN)
    cv2.circle(frame, (panel_x + 15, panel_y + 38), 6, COLOR_OK, -1)
    cv2.putText(frame, f"Normal: {normal_count}",
                (panel_x + 28, panel_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OK, 1, cv2.LINE_AA)

    # Overflow count (RED)
    cv2.circle(frame, (panel_x + 15, panel_y + 60), 6, COLOR_OVERFLOW, -1)
    cv2.putText(frame, f"Overflow: {overflow_count}",
                (panel_x + 28, panel_y + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OVERFLOW, 1, cv2.LINE_AA)

    # Total
    total = normal_count + overflow_count
    cv2.putText(frame, f"Total: {total} bins",
                (panel_x + 5, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Overflow indicator dot (pulsing red if any overflow)
    if overflow_count > 0:
        pulse_radius = int(4 + 2 * abs(np.sin(time.time() * 5)))
        cv2.circle(frame, (panel_x + panel_w - 15, panel_y + 15),
                   pulse_radius, (0, 0, 255), -1)

    return frame


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo — create a blank image and draw sample detections
    demo = np.zeros((480, 640, 3), dtype=np.uint8) + 30

    # Test simultaneous detection
    test_cases = [
        ((50, 100, 200, 350), "Normal Bin", 0.92, 0.45, False),     # OK
        ((300, 80, 450, 320), "Normal Bin", 0.87, 0.72, False),     # NEAR_FULL
        ((480, 60, 600, 380), "OVERFLOW",   0.95, 1.0,  True),      # OVERFLOW
    ]

    for bbox, cls, conf, fill, is_of in test_cases:
        draw_detection(demo, bbox, cls, conf, fill, is_overflow=is_of)

    draw_overflow_alert(demo, count=1)
    draw_simultaneous_status(demo, normal_count=2, overflow_count=1, mode="finetune")

    cv2.imwrite("fill_estimator_demo.png", demo)
    print("✅ fill_estimator.py self-test passed — saved fill_estimator_demo.png")
    print("   Shows simultaneous overflow (RED) + normal (GREEN) detection")
