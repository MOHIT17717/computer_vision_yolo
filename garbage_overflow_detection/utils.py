"""
utils.py — Shared Utility Functions
====================================
Helper functions used across the Garbage Overflow Detection system:
  - Directory creation & validation
  - Logging setup
  - Timestamp formatting
  - CSV logging for overflow events
"""

import os
import csv
import logging
from datetime import datetime


# ---------------------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------------------

def ensure_directories():
    """
    Create all required project directories if they don't already exist.
    Called at startup by any module that needs the folder structure.
    """
    dirs = [
        "data",
        "data/train/images", "data/train/labels",
        "data/valid/images", "data/valid/labels",
        "data/test/images",  "data/test/labels",
        "models",
        "logs",
        "static",
        "templates",
    ]
    base = os.path.dirname(os.path.abspath(__file__))
    for d in dirs:
        path = os.path.join(base, d)
        os.makedirs(path, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "garbage_detection", level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with both console and file handlers.
    Log file is saved to logs/detection.log.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Reconfigure stdout to utf-8 if needed to fix Windows emoji errors
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "detection.log"), encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Timestamp Helpers
# ---------------------------------------------------------------------------

def timestamp_now() -> str:
    """Return current timestamp as a human-readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_filename() -> str:
    """Return current timestamp safe for use in filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# CSV Event Logging
# ---------------------------------------------------------------------------

CSV_HEADER = ["timestamp", "bin_id", "location", "fill_percent", "status", "confidence"]


def init_event_csv(csv_path: str = None) -> str:
    """
    Initialize the event CSV file with headers if it doesn't exist.
    Returns the absolute path to the CSV file.
    """
    if csv_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, "logs", "events.csv")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    return csv_path


def log_event(csv_path: str, bin_id: int, location: str,
              fill_percent: float, status: str, confidence: float):
    """
    Append a single overflow / detection event row to the CSV log.

    Args:
        csv_path:      Path to events.csv
        bin_id:        Unique ID for the detected bin in this frame
        location:      Description string (e.g. "frame_center")
        fill_percent:  Estimated fill level 0-100
        status:        OK / NEAR_FULL / OVERFLOW
        confidence:    Model detection confidence 0-1
    """
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp_now(),
            bin_id,
            location,
            f"{fill_percent:.1f}",
            status,
            f"{confidence:.3f}",
        ])


# ---------------------------------------------------------------------------
# Misc Helpers
# ---------------------------------------------------------------------------

def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def model_path(filename: str = "best.pt") -> str:
    """Return the full path to a model weights file in models/."""
    return os.path.join(get_project_root(), "models", filename)


if __name__ == "__main__":
    # Quick self-test
    base = ensure_directories()
    logger = setup_logger()
    logger.info("✅ utils.py self-test passed. Project root: %s", base)
    csv_file = init_event_csv()
    log_event(csv_file, bin_id=0, location="test", fill_percent=75.0,
              status="NEAR_FULL", confidence=0.85)
    logger.info("✅ CSV event logging works. File: %s", csv_file)
