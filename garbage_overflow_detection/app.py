"""
app.py — Flask Web Dashboard for Garbage Overflow Detection
============================================================
Provides a dark-themed web dashboard with:
  • Live MJPEG video stream from the detection feed
  • Summary stats cards (Total, Overflow, Near Full, OK)
  • Recent overflow events table (from logs/events.csv)
  • Auto-refresh every 5 seconds
  • REST API for stats

Routes:
    /              — Dashboard HTML page
    /video_feed    — MJPEG live video stream
    /api/stats     — JSON stats endpoint

Usage:
    python app.py                           # Start dashboard only
    python app.py --with-detection           # Start dashboard + detection
    python app.py --source path/to/video.mp4 # Dashboard + detection on video
"""

import os
import sys
import csv
import threading
import argparse
import cv2

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, Response, jsonify
from utils import ensure_directories, setup_logger, get_project_root

logger = setup_logger("dashboard")

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------

app = Flask(__name__,
            template_folder=os.path.join(PROJECT_ROOT, "templates"),
            static_folder=os.path.join(PROJECT_ROOT, "static"))


# ---------------------------------------------------------------------------
# Video Streaming
# ---------------------------------------------------------------------------

def generate_frames():
    """
    Generator that yields MJPEG frames for the /video_feed route.
    Reads from the shared DetectionState in detect.py.
    """
    from detect import detection_state

    while True:
        frame = detection_state.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
        else:
            # No frame available — send a placeholder
            placeholder = create_placeholder_frame()
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')

        # ~30 FPS max for the stream
        import time
        time.sleep(0.033)


def create_placeholder_frame():
    """Create a dark placeholder frame when no detection is running."""
    frame = __import__('numpy').zeros((480, 640, 3), dtype=__import__('numpy').uint8)
    frame[:] = (30, 30, 30)
    cv2.putText(frame, "Waiting for detection feed...",
                (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (100, 100, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, "Start detect.py or use --with-detection flag",
                (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 80, 80), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Read Events from CSV
# ---------------------------------------------------------------------------

def read_recent_events(max_rows: int = 50) -> list:
    """
    Read the most recent events from logs/events.csv.

    Returns:
        List of dicts, newest first.
    """
    csv_path = os.path.join(PROJECT_ROOT, "logs", "events.csv")
    events = []

    if not os.path.exists(csv_path):
        return events

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(row)
    except Exception as e:
        logger.error("Error reading events CSV: %s", e)
        return []

    # Return newest first, limited to max_rows
    return list(reversed(events[-max_rows:]))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Live dashboard page."""
    from detect import detection_state
    stats = detection_state.get_stats()
    events = read_recent_events(max_rows=30)
    return render_template("dashboard.html", stats=stats, events=events)


@app.route("/video_feed")
def video_feed():
    """MJPEG streaming endpoint."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    """JSON API endpoint for detection statistics."""
    from detect import detection_state
    stats = detection_state.get_stats()

    # Also include recent events count
    events = read_recent_events(max_rows=100)
    overflow_events = [e for e in events if e.get("status") == "OVERFLOW"]

    stats["recent_events_count"] = len(events)
    stats["recent_overflow_events"] = len(overflow_events)
    stats["events"] = events[:100]  # Last 100 events in API

    return jsonify(stats)


# ---------------------------------------------------------------------------
# Background Detection Thread
# ---------------------------------------------------------------------------

def start_detection_thread(source, model_file=None):
    """Start detection in a background thread."""
    from detect import run_detection

    def _run():
        logger.info("🚀 Starting detection in background thread...")
        run_detection(
            source=source,
            model_file=model_file,
            headless=True,  # No GUI when running with Flask
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Detection thread started")
    return thread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Garbage Overflow Detection Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Flask host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Flask port (default: 5000)")
    parser.add_argument("--with-detection", action="store_true",
                        help="Also start the detection loop in a background thread")
    parser.add_argument("--source", type=str, default="test_video.mp4",
                        help="Video source for detection (default: test_video.mp4)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights")
    args = parser.parse_args()

    ensure_directories()

    print("\n" + "=" * 60)
    print("  GARBAGE OVERFLOW DETECTION — WEB DASHBOARD")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:{args.port}")
    print(f"  Video:     http://localhost:{args.port}/video_feed")
    print(f"  API:       http://localhost:{args.port}/api/stats")
    print("=" * 60 + "\n")

    # Optionally start detection in background
    if args.with_detection:
        start_detection_thread(source=args.source, model_file=args.model)

    # Start Flask
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
