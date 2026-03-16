"""
MIE491 Rover Tracker — CLI entry point.

Usage:
    python main.py                                    # use config.yaml defaults
    python main.py --config config.yaml --video data_/my_video.mp4
    python main.py --camera                           # live USB camera
    python main.py --calibrate                        # interactive homography calibration
    python main.py --display                          # show debug window while processing
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from rover_tracker.data.trial_logger import TrialLogger, TrialSummary
from rover_tracker.events.event_detector import EventDetector
from rover_tracker.perception.homography import HomographyTransform
from rover_tracker.perception.tracker import RoverTracker
from rover_tracker.sensor.video_source import VideoSource
from rover_tracker.state.rover_state import StateHistory


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> None:
    level = cfg.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level, logging.INFO),
    )


def run_calibration(cfg: dict) -> None:
    """Open first video frame and let user click 4 maze corners."""
    sensor_cfg = cfg.get("sensor", {})
    cap = cv2.VideoCapture(sensor_cfg.get("file_path", ""))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not read first frame for calibration.", file=sys.stderr)
        sys.exit(1)

    transform = HomographyTransform.from_four_point_click(frame, cfg)
    print("Calibration complete. Pixel points saved to cfg.")
    print("Homography matrix:\n", transform.get_matrix())
    # Optionally save updated pixel_points back to config.yaml
    config_path = cfg.get("_config_path", "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Updated config saved to {config_path}")


def prompt_rover_name() -> str:
    """Ask the student for their rover/team name before the trial begins."""
    print("\n" + "=" * 50)
    print("  🏁  MIE491 Rover Tracker")
    print("=" * 50)
    raw = input("  Enter your rover / team name: ").strip()
    if not raw:
        raw = "unnamed"
    # Sanitize: keep alphanumerics, hyphens, underscores; replace spaces with _
    import re
    safe = re.sub(r"[^\w\-]", "_", raw).strip("_") or "unnamed"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_id = f"{safe}_{timestamp}"
    print(f"\n  ✅  Trial ID: {trial_id}")
    print("=" * 50 + "\n")
    return trial_id, safe


def run_trial(cfg: dict, display: bool = False,
              trial_id: str | None = None, rover_name: str | None = None) -> TrialSummary:
    """Run the full tracking pipeline for one trial."""
    log = logging.getLogger("main")

    # Embed rover name in config snapshot so the dashboard can display it
    if rover_name:
        cfg.setdefault("trial", {})["rover_name"] = rover_name

    homography = HomographyTransform.from_config(cfg)
    tracker = RoverTracker(cfg, homography)
    detector = EventDetector(cfg)
    history = StateHistory()

    with VideoSource(cfg) as src, TrialLogger(cfg, trial_id=trial_id) as logger:
        meta = src.get_metadata()
        log.info("Video: %dx%d @ %.1f fps, %d frames",
                 meta["width"], meta["height"], meta["fps"], meta["total_frames"])

        trial_dir = logger.open(cfg)
        log.info("Trial output: %s", trial_dir)

        fps = meta["fps"] if meta["fps"] > 0 else 30.0
        total = meta["total_frames"]
        win_name = "Rover Tracker - press Q to quit"
        if display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 960, 540)
            # Face blur setup — Haar cascade ships with OpenCV
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            _cached_faces: list = []   # reused across frames to save CPU

        for frame_num, frame in enumerate(src):
            timestamp_s = frame_num / fps

            state = tracker.process_frame(frame, timestamp_s)

            if state is not None:
                events = detector.update(state, history)
                history.append(state)
                logger.log_state(state)
                for event in events:
                    logger.log_event(event)
                    log.debug("Event: %s at t=%.2fs", event.event_type, event.timestamp_s)

            if display:
                debug = tracker.get_debug_frame()
                if debug is None:
                    debug = frame.copy()

                # --- Face blur (privacy): run detector every 5 frames ---
                if frame_num % 5 == 0:
                    gray = cv2.cvtColor(debug, cv2.COLOR_BGR2GRAY)
                    _cached_faces = _face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                    )
                for (fx, fy, fw, fh) in _cached_faces:
                    roi = debug[fy:fy + fh, fx:fx + fw]
                    debug[fy:fy + fh, fx:fx + fw] = cv2.GaussianBlur(roi, (99, 99), 30)

                # Status overlay
                detected = state is not None
                status = f"Frame {frame_num}/{total}  t={timestamp_s:.1f}s  " \
                         f"{'DETECTED' if detected else 'searching...'}"
                if detected:
                    speed_fts = state.velocity_mms / 304.8
                    status += f"  spd={speed_fts:.2f}ft/s  pos=({state.x_mm/304.8:.1f},{state.y_mm/304.8:.1f})ft"
                cv2.putText(debug, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(win_name, debug)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Display quit by user.")
                    break

        if display:
            cv2.destroyAllWindows()

        summary = logger.close()

    log.info("Trial complete: %d frames, %.1f s, %.0f mm travelled, %d events",
             summary.total_frames, summary.duration_s,
             summary.total_distance_mm, sum(summary.event_counts.values()))
    log.info("Event breakdown: %s", summary.event_counts)
    log.info("Results saved to: %s", summary.output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="MIE491 Rover Tracker")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--video", default=None, help="Override video file path")
    parser.add_argument("--camera", action="store_true", help="Use live USB camera")
    parser.add_argument("--calibrate", action="store_true", help="Run homography calibration")
    parser.add_argument("--display", action="store_true", help="Show debug video window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config  # store path for calibration save

    if args.video:
        cfg.setdefault("sensor", {})["file_path"] = args.video
        cfg["sensor"]["source"] = "file"

    if args.camera:
        cfg.setdefault("sensor", {})["source"] = "camera"

    setup_logging(cfg)

    if args.calibrate:
        run_calibration(cfg)
        return

    trial_id, rover_name = prompt_rover_name()
    summary = run_trial(cfg, display=args.display, trial_id=trial_id, rover_name=rover_name)
    print(f"\nTrial ID:       {summary.trial_id}")
    print(f"Duration:       {summary.duration_s:.1f} s")
    print(f"Distance:       {summary.total_distance_mm / 1000:.3f} m")
    print(f"Avg speed:      {summary.average_speed_mms:.1f} mm/s")
    print(f"Events:         {summary.event_counts}")
    print(f"Output:         {summary.output_dir}")


if __name__ == "__main__":
    main()
