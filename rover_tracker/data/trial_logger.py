"""Trial data logger: trajectory CSV + events JSON + config snapshot."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from ..events.event_detector import DetectedEvent
from ..state.rover_state import RoverState


@dataclass
class TrialSummary:
    trial_id:          str
    output_dir:        Path
    total_frames:      int
    duration_s:        float
    total_distance_mm: float
    average_speed_mms: float
    event_counts:      dict


class TrialLogger:
    """Manages output directory for one trial."""

    def __init__(self, cfg: dict, trial_id: str | None = None):
        dcfg = cfg.get("data", {})
        output_root = Path(dcfg.get("output_root", "trials"))
        self._flush_every: int = dcfg.get("trajectory_flush_every", 100)

        if trial_id is None:
            trial_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trial_id = trial_id
        self._trial_dir = output_root / f"trial_{trial_id}"

        self._states: list[RoverState] = []
        self._events: list[DetectedEvent] = []
        self._csv_file = None
        self._csv_writer = None
        self._opened = False

    def open(self, config_snapshot: dict) -> Path:
        self._trial_dir.mkdir(parents=True, exist_ok=True)

        # Lock file: present while recording, deleted on close.
        # Dashboard uses this to detect a live trial.
        (self._trial_dir / "recording.lock").touch()

        snapshot_path = self._trial_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)

        traj_path = self._trial_dir / "trajectory.csv"
        self._csv_file = open(traj_path, "w", newline="")
        fieldnames = [
            "frame_idx", "timestamp_s", "x_mm", "y_mm",
            "px", "py", "velocity_mms", "event_flags",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        self._csv_file.flush()   # write header to disk immediately so dashboard sees the file
        self._opened = True
        return self._trial_dir

    def log_state(self, state: RoverState) -> None:
        if not self._opened:
            raise RuntimeError("TrialLogger not opened.")
        self._states.append(state)
        self._csv_writer.writerow(state.to_dict())
        if self._flush_every > 0 and len(self._states) % self._flush_every == 0:
            self._csv_file.flush()

    def log_event(self, event: DetectedEvent) -> None:
        self._events.append(event)
        # Flush events to disk immediately so the dashboard updates in real-time
        events_path = self._trial_dir / "events.json"
        with open(events_path, "w") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)

    def close(self) -> TrialSummary:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

        # Final events flush (already written incrementally, this ensures completeness)
        events_path = self._trial_dir / "events.json"
        with open(events_path, "w") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)

        # Remove lock file — signals to dashboard that recording is complete
        lock = self._trial_dir / "recording.lock"
        if lock.exists():
            lock.unlink()

        # Compute summary distance using 1-second time chunks so that jitter and
        # rotation-in-place cancel out within each window.
        import math
        total_distance = 0.0
        if len(self._states) >= 2:
            CHUNK_S   = 1.0
            MIN_STEP  = 30.0   # mm — ignore residual noise after averaging
            MAX_STEP  = 400.0  # mm — reject teleport glitches
            t0 = self._states[0].timestamp_s
            chunks: dict[int, list[tuple[float, float]]] = {}
            for s in self._states:
                key = int((s.timestamp_s - t0) / CHUNK_S)
                chunks.setdefault(key, []).append((s.x_mm, s.y_mm))
            positions = [
                (sum(p[0] for p in pts) / len(pts),
                 sum(p[1] for p in pts) / len(pts))
                for _, pts in sorted(chunks.items())
            ]
            for i in range(1, len(positions)):
                d = math.hypot(positions[i][0] - positions[i - 1][0],
                               positions[i][1] - positions[i - 1][1])
                if MIN_STEP < d <= MAX_STEP:
                    total_distance += d

        duration = 0.0
        if len(self._states) >= 2:
            duration = self._states[-1].timestamp_s - self._states[0].timestamp_s

        avg_speed = 0.0
        if duration > 0:
            avg_speed = total_distance / duration

        event_counts: dict[str, int] = {}
        for e in self._events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        self._opened = False
        return TrialSummary(
            trial_id=self._trial_id,
            output_dir=self._trial_dir,
            total_frames=len(self._states),
            duration_s=duration,
            total_distance_mm=total_distance,
            average_speed_mms=avg_speed,
            event_counts=event_counts,
        )

    def __enter__(self) -> "TrialLogger":
        return self

    def __exit__(self, *_) -> None:
        if self._opened:
            self.close()
        if self._csv_file:
            self._csv_file.close()
