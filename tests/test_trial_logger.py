"""Tests for TrialLogger."""

import csv
import json
from pathlib import Path

import pytest

from rover_tracker.data.trial_logger import TrialLogger
from rover_tracker.events.event_detector import DetectedEvent
from rover_tracker.state.rover_state import EventFlag, RoverState


def _state(i, x=600.0, y=1200.0):
    return RoverState(
        frame_idx=i, timestamp_s=i * 0.1,
        x_mm=x, y_mm=y, px=0, py=0,
        velocity_mms=50.0, heading_deg=0.0,
        event_flags=EventFlag.NONE,
    )


def _event(t=0.5):
    return DetectedEvent(
        event_type="wall_collision",
        timestamp_s=t, frame_idx=5,
        x_mm=20.0, y_mm=600.0,
        metadata={"wall": "west"},
    )


@pytest.fixture
def tmp_cfg(tmp_path, mock_cfg):
    cfg = mock_cfg.copy()
    cfg["data"] = {
        "output_root": str(tmp_path / "trials"),
        "trajectory_flush_every": 0,
        "events_format": "json",
    }
    return cfg


def test_logger_creates_output_dir(tmp_cfg):
    logger = TrialLogger(tmp_cfg, trial_id="test001")
    logger.open(tmp_cfg)
    assert (Path(tmp_cfg["data"]["output_root"]) / "trial_test001").exists()
    logger.close()


def test_trajectory_csv_written(tmp_cfg):
    logger = TrialLogger(tmp_cfg, trial_id="traj_test")
    logger.open(tmp_cfg)
    for i in range(5):
        logger.log_state(_state(i))
    summary = logger.close()

    csv_path = summary.output_dir / "trajectory.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert "x_mm" in rows[0]


def test_events_json_written(tmp_cfg):
    logger = TrialLogger(tmp_cfg, trial_id="evt_test")
    logger.open(tmp_cfg)
    logger.log_state(_state(0))
    logger.log_event(_event())
    summary = logger.close()

    events_path = summary.output_dir / "events.json"
    assert events_path.exists()
    events = json.loads(events_path.read_text())
    assert len(events) == 1
    assert events[0]["event_type"] == "wall_collision"


def test_config_snapshot_written(tmp_cfg):
    logger = TrialLogger(tmp_cfg, trial_id="cfg_test")
    logger.open(tmp_cfg)
    summary = logger.close()
    assert (summary.output_dir / "config_snapshot.yaml").exists()


def test_summary_metrics(tmp_cfg):
    logger = TrialLogger(tmp_cfg, trial_id="metrics_test")
    logger.open(tmp_cfg)
    # Rover moves 100mm in 1.0s → avg speed 100 mm/s
    logger.log_state(_state(0, x=0.0))
    logger.log_state(_state(10, x=100.0))  # t=1.0s
    logger.log_event(_event())
    summary = logger.close()

    assert summary.total_frames == 2
    assert abs(summary.total_distance_mm - 100.0) < 1.0
    assert summary.event_counts.get("wall_collision") == 1


def test_context_manager(tmp_cfg):
    with TrialLogger(tmp_cfg, trial_id="ctx_test") as logger:
        logger.open(tmp_cfg)
        logger.log_state(_state(0))
    # No exception raised — files should exist
    trial_dir = Path(tmp_cfg["data"]["output_root"]) / "trial_ctx_test"
    assert (trial_dir / "trajectory.csv").exists()
