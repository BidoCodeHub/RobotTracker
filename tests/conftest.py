"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from rover_tracker.state.rover_state import EventFlag, RoverState, StateHistory


MAZE_W_MM = 1219.2
MAZE_H_MM = 2438.4

MOCK_CFG = {
    "maze": {
        "length_ft": 8.0,
        "width_ft": 4.0,
        "length_mm": MAZE_H_MM,
        "width_mm": MAZE_W_MM,
    },
    "homography": {
        "pixel_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
        "world_points": [
            [0.0, 0.0],
            [MAZE_W_MM, 0.0],
            [MAZE_W_MM, MAZE_H_MM],
            [0.0, MAZE_H_MM],
        ],
    },
    "sensor": {
        "source": "file",
        "file_path": "data_/WIN_20260225_15_07_32_Pro.mp4",
        "camera_index": 0,
        "target_fps": None,
    },
    "perception": {
        "bgs_algorithm": "MOG2",
        "bgs_history": 5,
        "bgs_var_threshold": 40,
        "bgs_detect_shadows": False,
        "min_contour_area_px": 500,
        "max_contour_area_px": 50000,
        "min_aspect_ratio": 0.3,
        "max_aspect_ratio": 4.0,
        "max_miss_frames": 10,
        "max_jump_mm": 300,
    },
    "events": {
        "collision_margin_mm": 50.0,
        "stop_velocity_threshold_mms": 5.0,
        "stop_min_duration_s": 1.0,
        "intervention_velocity_jump_mms": 400.0,
        "intervention_min_gap_s": 2.0,
    },
    "data": {
        "output_root": "trials_test",
        "trajectory_flush_every": 0,
        "events_format": "json",
    },
    "visualization": {},
    "logging": {"level": "WARNING", "log_to_file": False},
}


@pytest.fixture
def mock_cfg() -> dict:
    return MOCK_CFG.copy()


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Black 640x320 frame with a white filled rectangle (fake rover) at centre."""
    frame = np.zeros((320, 640, 3), dtype=np.uint8)
    # Draw a rover-like rectangle ~50x30 px at the centre
    cv2 = __import__("cv2")
    cx, cy = 320, 160
    cv2.rectangle(frame, (cx - 25, cy - 15), (cx + 25, cy + 15), (255, 255, 255), -1)
    return frame


@pytest.fixture
def known_homography():
    """HomographyTransform with exact pixel->world mapping (1px = 1mm shortcut)."""
    from rover_tracker.perception.homography import HomographyTransform
    import cv2
    # 4 pixel corners map to 4 world corners of the maze
    src = np.array([[0, 0], [MAZE_W_MM, 0], [MAZE_W_MM, MAZE_H_MM], [0, MAZE_H_MM]],
                   dtype=np.float32)
    dst = src.copy()
    H, _ = cv2.findHomography(src, dst)
    return HomographyTransform(H)


@pytest.fixture
def state_sequence() -> list[RoverState]:
    """30 states simulating straight-line motion at 100 mm/s eastward."""
    states = []
    for i in range(30):
        t = i * 0.1  # 10 fps
        states.append(RoverState(
            frame_idx=i,
            timestamp_s=t,
            x_mm=100.0 + i * 10.0,  # 100 mm/s
            y_mm=600.0,
            px=i * 5,
            py=100,
            velocity_mms=100.0,
            event_flags=EventFlag.NONE,
        ))
    return states


@pytest.fixture
def state_history(state_sequence) -> StateHistory:
    h = StateHistory()
    for s in state_sequence:
        h.append(s)
    return h
