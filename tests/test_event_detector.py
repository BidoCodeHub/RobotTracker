"""Tests for EventDetector."""

import pytest

from rover_tracker.events.event_detector import DetectedEvent, EventDetector
from rover_tracker.state.rover_state import EventFlag, RoverState, StateHistory


MAZE_X_MAX = 2438.4   # length (8ft), x-axis
MAZE_Y_MAX = 1219.2   # width (4ft), y-axis


def _state(x, y, vel=50.0, t=0.0, frame=0):
    return RoverState(
        frame_idx=frame, timestamp_s=t,
        x_mm=x, y_mm=y, px=0, py=0,
        velocity_mms=vel, heading_deg=0.0,
        event_flags=EventFlag.NONE,
    )


@pytest.fixture
def detector(mock_cfg):
    return EventDetector(mock_cfg)


def test_wall_collision_west(detector):
    state = _state(x=10.0, y=600.0)
    history = StateHistory()
    events = detector.update(state, history)
    wall_events = [e for e in events if e.event_type == "wall_collision"]
    assert len(wall_events) == 1
    assert wall_events[0].metadata["wall"] == "west"


def test_wall_collision_east(detector):
    state = _state(x=MAZE_X_MAX - 10.0, y=600.0)
    history = StateHistory()
    events = detector.update(state, history)
    wall_events = [e for e in events if e.event_type == "wall_collision"]
    assert len(wall_events) == 1
    assert wall_events[0].metadata["wall"] == "east"


def test_no_collision_at_center(detector):
    state = _state(x=600.0, y=600.0)
    history = StateHistory()
    events = detector.update(state, history)
    assert not any(e.event_type == "wall_collision" for e in events)


def test_off_track(detector):
    state = _state(x=-100.0, y=600.0)
    history = StateHistory()
    events = detector.update(state, history)
    assert any(e.event_type == "off_track" for e in events)


def test_stop_detection(detector):
    history = StateHistory()
    # Feed 15 stopped states at 10fps → 1.5s of being stopped
    events_found = []
    for i in range(15):
        s = _state(x=600.0, y=1200.0, vel=0.0, t=i * 0.1, frame=i)
        events_found.extend(detector.update(s, history))
        history.append(s)

    stop_events = [e for e in events_found if e.event_type == "stop"]
    assert len(stop_events) == 1


def test_stop_not_triggered_if_too_short(detector):
    history = StateHistory()
    events_found = []
    # Only 0.5s of stopping — below the 1.0s threshold
    for i in range(5):
        s = _state(x=600.0, y=1200.0, vel=0.0, t=i * 0.1, frame=i)
        events_found.extend(detector.update(s, history))
        history.append(s)

    stop_events = [e for e in events_found if e.event_type == "stop"]
    assert len(stop_events) == 0


def test_manual_intervention(detector):
    history = StateHistory()
    prev = _state(x=600.0, y=1200.0, vel=10.0, t=0.0)
    history.append(prev)
    # Sudden velocity spike
    state = _state(x=600.0, y=1200.0, vel=500.0, t=0.1, frame=1)
    events = detector.update(state, history)
    assert any(e.event_type == "manual_intervention" for e in events)


def test_event_flag_set_on_state(detector):
    state = _state(x=10.0, y=600.0)
    history = StateHistory()
    detector.update(state, history)
    assert state.event_flags & EventFlag.WALL_COLLISION
