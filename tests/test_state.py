"""Tests for rover_state module."""

import math

import pytest

from rover_tracker.state.rover_state import EventFlag, RoverState, StateHistory


def make_state(i, x=100.0, y=100.0, vel=50.0):
    return RoverState(
        frame_idx=i, timestamp_s=i * 0.1,
        x_mm=x, y_mm=y, px=0, py=0,
        velocity_mms=vel, heading_deg=0.0,
        event_flags=EventFlag.NONE,
    )


def test_event_flag_bitmask():
    flags = EventFlag.WALL_COLLISION | EventFlag.STOPPED
    assert EventFlag.WALL_COLLISION in flags
    assert EventFlag.MANUAL_INTERVENTION not in flags


def test_rover_state_to_dict_roundtrip():
    s = make_state(5)
    d = s.to_dict()
    s2 = RoverState.from_dict(d)
    assert s == s2


def test_state_history_append_and_last():
    h = StateHistory()
    for i in range(5):
        h.append(make_state(i))
    assert len(h) == 5
    last2 = h.last(2)
    assert len(last2) == 2
    assert last2[-1].frame_idx == 4


def test_state_history_total_distance():
    h = StateHistory()
    # Three states in a line: (0,0), (100,0), (200,0)
    h.append(RoverState(0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, EventFlag.NONE))
    h.append(RoverState(1, 0.1, 100.0, 0.0, 1, 0, 0.0, 0.0, EventFlag.NONE))
    h.append(RoverState(2, 0.2, 200.0, 0.0, 2, 0, 0.0, 0.0, EventFlag.NONE))
    assert math.isclose(h.total_distance_mm(), 200.0, rel_tol=1e-6)


def test_state_history_average_speed():
    h = StateHistory()
    for i in range(4):
        h.append(make_state(i, vel=100.0))
    assert math.isclose(h.average_speed_mms(), 100.0)


def test_state_history_maxlen():
    h = StateHistory(maxlen=3)
    for i in range(10):
        h.append(make_state(i))
    assert len(h) == 3
    assert list(h)[-1].frame_idx == 9


def test_state_history_to_dataframe(state_history):
    df = state_history.to_dataframe()
    assert "x_mm" in df.columns
    assert len(df) == 30
