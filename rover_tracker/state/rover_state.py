"""Rover state dataclass and history buffer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntFlag
from collections import deque
from typing import Iterator


class EventFlag(IntFlag):
    NONE                = 0
    WALL_COLLISION      = 1
    STOPPED             = 2
    MANUAL_INTERVENTION = 4
    OFF_TRACK           = 8


@dataclass
class RoverState:
    """Immutable snapshot of rover state at a single point in time."""
    frame_idx:    int
    timestamp_s:  float
    x_mm:         float   # world coordinate
    y_mm:         float
    px:           int     # pixel coordinate (pre-transform, for debug)
    py:           int
    velocity_mms: float   # instantaneous speed mm/s
    heading_deg:  float   # 0=East, CCW positive; -1 if unknown
    event_flags:  int     # bitmask of EventFlag values

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoverState":
        return cls(**d)


class StateHistory:
    """Rolling buffer of RoverState objects for one trial."""

    def __init__(self, maxlen: int | None = None):
        self._states: deque[RoverState] = deque(maxlen=maxlen)

    def append(self, state: RoverState) -> None:
        self._states.append(state)

    def last(self, n: int = 1) -> list[RoverState]:
        states = list(self._states)
        return states[-n:] if n <= len(states) else states

    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self) -> Iterator[RoverState]:
        return iter(self._states)

    def total_distance_mm(self) -> float:
        states = list(self._states)
        total = 0.0
        for i in range(1, len(states)):
            dx = states[i].x_mm - states[i - 1].x_mm
            dy = states[i].y_mm - states[i - 1].y_mm
            total += math.hypot(dx, dy)
        return total

    def average_speed_mms(self) -> float:
        states = list(self._states)
        if len(states) < 2:
            return 0.0
        valid = [s.velocity_mms for s in states if s.velocity_mms >= 0]
        return sum(valid) / len(valid) if valid else 0.0

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([s.to_dict() for s in self._states])
