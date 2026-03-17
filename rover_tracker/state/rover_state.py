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

    def total_distance_mm(self, chunk_s: float = 1.0,
                          min_step_mm: float = 30.0,
                          max_step_mm: float = 400.0) -> float:
        """Return accumulated travel distance, robust to jitter and rotation-in-place.

        Groups frames into chunk_s-second windows and averages positions within each
        window.  Random jitter and rotation-in-place both cancel out over ~1 second so
        only genuine net translation shows up in the chunk-to-chunk steps.

        min_step_mm: ignore steps smaller than this (residual noise after averaging).
        max_step_mm: ignore steps larger than this (tracker teleport / glitch).
        """
        states = list(self._states)
        if len(states) < 2:
            return 0.0

        t0 = states[0].timestamp_s
        chunks: dict[int, list[tuple[float, float]]] = {}
        for s in states:
            key = int((s.timestamp_s - t0) / chunk_s)
            chunks.setdefault(key, []).append((s.x_mm, s.y_mm))

        positions = [
            (sum(p[0] for p in pts) / len(pts),
             sum(p[1] for p in pts) / len(pts))
            for _, pts in sorted(chunks.items())
        ]

        total = 0.0
        for i in range(1, len(positions)):
            d = math.hypot(positions[i][0] - positions[i - 1][0],
                           positions[i][1] - positions[i - 1][1])
            if min_step_mm < d <= max_step_mm:
                total += d
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
