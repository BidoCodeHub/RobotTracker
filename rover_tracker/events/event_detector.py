"""Event detection: wall collisions, stops, manual intervention, off-track."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..state.rover_state import EventFlag, RoverState, StateHistory


@dataclass
class DetectedEvent:
    event_type:  str
    timestamp_s: float
    frame_idx:   int
    x_mm:        float
    y_mm:        float
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":  self.event_type,
            "timestamp_s": self.timestamp_s,
            "frame_idx":   self.frame_idx,
            "x_mm":        self.x_mm,
            "y_mm":        self.y_mm,
            "metadata":    self.metadata,
        }


class EventDetector:
    """
    Stateless logic applied to StateHistory each frame.
    Maintains only small debounce counters.
    """

    def __init__(self, cfg: dict):
        ecfg = cfg.get("events", {})
        mcfg = cfg.get("maze", {})

        # Fixed: x uses length (8ft), y uses width (4ft)
        self._maze_x_max = mcfg.get("length_mm", 2438.4)
        self._maze_y_max = mcfg.get("width_mm", 1219.2)
        self._obstacles = mcfg.get("obstacles", [])

        self._collision_margin: float = ecfg.get("collision_margin_mm", 50.0)
        self._collision_debounce: float = ecfg.get("collision_debounce_s", 1.0)
        self._collision_decel_ratio: float = ecfg.get("collision_decel_ratio", 0.5)
        self._stop_vel: float = ecfg.get("stop_velocity_threshold_mms", 5.0)
        self._stop_min_dur: float = ecfg.get("stop_min_duration_s", 1.0)
        self._intervention_jump: float = ecfg.get("intervention_velocity_jump_mms", 400.0)
        self._intervention_gap: float = ecfg.get("intervention_min_gap_s", 2.0)
        self._off_track_margin: float = ecfg.get("off_track_margin_mm", 20.0)

        # Debounce state
        self._stop_start_t: float | None = None
        self._stop_fired: bool = False
        self._last_intervention_t: float = -999.0
        self._last_collision_t: float = -999.0

    def update(self, state: RoverState, history: StateHistory) -> list[DetectedEvent]:
        events: list[DetectedEvent] = []

        ev = self._check_wall_collision(state, history)
        if ev:
            events.append(ev)

        ev = self._check_off_track(state)
        if ev:
            events.append(ev)

        ev = self._check_stop(state, history)
        if ev:
            events.append(ev)

        ev = self._check_manual_intervention(state, history)
        if ev:
            events.append(ev)

        # Set event_flags bitmask on state (mutate in place)
        for e in events:
            flag_map = {
                "wall_collision":      EventFlag.WALL_COLLISION,
                "stop":                EventFlag.STOPPED,
                "manual_intervention": EventFlag.MANUAL_INTERVENTION,
                "off_track":           EventFlag.OFF_TRACK,
            }
            flag = flag_map.get(e.event_type, EventFlag.NONE)
            # RoverState is a dataclass — update event_flags
            object.__setattr__(state, "event_flags", state.event_flags | int(flag))

        return events

    def _check_wall_collision(self, state: RoverState, history: StateHistory) -> DetectedEvent | None:
        m = self._collision_margin
        wall = None

        # Check outer walls (fixed axes: x_max=length, y_max=width)
        if state.x_mm < m:
            wall = "west"
        elif state.x_mm > self._maze_x_max - m:
            wall = "east"
        elif state.y_mm < m:
            wall = "north"
        elif state.y_mm > self._maze_y_max - m:
            wall = "south"

        # Check internal obstacles
        if wall is None:
            for obs in self._obstacles:
                ox0, oy0, ox1, oy1 = obs
                in_x_band = ox0 - m <= state.x_mm <= ox1 + m
                in_y_band = oy0 - m <= state.y_mm <= oy1 + m
                if in_x_band and in_y_band:
                    wall = "obstacle"
                    break

        if wall is None:
            return None

        # Debounce: skip if too soon after last collision
        if state.timestamp_s - self._last_collision_t < self._collision_debounce:
            return None

        # Require trajectory disruption: velocity must have dropped significantly
        if len(history) >= 3:
            recent = history.last(3)
            avg_speed = sum(s.velocity_mms for s in recent) / len(recent)
            if avg_speed > 0 and state.velocity_mms > avg_speed * self._collision_decel_ratio:
                # Rover is still moving at >50% of recent speed — not a real collision
                return None

        self._last_collision_t = state.timestamp_s
        return DetectedEvent(
            event_type="wall_collision",
            timestamp_s=state.timestamp_s,
            frame_idx=state.frame_idx,
            x_mm=state.x_mm,
            y_mm=state.y_mm,
            metadata={"wall": wall},
        )

    def _check_off_track(self, state: RoverState) -> DetectedEvent | None:
        m = self._off_track_margin
        if (state.x_mm < -m or state.x_mm > self._maze_x_max + m or
                state.y_mm < -m or state.y_mm > self._maze_y_max + m):
            return DetectedEvent(
                event_type="off_track",
                timestamp_s=state.timestamp_s,
                frame_idx=state.frame_idx,
                x_mm=state.x_mm,
                y_mm=state.y_mm,
            )
        return None

    def _check_stop(self, state: RoverState, history: StateHistory) -> DetectedEvent | None:
        is_stopped = state.velocity_mms < self._stop_vel

        if is_stopped:
            if self._stop_start_t is None:
                self._stop_start_t = state.timestamp_s
                self._stop_fired = False
            elif (not self._stop_fired and
                  state.timestamp_s - self._stop_start_t >= self._stop_min_dur):
                self._stop_fired = True
                return DetectedEvent(
                    event_type="stop",
                    timestamp_s=state.timestamp_s,
                    frame_idx=state.frame_idx,
                    x_mm=state.x_mm,
                    y_mm=state.y_mm,
                    metadata={"duration_s": state.timestamp_s - self._stop_start_t},
                )
        else:
            self._stop_start_t = None
            self._stop_fired = False

        return None

    def _check_manual_intervention(self, state: RoverState, history: StateHistory) -> DetectedEvent | None:
        if len(history) < 1:
            return None
        prev = history.last(1)[0]
        # Fixed: check velocity JUMP (delta), not absolute velocity
        delta = state.velocity_mms - prev.velocity_mms
        if delta > self._intervention_jump:
            if state.timestamp_s - self._last_intervention_t > self._intervention_gap:
                self._last_intervention_t = state.timestamp_s
                return DetectedEvent(
                    event_type="manual_intervention",
                    timestamp_s=state.timestamp_s,
                    frame_idx=state.frame_idx,
                    x_mm=state.x_mm,
                    y_mm=state.y_mm,
                    metadata={"velocity_mms": state.velocity_mms,
                              "prev_velocity_mms": prev.velocity_mms},
                )
        return None
