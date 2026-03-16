"""RoverTracker: per-frame perception pipeline."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..state.rover_state import EventFlag, RoverState
from .homography import HomographyTransform


class RoverTracker:
    """
    Stateful rover tracker. Call process_frame() for each video frame.
    Owns background subtractor, contour filtering, and heading estimation.
    """

    def __init__(self, cfg: dict, homography: HomographyTransform):
        pcfg = cfg.get("perception", {})

        # Background subtractor
        algo = pcfg.get("bgs_algorithm", "MOG2").upper()
        history = pcfg.get("bgs_history", 200)
        threshold = pcfg.get("bgs_var_threshold", 40)
        shadows = pcfg.get("bgs_detect_shadows", False)

        if algo == "KNN":
            self._bgs = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=threshold, detectShadows=shadows
            )
        else:
            self._bgs = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=shadows
            )

        # Contour filtering thresholds
        self._min_area: int = pcfg.get("min_contour_area_px", 500)
        self._max_area: int = pcfg.get("max_contour_area_px", 50000)
        self._min_ar: float = pcfg.get("min_aspect_ratio", 0.3)
        self._max_ar: float = pcfg.get("max_aspect_ratio", 4.0)

        # Area-consistency lock: tracks rover's expected size to reject hands/arms
        self._area_max_factor: float = pcfg.get("area_tolerance_factor", 2.5)
        self._area_alpha: float = 0.3          # EMA decay for size estimate
        self._rover_area: float | None = None  # learned on first detection

        # Heading
        self._heading_method: str = pcfg.get("heading_method", "ellipse")
        self._min_frames_heading: int = pcfg.get("min_frames_for_heading", 3)

        # Tracking continuity
        self._max_miss: int = pcfg.get("max_miss_frames", 10)
        self._max_jump_mm: float = pcfg.get("max_jump_mm", 300.0)
        # Physical velocity cap — rejects spikes from tracking noise
        self._max_velocity_mms: float = pcfg.get("max_velocity_mms", 1500.0)  # ~5 ft/s

        # Position smoothing — EMA dampens frame-to-frame centroid jitter
        self._pos_alpha: float = pcfg.get("position_smoothing_alpha", 0.4)
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

        self._homography = homography

        # Maze ROI mask — built lazily on first frame (need frame shape)
        self._maze_pixels: np.ndarray = np.array(
            cfg.get("homography", {}).get("pixel_points", []), dtype=np.int32
        )
        self._roi_mask: np.ndarray | None = None  # built on first frame

        # Internal state
        self._last_state: RoverState | None = None
        self._miss_count: int = 0
        self._frame_count: int = 0
        self._debug_frame: np.ndarray | None = None
        # Last raw (pre-EMA) world position — used by teleport guard so the
        # jump check is not fooled by the lagged smoothed position.
        self._last_raw_x: float | None = None
        self._last_raw_y: float | None = None

    def _build_roi_mask(self, shape: tuple) -> np.ndarray:
        """Build a binary mask that is 255 inside the maze polygon, 0 outside."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if len(self._maze_pixels) == 4:
            cv2.fillPoly(mask, [self._maze_pixels], 255)
        else:
            # No valid corners — allow entire frame (fallback)
            mask[:] = 255
        return mask

    def process_frame(self, frame: np.ndarray, timestamp_s: float) -> RoverState | None:
        """Run the full perception pipeline on one frame."""
        self._frame_count += 1
        debug = frame.copy()

        # Build ROI mask once we know the frame shape
        if self._roi_mask is None:
            self._roi_mask = self._build_roi_mask(frame.shape)

        # Always draw the maze boundary on the debug frame
        if len(self._maze_pixels) == 4:
            cv2.polylines(debug, [self._maze_pixels], isClosed=True,
                          color=(0, 255, 255), thickness=2)

        fg_mask = self._bgs.apply(frame)
        # Remove shadows (value 127) and noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # fill holes → stable centroid
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # Mask out everything outside the maze — ignores people walking around it
        fg_mask = cv2.bitwise_and(fg_mask, self._roi_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        last_px = self._last_state.px if self._last_state is not None else None
        last_py = self._last_state.py if self._last_state is not None else None
        best = self._select_best_contour(contours, last_px, last_py)

        if best is None:
            self._miss_count += 1
            if self._miss_count > self._max_miss:
                self._last_state = None
            self._debug_frame = debug
            return None

        self._miss_count = 0

        # Update rover area EMA — only when not in a miss streak (confident detection)
        detected_area = cv2.contourArea(best)
        if self._rover_area is None:
            self._rover_area = detected_area
        else:
            self._rover_area = (self._area_alpha * detected_area
                                + (1 - self._area_alpha) * self._rover_area)

        cx, cy, heading = self._extract_pose(best)

        # Teleport guard — compare raw detected positions across consecutive frames.
        # Using smoothed positions here would incorrectly flag fast movement because
        # the adaptive EMA deliberately lags during rotation / slow motion.
        x_mm, y_mm = self._homography.pixel_to_world(cx, cy)
        if self._last_raw_x is not None:
            jump = math.hypot(x_mm - self._last_raw_x, y_mm - self._last_raw_y)
            if jump > self._max_jump_mm:
                self._miss_count += 1
                self._debug_frame = debug
                return None
        self._last_raw_x, self._last_raw_y = x_mm, y_mm

        # Adaptive position EMA — alpha scales with how far the raw centroid has moved
        # from the current smooth estimate.  When the rover rotates in place the centroid
        # circles within ~30 mm of the true centre; translation moves it much further.
        # Low alpha  → barely follows small jitter (rotation / noise).
        # Full alpha → responds quickly to genuine displacement (translation).
        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = x_mm, y_mm
        else:
            raw_disp = math.hypot(x_mm - self._smooth_x, y_mm - self._smooth_y)
            # Scale: 0.05 at 0 mm displacement, ramps to pos_alpha at 60 mm
            alpha = min(self._pos_alpha, max(0.05, raw_disp / 60.0) * self._pos_alpha)
            self._smooth_x = alpha * x_mm + (1 - alpha) * self._smooth_x
            self._smooth_y = alpha * y_mm + (1 - alpha) * self._smooth_y
        x_mm, y_mm = self._smooth_x, self._smooth_y

        # Velocity from last state
        velocity_mms = 0.0
        if self._last_state is not None:
            dt = timestamp_s - self._last_state.timestamp_s
            if dt > 0:
                dx = x_mm - self._last_state.x_mm
                dy = y_mm - self._last_state.y_mm
                velocity_mms = math.hypot(dx, dy) / dt
                # Clamp to physical maximum — spikes above this are tracking noise
                if velocity_mms > self._max_velocity_mms:
                    velocity_mms = self._last_state.velocity_mms
            if self._heading_method == "delta_position" and velocity_mms > 1.0:
                heading = math.degrees(math.atan2(-(y_mm - self._last_state.y_mm),
                                                   x_mm - self._last_state.x_mm))

        state = RoverState(
            frame_idx=self._frame_count,
            timestamp_s=timestamp_s,
            x_mm=x_mm,
            y_mm=y_mm,
            px=int(cx),
            py=int(cy),
            velocity_mms=velocity_mms,
            heading_deg=heading,
            event_flags=EventFlag.NONE,
        )
        self._last_state = state

        # Annotate debug frame
        rect = cv2.boundingRect(best)
        cv2.rectangle(debug, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
        cv2.circle(debug, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        if heading != -1:
            length = 40
            ex = int(cx + length * math.cos(math.radians(heading)))
            ey = int(cy - length * math.sin(math.radians(heading)))
            cv2.arrowedLine(debug, (int(cx), int(cy)), (ex, ey), (255, 0, 0), 2)
        self._debug_frame = debug
        return state

    def _select_best_contour(self, contours, last_px=None, last_py=None) -> np.ndarray | None:
        """Return the best contour passing area/aspect-ratio filters.

        Two-stage defense against hands/arms entering the maze:

        1. Area-consistency filter (primary): once the rover's size is learned,
           reject any blob whose area exceeds `area_tolerance_factor` × expected
           rover area.  A human hand is typically 3-10× larger than the rover
           so it is filtered out even when it is adjacent to the rover.

        2. Proximity lock (secondary): among the remaining size-consistent
           candidates, pick the one closest to the rover's last known pixel
           position, preventing any stray similarly-sized object from stealing
           the track.

        Without a prior position the largest valid candidate is returned so the
        initial acquisition still works.
        """
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area or area > self._max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / h if h > 0 else 0
            if ar < self._min_ar or ar > self._max_ar:
                ar = h / w if w > 0 else 0
                if ar < self._min_ar or ar > self._max_ar:
                    continue
            candidates.append(c)

        if not candidates:
            return None

        # Stage 1 — area-consistency filter (active once rover size is learned)
        if self._rover_area is not None:
            max_allowed = self._rover_area * self._area_max_factor
            size_consistent = [c for c in candidates
                               if cv2.contourArea(c) <= max_allowed]
            # Only apply the filter if it leaves at least one candidate
            if size_consistent:
                candidates = size_consistent

        # Stage 2 — proximity lock
        if last_px is not None and last_py is not None:
            def _dist(c):
                M = cv2.moments(c)
                if M["m00"] == 0:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cx, cy = bx + bw / 2, by + bh / 2
                else:
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                return math.hypot(cx - last_px, cy - last_py)
            return min(candidates, key=_dist)

        # No prior position — pick the largest valid contour for initial acquisition
        return max(candidates, key=cv2.contourArea)

    def _extract_pose(self, contour: np.ndarray) -> tuple[float, float, float]:
        """Return (cx, cy, heading_deg). heading=-1 if undetermined."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            x, y, w, h = cv2.boundingRect(contour)
            return x + w / 2, y + h / 2, -1.0

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        heading = -1.0

        if self._heading_method == "ellipse" and len(contour) >= 5:
            try:
                (_, _), (_, _), angle = cv2.fitEllipse(contour)
                # OpenCV angle is CCW from vertical; convert to CCW from East
                heading = 90.0 - angle
            except cv2.error:
                pass

        return cx, cy, heading

    def get_debug_frame(self) -> np.ndarray | None:
        return self._debug_frame

    def reset(self) -> None:
        """Reset BGS history and tracking state (call between trials)."""
        self._last_state = None
        self._miss_count = 0
        self._frame_count = 0
        self._rover_area = None
        self._smooth_x = None
        self._smooth_y = None
        self._last_raw_x = None
        self._last_raw_y = None
        self._bgs = self._bgs  # BGS history resets on next apply() call after recreate
