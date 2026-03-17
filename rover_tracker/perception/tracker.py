"""RoverTracker: per-frame perception pipeline."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

_PERCEPTION_DIR = os.path.dirname(os.path.abspath(__file__))

from ..state.rover_state import EventFlag, RoverState
from .homography import HomographyTransform


class RoverTracker:
    """
    Stateful rover tracker. Call process_frame() for each video frame.
    Owns background subtractor, contour filtering.
    """

    def __init__(self, cfg: dict, homography: HomographyTransform):
        pcfg = cfg.get("perception", {})

        # Background subtractor
        algo = pcfg.get("bgs_algorithm", "MOG2").upper()
        history = pcfg.get("bgs_history", 400)
        threshold = pcfg.get("bgs_var_threshold", 200)
        shadows = pcfg.get("bgs_detect_shadows", False)
        bright_threshold = pcfg.get("bright_threshold", 700)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([50, 50, 50], dtype=np.uint8)
        kernal = np.ones((99, 99), np.float32) / 9801

        # Store BGS params so reset() can recreate the model between trials
        self._bgs_algo      = algo
        self._bgs_history   = history
        self._bgs_threshold = threshold
        self._bgs_shadows   = shadows

        if algo == "KNN":
            self._bgs = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=threshold, detectShadows=shadows
            )
        else:
            self._bgs = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=shadows
            )

        # Pre-load initial-detection reference images once (not on every lost-track event)
        bg_path  = os.path.join(_PERCEPTION_DIR, "background1new.jpg")
        bm_path  = os.path.join(_PERCEPTION_DIR, "background_mat.jpg")
        self._bg1   = cv2.imread(bg_path)
        _bm_raw     = cv2.imread(bm_path, cv2.IMREAD_GRAYSCALE)
        if _bm_raw is not None:
            _, self._bmask = cv2.threshold(_bm_raw, 127, 255, cv2.THRESH_BINARY)
        else:
            self._bmask = None

        # Contour filtering thresholds
        self._min_area: int = pcfg.get("min_contour_area_px", 500)
        self._max_area: int = pcfg.get("max_contour_area_px", 50000)
        self._min_ar: float = pcfg.get("min_aspect_ratio", 0.3)
        self._max_ar: float = pcfg.get("max_aspect_ratio", 4.0)

        # Area-consistency lock: tracks rover's expected size to reject hands/arms
        self._area_max_factor: float = pcfg.get("area_tolerance_factor", 2.5)
        self._area_alpha: float = 0.3          # EMA decay for size estimate
        self._size: int = 200

        # Tracking continuity
        self._max_miss: int = pcfg.get("max_miss_frames", 10)
        self._max_jump_mm: float = pcfg.get("max_jump_mm", 300.0)
        # Physical velocity cap — rejects spikes from tracking noise
        self._max_velocity_mms: float = pcfg.get("max_velocity_mms", 1500.0)

        # MeanShift
        self._term_crit = term_crit
        self._bright_threshold = bright_threshold

        # Initial detection
        self._lower_black = lower_black
        self._upper_black = upper_black
        self._kernal = kernal

        # Position smoothing — EMA dampens frame-to-frame centroid jitter
        self._pos_alpha: float = pcfg.get("position_smoothing_alpha", 0.4)
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

        self._homography = homography

        # Maze ROI mask — built lazily on first frame (need frame shape)
        self._maze_pixels: np.ndarray = np.array(
            cfg.get("homography", {}).get("pixel_points", []), dtype=np.int32
        )
        self._roi_mask: np.ndarray | None = None

        # Internal state
        self._last_state: RoverState | None = None
        self._miss_count: int = 0
        self._frame_count: int = 0
        self._debug_frame: np.ndarray | None = None
        self._lost_track: int = 0
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

    def initialDetection(self, frame: np.ndarray, timestamp_s: float):
        background1 = self._bg1
        bmask       = self._bmask
        if background1 is None or bmask is None:
            # Fallback if reference images are missing: centre of frame
            h, w = frame.shape[:2]
            x1 = w // 2 - self._size // 2
            y1 = h // 2 - self._size // 2
            bbox1 = (x1, y1, self._size, self._size)
            x_mm, y_mm = self._homography.pixel_to_world(x1, y1)
            state = RoverState(
                frame_idx=self._frame_count,
                timestamp_s=timestamp_s,
                x_mm=x_mm, y_mm=y_mm,
                px=int(x1), py=int(y1),
                velocity_mms=0,
                event_flags=EventFlag.NONE,
            )
            self._last_state = state
            return bbox1

        diff1 = cv2.absdiff(background1, frame)
        dim = cv2.inRange(diff1, self._lower_black, self._upper_black)

        frame = frame.copy()
        frame[dim > 0] = (0, 0, 0)
        frame = cv2.bitwise_and(frame, frame, mask=bmask)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.filter2D(frame, -1, self._kernal)

        y1, x1 = np.unravel_index(frame.argmax(), frame.shape)
        x1 = x1 - (int)(self._size / 2)
        y1 = y1 - (int)(self._size / 2)
        bbox1 = (x1, y1, self._size, self._size)
        x_mm, y_mm = self._homography.pixel_to_world(x1, y1)
        state = RoverState(
            frame_idx=self._frame_count,
            timestamp_s=timestamp_s,
            x_mm=x_mm,
            y_mm=y_mm,
            px=int(x1),
            py=int(y1),
            velocity_mms=0,
            event_flags=EventFlag.NONE,
        )
        self._last_state = state
        return bbox1

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

        fg_mask = self._bgs.apply(debug)
        # Remove shadows (value 127) and noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # Mask out everything outside the maze — ignores people walking around it
        fg_mask = cv2.bitwise_and(fg_mask, self._roi_mask)

        if self._last_state is None:
            self.initialDetection(frame, timestamp_s)
        bbox1 = ((int)(self._last_state.px - self._size / 2),
                 (int)(self._last_state.py - self._size / 2),
                 self._size, self._size)
        _, bbox1 = cv2.meanShift(fg_mask, bbox1, self._term_crit)
        oldx1, oldy1 = ((int)(self._last_state.px - self._size / 2),
                        (int)(self._last_state.py - self._size / 2))
        x1, y1, _, _ = bbox1
        x1 = (int)(x1)
        y1 = (int)(y1)
        # Clamp bbox to frame bounds before pixel access
        h_frame, w_frame = fg_mask.shape[:2]
        x1 = max(0, min(x1, w_frame - self._size))
        y1 = max(0, min(y1, h_frame - self._size))
        bright = 0
        for xi in range(self._size):
            for yi in range(self._size):
                if fg_mask[yi + y1][xi + x1] > 0:
                    bright += 1
        if bright < self._bright_threshold:
            x1, y1 = oldx1, oldy1
        if bright < 5:
            self._lost_track += 1
        else:
            self._lost_track = 0
        if self._lost_track >= 30:
            self._lost_track = 0
            x1, y1, _, _ = self.initialDetection(frame, timestamp_s)

        cx, cy = (int)(x1 + 0.5 * self._size), (int)(y1 + 0.5 * self._size)

        x_mm, y_mm = self._homography.pixel_to_world(cx, cy)

        # Adaptive position EMA — alpha scales with how far the raw centroid has moved
        # from the current smooth estimate.  Low alpha → barely follows small jitter
        # (rotation / noise).  Full alpha → responds quickly to genuine displacement.
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

        state = RoverState(
            frame_idx=self._frame_count,
            timestamp_s=timestamp_s,
            x_mm=x_mm,
            y_mm=y_mm,
            px=int(cx),
            py=int(cy),
            velocity_mms=velocity_mms,
            event_flags=EventFlag.NONE,
        )
        self._last_state = state

        # Annotate debug frame
        cv2.rectangle(debug, (x1, y1), (x1 + self._size, y1 + self._size), (0, 255, 0), 2)
        self._debug_frame = debug
        return state

    def _select_best_contour(self, contours, last_px=None, last_py=None) -> np.ndarray | None:
        """Return the best contour passing area/aspect-ratio filters.

        Two-stage defense against hands/arms entering the maze:
        1. Area-consistency filter: reject blobs larger than area_tolerance_factor × rover area.
        2. Proximity lock: pick the candidate closest to the last known pixel position.
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

    def _extract_pose(self, contour: np.ndarray) -> tuple[float, float]:
        """Return (cx, cy)."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            x, y, w, h = cv2.boundingRect(contour)
            return x + w / 2, y + h / 2
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return cx, cy

    def get_debug_frame(self) -> np.ndarray | None:
        return self._debug_frame

    def reset(self) -> None:
        """Reset BGS model and tracking state (call between trials)."""
        # Recreate the background subtractor so the previous trial's background
        # model does not bleed into the new trial.
        if self._bgs_algo == "KNN":
            self._bgs = cv2.createBackgroundSubtractorKNN(
                history=self._bgs_history,
                dist2Threshold=self._bgs_threshold,
                detectShadows=self._bgs_shadows,
            )
        else:
            self._bgs = cv2.createBackgroundSubtractorMOG2(
                history=self._bgs_history,
                varThreshold=self._bgs_threshold,
                detectShadows=self._bgs_shadows,
            )
        self._last_state  = None
        self._miss_count  = 0
        self._frame_count = 0
        self._smooth_x    = None
        self._smooth_y    = None
        self._last_raw_x  = None
        self._last_raw_y  = None
        self._lost_track  = 0
