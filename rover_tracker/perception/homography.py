"""Pixel <-> real-world coordinate transform using a homography matrix."""

from __future__ import annotations

import cv2
import numpy as np


class HomographyTransform:
    """Bidirectional pixel <-> real-world (mm) mapping via perspective homography."""

    def __init__(self, H: np.ndarray):
        self._H = H
        self._H_inv = np.linalg.inv(H)

    @classmethod
    def from_config(cls, cfg: dict) -> "HomographyTransform":
        """Build transform from pixel_points / world_points in config."""
        hcfg = cfg.get("homography", {})
        src = np.array(hcfg["pixel_points"], dtype=np.float32)
        dst = np.array(hcfg["world_points"], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return cls(H)

    @classmethod
    def from_four_point_click(cls, frame: np.ndarray, cfg: dict) -> "HomographyTransform":
        """
        Interactive calibration: click the 4 maze corners in order
        (top-left, top-right, bottom-right, bottom-left), then press any key.
        Saves chosen pixel_points back into cfg['homography']['pixel_points'].
        """
        points: list[list[int]] = []

        def _on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.imshow("Calibration - click 4 maze corners (TL TR BR BL)", frame)

        win = "Calibration - click 4 maze corners (TL TR BR BL)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, frame)
        cv2.waitKey(1)  # process events so the window handle is created
        cv2.setMouseCallback(win, _on_click)

        while len(points) < 4:
            cv2.waitKey(50)

        cv2.waitKey(500)
        cv2.destroyWindow(win)

        cfg.setdefault("homography", {})["pixel_points"] = points
        return cls.from_config(cfg)

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        """Return (x_mm, y_mm) in maze coordinate frame."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        return float(result[0][0][0]), float(result[0][0][1])

    def world_to_pixel(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """Return (px, py) in image coordinates."""
        pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H_inv)
        return int(round(result[0][0][0])), int(round(result[0][0][1]))

    def get_matrix(self) -> np.ndarray:
        return self._H.copy()
