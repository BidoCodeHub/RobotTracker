"""Video source abstraction: file or live USB camera."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


class VideoSource:
    """Wraps cv2.VideoCapture for both file and live camera input."""

    def __init__(self, cfg: dict):
        sensor = cfg.get("sensor", {})
        self._source_type: str = sensor.get("source", "file")
        self._file_path: str = sensor.get("file_path", "")
        self._camera_index: int = sensor.get("camera_index", 0)
        self._target_fps: float | None = sensor.get("target_fps", None)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._source_type == "camera":
            self._cap = cv2.VideoCapture(self._camera_index)
        else:
            self._cap = cv2.VideoCapture(self._file_path)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: "
                f"{'camera index ' + str(self._camera_index) if self._source_type == 'camera' else self._file_path}"
            )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            raise RuntimeError("VideoSource not opened. Call open() first.")
        return self._cap.read()

    def get_metadata(self) -> dict:
        if self._cap is None:
            raise RuntimeError("VideoSource not opened. Call open() first.")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total,
            "source": self._source_type,
        }

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("VideoSource not opened.")
        meta = self.get_metadata()
        native_fps = meta["fps"]
        skip_ratio = 1
        if self._target_fps and native_fps > 0 and self._target_fps < native_fps:
            skip_ratio = max(1, round(native_fps / self._target_fps))

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_idx % skip_ratio == 0:
                yield frame
            frame_idx += 1
