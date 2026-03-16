"""Tests for HomographyTransform."""

import math

import numpy as np
import pytest

from rover_tracker.perception.homography import HomographyTransform


MAZE_W = 1219.2
MAZE_H = 2438.4


@pytest.fixture
def exact_transform():
    """Homography where pixel coords equal world coords (identity-like)."""
    import cv2
    src = np.array([[0, 0], [MAZE_W, 0], [MAZE_W, MAZE_H], [0, MAZE_H]], dtype=np.float32)
    dst = src.copy()
    H, _ = cv2.findHomography(src, dst)
    return HomographyTransform(H)


def test_pixel_to_world_corners(exact_transform):
    x, y = exact_transform.pixel_to_world(0.0, 0.0)
    assert math.isclose(x, 0.0, abs_tol=1.0)
    assert math.isclose(y, 0.0, abs_tol=1.0)

    x, y = exact_transform.pixel_to_world(MAZE_W, MAZE_H)
    assert math.isclose(x, MAZE_W, abs_tol=1.0)
    assert math.isclose(y, MAZE_H, abs_tol=1.0)


def test_world_to_pixel_roundtrip(exact_transform):
    px, py = exact_transform.world_to_pixel(600.0, 1200.0)
    x, y = exact_transform.pixel_to_world(px, py)
    assert math.isclose(x, 600.0, abs_tol=2.0)
    assert math.isclose(y, 1200.0, abs_tol=2.0)


def test_from_config(mock_cfg):
    # mock_cfg has a simplified homography; just verify it constructs without error
    transform = HomographyTransform.from_config(mock_cfg)
    assert transform.get_matrix().shape == (3, 3)


def test_get_matrix_returns_copy(exact_transform):
    m1 = exact_transform.get_matrix()
    m1[0, 0] = 999
    m2 = exact_transform.get_matrix()
    assert m2[0, 0] != 999
