"""Geometry of the pile of sheets (stack mode), shared by the sequencer and the renderer.

A sheet is a page of size (pw, ph) laid on the desk rotated by `angle` degrees about its
centre, which sits at (W/2 + dx*H, H/2 + dy*H) in the frame. The same 2x3 matrix is used to
draw it (renderer) and to reason about what is visible (sequencer), so a writing planned
on the visible part of a lower sheet lands exactly there in the video.
"""

from __future__ import annotations

import cv2
import numpy as np


def centre(pose, W: float, H: float) -> tuple[float, float, float]:
    angle, dx, dy = pose
    return angle, W / 2 + dx * H, H / 2 + dy * H


def matrix(pw: float, ph: float, angle: float, cx: float, cy: float, scale: float = 1.0) -> np.ndarray:
    """Page pixel -> frame pixel."""
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, scale)
    M[0, 2] += cx - pw / 2
    M[1, 2] += cy - ph / 2
    return M


def corners(M: np.ndarray, pw: float, ph: float) -> np.ndarray:
    return np.array([[0, 0, 1], [pw, 0, 1], [pw, ph, 1], [0, ph, 1]], np.float64) @ M.T


def visible_masks(sheets: list[tuple[float, float, np.ndarray]], W: int, H: int, grid: int = 8) -> list[np.ndarray]:
    """For sheets listed bottom to top as (pw, ph, M), the part of each that is visible
    (inside the frame, not covered by any sheet above it), on a grid of W/grid x H/grid."""
    gw, gh = int(np.ceil(W / grid)), int(np.ceil(H / grid))
    polys = []
    for pw, ph, M in sheets:
        m = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(m, [np.round(corners(M, pw, ph) / grid).astype(np.int32)], 1)
        polys.append(m.astype(bool))
    out = []
    covered = np.zeros((gh, gw), bool)
    for m in reversed(polys):  # from the top down
        out.append(m & ~covered)
        covered |= m
    return list(reversed(out))


def footprint(w: float, h: float, angle: float) -> np.ndarray:
    """Normalized kernel of a w x h rectangle rotated by `angle` degrees (image coordinates,
    counter-clockwise like cv2.getRotationMatrix2D), for measuring how much of it is visible."""
    size = int(np.ceil(np.hypot(w, h))) | 1
    k = np.zeros((size, size), np.float32)
    box = cv2.boxPoints(((size / 2, size / 2), (max(1.0, w), max(1.0, h)), -angle))
    cv2.fillPoly(k, [np.round(box).astype(np.int32)], 1.0)
    return k / max(k.sum(), 1.0)


def to_page(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Frame pixel -> page pixel."""
    inv = cv2.invertAffineTransform(M)
    px, py = inv @ np.array([x, y, 1.0])
    return float(px), float(py)
