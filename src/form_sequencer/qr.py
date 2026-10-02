"""QR code exclusion.

Two detectors, results merged:
  * OpenCV's Aruco-based QR detector (finds codes even when they cannot be decoded);
  * a finder-pattern search: QR codes carry three nested-square position marks
    (dark 7x7, light 5x5, dark 3x3). Three such marks of similar size forming a right
    angle identify a code even if it is blurred, cropped or partly printed over.
Every code becomes a forbidden area (verdict "qr"), like photos and email addresses.
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


def _finder_marks(gray: np.ndarray, dpi: int) -> list[tuple[float, float, float]]:
    """Centres and sizes of nested-square marks (contour -> child -> grandchild, all square)."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    contours, hier = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    lo, hi = dpi * 0.04, dpi * 0.6
    marks = []

    def square(i):
        x, y, w, h = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])
        return (lo <= w <= hi and 0.75 <= w / max(h, 1) <= 1.33 and area > 0.6 * w * h), (x, y, w, h)

    for i in range(len(contours)):
        ok, (x, y, w, h) = square(i)
        if not ok:
            continue
        # walk down to a nested square two levels deep (outer dark ring -> light -> dark core)
        depth, j = 0, hier[i][2]
        while j >= 0 and depth < 4:
            ok_j, (xj, yj, wj, hj) = square(j) if cv2.contourArea(contours[j]) > 4 else (False, (0, 0, 0, 0))
            if ok_j and 0.2 < wj / w < 0.9 and abs((xj + wj / 2) - (x + w / 2)) < w * 0.15:
                depth += 1
                j = hier[j][2]
            else:
                j = hier[j][0]  # next sibling
        if depth >= 2:
            marks.append((x + w / 2, y + h / 2, float(w)))
    # de-duplicate nested detections of the same mark
    marks.sort(key=lambda m: -m[2])
    out = []
    for m in marks:
        if all(math.hypot(m[0] - o[0], m[1] - o[1]) > o[2] * 0.4 for o in out):
            out.append(m)
    return out


def _codes_from_marks(marks) -> list[list[int]]:
    """Group three marks of similar size at a right angle into a code's bounding box."""
    boxes = []
    used = set()
    for a, b, c in itertools.combinations(range(len(marks)), 3):
        if {a, b, c} & used:
            continue
        pts = [marks[a], marks[b], marks[c]]
        sizes = [p[2] for p in pts]
        if max(sizes) > 1.5 * min(sizes):
            continue
        s = float(np.mean(sizes))
        for corner in range(3):
            p0 = pts[corner]
            p1, p2 = [pts[k] for k in range(3) if k != corner]
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p0[0], p2[1] - p0[1])
            l1, l2 = math.hypot(*v1), math.hypot(*v2)
            if not (1.5 * s < l1 < 25 * s and 0.8 < l1 / max(l2, 1e-6) < 1.25):
                continue
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            if abs(cos) < 0.2:
                xs = [p[0] for p in pts] + [p1[0] + v2[0]]
                ys = [p[1] for p in pts] + [p1[1] + v2[1]]
                pad = s * 0.7
                boxes.append([int(min(xs) - pad), int(min(ys) - pad), int(max(xs) - min(xs) + 2 * pad),
                              int(max(ys) - min(ys) + 2 * pad)])
                used |= {a, b, c}
                break
    return boxes


def detect(rgb: np.ndarray, dpi: int, cfg: dict) -> list[dict]:
    if not cfg.get("enabled", True):
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    boxes = []
    try:
        ok, pts = cv2.QRCodeDetectorAruco().detectMulti(gray)
        if ok and pts is not None:
            for quad in pts:
                xs, ys = quad[:, 0], quad[:, 1]
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())])
    except (cv2.error, AttributeError):
        pass
    ink = gray < 128
    for b in _codes_from_marks(_finder_marks(gray, dpi)):
        x, y, w, h = b
        region = ink[max(0, y):y + h, max(0, x):x + w]
        # real codes are dense with modules and small; nested boxes on forms are sparse or large
        if region.size and 0.25 <= region.mean() <= 0.75 and max(w, h) <= 3 * dpi:
            boxes.append(b)
    # merge overlapping boxes, add a margin, clip to the page
    margin = int(float(cfg.get("margin_inches", 0.08)) * dpi)
    merged: list[list[int]] = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        b = [b[0] - margin, b[1] - margin, b[2] + 2 * margin, b[3] + 2 * margin]
        for m in merged:
            if b[0] < m[0] + m[2] and m[0] < b[0] + b[2] and b[1] < m[1] + m[3] and m[1] < b[1] + b[3]:
                x0, y0 = min(m[0], b[0]), min(m[1], b[1])
                m[:] = [x0, y0, max(m[0] + m[2], b[0] + b[2]) - x0, max(m[1] + m[3], b[1] + b[3]) - y0]
                break
        else:
            merged.append(b)
    out = []
    for i, (x, y, w, h) in enumerate(sorted(merged, key=lambda b: (b[1], b[0])), start=1):
        x0, y0 = max(0, x), max(0, y)
        out.append({"id": f"q{i}", "bbox": [x0, y0, min(W, x + w) - x0, min(H, y + h) - y0], "verdict": "qr"})
    return out
