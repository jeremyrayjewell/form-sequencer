"""Ink libraries (stamps, writing): isolate imprints/handwriting from photos or scans as
transparent PNGs.

Source images (a folder of photos or scans of stamp imprints on paper) are processed to
  1. flatten the paper: divide by a large-kernel local maximum, so creases, shading and
     uneven light become flat white;
  2. find the ink: its absorption signature (which colour channels it darkens) is taken
     from the most strongly absorbing pixels, so black, blue, violet and faded red inks all
     work; a pixel is ink when it absorbs along that signature. Stains, creases, dark edges,
     pen marks and faint security patterns absorb differently and are rejected;
  3. isolate the stamp(s): each large ink cluster becomes a stamp together with every ink
     shape inside its box (text, dates); specks elsewhere are dropped; tiny or empty results
     are skipped;
  4. un-mix the ink from the paper: alpha is the ink coverage, colour is the ink colour
     with the paper removed, so the impression keeps its rubber texture.
Results: build/stamps/<name>.png (RGBA) and build/stamps/review.jpg.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import IMAGE_EXTS, safe_key
from .fonts import font


def _flatten(rgb: np.ndarray, frac: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Paper-normalized image (paper -> ~1.0) and the estimated paper colour per pixel."""
    f = rgb.astype(np.float32) / 255.0
    k = max(15, int(min(rgb.shape[:2]) * frac)) | 1
    bg = np.stack([cv2.GaussianBlur(cv2.dilate(f[..., c], np.ones((k, k), np.uint8)), (0, 0), k / 3)
                   for c in range(3)], -1)
    return np.clip(f / np.maximum(bg, 0.15), 0, 1), bg


def _paper_from(rgb: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Paper colour everywhere, interpolated from paper pixels only (ink masked out).

    Follows lighting gradients and creases, and is not fooled by solid-filled lettering
    however wide it is (normalized convolution of the unmasked pixels)."""
    f = rgb.astype(np.float32) / 255.0
    k = max(3, int(min(rgb.shape[:2]) * 0.03))
    w = 1.0 - cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    sigma = max(4.0, min(rgb.shape[:2]) * 0.06)
    den = cv2.GaussianBlur(w, (0, 0), sigma) + 1e-4
    bg = np.stack([cv2.GaussianBlur(f[..., c] * w, (0, 0), sigma) / den for c in range(3)], -1)
    # far from any paper pixel the estimate is unreliable: fall back to the brightest nearby
    _, fallback = _flatten(rgb, 0.2)
    return np.where((den < 0.05)[..., None], fallback, bg)


def _ink_alpha(rgb: np.ndarray, bg: np.ndarray, cfg: dict):
    """Alpha of the ink, the ink's absorption signature, and the absorption image."""
    f = rgb.astype(np.float32) / 255.0
    d = 1.0 - np.clip(f / np.maximum(bg, 0.15), 0, 1)   # absorption per channel
    mag = np.linalg.norm(d, axis=2)
    if (mag > 0.15).sum() < 30:
        raise ValueError("no ink found")
    top = mag >= max(0.15, float(np.percentile(mag, 98)))
    v = d[top].mean(axis=0)
    v /= np.linalg.norm(v) + 1e-9                      # the ink's absorption signature
    proj = d @ v
    cos = proj / (mag + 1e-6)
    tol = float(cfg.get("direction_tolerance", 0.9))
    same_ink = np.clip((cos - tol) / max(1e-3, (1 - tol) * 0.5), 0, 1)
    strength = float(np.percentile(proj[top], 60))
    # the paper's own absorption (tinted, stained or noisy paper) sets the floor for ink
    med = float(np.median(proj))
    mad = float(np.median(np.abs(proj - med))) + 1e-4
    t0 = max(0.05, 0.15 * strength, med + float(cfg.get("paper_margin", 3.0)) * 1.4826 * mad)
    t1 = max(t0 + 0.06, 0.6 * strength)
    return np.clip((proj - t0) / (t1 - t0), 0, 1) * same_ink, v


def extract_stamps(rgb: np.ndarray, cfg: dict) -> list[tuple[np.ndarray, dict]]:
    """Every stamp in an image, as (RGBA uint8, info). Works for any ink colour, black included.

    After flattening the paper, each pixel's absorption per channel (1 - normalized value)
    is measured. The ink's absorption signature is taken from the most strongly absorbing
    pixels: (1,1,1)-like for black ink, red+green for blue ink, green for red/pink ink.
    Alpha is the absorption along that signature; pixels absorbing in another direction
    (yellow stains or brown creases under a pink stamp, a black line beside a blue one, a
    blue pen signature over a black stamp) are rejected. Two passes: a first ink estimate
    against a local paper estimate, then the paper is re-estimated from paper pixels only.
    """
    _, bg0 = _flatten(rgb)
    alpha0, _ = _ink_alpha(rgb, bg0, cfg)
    bg = _paper_from(rgb, alpha0 > 0.2)
    alpha, v = _ink_alpha(rgb, bg, cfg)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)

    # one stamp per large ink cluster; each keeps every ink shape inside its box
    mask = (alpha > 0.3).astype(np.uint8)
    k = max(3, int(min(rgb.shape[:2]) * 0.04))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((k, k), np.uint8)), 8)
    if n < 2:
        raise ValueError("no ink found")
    areas = stats[1:, cv2.CC_STAT_AREA]
    H, W = rgb.shape[:2]
    # several stamps only when there are several stamp-sized clusters (e.g. two round seals);
    # a frameless stamp made of separate text lines or handwritten words stays one stamp
    mains = [1 + i for i in np.argsort(-areas)
             if areas[i] >= 0.25 * areas.max() and stats[1 + i, 2] >= 0.3 * W and stats[1 + i, 3] >= 0.3 * H]
    if not cfg.get("split", True):
        mains = mains[:1]  # a handwriting image is always one piece
    significant = np.zeros(n, bool)
    significant[1:] = areas >= 0.01 * areas.sum()
    groups = []
    if len(mains) <= 1:
        groups.append(significant.copy())
    else:
        taken = np.zeros(n, bool)
        for main in mains:
            mx, my, mw, mh = stats[main, :4]
            keep = np.zeros(n, bool)
            for i in range(1, n):
                x, y, w, h = stats[i, :4]
                inside = x < mx + mw and mx < x + w and y < my + mh and my < y + h
                keep[i] = not taken[i] and inside and (i == main or significant[i])
            taken |= keep
            groups.append(keep)
    out = []
    for keep in groups:
        a_st = alpha * keep[lab]
        ys, xs = np.nonzero(a_st > 0.1)
        if ys.size == 0:
            continue
        pad = 4
        y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
        a_st = a_st[y0:y1, x0:x1]
        cover = float((a_st > 0.5).mean())
        if max(a_st.shape) < 40 or cover < 0.01:
            continue
        # un-mix: pixel = paper * (1 - a) + ink * a  ->  ink = (pixel - paper * (1 - a)) / a
        pix = rgb[y0:y1, x0:x1].astype(np.float32) / 255
        paper = bg[y0:y1, x0:x1]
        am = np.maximum(a_st, 0.15)[..., None]
        ink = np.clip((pix - paper * (1 - am)) / am, 0, 1)
        ref = np.median(ink[a_st > 0.6], axis=0) if (a_st > 0.6).any() else ink.reshape(-1, 3).mean(0)
        ink = np.where((a_st < 0.3)[..., None], ref, ink)  # faint edges take the stamp's ink colour
        hue = float(cv2.cvtColor(np.uint8([[ref * 255]]), cv2.COLOR_RGB2HSV)[0, 0, 0]) * 2
        rgba = (np.dstack([ink, a_st]) * 255).clip(0, 255).astype(np.uint8)
        _, n_lines = writing_order(a_st, cfg.get("vertical_ratio"))
        out.append((rgba, {"ink_hue": round(hue, 1), "ink_rgb": [int(c * 255) for c in ref],
                           "ink": "black" if np.ptp(v) < 0.25 else "colour",
                           "coverage": round(cover, 3), "size": [int(x1 - x0), int(y1 - y0)],
                           "vertical": is_vertical(a_st, cfg.get("vertical_ratio")),
                           "lines": n_lines}))
    if not out:
        raise ValueError("no stamp-sized ink found")
    return out


def line_bands(alpha: np.ndarray) -> list[tuple[int, int]]:
    """Rows occupied by each line of writing (bands of inked rows separated by gaps)."""
    # a row belongs to a line when it carries a fair share of ink: slanted handwriting never
    # leaves fully empty rows between lines, but the rows between them are much lighter
    density = (alpha > 0.3).mean(axis=1)
    density = np.convolve(density, np.ones(3) / 3, mode="same")
    rows = density > 0.15 * density.max()
    gap = max(2, int(alpha.shape[0] * 0.03))
    bands, start, last = [], None, -10**6
    for y, on in enumerate(rows):
        if on:
            if start is None:
                start = y
            elif y - last > gap:
                bands.append((start, last + 1))
                start = y
            last = y
    if start is not None:
        bands.append((start, last + 1))
    # merge slivers (dots, descenders) into their neighbours
    merged: list[tuple[int, int]] = []
    for b in bands:
        if merged and (b[1] - b[0] < 0.25 * (merged[-1][1] - merged[-1][0]) or merged[-1][1] - merged[-1][0] < 4):
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged or [(0, alpha.shape[0])]


def is_vertical(alpha: np.ndarray, ratio: float | None) -> bool:
    """A writing much taller than wide is taken to be written at 90 degrees (read bottom to top)."""
    h, w = alpha.shape
    return bool(ratio) and h > float(ratio) * w


def writing_order(alpha: np.ndarray, vertical_ratio: float | None = None) -> tuple[np.ndarray, int]:
    """Stroke order map (0..1) and line count of a writing: line by line, left to right.

    A vertical writing (see is_vertical) is turned upright first, so it is revealed bottom to
    top, column after column from the left."""
    vertical = is_vertical(alpha, vertical_ratio)
    a = np.rot90(alpha, k=-1) if vertical else alpha  # clockwise: bottom -> left, left column -> top line
    h, w = a.shape
    bands = line_bands(a)
    order = np.ones((h, w), np.float32)
    centres = np.array([(b0 + b1) / 2 for b0, b1 in bands])
    nearest = np.abs(np.arange(h)[:, None] - centres[None]).argmin(axis=1)  # each row belongs to a line
    for li in range(len(bands)):
        rows = nearest == li
        cols = np.nonzero(a[rows].max(axis=0) > 0.2)[0]
        x0, x1 = (cols.min(), cols.max() + 1) if cols.size else (0, w)
        xs = np.clip((np.arange(w) - x0) / max(1, x1 - x0), 0, 1)
        order[rows] = (li + xs[None, :]) / len(bands)
    return (np.rot90(order, k=1).copy() if vertical else order), len(bands)


def extract_stamp(rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """The largest stamp in an image."""
    return extract_stamps(rgb, cfg)[0]


def _key(path: Path) -> str:
    return safe_key(path.stem, 60)


def process(project, settings: dict, log=lambda m: None, kind: str = "stamps") -> list[dict]:
    """Extract every image in a library folder (kind = "stamps" or "writing") into
    build/<kind>/ as transparent PNGs; returns the index (also saved as index.json)."""
    cfg = settings[kind]
    src = project.root / cfg["source"]
    out_dir = project.build / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return []
    for old in out_dir.glob("*.png"):
        old.unlink()
    index = []
    for f in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        try:
            found = extract_stamps(np.asarray(Image.open(f).convert("RGB")), cfg)
        except ValueError as e:
            log(f"  {kind} {f.name}: skipped ({e})")
            continue
        for n, (rgba, info) in enumerate(found, start=1):
            key = _key(f) + (f"_{n}" if len(found) > 1 else "")
            Image.fromarray(rgba).save(out_dir / f"{key}.png")
            index.append({"id": key, "source": str(f.relative_to(project.root) if f.is_relative_to(project.root)
                                                   else f), "file": f"build/{kind}/{key}.png", **info})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    _review(out_dir, index, project)
    log(f"{kind}: {len(index)} extracted from {cfg['source']} -> build/{kind} (review.jpg)")
    return index


def _review(out_dir: Path, index: list[dict], project) -> None:
    """Each stamp before/after, the result shown on a checkerboard and on paper."""
    if not index:
        return
    tiles = []
    f = font("sans", 14)
    for it in index:
        rgba = Image.open(project.resolve(it["file"]))
        rgba.thumbnail((260, 260))
        w, h = rgba.size
        checker = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(checker)
        for yy in range(0, h, 12):
            for xx in range(0, w, 12):
                if (xx // 12 + yy // 12) % 2:
                    d.rectangle([xx, yy, xx + 11, yy + 11], fill=(205, 205, 205))
        checker.paste(rgba, (0, 0), rgba)
        paper = Image.new("RGB", (w, h), (243, 240, 230))
        paper.paste(rgba, (0, 0), rgba)
        tile = Image.new("RGB", (2 * w + 30, h + 26), (60, 60, 60))
        tile.paste(checker, (0, 0))
        tile.paste(paper, (w + 30, 0))
        ImageDraw.Draw(tile).text((4, h + 5), it["id"][:40], font=f, fill=(235, 235, 235))
        tiles.append(tile)
    W, cols = max(t.width for t in tiles), 4
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    sheet = Image.new("RGB", (W * cols + 10 * cols, sum(max(t.height for t in r) + 8 for r in rows)), (40, 40, 40))
    y = 0
    for r in rows:
        for c, t in enumerate(r):
            sheet.paste(t, (c * (W + 10), y))
        y += max(t.height for t in r) + 8
    sheet.save(out_dir / "review.jpg", quality=88)


def load(project, kind: str = "stamps") -> list[dict]:
    path = project.build / kind / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
