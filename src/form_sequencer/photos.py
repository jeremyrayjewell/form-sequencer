"""Photo exclusion: find photographic regions on rendered pages and forbid them.

Every page is rendered to an image first (documents.py). On that image:

1. Lighting is flattened (divide by a large-kernel local maximum), so aged paper,
   uneven scans and tinted forms read as white paper while photos keep their tones.
2. Mid-tone pixels are found after a slight blur (which also merges print halftones).
3. A morphological opening removes everything built from thin strokes: text,
   handwriting, rules, line diagrams, charts, stamps, signatures.
4. What survives is examined in small tiles. A tile is *continuous-tone* when it is
   mostly mid-tone, has many tone levels (entropy) and soft edges (low Laplacian).
   Bold type, hatching, QR codes and flat-colour seals have hard edges and stay eligible.
5. Continuous-tone tiles are merged into regions. A region is a ``photo`` if a face is
   found in it or its tones are clearly photographic; otherwise it is ``uncertain``.
   Both are forbidden — uncertainty is resolved by excluding.
6. A page that is mostly continuous tone, or mostly forbidden, is rejected whole.

Manual corrections live in ``<project>/exclusions.yaml`` (see TEMPLATE below) and are
applied on top of detection, so they survive re-analysis.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from .fonts import font

EXCLUSIONS_TEMPLATE = """\
# Manual corrections to photo exclusion. Page keys and region ids are shown on the
# review sheets in build/exclusions/. Coordinates are pixels of build/pages/<page>.png.
#
# <page_key>:
#   reject_page: true          # never use anything from this page
#   allow_page: true           # override an automatic whole-page rejection
#   allow: [p1, p3]            # detected regions that are NOT photos
#   add: [[x, y, w, h]]        # extra forbidden rectangles
#   keep: [s2]                 # detected logo/title areas to leave unscrambled
#   scramble: [[x, y, w, h]]   # extra areas to scramble
{}
"""


def _entropy(v: np.ndarray, bins: int = 64) -> float:
    h = np.histogram(v, bins, (0, 1))[0].astype(np.float64)
    h = h[h > 0] / h.sum()
    return float(-(h * np.log2(h)).sum())


@functools.lru_cache(maxsize=1)
def _face_cascade():
    """OpenCV's frontal-face detector, or None where this OpenCV build lacks it (OpenCV 5
    dropped the Haar cascades). Without it, photos are still found by tone analysis; regions
    that would have been confirmed as 'photo' by a face come out 'uncertain', also excluded."""
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    c = cv2.CascadeClassifier(str(path))
    return None if c.empty() else c


def _normalized(rgb: np.ndarray, dpi: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    k = int(1.0 * dpi) | 1  # wider than a typical photo, so its highlights are not taken for paper
    bg = cv2.GaussianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), (0, 0), k / 4)
    return np.clip(gray / np.maximum(bg, 0.2), 0, 1)


def detect(rgb: np.ndarray, dpi: int, cfg: dict) -> dict:
    """Return {"regions": [{id, bbox, verdict, entropy, texture, faces}], "page": ok|rejected, "reason"}."""
    H, W = rgb.shape[:2]
    n = _normalized(rgb, dpi)
    nb = cv2.GaussianBlur(n, (0, 0), dpi / 100)
    mid = ((nb > 0.2) & (nb < 0.88)).astype(np.uint8)
    r = max(3, int(dpi * 0.04))
    solid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    lap = np.abs(cv2.Laplacian(n, cv2.CV_32F))

    # tile classification
    t = max(8, int(dpi * float(cfg["tile_inches"])))
    tex_max, ent_min = float(cfg["texture_max"]), float(cfg["entropy_min"])
    tiles = np.zeros((H, W), np.uint8)
    for y in range(0, H - t // 2, t // 2):
        for x in range(0, W - t // 2, t // 2):
            s = solid[y:y + t, x:x + t]
            if s.size == 0 or s.mean() < 0.6:
                continue
            m = s > 0
            v = n[y:y + t, x:x + t][m]
            paper = float((nb[y:y + t, x:x + t] > 0.9).mean())  # blurry text still shows paper between letters
            if paper < 0.2 and lap[y:y + t, x:x + t][m].mean() <= tex_max and v.std() >= 0.012:  # soft, not flat
                tiles[y:y + t, x:x + t] = 1

    faces = []
    cascade = _face_cascade()
    if cascade is not None:
        gray8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        min_face = max(20, int(dpi * 0.25))
        faces = [tuple(int(v) for v in f) for f in cascade.detectMultiScale(gray8, 1.1, 6, minSize=(min_face, min_face))]

    regions = []
    min_area = (float(cfg["min_inches"]) * dpi) ** 2
    joined = cv2.morphologyEx(tiles, cv2.MORPH_CLOSE, np.ones((t, t), np.uint8))
    cnt, lab, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    for i in range(1, cnt):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        m = (lab[y:y + h, x:x + w] == i) & (solid[y:y + h, x:x + w] > 0)
        if m.sum() < min_area * 0.5:
            continue
        v = n[y:y + h, x:x + w][m]
        ent = _entropy(v)
        tex = float(lap[y:y + h, x:x + w][m].mean())
        inside = [f for f in faces if x <= f[0] + f[2] / 2 <= x + w and y <= f[1] + f[3] / 2 <= y + h]
        if inside or (ent >= float(cfg["photo_entropy"]) and tex <= tex_max * 0.6):
            verdict = "photo"
        elif ent >= ent_min and tex <= tex_max:
            verdict = "uncertain"
        else:
            continue
        regions.append({"bbox": [x, y, w, h], "verdict": verdict, "entropy": round(ent, 2),
                        "texture": round(tex, 3), "faces": len(inside)})

    # A face inside mid-tone content is a portrait even if its tile area was small.
    for fx, fy, fw, fh in faces:
        box = (max(0, fx - fw // 2), max(0, fy - fh // 2), min(W, fx + fw * 3 // 2), min(H, fy + fh * 2))
        region = solid[box[1]:box[3], box[0]:box[2]]
        covered = any(_overlap(r["bbox"], (box[0], box[1], box[2] - box[0], box[3] - box[1])) for r in regions)
        if not covered and region.size and region.mean() > 0.35:
            regions.append({"bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "verdict": "photo",
                            "entropy": round(_entropy(n[box[1]:box[3], box[0]:box[2]].ravel()), 2),
                            "texture": 0.0, "faces": 1})

    # Grow each region to the full non-paper blobs it touches, so a partly detected
    # photo (bright sky, white shirt) is still excluded in full.
    tone = cv2.morphologyEx((nb < 0.95).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    tcnt, tlab, tstats, _ = cv2.connectedComponentsWithStats(tone, 8)
    for reg in regions:
        x, y, w, h = reg["bbox"]
        ids = np.unique(tlab[y:y + h, x:x + w])
        for j in ids[ids > 0]:
            bx, by, bw, bh, _ = (int(v) for v in tstats[j])
            if bw * bh > 4 * w * h:  # a huge blob is a shaded page area, not this photo
                continue
            x0, y0 = min(x, bx), min(y, by)
            x, y, w, h = x0, y0, max(x + w, bx + bw) - x0, max(y + h, by + bh) - y0
        reg["bbox"] = [x, y, w, h]
    # merge regions that now overlap
    merged = []
    for reg in sorted(regions, key=lambda r: -r["bbox"][2] * r["bbox"][3]):
        for m in merged:
            if _overlap(m["bbox"], reg["bbox"]):
                a, b = m["bbox"], reg["bbox"]
                x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                m["bbox"] = [x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0]
                if reg["verdict"] == "photo":
                    m["verdict"] = "photo"
                m["faces"] += reg["faces"]
                break
        else:
            merged.append(reg)
    regions = merged

    margin = int(float(cfg["margin_inches"]) * dpi)
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for k, reg in enumerate(regions, start=1):
        x, y, w, h = reg["bbox"]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        reg["bbox"] = [x0, y0, min(W, x + w + margin) - x0, min(H, y + h + margin) - y0]
        reg["id"] = f"p{k}"

    page, reason = "ok", ""
    forbidden = np.zeros((H, W), bool)
    for reg in regions:
        x, y, w, h = reg["bbox"]
        forbidden[y:y + h, x:x + w] = True
    if tiles.mean() > float(cfg["reject_page_tone"]):
        page, reason = "rejected", f"page is {tiles.mean():.0%} continuous tone"
    elif forbidden.mean() > float(cfg["reject_page_area"]):
        page, reason = "rejected", f"{forbidden.mean():.0%} of the page is forbidden"
    return {"regions": regions, "page": page, "reason": reason}


def _overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


# ------------------------------------------------------------------ manual corrections

def load_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): (v or {}) for k, v in data.items()}


def ensure_overrides_file(path: Path) -> None:
    if not path.exists():
        path.write_text(EXCLUSIONS_TEMPLATE.format(""), encoding="utf-8")


def apply_overrides(page_key: str, width: int, height: int, detected: dict, overrides: dict) -> dict:
    """Final forbidden areas for a page: detection plus manual corrections."""
    o = overrides.get(page_key, {})
    allow = set(o.get("allow", []) or [])
    rects = [r["bbox"] for r in detected["regions"] if r["id"] not in allow]
    rects += [list(map(int, r)) for r in (o.get("add", []) or [])]
    rejected = detected["page"] == "rejected" and not o.get("allow_page")
    reason = detected["reason"] if rejected else ""
    if o.get("reject_page"):
        rejected, reason = True, "rejected in exclusions.yaml"
    if rejected:
        rects = [[0, 0, width, height]]
    return {"forbidden": rects, "rejected": rejected, "reason": reason}


def intersects(bbox, rects) -> bool:
    return any(_overlap(bbox, r) for r in rects)


# ------------------------------------------------------------------ review sheets

def write_review(out_dir: Path, pages: list[dict], results: dict, load_page) -> list[Path]:
    """Contact sheets: every page with its excluded areas hatched red and labelled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    f_small, f_big = font("sans-bold", 13), font("sans-bold", 30)
    thumbs = []
    for page in pages:
        key = page["key"]
        res = results[key]
        img = Image.fromarray(load_page(page)).convert("RGB")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        allowed = set(res.get("allowed", []))
        for reg in res["detected"]["regions"]:
            x, y, w, h = reg["bbox"]
            ok = reg["id"] in allowed
            col = (40, 150, 40) if ok else {"photo": (210, 30, 30), "email": (30, 90, 220),
                                            "qr": (0, 140, 140)}.get(reg["verdict"], (230, 130, 0))
            if not ok:
                hatch = (np.add.outer(np.arange(h), np.arange(w)) % 18 < 4)
                alpha = np.where(hatch, 150, 60).astype(np.uint8)
                layer = Image.new("RGBA", (w, h), col + (0,))
                layer.putalpha(Image.fromarray(alpha))
                over.alpha_composite(layer, (x, y))
            d.rectangle([x, y, x + w, y + h], outline=col + (255,), width=5)
            d.text((x + 6, y + 4), f"{reg['id']} {'allowed' if ok else reg['verdict']}", font=f_big, fill=col + (255,))
        kept = {r["id"] for r in res.get("scramble", [])} - {r["id"] for r in res.get("scramble_final", [])}
        for reg in res.get("scramble", []) + [r for r in res.get("scramble_final", []) if r["id"].startswith("m")]:
            x, y, w, h = reg["bbox"]
            col = (40, 150, 40) if reg["id"] in kept else (140, 40, 190)
            d.rectangle([x, y, x + w, y + h], outline=col + (255,), width=5)
            d.text((x + 6, y + h + 2), f"{reg['id']} {'kept' if reg['id'] in kept else 'scramble ' + reg['verdict']}",
                   font=f_big, fill=col + (255,))
        for rect in res["final"]["forbidden"]:
            if res["final"]["rejected"]:
                x, y, w, h = rect
                d.line([x, y, x + w, y + h], fill=(210, 30, 30, 255), width=12)
                d.line([x + w, y, x, y + h], fill=(210, 30, 30, 255), width=12)
        img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
        img.thumbnail((300, 390))
        thumbs.append((key, res, img))
    per, cols = 40, 8
    written = []
    for n, start in enumerate(range(0, len(thumbs), per), start=1):
        chunk = thumbs[start:start + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 440), (232, 232, 228))
        d = ImageDraw.Draw(sheet)
        for i, (key, res, img) in enumerate(chunk):
            cx, cy = (i % cols) * 320 + 10, (i // cols) * 440 + 8
            sheet.paste(img, (cx, cy))
            n_reg = len(res["detected"]["regions"])
            kinds = [r["verdict"] for r in res["detected"]["regions"]]
            n_scr = len(res.get("scramble_final", []))
            parts = [f"{kinds.count(v)} {v}" for v in ("photo", "uncertain", "email", "qr") if kinds.count(v)]
            parts += [f"{n_scr} scrambled"] if n_scr else []
            status = "REJECTED: " + res["final"]["reason"] if res["final"]["rejected"] else (
                ", ".join(parts) if parts else "nothing excluded")
            d.text((cx, cy + 395), key[:44], font=f_small, fill=(20, 20, 20))
            d.text((cx, cy + 412), status[:46], font=f_small,
                   fill=(200, 30, 30) if res["final"]["rejected"] or n_reg else (60, 120, 60))
        path = out_dir / f"review_{n:02d}.jpg"
        sheet.save(path, quality=85)
        written.append(path)
    return written
