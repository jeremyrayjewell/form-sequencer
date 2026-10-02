"""Text lines on every page, with boxes in pixels of the cached page image.

Two sources:
  * the PDF text layer (exact, instant, with font sizes) for pages that have one;
  * OCR (RapidOCR, local ONNX models; nothing leaves the machine) for raster content:
    image files, scanned PDF pages, and images embedded in otherwise vector PDFs.

Used by email exclusion and title scrambling. OCR results are cached per page image
(content hash) in build/ocr/, so only new or changed pages are read again.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

OCR_VERSION = 1


def _text_layer(pdf: Path, page_no: int, width_px: int) -> tuple[list[dict], list[list[int]], bool]:
    """(text lines, raster image boxes to OCR, page has a text layer)."""
    with pymupdf.open(pdf) as doc:
        page = doc[page_no - 1]
        scale = width_px / page.rect.width
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                x0, y0, x1, y1 = line["bbox"]
                lines.append({"text": " ".join(s["text"].strip() for s in spans), "source": "text layer",
                              "size": round(max(s["size"] for s in spans) * scale, 1),
                              "bbox": [int(x0 * scale), int(y0 * scale), int((x1 - x0) * scale) + 1,
                                       int((y1 - y0) * scale) + 1]})
        images = []
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            if (x1 - x0) * (y1 - y0) > 400:  # ignore tiny rules/bullets drawn as images
                images.append([int(x0 * scale), int(y0 * scale), int((x1 - x0) * scale) + 1, int((y1 - y0) * scale) + 1])
        return lines, images, bool(lines)


# ------------------------------------------------------------------ OCR (in worker processes)

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _ocr_job(args) -> tuple[str, list[dict]]:
    key, png, regions = args
    rgb = np.asarray(Image.open(png).convert("RGB"))
    H, W = rgb.shape[:2]
    lines = []
    for x, y, w, h in regions:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 - x0 < 24 or y1 - y0 < 12:
            continue
        result, _ = _engine()(rgb[y0:y1, x0:x1])
        for box, text, conf in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lines.append({"text": text, "conf": round(float(conf), 3),
                          "bbox": [int(x0 + min(xs)), int(y0 + min(ys)), int(max(xs) - min(xs)) + 1,
                                   int(max(ys) - min(ys)) + 1]})
    return key, lines


def _stamp(png: Path, regions) -> list:
    """Cache key: page image content (not its timestamp: pages are re-rasterized each run)."""
    return [OCR_VERSION, hashlib.sha1(png.read_bytes()).hexdigest(), [list(r) for r in regions]]


def _cache_path(project, key: str) -> Path:
    return project.build / "ocr" / f"{key}.json"


def _cached(project, key: str, png: Path, regions) -> list[dict] | None:
    path = _cache_path(project, key)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["lines"] if data.get("stamp") == _stamp(png, regions) else None


def _ocr_line(ln: dict) -> dict:
    return {"text": ln["text"], "bbox": ln["bbox"], "source": "ocr", "size": float(ln["bbox"][3]),
            "conf": float(ln.get("conf", 1.0))}


def page_lines(project, pages: list[dict], cfg: dict, log=lambda m: None) -> dict[str, list[dict]]:
    """All text lines per page key: [{text, bbox, source, size}] (size = text height in px)."""
    out: dict[str, list[dict]] = {p["key"]: [] for p in pages}
    jobs = []
    for p in pages:
        png = project.resolve(p["path"])
        src = project.root / p["source"]
        regions = [[0, 0, p["width"], p["height"]]]
        if src.suffix.lower() == ".pdf":
            lines, images, has_text = _text_layer(src, p["page"], p["width"])
            out[p["key"]] += lines
            regions = images if has_text else regions
        if not cfg.get("ocr", True) or not regions:
            continue
        cached = _cached(project, p["key"], png, regions)
        if cached is not None:
            out[p["key"]] += [_ocr_line(ln) for ln in cached]
        else:
            jobs.append((p["key"], str(png), regions))
    if jobs:
        workers = int(cfg.get("workers", 0)) or max(1, (os.cpu_count() or 2) - 1)
        log(f"  OCR: {len(jobs)} pages on {min(workers, len(jobs))} processes (cached afterwards)")
        (project.build / "ocr").mkdir(parents=True, exist_ok=True)
        by_key = {j[0]: j for j in jobs}
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for key, lines in pool.map(_ocr_job, jobs, chunksize=1):
                png, regions = Path(by_key[key][1]), by_key[key][2]
                _cache_path(project, key).write_text(json.dumps({"stamp": _stamp(png, regions), "lines": lines}),
                                                     encoding="utf-8")
                out[key] += [_ocr_line(ln) for ln in lines]
    return out
