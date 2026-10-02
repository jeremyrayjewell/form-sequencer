"""Rasterize input documents (images, multi-frame TIFFs, PDFs) into a page cache.

All pages are brought to a common scale (settings.documents.dpi) so that text in
fragments from different sources has a comparable size on the composite form.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageSequence

from .config import IMAGE_EXTS, PDF_EXTS, Project, safe_key


@dataclass
class PageRef:
    key: str      # unique page id, e.g. "annex_bundle_pdf_p2"
    source: str   # document path relative to the project folder
    page: int     # 1-based page number within the source
    path: str     # cached PNG relative to the project folder
    width: int
    height: int

    def asdict(self) -> dict:
        return asdict(self)


def _key(path: Path, page: int) -> str:
    return safe_key(path.name) + f"_p{page}"


def _image_pages(path: Path, dpi: int, assume_inches: float) -> list[Image.Image]:
    img = Image.open(path)
    pages = []
    for frame in ImageSequence.Iterator(img):
        im = frame.convert("RGB")
        src_dpi = frame.info.get("dpi", (0, 0))[0] or 0
        if 50 <= src_dpi <= 1200:
            scale = dpi / float(src_dpi)
        else:
            scale = assume_inches * dpi / max(im.size)
        if abs(scale - 1) > 0.02:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        pages.append(im)
    return pages


def rasterize(project: Project, settings: dict) -> list[PageRef]:
    cfg = settings["documents"]
    dpi = int(cfg["dpi"])
    docs = project.documents_dir(settings)
    if not docs.exists():
        raise FileNotFoundError(f"documents folder not found: {docs}")
    project.pages_dir.mkdir(parents=True, exist_ok=True)
    refs: list[PageRef] = []
    used: set[str] = set()
    files = sorted(p for p in docs.rglob("*") if p.suffix.lower() in IMAGE_EXTS | PDF_EXTS)
    if not files:
        raise FileNotFoundError(f"no scans or PDFs in {docs}")
    for f in files:
        rel = Path(os.path.relpath(f, project.root)).as_posix()  # inputs may live outside the project
        if f.suffix.lower() in PDF_EXTS:
            doc = pymupdf.open(f)
            images = []
            max_in = 1.6 * float(cfg["assume_page_inches"])  # larger than ~A3: wrongly sized scan or poster
            for page in doc:
                longest = max(page.rect.width, page.rect.height) / 72.0
                page_dpi = dpi if longest <= max_in else max(36, int(dpi * float(cfg["assume_page_inches"]) / longest))
                pix = page.get_pixmap(dpi=page_dpi, alpha=False)
                images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            doc.close()
        else:
            images = _image_pages(f, dpi, float(cfg["assume_page_inches"]))
        for i, im in enumerate(images, start=1):
            key = _key(f, i)
            if key in used:  # two different names that still map to one key: tell them apart
                key = safe_key(f.name + "#" + rel) + f"_p{i}"
            used.add(key)
            out = project.pages_dir / f"{key}.png"
            im.save(out, optimize=False, compress_level=1)
            refs.append(PageRef(key, rel, i, f"build/pages/{key}.png", im.width, im.height))
    return refs


def load_page(project: Project, ref: dict) -> np.ndarray:
    return np.asarray(Image.open(project.resolve(ref["path"])).convert("RGB"))
