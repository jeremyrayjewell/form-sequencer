"""Locate a few system fonts, falling back to Pillow's built-in font."""

from __future__ import annotations

import functools
import os
from pathlib import Path

from PIL import ImageFont

_DIRS = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path.home() / ".fonts",
]

FAMILIES = {
    "sans": ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttc"],
    "sans-bold": ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Helvetica.ttc"],
    "serif": ["times.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "Times.ttc"],
    "mono": ["cour.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Courier.ttc"],
    "mono-bold": ["courbd.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "Courier.ttc"],
}


@functools.lru_cache(maxsize=None)
def _find(name: str) -> str | None:
    for d in _DIRS:
        if not d.exists():
            continue
        direct = d / name
        if direct.exists():
            return str(direct)
        for hit in d.rglob(name):
            return str(hit)
    return None


@functools.lru_cache(maxsize=None)
def font(family: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FAMILIES.get(family, FAMILIES["sans"]):
        path = _find(name)
        if path:
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)
