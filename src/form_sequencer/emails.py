"""Email exclusion: text lines containing email addresses become forbidden areas.

Lines come from textscan (PDF text layer, or local OCR for raster content). Matching is
deliberately generous — excluding a little too much is the safe failure: a regular
address pattern, an '@' (or its OCR misreading '©') next to a domain-like word, an '@'
between letters (truncated addresses), or a well-known mail provider name. The whole
text line containing a hit is forbidden.
"""

from __future__ import annotations

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+\s?[@©]\s?[A-Za-z0-9\-]+(\s?\.\s?[A-Za-z0-9\-]+)*\s?\.\s?[A-Za-z]{2,}")
AT_DOMAIN = re.compile(r"[@©]\s?[A-Za-z0-9\-]+\s?\.\s?[A-Za-z]{2,}")
AT_WORD = re.compile(r"[A-Za-z0-9][@][A-Za-z0-9]")  # truncated or partly misread addresses
PROVIDERS = re.compile(r"(gmail|googlemail|yahoo|hotmail|outlook|live\.com|icloud|proton|aol\.|gmx\.|mail\.ru|"
                       r"yandex|qq\.com|163\.com|126\.com|naver|daum)", re.I)


def looks_like_email(text: str) -> bool:
    return bool(EMAIL.search(text) or AT_DOMAIN.search(text) or AT_WORD.search(text) or PROVIDERS.search(text))


def find(lines: list[dict], width: int, height: int, cfg: dict, dpi: int = 150) -> list[dict]:
    """Forbidden regions (verdict "email") for the lines of one page."""
    if not cfg.get("enabled", True):
        return []
    margin = int(float(cfg.get("margin_inches", 0.05)) * dpi)
    out = []
    for ln in lines:
        if not looks_like_email(ln["text"]):
            continue
        x, y, w, h = ln["bbox"]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        out.append({"id": f"e{len(out) + 1}", "verdict": "email", "source": ln["source"], "text": ln["text"],
                    "bbox": [x0, y0, min(width, x + w + margin) - x0, min(height, y + h + margin) - y0]})
    return out
