"""Slots and fitting.

Slot rectangles are expressed in *normalized page coordinates* (0..1 of page width and
height) so the same timeline renders at any resolution and can be consumed by other
front-ends (e.g. TouchDesigner) without knowing about the grid. Templates (templates.py)
build the slots; this module only knows how fragments fit into them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    layout: str                               # template name
    name: str
    number: int                               # printed section number, in reading order
    cell: tuple[int, int, int, int]           # col, row, width, height in grid units
    rect: tuple[float, float, float, float]   # x, y, w, h normalized to the page
    prefer: tuple[str, ...]
    page_aspect: float = 1.294                # width / height of the template's page

    def aspect(self) -> float:
        """Physical width / height of the slot."""
        return self.rect[2] / self.rect[3] * self.page_aspect


def fill_fraction(frag_aspect: float, slot_aspect: float) -> float:
    """Share of the slot area a contained (aspect-preserving) fragment covers."""
    return min(frag_aspect / slot_aspect, slot_aspect / frag_aspect)


def inner_box(x: int, y: int, w: int, h: int, label_h: int, pad: int) -> tuple[int, int, int, int]:
    """Area inside a slot available to a fragment (below the section-number label)."""
    return x + pad, y + label_h + pad, max(1, w - 2 * pad), max(1, h - label_h - 2 * pad)


def fit(fw: int, fh: int, box: tuple[int, int, int, int], center: bool = False) -> tuple[int, int, int, int]:
    """Scale (fw, fh) to fit inside box keeping aspect ratio. Forms align top-left; stamps centre."""
    bx, by, bw, bh = box
    s = min(bw / fw, bh / fh)
    w, h = max(1, round(fw * s)), max(1, round(fh * s))
    if center:
        return bx + (bw - w) // 2, by + (bh - h) // 2, w, h
    return bx, by + (bh - h) // 2 if h < bh * 0.5 else by, w, h
