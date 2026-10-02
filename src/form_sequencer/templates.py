"""Form templates: page proportions, grid, header position, rule style, slots.

A template is turned into
  * slots  — non-overlapping rectangles (normalized page coordinates) that hold fragments
  * prims  — the drawn structure of the form: rules, shaded bands and printed labels,
             each with a *role*, a stable *key* and a drawing *order*

Roles starting with ``chrome:`` (page frame, header rule, title, footer) exist in every
template; during a form change they morph from one template's position to the next so
the page always reads as a form. Other prims (cell rules, labels) erase and redraw. Two
templates that share a prim key (same geometry, same line) keep it on screen, which is
how sibling templates express "a section was replaced" or "columns shifted".

Line widths and text sizes are in page units u = page height / 1000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .layout import Slot

HEADER_POSITIONS = ("top", "bottom", "left")
RULE_STYLES = {
    #          width, darkness, dash, double inset, shaded label band
    "single": (1.3, 0.80, 0.0, 0.0, False),
    "double": (1.0, 0.80, 0.0, 3.2, False),
    "heavy": (2.8, 0.85, 0.0, 0.0, False),
    "dotted": (1.3, 0.75, 5.0, 0.0, False),
    "boxed": (1.2, 0.80, 0.0, 0.0, True),
}
LABEL_H = 15.0  # section-number strip height, in u
PAD = 5.0       # inner padding of a slot, in u


@dataclass(frozen=True)
class Prim:
    kind: str                      # "line" | "fill" | "text"
    role: str                      # "chrome:..." morphs; anything else erases/redraws
    a: tuple[float, float]         # line start / fill top-left / text anchor (normalized)
    b: tuple[float, float] = (0.0, 0.0)  # line end / fill bottom-right
    width: float = 1.0             # line width (u)
    dark: float = 0.8              # ink darkness 0..1
    dash: float = 0.0              # dash length (u), 0 = solid
    text: str = ""
    size: float = 12.0             # font size (u)
    font: str = "sans-bold"
    anchor: str = "la"             # PIL anchor
    rotate: int = 0                # degrees (90 = reads bottom-to-top)
    knockout: bool = False         # light text on a dark label box
    order: float = 0.0             # 0..1: position in the drawing sequence

    @property
    def key(self) -> tuple:
        r = lambda p: (round(p[0], 4), round(p[1], 4))  # noqa: E731
        return (self.kind, self.role, r(self.a), r(self.b), self.width, self.dark, self.dash, self.text,
                self.size, self.rotate, self.knockout)


@dataclass(frozen=True)
class Template:
    name: str
    family: str
    aspect: float
    margins: dict = field(hash=False, compare=False)
    columns: int = 12
    rows: int = 12
    gutter: float = 0.012
    header_pos: str = "top"
    header_size: float = 0.08
    rules: str = "single"
    title: str = ""
    form_number: str = ""
    slots: tuple[Slot, ...] = ()
    grid: tuple[float, float, float, float] = (0, 0, 1, 1)
    header_box: tuple[float, float, float, float] = (0, 0, 0, 0)
    prims: tuple[Prim, ...] = ()

    def slot(self, name: str) -> Slot:
        return next(s for s in self.slots if s.name == name)


# ---------------------------------------------------------------------- building

def page_size(aspect: float, frame_w: int, frame_h: int, height_fraction: float) -> tuple[int, int]:
    """Page size in pixels for a template aspect: fill the frame height, never overflow its width."""
    ph = frame_h * height_fraction
    pw = ph * aspect
    if pw > frame_w * 0.96:
        pw = frame_w * 0.96
        ph = pw / aspect
    return int(round(pw)), int(round(ph))


def build_templates(settings: dict) -> list[Template]:
    out = [build_template(spec, settings) for spec in settings["templates"]]
    if not out:
        raise ValueError("settings.templates is empty")
    names = [t.name for t in out]
    if len(set(names)) != len(names):
        raise ValueError("template names must be unique")
    return out


def build_template(spec: dict, settings: dict) -> Template:
    name = spec["name"]
    aspect = float(spec["aspect"])
    m = spec["margins"]
    cols, rows = int(spec["columns"]), int(spec["rows"])
    gx = float(spec["gutter"])
    gy = gx * aspect  # same physical gutter vertically
    header = spec.get("header", {}) or {}
    hpos = header.get("position", "top")
    hsize = float(header.get("size", 0.08))
    if hpos not in HEADER_POSITIONS:
        raise ValueError(f"template '{name}': header.position must be one of {HEADER_POSITIONS}")
    rules = spec.get("rules", "single")
    if rules not in RULE_STYLES:
        raise ValueError(f"template '{name}': rules must be one of {sorted(RULE_STYLES)}")

    x0, y0, x1, y1 = m["left"], m["top"], 1 - m["right"], 1 - m["bottom"]
    if hpos == "top":
        hb = (x0, y0, x1 - x0, hsize)
        y0 += hsize + gy
    elif hpos == "bottom":
        hb = (x0, y1 - hsize, x1 - x0, hsize)
        y1 -= hsize + gy
    else:
        hb = (x0, y0, hsize, y1 - y0)
        x0 += hsize + gx
    grid = (x0, y0, x1 - x0, y1 - y0)
    cw = (grid[2] - gx * (cols - 1)) / cols
    rh = (grid[3] - gy * (rows - 1)) / rows
    if cw <= 0 or rh <= 0:
        raise ValueError(f"template '{name}': margins/header leave no room for the grid")

    occupied: dict[tuple[int, int], str] = {}
    raw = []
    for slot_name, s in spec["slots"].items():
        c, r, w, h = (int(v) for v in s["cell"])
        if w < 1 or h < 1 or c < 0 or r < 0 or c + w > cols or r + h > rows:
            raise ValueError(f"template '{name}' slot '{slot_name}': cell {s['cell']} is outside the {cols}x{rows} grid")
        for cc in range(c, c + w):
            for rr in range(r, r + h):
                if (cc, rr) in occupied:
                    raise ValueError(f"template '{name}': slots '{slot_name}' and '{occupied[(cc, rr)]}' "
                                     f"overlap at cell ({cc}, {rr})")
                occupied[(cc, rr)] = slot_name
        rect = (x0 + c * (cw + gx), y0 + r * (rh + gy), w * cw + (w - 1) * gx, h * rh + (h - 1) * gy)
        raw.append((slot_name, (c, r, w, h), rect, tuple(s.get("prefer", []))))
    raw.sort(key=lambda it: (it[1][1], it[1][0]))
    slots = tuple(Slot(name, n, i + 1, cell, rect, pref, aspect) for i, (n, cell, rect, pref) in enumerate(raw))

    form_number = str(spec.get("form_number") or settings["page"].get("form_number", "FS-01"))
    t = Template(name=name, family=str(spec.get("family", name)), aspect=aspect, margins=dict(m), columns=cols,
                 rows=rows, gutter=gx, header_pos=hpos, header_size=hsize, rules=rules,
                 title=str(spec.get("title", "")), form_number=form_number, slots=slots, grid=grid, header_box=hb)
    return Template(**{**t.__dict__, "prims": tuple(_prims(t))})


# ---------------------------------------------------------------------- drawing primitives

def _hline(role, x0, x1, y, **kw) -> Prim:
    return Prim("line", role, (x0, y), (x1, y), **kw)


def _vline(role, x, y0, y1, **kw) -> Prim:
    return Prim("line", role, (x, y0), (x, y1), **kw)


def _prims(t: Template) -> list[Prim]:
    """The printed structure of a template, in drawing order."""
    ux = 1 / 1000 / t.aspect  # one u, horizontally, in normalized page units
    uy = 1 / 1000
    width, dark, dash, inset, banded = RULE_STYLES[t.rules]
    m = t.margins
    x0, y0, x1, y1 = m["left"], m["top"], 1 - m["right"], 1 - m["bottom"]
    out: list[Prim] = []

    # --- chrome: page frame, header rule, title, footer (present in every template)
    fw = 1.6 if t.rules != "heavy" else 3.0
    out += [
        _hline("chrome:frame-top", x0, x1, y0, width=fw, dark=0.85),
        _hline("chrome:frame-bottom", x0, x1, y1, width=fw, dark=0.85),
        _vline("chrome:frame-left", x0, y0, y1, width=fw, dark=0.85),
        _vline("chrome:frame-right", x1, y0, y1, width=fw, dark=0.85),
    ]
    hx, hy, hw, hh = t.header_box
    rule_w = 3.0 if t.rules in ("heavy", "double") else 2.2
    if t.header_pos == "top":
        out.append(_hline("chrome:header-rule", hx, hx + hw, hy + hh, width=rule_w, dark=0.9))
        if t.rules == "double":
            out.append(_hline("chrome:header-rule-2", hx, hx + hw, hy + hh + 5 * uy, width=1.0, dark=0.9))
        out.append(Prim("text", "chrome:title", (hx + 8 * ux, hy + hh * 0.55), text=t.title, size=hh * 1000 * 0.36,
                        anchor="lm"))
        bx = hx + hw - 170 * ux
        out += [_vline("chrome:number-box", bx, hy + 6 * uy, hy + hh - 6 * uy, width=1.2, dark=0.85)]
        out.append(Prim("text", "chrome:number", (hx + hw - 10 * ux, hy + hh * 0.55), text=f"FORM {t.form_number}",
                        size=hh * 1000 * 0.3, anchor="rm"))
    elif t.header_pos == "bottom":
        # engineering-style title block
        out.append(_hline("chrome:header-rule", hx, hx + hw, hy, width=rule_w, dark=0.9))
        if t.rules == "double":
            out.append(_hline("chrome:header-rule-2", hx, hx + hw, hy - 5 * uy, width=1.0, dark=0.9))
        split = hx + hw * 0.62
        out += [
            _vline("chrome:number-box", split, hy, hy + hh, width=1.2, dark=0.85),
            _hline("chrome:block-mid", split, hx + hw, hy + hh / 2, width=1.0, dark=0.8),
        ]
        out.append(Prim("text", "chrome:title", (hx + 10 * ux, hy + hh * 0.5), text=t.title, size=hh * 1000 * 0.34,
                        anchor="lm"))
        out.append(Prim("text", "chrome:number", (split + 10 * ux, hy + hh * 0.25), text=f"FORM {t.form_number}",
                        size=hh * 1000 * 0.24, anchor="lm"))
        out.append(Prim("text", "chrome:sheet", (split + 10 * ux, hy + hh * 0.75), text="SHEET 1 OF 1",
                        size=hh * 1000 * 0.2, anchor="lm", font="mono-bold"))
    else:  # left strip, title reads bottom-to-top
        out.append(_vline("chrome:header-rule", hx + hw, hy, hy + hh, width=rule_w, dark=0.9))
        if t.rules == "double":
            out.append(_vline("chrome:header-rule-2", hx + hw + 5 * ux, hy, hy + hh, width=1.0, dark=0.9))
        out.append(Prim("text", "chrome:title", (hx + hw * 0.5, hy + hh * 0.6), text=t.title,
                        size=hw * t.aspect * 1000 * 0.36, anchor="mm", rotate=90))
        out += [_hline("chrome:number-box", hx, hx + hw, hy + hh * 0.16, width=1.2, dark=0.85)]
        out.append(Prim("text", "chrome:number", (hx + hw * 0.5, hy + hh * 0.08), text=f"FORM {t.form_number}",
                        size=hw * t.aspect * 1000 * 0.26, anchor="mm", rotate=90))
    fy = 1 - m["bottom"] * 0.45
    out += [
        _hline("chrome:footer-rule", x0, x1, fy - m["bottom"] * 0.28, width=1.2, dark=0.8),
        Prim("text", "chrome:footer-left", (x0, fy),
             text=f"FORM {t.form_number}" + ("  ·  COMPOSITE OF RECORDED FRAGMENTS" if t.aspect >= 1.2 else ""),
             size=13, anchor="lm"),
        Prim("text", "chrome:footer-right", (x1, fy), text="PAGE 1 OF 1", size=13, anchor="rm"),
    ]
    n_chrome = len(out)
    for i, p in enumerate(out):
        out[i] = Prim(**{**p.__dict__, "order": 0.2 * i / max(1, n_chrome - 1)})

    # --- cells, in reading order
    cells: list[Prim] = []
    n = len(t.slots)
    for k, s in enumerate(t.slots):
        x, y, w, h = s.rect
        order = 0.25 + 0.75 * k / max(1, n - 1)
        role = f"cell:{s.name}"
        kw = dict(width=width, dark=dark, dash=dash, order=order)
        if banded:
            cells.append(Prim("fill", role, (x, y), (x + w, y + LABEL_H * uy), dark=0.13, order=order))
        cells += [_hline(role, x, x + w, y, **kw), _hline(role, x, x + w, y + h, **kw),
                  _vline(role, x, y, y + h, **kw), _vline(role, x + w, y, y + h, **kw)]
        if inset:
            ix, iy = inset * ux, inset * uy
            ikw = dict(width=0.8, dark=dark, order=order)
            cells += [_hline(role, x + ix, x + w - ix, y + iy, **ikw), _hline(role, x + ix, x + w - ix, y + h - iy, **ikw),
                      _vline(role, x + ix, y + iy, y + h - iy, **ikw), _vline(role, x + w - ix, y + iy, y + h - iy, **ikw)]
        cells.append(_hline(role, x, x + w, y + LABEL_H * uy, width=0.7, dark=0.55, order=order))
        cells.append(Prim("fill", role, (x, y), (x + 24 * ux, y + LABEL_H * uy), dark=0.78, order=order))
        cells.append(Prim("text", role, (x + 12 * ux, y + LABEL_H * uy / 2), text=str(s.number), size=11.5,
                          anchor="mm", knockout=True, order=order))
        if s.prefer and s.prefer[0] == "diagram":
            cells.append(Prim("text", role, (x + w - 8 * ux, y + h - 8 * uy), text=f"FIG. {s.number}", size=11,
                              anchor="rs", font="mono-bold", dark=0.7, order=order))
    return out + cells


# ---------------------------------------------------------------------- comparing templates

def shared_slots(a: Template, b: Template) -> int:
    if abs(a.aspect - b.aspect) > 1e-6:
        return 0
    rects = {tuple(round(v, 5) for v in s.rect) for s in a.slots}
    return sum(tuple(round(v, 5) for v in s.rect) in rects for s in b.slots)


def distance(a: Template, b: Template) -> float:
    """How different two forms look: proportions, header, rules, family, shared cells."""
    d = min(1.0, abs(math.log(a.aspect / b.aspect)) / 0.4)
    d += 0.5 * (a.header_pos != b.header_pos)
    d += 0.3 * (a.rules != b.rules)
    d += 0.3 * (a.family != b.family)
    d += 0.5 * (1 - shared_slots(a, b) / max(len(a.slots), len(b.slots)))
    return d


def match_cells(a: Template, b: Template) -> list[tuple[str, str]]:
    """Pair each cell of form `a` with a nearby cell of form `b` (greedy, deterministic).

    During a form change paired cells glide and resize into each other, so the page
    always shows a ruled grid; only unpaired cells erase or draw. Distances are physical
    (x scaled by page aspect) so pairing works across different page proportions.
    """
    def centre(s: Slot, aspect: float) -> tuple[float, float]:
        x, y, w, h = s.rect
        return (x + w / 2) * aspect, y + h / 2

    costs = []
    for sa in a.slots:
        ca = centre(sa, a.aspect)
        for sb in b.slots:
            cb = centre(sb, b.aspect)
            area = (sa.rect[2] * sa.rect[3] * a.aspect) / (sb.rect[2] * sb.rect[3] * b.aspect)
            cost = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) + 0.25 * abs(math.log(area))
            costs.append((round(cost, 6), sa.name, sb.name))
    pairs, used_a, used_b = [], set(), set()
    for _, na, nb in sorted(costs):
        if na not in used_a and nb not in used_b:
            pairs.append((na, nb))
            used_a.add(na)
            used_b.add(nb)
    return pairs
