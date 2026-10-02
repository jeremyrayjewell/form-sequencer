"""Deterministic frame renderer.

``FrameRenderer.frame(t)`` is a pure function of (inputs, settings, timeline, t): no
wall-clock, no unseeded randomness. Fragments are *multiplied* onto one sheet of paper
(after their own scanned paper tone is normalized away), so pieces from different sources
read as ink printed on the same document.

The form itself is drawn from template primitives (templates.py). With
``form_changes.style: stack`` each form is a separate sheet: a new sheet slides or drops onto
a pile, covered sheets stay visible underneath exactly as they were, each at its own small
rotation and offset. With ``style: morph`` the form reshapes in place; during a change the
renderer
  1. erases the old cells' rules (staggered, reverse reading order) as their fragments fade,
  2. reshapes the page: its size eases to the new proportions while the frame, header and
     footer slide to their new places and carried fragments travel as paper slips,
  3. draws the new rules (staggered, reading order); shared rules never leave,
  4. lets new fragments print in (their times come from the sequencer).
"""

from __future__ import annotations

import bisect
import math
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import Project
from .fonts import font
from .fragments import crop
from .layout import fit, inner_box
from .sequencer import Segment, Timeline
from .templates import LABEL_H, PAD, Prim, Template, build_templates, match_cells, page_size

RED = np.array([0.78, 0.16, 0.14], np.float32)
INK = 0.9  # darkness of a fully inked rule


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def ease_in_out(p: float) -> float:
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def _progress(t: float, span) -> float:
    a, b = span
    if b <= a:
        return 1.0 if t >= a else 0.0
    return float(np.clip((t - a) / (b - a), 0, 1))


def _stagger(p: float, order: float, spread: float = 0.6) -> float:
    """Per-primitive progress: primitives start one after another along `order`."""
    return float(np.clip((p - order * spread) / (1 - spread), 0, 1))


def normalize_fragment(rgb: np.ndarray) -> np.ndarray:
    """Turn a scanned crop into a multiply map: paper -> 1.0, ink keeps its colour."""
    f = rgb.astype(np.float32) / 255.0
    lum = f.mean(axis=2)
    paper = lum >= np.percentile(lum, 60)
    white = np.array([np.median(f[..., c][paper]) for c in range(3)], np.float32)
    f = np.clip(f / np.maximum(white, 0.2), 0, 1)
    d = 1.0 - f
    d = np.clip((d - 0.08) / 0.92, 0, 1) ** 0.9  # drop paper grain, keep ink
    return (1.0 - d).astype(np.float32)


def _text_mask(size: tuple[int, int], draw_fn) -> np.ndarray:
    img = Image.new("L", size, 0)
    draw_fn(ImageDraw.Draw(img))
    return np.asarray(img, np.float32) / 255.0


def _fade(p: float) -> float:
    """Revision tags: full for the first half, then fade out in four visible steps."""
    return 1.0 if p < 0.5 else round(math.ceil((1 - p) * 8) / 4, 2)


def _lerp(a, b, e):
    return tuple(x + (y - x) * e for x, y in zip(a, b))


def _map(pt, src, dst):
    """Move a point from rectangle src to the same relative place in rectangle dst."""
    return (dst[0] + (pt[0] - src[0]) / src[2] * dst[2], dst[1] + (pt[1] - src[1]) / src[3] * dst[3])


def _same_rect(a, b) -> bool:
    return all(abs(x - y) < 1e-5 for x, y in zip(a, b))


def _orientation(p: Prim) -> str:
    if p.kind != "line":
        return p.kind
    return "h" if abs(p.b[1] - p.a[1]) < 1e-9 else "v" if abs(p.b[0] - p.a[0]) < 1e-9 else "d"


def _morphable(pa: Prim, pb: Prim) -> bool:
    """Chrome that can slide from one form to the next without rotating or changing wording."""
    if pa.kind != pb.kind or _orientation(pa) != _orientation(pb):
        return False
    return pa.kind != "text" or (pa.text == pb.text and pa.rotate == pb.rotate)


def _cells(tpl: Template) -> dict[str, list[Prim]]:
    out: dict[str, list[Prim]] = {s.name: [] for s in tpl.slots}
    for p in tpl.prims:
        if p.role.startswith("cell:"):
            out[p.role[5:]].append(p)
    return out


class FrameRenderer:
    def __init__(self, project: Project, settings: dict, manifest: dict, timeline: Timeline, scale: float = 1.0):
        self.project, self.settings, self.manifest, self.tl = project, settings, manifest, timeline
        v, pg = settings["video"], settings["page"]
        self.W, self.H = _even(v["width"] * scale), _even(v["height"] * scale)
        self.templates: dict[str, Template] = {t.name: t for t in build_templates(settings)}
        hf = float(pg["height_fraction"])
        self.geo = {n: page_size(t.aspect, self.W, self.H, hf) for n, t in self.templates.items()}
        self.u = self.H * hf / 1000.0  # page unit: line widths and type sizes scale with it
        self.label_h = max(6, int(round(LABEL_H * self.u)))
        self.pad = max(2, int(round(PAD * self.u)))
        self.frags = {f["id"]: f for f in manifest["fragments"]}
        self._maps: dict[str, np.ndarray] = {}
        self._resized: OrderedDict = OrderedDict()
        self._sprites: dict = {}
        self._rules: dict = {}
        self._bgs: OrderedDict = OrderedDict()
        self._last_key = None
        self._last_frame: np.ndarray | None = None

        rng = np.random.default_rng(int(settings["seed"]) + 1000)
        self.paper = self._make_paper(rng)
        self.vignette = self._make_vignette()
        # Alpha mode: everything is drawn over a black, fully transparent desk, so the frame holds
        # premultiplied colour, and a transparency plane (1 = desk shows) is kept alongside it.
        # Shadows then fall on the desk as see-through darkness.
        self.alpha = bool(settings["video"].get("alpha", False))
        self.forms = sorted(timeline.forms, key=lambda f: f["t"])
        self.rev_times = sorted([m["t"] for m in timeline.marks] + [f["t"] for f in self.forms])
        self.segments = sorted(timeline.segments, key=lambda s: (s.t0, s.slot))
        self._starts = [s.t0 for s in self.segments]
        self.stack = any(f.get("style") == "stack" for f in self.forms)
        self.pile_depth = int(settings["form_changes"].get("pile_depth", 6))
        self.intro = float(settings["video"].get("intro", 0.0))
        self.outro = float(settings["video"].get("outro", 0.0))
        # with an intro the desk is empty until the music starts; then the first sheet slides in
        first_land = 0.8 if self.intro > 0 else 0.0
        self.sheets = [{"tpl": timeline.initial, "pose": (0.0, 0.0, 0.0), "t": 0.0, "land": first_land,
                        "entry": "bottom" if self.intro > 0 else None}]
        for f in self.forms:
            self.sheets.append({"tpl": f["to"], "pose": tuple(f.get("pose", (0, 0, 0))), "t": f["t"],
                                "land": f.get("land", f["t"]), "entry": f.get("entry")})
        for i, sh in enumerate(self.sheets):
            sh["freeze"] = self.sheets[i + 1]["t"] if i + 1 < len(self.sheets) else math.inf
        self._sheet_cache: dict[int, np.ndarray] = {}
        self.stamps = sorted(getattr(timeline, "stamps", []) or [], key=lambda st: st["t"])
        self._stamp_src: dict[str, np.ndarray] = {}
        self.writings = sorted(getattr(timeline, "writings", []) or [], key=lambda w: w["t"])
        self._writing_src: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._stamp_warp: OrderedDict = OrderedDict()
        self._base_pages: dict[str, np.ndarray] = {}
        self._page_memo = None
        self._placed: OrderedDict = OrderedDict()
        self._pile_cache: OrderedDict = OrderedDict()
        self._desk = cv2.convertScaleAbs(self.vignette, alpha=255.0)
        self._backdrop = self._desk.astype(np.float32)  # what transparent areas show in frame()
        self._backdrop_u8 = self._desk.copy()
        if self.alpha:
            self._desk = np.zeros_like(self._desk)

        self.motion = settings.get("motion", {}) if settings.get("motion", {}).get("enabled", True) else {}
        self.pulses = sorted(getattr(timeline, "pulses", []) or []) if self.motion else []
        self._pulse_t = [p[0] for p in self.pulses]
        focus = []
        if self.motion.get("camera", True):
            for item in self.stamps + self.writings:
                focus.append((item["t"],) + self._page_point(item.get("sheet"), item["layout"], item["x"], item["y"]))
        self.focus = sorted(focus)
        self._focus_t = [f[0] for f in self.focus]

    # ------------------------------------------------------------------ static layers
    def _make_paper(self, rng) -> np.ndarray:
        base = np.array(self.settings["page"]["paper"], np.float32) / 255.0
        low = cv2.GaussianBlur(rng.standard_normal((self.H, self.W)).astype(np.float32), (0, 0), 40 * self.u)
        low /= np.abs(low).max() + 1e-6
        grain = rng.standard_normal((self.H, self.W)).astype(np.float32) * 0.008
        return np.clip(base[None, None, :] * (1 + 0.02 * low + grain)[..., None], 0, 1)

    def _make_vignette(self) -> np.ndarray:
        bg = np.empty((self.H, self.W, 3), np.float32)
        bg[:] = np.array(self.settings["video"]["background"], np.float32) / 255.0
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        v = float(self.settings["video"].get("vignette", 0.35))
        if v <= 0:
            return bg  # flat background, e.g. a chroma-key green
        return bg * (1 - v * (((xx / self.W - 0.5) ** 2 + (yy / self.H - 0.5) ** 2) * 2))[..., None]

    def background(self, pw: int, ph: int) -> np.ndarray:
        """Desk with the page's drop shadow, as uint8 (cached per page size)."""
        key = (pw, ph)
        if key in self._bgs:
            self._bgs.move_to_end(key)
            return self._bgs[key]
        if self.alpha:
            out = np.zeros((self.H, self.W, 3), np.uint8)
            self._bgs[key] = out
            return out
        if not self.settings["video"].get("desk_shadows", True):
            out = cv2.convertScaleAbs(self.vignette, alpha=255.0)  # keep a keyable background clean
            self._bgs[key] = out
            return out
        q = 4  # blur at quarter resolution: the shadow is soft anyway
        sh = np.zeros((self.H // q, self.W // q), np.float32)
        ox, oy = (self.W - pw) // 2, (self.H - ph) // 2
        off = int(8 * self.u)
        sh[(oy + off) // q:(oy + ph + off) // q, (ox + off) // q:(ox + pw + off) // q] = 1
        sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 14 * self.u / q), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        out = cv2.convertScaleAbs(self.vignette * (1 - 0.55 * sh)[..., None], alpha=255.0)
        self._bgs[key] = out
        if len(self._bgs) > 12:
            self._bgs.popitem(last=False)
        return out

    # ------------------------------------------------------------------ form drawing
    def sprite(self, text: str, fnt: str, px: int, rotate: int) -> np.ndarray:
        key = (text, fnt, px, rotate)
        if key not in self._sprites:
            f = font(fnt, max(6, px))
            left, top, right, bottom = f.getbbox(text)
            m = _text_mask((max(1, right - left + 2), max(1, bottom - top + 2)),
                           lambda d: d.text((1 - left, 1 - top), text, font=f, fill=255))
            if rotate:
                m = np.rot90(m, k=rotate // 90).copy()
            self._sprites[key] = m
        return self._sprites[key]

    def _place_sprite(self, mask: np.ndarray, p: Prim, pos, pw: int, ph: int, q: float, u: float) -> None:
        spr = self.sprite(p.text, p.font, int(round(p.size * u)), p.rotate)
        h, w = spr.shape
        x, y = pos[0] * pw, pos[1] * ph
        x -= {"l": 0, "m": w / 2, "r": w}.get(p.anchor[0], 0)
        y -= {"a": 0, "t": 0, "m": h / 2, "s": h, "b": h, "d": h}.get(p.anchor[1], 0)
        x, y = int(round(x)), int(round(y))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(pw, x + w), min(ph, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        glyph = spr[y0 - y:y1 - y, x0 - x:x1 - x] * q
        region = mask[y0:y1, x0:x1]
        if p.knockout:
            region[:] = region * (1 - glyph)
        else:
            np.maximum(region, glyph * p.dark, out=region)

    def _draw(self, mask: np.ndarray, items, pw: int, ph: int) -> None:
        """Draw (prim, q, a, b) items into a darkness mask (0..1). q = how much is drawn."""
        u = ph / 1000.0
        for kind in ("fill", "line", "text"):
            for p, q, a, b in items:
                if p.kind != kind or q <= 0.001:
                    continue
                if kind == "fill":
                    x0, y0 = int(round(a[0] * pw)), int(round(a[1] * ph))
                    x1, y1 = int(round(b[0] * pw)), int(round(b[1] * ph))
                    region = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
                    np.maximum(region, p.dark * q, out=region)
                elif kind == "line":
                    p0 = np.array([a[0] * pw, a[1] * ph])
                    p1 = np.array([b[0] * pw, b[1] * ph])
                    p1 = p0 + (p1 - p0) * q  # partly drawn rules grow from their start
                    th = max(1, int(round(p.width * u)))
                    length = float(np.hypot(*(p1 - p0)))
                    if p.dash > 0 and length > 0:
                        step = p.dash * u
                        segs = [(p0 + (p1 - p0) * (s / length), p0 + (p1 - p0) * (min(s + step, length) / length))
                                for s in np.arange(0, length, 2 * step)]
                    else:
                        segs = [(p0, p1)]
                    for s0, s1 in segs:
                        cv2.line(mask, (int(s0[0] * 16), int(s0[1] * 16)), (int(s1[0] * 16), int(s1[1] * 16)),
                                 float(p.dark), th, cv2.LINE_AA, shift=4)
                else:
                    self._place_sprite(mask, p, a, pw, ph, q, u)

    def static_rules(self, name: str) -> np.ndarray:
        pw, ph = self.geo[name]
        key = (name, pw, ph)
        if key not in self._rules:
            mask = np.zeros((ph, pw), np.float32)
            self._draw(mask, [(p, 1.0, p.a, p.b) for p in self.templates[name].prims], pw, ph)
            self._rules[key] = (1.0 - INK * np.clip(mask, 0, 1))[..., None]
        return self._rules[key]

    def transition_items(self, f: dict, t: float) -> list:
        """Every primitive of both forms with how much of it is drawn at time t.

        Chrome (frame, header, footer) slides to its new position. Cells paired by
        match_cells glide and resize from their old to their new rectangle, crossfading
        from the old rule style to the new, so a ruled grid is on the page throughout.
        Unpaired old cells erase first; unpaired new cells draw last.
        """
        A, B = self.templates[f["from"]], self.templates[f["to"]]
        pe, pr, pd = _progress(t, f["erase"]), ease_in_out(_progress(t, f["reshape"])), _progress(t, f["draw"])
        items = []
        ca = {p.role: p for p in A.prims if p.role.startswith("chrome:")}
        cb = {p.role: p for p in B.prims if p.role.startswith("chrome:")}
        for role in sorted(set(ca) | set(cb)):
            pa, pb = ca.get(role), cb.get(role)
            if pa and pb and _morphable(pa, pb):
                m = Prim(**{**pb.__dict__, "width": pa.width + (pb.width - pa.width) * pr,
                            "size": pa.size + (pb.size - pa.size) * pr})
                items.append((m, 1.0, _lerp(pa.a, pb.a, pr), _lerp(pa.b, pb.b, pr)))
                continue
            if pa:  # e.g. a top header rule becoming a left strip: fade out, never rotate
                items.append((pa, 1 - _stagger(pe, 1 - pa.order), pa.a, pa.b))
            if pb:
                items.append((pb, _stagger(pd, pb.order), pb.a, pb.b))

        cells_a = _cells(A)
        cells_b = _cells(B)
        pairs = self._pairs(A.name, B.name)
        paired_a = {na for na, _ in pairs}
        paired_b = {nb for _, nb in pairs}
        for na, nb in pairs:
            ra, rb = A.slot(na).rect, B.slot(nb).rect
            if f["same_geometry"] and _same_rect(ra, rb) and A.rules == B.rules:
                items += [(p, 1.0, p.a, p.b) for p in cells_b[nb]]  # unchanged cell
                continue
            rt = _lerp(ra, rb, pr)
            items += [(p, 1 - pr, _map(p.a, ra, rt), _map(p.b, ra, rt)) for p in cells_a[na]]
            items += [(p, pr, _map(p.a, rb, rt), _map(p.b, rb, rt)) for p in cells_b[nb]]
        for na, prims in cells_a.items():
            if na not in paired_a:
                items += [(p, 1 - _stagger(pe, 1 - p.order), p.a, p.b) for p in prims]
        for nb, prims in cells_b.items():
            if nb not in paired_b:
                items += [(p, _stagger(pd, p.order), p.a, p.b) for p in prims]
        return items

    def _pairs(self, a: str, b: str) -> list[tuple[str, str]]:
        key = ("pairs", a, b)
        if key not in self._sprites:
            self._sprites[key] = match_cells(self.templates[a], self.templates[b])
        return self._sprites[key]

    def transition_rules(self, f: dict, t: float, pw: int, ph: int) -> np.ndarray:
        mask = np.zeros((ph, pw), np.float32)
        self._draw(mask, self.transition_items(f, t), pw, ph)
        return (1.0 - INK * np.clip(mask, 0, 1))[..., None]

    # ------------------------------------------------------------------ state at time t
    def form_state(self, t: float) -> tuple[str, dict | None]:
        """Current template, and the form change in progress (if any)."""
        cur = self.tl.initial
        for f in self.forms:
            if t < f["t"]:
                break
            if t >= max(f["draw"][1], f["reshape"][1]):
                cur = f["to"]
                continue
            return cur, f
        return cur, None

    def geometry(self, cur: str, f: dict | None, t: float) -> tuple[int, int]:
        if f is None:
            return self.geo[cur]
        (aw, ah), (bw, bh) = self.geo[f["from"]], self.geo[f["to"]]
        e = ease_in_out(_progress(t, f["reshape"]))
        return int(round(aw + (bw - aw) * e)), int(round(ah + (bh - ah) * e))

    def frame_rect(self, rect, layout: str) -> tuple[float, float, float, float]:
        """A slot rect of a template, in frame pixels, with that template's page centred."""
        pw, ph = self.geo[layout]
        ox, oy = (self.W - pw) // 2, (self.H - ph) // 2
        return ox + rect[0] * pw, oy + rect[1] * ph, rect[2] * pw, rect[3] * ph

    # ------------------------------------------------------------------ fragments
    @staticmethod
    def rect_px(r, pw: int, ph: int) -> tuple[int, int, int, int]:
        return int(round(r[0] * pw)), int(round(r[1] * ph)), int(round(r[2] * pw)), int(round(r[3] * ph))

    def frag_map(self, fid: str) -> np.ndarray:
        if fid not in self._maps:
            self._maps[fid] = normalize_fragment(crop(self.project, self.manifest, self.frags[fid]))
        return self._maps[fid]

    def resized(self, fid: str, w: int, h: int) -> np.ndarray:
        key = (fid, w, h)
        if key in self._resized:
            self._resized.move_to_end(key)
            return self._resized[key]
        src = self.frag_map(fid)
        interp = cv2.INTER_AREA if w < src.shape[1] else cv2.INTER_CUBIC
        out = np.clip(cv2.resize(src, (w, h), interpolation=interp), 0, 1)
        self._resized[key] = out
        if len(self._resized) > 96:
            self._resized.popitem(last=False)
        return out

    def seg_state(self, seg: Segment, t: float):
        """(alpha, reveal, zoom, move progress or None) of a segment at t, or None if not visible."""
        if t < seg.t0 or t >= seg.t1 + seg.exit_dur:
            return None
        alpha, reveal, zoom, move = 1.0, 1.0, 1.0, None
        if seg.enter_dur > 0 and t < seg.t0 + seg.enter_dur and seg.enter != "cut":
            p = (t - seg.t0) / seg.enter_dur
            if seg.enter == "print":
                reveal = ease_out(p)
            elif seg.enter == "stamp":
                zoom = 1 + 0.22 * (1 - ease_out(p))
                alpha = min(1.0, p * 2.5)
            elif seg.enter == "move" and seg.from_rect:
                move = round(ease_in_out(p), 4)
        if t >= seg.t1:
            if seg.exit_dur <= 0:
                return None
            alpha *= 1 - (t - seg.t1) / seg.exit_dur
        return round(alpha, 3), round(reveal, 3), round(zoom, 3), move

    def placement(self, seg: Segment, box: tuple[int, int, int, int], zoom: float) -> tuple[int, int, int, int]:
        f = self.frags[seg.frag]
        x, y, w, h = fit(f["bbox"][2], f["bbox"][3], inner_box(*box, self.label_h, self.pad), center=seg.center)
        if zoom != 1.0:
            cx, cy = x + w / 2, y + h / 2
            w, h = max(1, int(w * zoom)), max(1, int(h * zoom))
            x, y = int(cx - w / 2), int(cy - h / 2)
        return x, y, w, h

    def active(self, t: float) -> list[Segment]:
        hi = bisect.bisect_right(self._starts, t)
        return [s for s in self.segments[:hi] if s.t1 + s.exit_dur > t]

    def visible(self, t: float):
        """Segments on the page (in the current geometry) and slips travelling above it."""
        cur, f = self.form_state(t)
        pw, ph = self.geometry(cur, f, t)
        on_page, slips = [], []
        for s in self.active(t):
            st = self.seg_state(s, t)
            if st is None:
                continue
            if st[3] is not None and s.from_layout and s.from_layout != s.layout:
                # carried through a form change: drawn inside its gliding cell, on the page
                on_page.append((s, st, tuple(round(v, 5) for v in _lerp(s.from_rect, s.rect, st[3]))))
            elif st[3] is not None:
                a = self.frame_rect(s.from_rect, s.layout)
                b = self.frame_rect(s.rect, s.layout)
                slips.append((s, st, tuple(round(v, 1) for v in _lerp(a, b, st[3]))))
            elif self.geo[s.layout] == (pw, ph):
                on_page.append((s, st, s.rect))
        return cur, f, (pw, ph), on_page, slips

    # ------------------------------------------------------------------ stacked sheets
    def top_sheet(self, t: float) -> int:
        return bisect.bisect_right([sh["t"] for sh in self.sheets[1:]], t)

    def page_at(self, t: float, tpl: str, sheet: int | None = None) -> tuple[np.ndarray, tuple]:
        """One sheet's page (float RGB) at time t: its form, fragments, swaps, tags, counter."""
        pw, ph = self.geo[tpl]
        on_page, slips = [], []
        for s in self.active(t):
            if s.layout != tpl:
                continue
            st = self.seg_state(s, t)
            if st is None:
                continue
            if st[3] is not None:
                slips.append((s, st, tuple(round(v, 5) for v in _lerp(s.from_rect, s.rect, st[3]))))
            else:
                on_page.append((s, st))
        rev = bisect.bisect_right(self.rev_times, t)
        marks = []
        for m in self.tl.marks:
            if m["t"] <= t < m["t"] + m["dur"] and m.get("layout") == tpl:
                p = (t - m["t"]) / m["dur"]
                marks.append((m["n"], tuple(m["rect"]), _fade(p)))
        stamps = self.stamp_state(t, tpl)
        writing = self.writing_state(t, sheet) if sheet is not None else ()
        key = (tpl, tuple((id(s), st) for s, st in on_page), tuple((id(s), r) for s, _, r in slips), rev,
               tuple(marks), stamps, writing)
        if self._page_memo is not None and self._page_memo[0] == key:
            return self._page_memo[1], key
        if tpl not in self._base_pages:
            self._base_pages[tpl] = np.ascontiguousarray(self.paper[:ph, :pw] * self.static_rules(tpl), np.float32)
        page = self._base_pages[tpl].copy()
        for s, (alpha, reveal, zoom, _) in on_page:
            x, y, w, h = self.placement(s, self.rect_px(s.rect, pw, ph), zoom)
            self._ink(page, s.frag, x, y, w, h, alpha, reveal)
        for s, st, rect in slips:  # swaps: slips of paper moving across this sheet
            x, y, w, h = self.placement(s, self.rect_px(rect, pw, ph), 1.0)
            m = max(2, int(4 * self.u))
            y0, y1, x0, x1 = max(0, y - m), min(ph, y + h + m), max(0, x - m), min(pw, x + w + m)
            page[y0:y1, x0:x1] = self.paper[y0:y1, x0:x1] * 1.01
            self._ink(page, s.frag, x, y, w, h, st[0], 1.0)
        for i, prog in stamps:
            self._stamp(page, self.stamps[i], prog)
        for i, prog in writing:
            self._write(page, self.writings[i], prog)
        for n, rect, alpha in marks:
            self._mark(page, n, self.rect_px(rect, pw, ph), alpha)
        self._footer(page, rev, pw, ph, self.templates[tpl])
        self._page_memo = (key, page)
        return page, key

    def _put_sheet(self, frame: np.ndarray, page_u8: np.ndarray, angle: float, cx: float, cy: float,
                   scale: float = 1.0, shadow: float = 0.5, cover: np.ndarray | None = None,
                   trans: np.ndarray | None = None) -> None:
        """Lay a sheet on the desk: rotated/scaled about its centre, with a soft drop shadow.
        With `cover` (paper coverage of the frame, uint8) the shadow only falls on paper and the
        coverage is updated, so a keyable background is never darkened."""
        placed = self._placement(frame, page_u8.shape[0], page_u8.shape[1], angle, cx, cy, scale, shadow, cover, trans)
        if placed is not None:
            self._blend(frame, page_u8, placed, trans)
            if cover is not None:
                (x0, y0, x1, y1), _, _, a1, _ = placed
                np.maximum(cover[y0:y1, x0:x1], (a1 * 255).astype(np.uint8), out=cover[y0:y1, x0:x1])

    def _placement(self, frame: np.ndarray, h: int, w: int, angle: float, cx: float, cy: float,
                   scale: float = 1.0, shadow: float = 0.5, cover: np.ndarray | None = None,
                   trans: np.ndarray | None = None):
        """Warp matrix, clipped box and alpha of a sheet pose; applies its shadow to `frame`
        (and to the transparency plane `trans`, in alpha mode)."""
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += cx - w / 2
        M[1, 2] += cy - h / 2
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
        pad = int(30 * self.u * scale)
        x0 = max(0, int(corners[:, 0].min()) - pad)
        y0 = max(0, int(corners[:, 1].min()) - pad)
        x1 = min(self.W, int(corners[:, 0].max()) + pad + 1)
        y1 = min(self.H, int(corners[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        M[0, 2] -= x0
        M[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        alpha8 = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        q = 4  # the shadow is soft: blur it at quarter resolution, apply it in 8-bit
        off = int(round(7 * self.u * scale))
        small = cv2.resize(alpha8, (max(1, size[0] // q), max(1, size[1] // q)), interpolation=cv2.INTER_AREA)
        o = off // q  # shift down-right without wrapping (np.roll would bring a sheet's clipped
        if o > 0:     # bottom edge back in at the top as a stray band of shadow)
            shifted = np.zeros_like(small)
            shifted[o:, o:] = small[:-o, :-o]
            small = shifted
        small = small.astype(np.float32)
        small = cv2.GaussianBlur(small, (0, 0), max(1.0, 10 * self.u * scale / q))
        dark = cv2.resize(shadow * small, size, interpolation=cv2.INTER_LINEAR)
        if cover is not None:
            dark *= cover[y0:y1, x0:x1].astype(np.float32) / 255.0  # shadows fall on paper only
        light = (255.0 - dark).astype(np.uint8)
        region = frame[y0:y1, x0:x1]
        region[:] = cv2.multiply(region, cv2.cvtColor(light, cv2.COLOR_GRAY2RGB), scale=1 / 255.0)
        if trans is not None:
            trans[y0:y1, x0:x1] *= 1.0 - dark * (1 / 255.0)
        alpha = alpha8.astype(np.float32) * (1 / 255.0)
        return (x0, y0, x1, y1), M, size, alpha, np.ascontiguousarray(1.0 - alpha)

    @staticmethod
    def _blend(frame: np.ndarray, page_u8: np.ndarray, placed, trans: np.ndarray | None = None) -> None:
        (x0, y0, x1, y1), M, size, a1, a0 = placed
        warped = cv2.warpAffine(page_u8, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frame[y0:y1, x0:x1] = cv2.blendLinear(frame[y0:y1, x0:x1], warped, a0, a1)
        if trans is not None:
            trans[y0:y1, x0:x1] *= a0

    def _sheet_pos(self, pose) -> tuple[float, float, float]:
        angle, dx, dy = pose
        return angle, self.W / 2 + dx * self.H, self.H / 2 + dy * self.H

    def _frozen(self, i: int) -> np.ndarray:
        """A covered sheet, exactly as it was when the next sheet landed on it."""
        if i not in self._sheet_cache:
            sh = self.sheets[i]
            page, _ = self.page_at(sh["freeze"] - 1e-4, sh["tpl"])
            self._sheet_cache[i] = cv2.convertScaleAbs(page, alpha=255.0)
            for old in [k for k in self._sheet_cache if k < i - self.pile_depth - 1]:
                del self._sheet_cache[old]
        return self._sheet_cache[i]

    def pile_key(self, k: int, t: float) -> tuple:
        lo = max(0, k - self.pile_depth)
        return (k, tuple(self.writing_state(t, i) for i in range(lo, k)),
                tuple(self.exit_progress(i, k, t) for i in range(lo, k)))

    def exit_progress(self, i: int, k: int, t: float) -> float:
        """Outro: how far sheet i has slid off the desk (0 = in place, 1 = gone). The top sheet
        k goes first, then each one below it in turn."""
        if self.outro <= 0 or t <= self.tl.duration:
            return 0.0
        start = self.tl.duration + 0.3 + (k - i) * 0.4
        return round(float(np.clip((t - start) / 0.8, 0, 1)), 3)

    def _exit_offset(self, i: int, p: float) -> tuple[float, float, float]:
        """(dx, dy, degrees) of sheet i that has slid off by p, each sheet its own way."""
        dirs = [(1.0, 0.15), (-1.0, 0.1), (0.2, 1.0), (1.0, -0.25), (-1.0, -0.2), (-0.15, 1.0)]
        dx, dy = dirs[(i * 7) % len(dirs)]
        e = p * p  # pulled away: slow start, fast exit
        return dx * self.W * 1.25 * e, dy * self.H * 1.35 * e, (12.0 if dx > 0 else -12.0) * e

    def pile(self, k: int, t: float | None = None) -> np.ndarray:
        """Desk with the covered sheets under sheet k, each with the writing on it at time t
        (cached; it only changes when a sheet arrives or a writing is being written below)."""
        return self.pile_with_cover(k, t)[0]

    def pile_with_cover(self, k: int, t: float | None = None):
        """(pile frame, paper coverage when shadows must stay off the background, transparency
        plane in alpha mode)."""
        key = self.pile_key(k, t if t is not None else self.sheets[k]["t"])
        if key not in self._pile_cache:
            frame = self._desk.copy()
            shadows = self.alpha or self.settings["video"].get("desk_shadows", True)
            cover = None if shadows else np.zeros((self.H, self.W), np.uint8)
            trans = np.ones((self.H, self.W), np.float32) if self.alpha else None
            lo = max(0, k - self.pile_depth)
            for i, writing, gone in zip(range(lo, k), key[1], key[2]):
                if gone >= 1:
                    continue
                page_u8 = self._frozen(i)
                if writing:
                    page = page_u8.astype(np.float32) / 255.0
                    for w_i, prog in writing:
                        self._write(page, self.writings[w_i], prog)
                    page_u8 = cv2.convertScaleAbs(page, alpha=255.0)
                angle, cx, cy = self._sheet_pos(self.sheets[i]["pose"])
                if gone > 0:
                    dx, dy, da = self._exit_offset(i, gone)
                    angle, cx, cy = angle + da, cx + dx, cy + dy
                self._put_sheet(frame, page_u8, angle, cx, cy, shadow=0.35, cover=cover, trans=trans)
            self._pile_cache[key] = (frame, cover, trans)
            if len(self._pile_cache) > 4:
                self._pile_cache.popitem(last=False)
        return self._pile_cache[key]

    def _frame_stack(self, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        k = self.top_sheet(t)
        sh = self.sheets[k]
        page, page_key = self.page_at(t, sh["tpl"], sheet=k)
        pkey = self.pile_key(k, t)
        angle, cx, cy = self._sheet_pos(sh["pose"])
        scale = 1.0
        if t < sh["land"] and sh["entry"]:
            p = ease_out(_progress(t, (sh["t"], sh["land"])))
            ph, pw = page.shape[:2]
            far = {"left": (-(self.W / 2 + pw * 0.6), 0), "right": (self.W / 2 + pw * 0.6, 0),
                   "top": (0, -(self.H / 2 + ph * 0.6)), "bottom": (0, self.H / 2 + ph * 0.6)}
            ox, oy = far.get(sh["entry"], (0, 0))
            cx, cy = cx + ox * (1 - p), cy + oy * (1 - p)
            angle += (1 - p) * (7.0 if sh["entry"] in ("left", "top") else -7.0)
            if sh["entry"] == "drop":
                scale = 1 + 0.18 * (1 - p)  # falling onto the pile from above
        # a stamp slamming down jolts the sheet: it dips and twists a little, then settles
        for i, prog in self.stamp_state(t, sh["tpl"]):
            st = self.stamps[i]
            if prog < 1 and st.get("sheet", k) == k:
                kick = (1 - prog) ** 2 * min(1.0, prog * 6)
                cy += kick * 6 * self.u
                angle += kick * 0.35 * (1 if st["x"] > 0.5 else -1)
        # every kick jostles the pile: the top sheet is pushed, the sheets below follow a beat later
        jx, jy, ja, _ = self.jostle(t)
        cx, cy, angle = cx + jx, cy + jy, angle + ja
        pj = float(self.motion.get("pile_jostle", 0.5))
        if pj > 0 and self.pulses:
            px, py, pa, _ = self.jostle(t - 0.045)
            pile_move = (round(px * pj, 1), round(py * pj, 1), round(pa * pj, 3))
        else:
            pile_move = (0.0, 0.0, 0.0)
        top_gone = self.exit_progress(k, k, t)
        if top_gone > 0:
            dx, dy, da = self._exit_offset(k, top_gone)
            angle, cx, cy = angle + da, cx + dx, cy + dy
        key = ("stack", pkey, page_key, round(angle, 3), round(cx, 1), round(cy, 1), round(scale, 4), pile_move,
               top_gone >= 1)
        if key == self._last_key and self._last_frame is not None:
            return self._last_frame
        page_u8 = cv2.convertScaleAbs(page, alpha=255.0)
        jolting = (round(angle, 3), round(cx, 1), round(cy, 1)) != tuple(round(v, 3 if j == 0 else 1)
                                                                       for j, v in enumerate(self._sheet_pos(sh["pose"])))
        if t >= sh["land"] and not jolting and pile_move == (0.0, 0.0, 0.0):
            # landed: the pile with this sheet's shadow, and its placement, are computed once
            if pkey not in self._placed:
                pile, cover, trans = self.pile_with_cover(k, t)
                base = pile.copy()
                base_trans = None if trans is None else trans.copy()
                placed = self._placement(base, page_u8.shape[0], page_u8.shape[1], angle, cx, cy, 1.0, 0.5, cover,
                                         base_trans)
                self._placed[pkey] = (base, placed, base_trans)
                if len(self._placed) > 3:
                    self._placed.popitem(last=False)
            base, placed, base_trans = self._placed[pkey]
            frame = base.copy()
            trans = None if base_trans is None else base_trans.copy()
            if placed is not None:
                self._blend(frame, page_u8, placed, trans)
        else:
            pile, cover, trans = self.pile_with_cover(k, t)
            if pile_move != (0.0, 0.0, 0.0):
                M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), pile_move[2], 1.0)
                M[0, 2] += pile_move[0]
                M[1, 2] += pile_move[1]
                pile = cv2.warpAffine(pile, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if cover is not None:
                    cover = cv2.warpAffine(cover, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
                if trans is not None:
                    trans = cv2.warpAffine(trans, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
            frame = pile.copy()
            trans = None if trans is None else trans.copy()
            if top_gone < 1:
                self._put_sheet(frame, page_u8, angle, cx, cy, scale, shadow=0.5 + 0.2 * (scale - 1) / 0.18,
                                cover=None if cover is None else cover.copy(), trans=trans)
        self._last_key, self._last_frame = key, (frame, trans)
        return frame, trans

    # ------------------------------------------------------------------ motion
    def _page_point(self, sheet: int | None, layout: str, x: float, y: float) -> tuple[float, float]:
        """A point given as a share of a sheet's page, in frame pixels (at the sheet's resting pose)."""
        pw, ph = self.geo[layout]
        pose = self.sheets[sheet]["pose"] if self.stack and sheet is not None and sheet < len(self.sheets) else (0, 0, 0)
        angle, cx, cy = self._sheet_pos(pose)
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        M[0, 2] += cx - pw / 2
        M[1, 2] += cy - ph / 2
        px, py = M @ np.array([x * pw, y * ph, 1.0])
        return float(px), float(py)

    def jostle(self, t: float) -> tuple[float, float, float, float]:
        """(dx px, dy px, degrees, zoom bump) of the kick jostle at t. Each hit gives a quick
        push with a small rebound, each in its own direction (golden-angle sequence)."""
        if not self.pulses:
            return 0.0, 0.0, 0.0, 0.0
        hi = bisect.bisect_right(self._pulse_t, t)
        lo = bisect.bisect_left(self._pulse_t, t - 0.6)
        dx = dy = da = dz = 0.0
        for i in range(lo, hi):
            pt, s = self.pulses[i]
            d = t - pt
            spring = 1.9 * math.sin(2 * math.pi * 3.5 * d) * math.exp(-d / 0.11)  # peaks near 1
            phi = i * 2.399963
            dx += s * spring * math.cos(phi)
            dy += s * spring * math.sin(phi)
            da += s * spring * math.cos(1.7 * phi + 1.0)
            dz += s * math.exp(-d / 0.09) * min(1.0, d / 0.012)
        m, u = self.motion, self.u
        px = float(m.get("jostle_px", 9)) * u
        return (round(dx * px, 1), round(dy * px, 1), round(da * float(m.get("jostle_deg", 0.6)), 3),
                round(dz * float(m.get("pulse_zoom", 0.01)), 4))

    def _cam_target(self, t: float) -> tuple[float, float, float]:
        i = bisect.bisect_right(self._focus_t, t) - 1
        if i >= 0 and t < self.focus[i][0] + float(self.motion.get("camera_hold", 2.5)):
            pan = float(self.motion.get("camera_pan", 0.4))
            _, fx, fy = self.focus[i]
            return float(self.motion.get("camera_zoom", 0.07)), (fx - self.W / 2) * pan, (fy - self.H / 2) * pan
        return 0.0, 0.0, 0.0

    def camera(self, t: float) -> tuple[float, float, float]:
        """(zoom, view centre x, y). The view follows the latest stamp/writing through a smooth
        (second-order) lag, so it eases toward each new one and drifts back when they stop."""
        z, ox, oy = 0.0, 0.0, 0.0
        if self.focus:
            tau = max(0.05, float(self.motion.get("camera_ease", 0.8)) / 3)
            dt = 1 / 60
            total = 0.0
            for j in range(int(8 * tau / dt) + 1):
                x = j * dt
                w = (x + dt / 2) * math.exp(-x / tau)
                tz, tx, ty = self._cam_target(t - x)
                z, ox, oy, total = z + w * tz, ox + w * tx, oy + w * ty, total + w
            z, ox, oy = z / total, ox / total, oy / total
        zoom = (1 + z) * (1 + self.jostle(t)[3])
        half_w, half_h = self.W / (2 * zoom), self.H / (2 * zoom)
        vx = float(np.clip(self.W / 2 + ox, half_w, self.W - half_w))
        vy = float(np.clip(self.H / 2 + oy, half_h, self.H - half_h))
        return round(zoom, 4), round(vx, 1), round(vy, 1)

    # ------------------------------------------------------------------ frame
    def layers(self, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """The frame at t through the camera: RGB uint8 and, in alpha mode, its transparency
        plane (float, 1 = background shows; the RGB is then premultiplied over black)."""
        if t < 0:  # intro: nothing on the desk yet
            if self.alpha:
                return np.zeros((self.H, self.W, 3), np.uint8), np.ones((self.H, self.W), np.float32)
            return self._backdrop_u8, None
        scene, trans = self._frame_stack(t) if self.stack else self._frame_flat(t)
        if not self.motion:
            return scene, trans
        z, vx, vy = self.camera(t)
        if z == 1.0 and vx == self.W / 2 and vy == self.H / 2:
            return scene, trans
        M = np.float32([[z, 0, self.W / 2 - z * vx], [0, z, self.H / 2 - z * vy]])

        def warp(a):
            return cv2.warpAffine(a, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warp(scene), None if trans is None else warp(trans)

    def composite(self, rgb: np.ndarray, trans: np.ndarray | None) -> np.ndarray:
        """Premultiplied layers over the configured background colour (for viewing copies)."""
        if trans is None:
            return rgb
        t8 = cv2.convertScaleAbs(trans, alpha=255.0)
        return cv2.add(rgb, cv2.multiply(self._backdrop_u8, cv2.merge([t8, t8, t8]), scale=1 / 255.0))

    @staticmethod
    def rgba(rgb: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """Straight (unpremultiplied) RGBA uint8, as video codecs with alpha expect."""
        a8 = cv2.convertScaleAbs(1.0 - trans, alpha=255.0)
        colour = cv2.divide(rgb, cv2.merge([a8, a8, a8]), scale=255)  # 0 where fully transparent
        return cv2.merge([colour, a8])

    def frame(self, t: float) -> np.ndarray:
        """RGB uint8 frame at t (in alpha mode: composited over the background colour)."""
        return self.composite(*self.layers(t))

    def frame_rgba(self, t: float) -> np.ndarray:
        rgb, trans = self.layers(t)
        if trans is None:
            return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        return self.rgba(rgb, trans)

    def _frame_flat(self, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        cur, f, (pw, ph), on_page, slips = self.visible(t)
        rules_key = (cur, None) if f is None else (
            cur, f["t"], round(_progress(t, f["erase"]), 3), round(_progress(t, f["reshape"]), 3),
            round(_progress(t, f["draw"]), 3))
        rev = bisect.bisect_right(self.rev_times, t)
        marks = []
        if f is None:
            for m in self.tl.marks:
                if m["t"] <= t < m["t"] + m["dur"] and m.get("layout", cur) == cur:
                    p = (t - m["t"]) / m["dur"]
                    marks.append((m["n"], tuple(m["rect"]), _fade(p)))
        key = ((pw, ph), rules_key, self.stamp_state(t, cur) if f is None else (),
               self.writing_state(t, self.top_sheet(t)) if f is None else (),
               tuple((id(s), st, r) for s, st, r in on_page),
               tuple((id(s), r) for s, _, r in slips), rev, tuple(marks))
        if key == self._last_key and self._last_frame is not None:
            return self._last_frame

        rules = self.static_rules(cur) if f is None else self.transition_rules(f, t, pw, ph)
        page = self.paper[:ph, :pw] * rules
        for s, (alpha, reveal, zoom, _), rect in on_page:
            x, y, w, h = self.placement(s, self.rect_px(rect, pw, ph), zoom)
            self._ink(page, s.frag, x, y, w, h, alpha, reveal)
        if f is None:
            for i, prog in self.stamp_state(t, cur):
                self._stamp(page, self.stamps[i], prog)
            for i, prog in self.writing_state(t, self.top_sheet(t)):
                self._write(page, self.writings[i], prog)
        for n, rect, alpha in marks:
            self._mark(page, n, self.rect_px(rect, pw, ph), alpha)
        footer_tpl = self.templates[cur if f is None or t < f["reshape"][0] else f["to"]]
        self._footer(page, rev, pw, ph, footer_tpl)

        frame = self.background(pw, ph).copy()
        ox, oy = (self.W - pw) // 2, (self.H - ph) // 2
        frame[oy:oy + ph, ox:ox + pw] = cv2.convertScaleAbs(page, alpha=255.0)
        if slips:
            self._slips(frame, slips)
        trans = self._flat_trans(pw, ph) if self.alpha else None  # opaque page, see-through shadow
        self._last_key, self._last_frame = key, (frame, trans)
        return frame, trans

    def _flat_trans(self, pw: int, ph: int) -> np.ndarray:
        key = ("trans", pw, ph)
        if key not in self._bgs:
            q = 4
            sh = np.zeros((self.H // q, self.W // q), np.float32)
            ox, oy = (self.W - pw) // 2, (self.H - ph) // 2
            off = int(8 * self.u)
            sh[(oy + off) // q:(oy + ph + off) // q, (ox + off) // q:(ox + pw + off) // q] = 1
            sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 14 * self.u / q), (self.W, self.H),
                            interpolation=cv2.INTER_LINEAR)
            trans = 1 - 0.55 * sh
            trans[oy:oy + ph, ox:ox + pw] = 0
            self._bgs[key] = trans
        return self._bgs[key]

    def _ink(self, canvas, fid, x, y, w, h, alpha, reveal) -> None:
        """Multiply a fragment onto a float canvas at (x, y, w, h)."""
        fmap = self.resized(fid, w, h)
        eff = 1.0 - fmap
        if alpha < 1:
            eff = eff * alpha
        if reveal < 1:
            band = max(2.0, 0.12 * h)
            ramp = np.clip((reveal * (h + band) - np.arange(h, dtype=np.float32)) / band, 0, 1)
            eff = eff * ramp[:, None, None]
        H, W = canvas.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] *= 1.0 - eff[y0 - y:y1 - y, x0 - x:x1 - x]

    def _slips(self, frame: np.ndarray, slips) -> None:
        """Carried fragments travel above the page as slips of paper with a soft shadow."""
        boxes = [self.placement(s, tuple(int(round(v)) for v in r), 1.0) for s, _, r in slips]
        m = max(2, int(4 * self.u))
        pad = int(20 * self.u)
        x0 = max(0, min(b[0] for b in boxes) - pad)
        y0 = max(0, min(b[1] for b in boxes) - pad)
        x1 = min(self.W, max(b[0] + b[2] for b in boxes) + pad)
        y1 = min(self.H, max(b[1] + b[3] for b in boxes) + pad)
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1].astype(np.float32) / 255.0
        paper = self.paper[y0:y1, x0:x1]
        for (s, st, _), (x, y, w, h) in zip(slips, boxes):
            x, y = x - x0, y - y0
            off = max(1, int(3 * self.u))
            shadow = np.zeros(region.shape[:2], np.float32)
            shadow[max(0, y - m + off):max(0, y + h + m + off), max(0, x - m + off):max(0, x + w + m + off)] = 1
            shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, 5 * self.u))
            region *= (1 - 0.35 * shadow)[..., None]
            sx0, sy0 = max(0, x - m), max(0, y - m)
            sx1, sy1 = min(region.shape[1], x + w + m), min(region.shape[0], y + h + m)
            if sx1 > sx0 and sy1 > sy0:
                region[sy0:sy1, sx0:sx1] = paper[sy0:sy1, sx0:sx1] * 1.01
            self._ink(region, s.frag, x, y, w, h, st[0], 1.0)
        frame[y0:y1, x0:x1] = cv2.convertScaleAbs(region, alpha=255.0)

    # ------------------------------------------------------------------ rubber stamps
    def stamp_state(self, t: float, tpl: str) -> tuple:
        """(index, slam progress) of the stamps inked on this sheet at time t."""
        out = []
        for i, st in enumerate(self.stamps):
            if st["t"] > t:
                break
            until = st["until"] if st["until"] is not None else math.inf
            if st["layout"] == tpl and t < until:
                out.append((i, round(min(1.0, (t - st["t"]) / max(st["dur"], 1e-3)), 2)))
        return tuple(out)

    def _stamp_rgba(self, sid: str) -> np.ndarray:
        if sid not in self._stamp_src:
            path = self.project.build / "stamps" / f"{sid}.png"
            self._stamp_src[sid] = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
        return self._stamp_src[sid]

    def _stamp(self, page: np.ndarray, st: dict, prog: float) -> None:
        """A stamp slamming down: it drops from 150% to its size and inks in, then stays."""
        ph, pw = page.shape[:2]
        e = ease_out(prog)
        zoom = 1 + 0.5 * (1 - e)
        ink_in = min(1.0, prog * 3)
        size = st["size"] * ph * zoom
        key = (st["stamp"], int(size), round(st["angle"], 1))
        if key not in self._stamp_warp:
            src = self._stamp_rgba(st["stamp"])
            h, w = src.shape[:2]
            scale = size / max(h, w)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), st["angle"], scale)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            bw, bh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
            M[0, 2] += bw / 2 - w / 2
            M[1, 2] += bh / 2 - h / 2
            self._stamp_warp[key] = cv2.warpAffine(src, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_CONSTANT)
            if len(self._stamp_warp) > 48:
                self._stamp_warp.popitem(last=False)
        spr = self._stamp_warp[key]
        h, w = spr.shape[:2]
        x, y = int(st["x"] * pw - w / 2), int(st["y"] * ph - h / 2)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(pw, x + w), min(ph, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        part = spr[y0 - y:y1 - y, x0 - x:x1 - x]
        a = part[..., 3:4] * (st["opacity"] * ink_in)
        page[y0:y1, x0:x1] *= 1 - a * (1 - part[..., :3])

    # ------------------------------------------------------------------ handwriting
    def writing_state(self, t: float, sheet: int) -> tuple:
        """(index, writing progress) of the writings on a sheet at time t."""
        out = []
        for i, w in enumerate(self.writings):
            if w["t"] > t:
                break
            if w["sheet"] == sheet:
                out.append((i, round(min(1.0, (t - w["t"]) / max(w["dur"], 1e-3)), 2)))
        return tuple(out)

    def _writing_rgba(self, wid: str) -> tuple[np.ndarray, np.ndarray]:
        """The writing image and its stroke order map: line by line, left to right in each line
        (bottom to top, column by column, for a writing much taller than wide)."""
        if wid not in self._writing_src:
            from .stamps import writing_order
            rgba = np.asarray(Image.open(self.project.build / "writing" / f"{wid}.png").convert("RGBA"),
                              np.float32) / 255.0
            order, _ = writing_order(rgba[..., 3], self.settings["writing"].get("vertical_ratio"))
            self._writing_src[wid] = (rgba, order)
        return self._writing_src[wid]

    def _write(self, page: np.ndarray, w: dict, prog: float) -> None:
        """Handwriting appearing as if written: line by line, left to right, soft pen front."""
        ph, pw = page.shape[:2]
        key = ("w", w["writing"], int(w["size"] * ph), round(w["angle"], 1), prog if prog < 1 else 1.0)
        if key not in self._stamp_warp:
            rgba, order = self._writing_rgba(w["writing"])
            src = rgba.copy()
            if prog < 1:
                soft = 0.06
                src[..., 3] *= np.clip((prog * (1 + soft) - order) / soft, 0, 1)
            h, wd = src.shape[:2]
            scale = w["size"] * ph / max(h, wd)
            M = cv2.getRotationMatrix2D((wd / 2, h / 2), w["angle"], scale)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            bw, bh = int(h * sin + wd * cos) + 2, int(h * cos + wd * sin) + 2
            M[0, 2] += bw / 2 - wd / 2
            M[1, 2] += bh / 2 - h / 2
            self._stamp_warp[key] = cv2.warpAffine(src, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_CONSTANT)
            if len(self._stamp_warp) > 64:
                self._stamp_warp.popitem(last=False)
        spr = self._stamp_warp[key]
        h, wd = spr.shape[:2]
        x, y = int(w["x"] * pw - wd / 2), int(w["y"] * ph - h / 2)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(pw, x + wd), min(ph, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        part = spr[y0 - y:y1 - y, x0 - x:x1 - x]
        page[y0:y1, x0:x1] *= 1 - part[..., 3:4] * (1 - part[..., :3])

    def _mark(self, page, n: int, rect_px, alpha: float) -> None:
        """Red revision tag in the cell's label strip: the form was just amended here."""
        x, y, w, h = rect_px
        key = ("mark", n, self.label_h)
        if key not in self._sprites:
            f = font("sans-bold", max(7, int(11 * self.u)))
            text = f"AMENDED  R{n:03d}"
            tw = int(f.getlength(text)) + int(10 * self.u)
            self._sprites[key] = _text_mask((tw, self.label_h), lambda d: (
                d.rectangle([0, 0, tw - 1, self.label_h - 1], outline=255, width=max(1, int(self.u))),
                d.text((tw // 2, self.label_h // 2 + 1), text, font=f, fill=255, anchor="mm")))
        m = self._sprites[key]
        mh, mw = m.shape
        mx, my = x + w - mw - self.pad, y
        if mx < 0 or my + mh > page.shape[0] or mx + mw > page.shape[1]:
            return
        page[my:my + mh, mx:mx + mw] *= 1 - (alpha * m)[..., None] * (1 - RED)

    def _footer(self, page, rev: int, pw: int, ph: int, tpl: Template) -> None:
        spr = self.sprite(f"REV. {rev:03d}", "mono-bold", int(16 * self.u), 0)
        mh, mw = spr.shape
        x = (pw - mw) // 2
        y = int((1 - tpl.margins["bottom"] * 0.45) * ph - mh / 2)
        if 0 <= y and y + mh <= ph and x >= 0:
            page[y:y + mh, x:x + mw] *= (1 - 0.85 * spr)[..., None]


def frame_times(start: float, end: float, fps: float) -> list[float]:
    n = int(math.floor((end - start) * fps + 1e-9))
    return [start + i / fps for i in range(n)]
