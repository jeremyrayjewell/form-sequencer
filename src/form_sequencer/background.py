"""Background scenes for compositing under the forms in post (``form-seq background``).

Three scenes, chosen for each section of the track by its energy:

  light    quiet sections: soft fields of office-coloured light drifting in the dark
  archive  middle sections: a dim, out-of-focus archive aisle under fluorescent tubes
  sea      loud sections: a deep field of out-of-focus forms drifting far below the pile

Sections come from the music itself: peaks of timbre/loudness novelty, snapped to downbeats,
at least `min_section_bars` apart. Each section is scored by loudness and kick density and
the scenes are assigned by rank (quiet / middle / loud thirds). Scene changes crossfade over a
few beats. Kicks lift the light a little; high hits flicker the tubes and flare the grain.

The forms' camera move is baked in at `parallax` strength, so the background moves less than
the pile and reads as further away: stack final.mov on top and nothing needs keyframing.
Everything is out of focus, so scenes are drawn at reduced resolution (`scale`), then
upscaled, and fine grain is added at full resolution. Every frame is a function of t alone.
"""

from __future__ import annotations

import bisect
import json
import math
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import Project

Log = Callable[[str], None]

FLUORESCENT = (0.86, 0.96, 0.88)
CARBON = (0.26, 0.33, 0.58)
MANILA = (0.82, 0.69, 0.45)
STAMP_RED = (0.62, 0.20, 0.16)
TEAL = (0.20, 0.62, 0.58)
DARK = np.array([0.065, 0.07, 0.066], np.float32)


# ---------------------------------------------------------------------- section plan
def plan_sections(beats: dict, audio: Path, cfg: dict) -> list[dict]:
    """Sections of the track with the scene each one gets."""
    import librosa

    b = np.asarray(beats["beats"], float)
    duration = float(beats["duration"])
    bpb = int(beats.get("beats_per_bar", 4))
    phase = int(beats.get("downbeat_offset") or 0)
    novelty = np.asarray(beats["novelty"], float)
    min_beats = int(cfg["min_section_bars"]) * bpb

    # candidate boundaries: downbeats where the music changes most, kept apart greedily
    downs = [i for i in range(phase, len(b), bpb) if min_beats <= i <= len(b) - min_beats]
    downs.sort(key=lambda i: -novelty[max(0, i - 2):i + 3].max())
    cuts: list[int] = []
    for i in downs:
        if novelty[max(0, i - 2):i + 3].max() < float(cfg["novelty_threshold"]):
            break
        if all(abs(i - c) >= min_beats for c in cuts):
            cuts.append(i)
    edges = [0] + sorted(cuts) + [len(b)]

    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    loud = 20 * np.log10(rms + 1e-6)
    kicks = beats.get("kicks", [])
    sections = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        t0 = 0.0 if i0 == 0 else float(b[i0])
        t1 = duration if i1 >= len(b) else float(b[i1])
        sel = (rms_t >= t0) & (rms_t < t1)
        level = float(np.percentile(loud[sel], 75)) if sel.any() else -60.0
        hits = sum(k["s"] for k in kicks if t0 <= k["t"] < t1 and k["s"] >= 0.4) / max(t1 - t0, 1e-3)
        sections.append({"t0": round(t0, 4), "t1": round(t1, 4), "level": round(level, 2), "kicks": round(hits, 3)})

    def z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / (v.std() + 1e-9)

    score = z([s["level"] for s in sections]) + 0.5 * z([s["kicks"] for s in sections])
    order = np.argsort(np.argsort(score))  # rank 0 = quietest
    n = len(sections)
    names = cfg["scenes"]
    for s, rank, sc in zip(sections, order, score):
        third = 0 if n < 3 else min(2, int(3 * rank / n))
        s["scene"] = names[("quiet", "middle", "loud")[third if n >= 3 else (0 if sc < 0 else 2)]]
        s["score"] = round(float(sc), 3)
    return sections


# ---------------------------------------------------------------------- renderer
class BackgroundRenderer:
    def __init__(self, project: Project, settings: dict, beats: dict, sections: list[dict], camera=None,
                 sheets: list[np.ndarray] | None = None, scale: float = 1.0):
        self.cfg = settings["background"]
        v = settings["video"]
        self.W, self.H = int(v["width"] * scale) // 2 * 2, int(v["height"] * scale) // 2 * 2
        s = float(self.cfg["scale"])
        self.lw, self.lh = max(64, int(self.W * s) // 2 * 2), max(36, int(self.H * s) // 2 * 2)
        self.sections = sections
        self._sec_t = [x["t0"] for x in sections]
        self.camera = camera  # t -> (zoom, view x, view y) in full-frame pixels, or None
        self.cam_W, self.cam_H = int(v["width"]), int(v["height"])
        self.intro, self.outro = float(v.get("intro", 0)), float(v.get("outro", 0))
        self.duration = float(beats["duration"])
        self.period = float(np.median(np.diff(beats["beats"]))) if len(beats["beats"]) > 1 else 0.5
        self.kicks = [(k["t"], k["s"]) for k in beats.get("kicks", [])]
        self.highs = [(h["t"], h["s"]) for h in beats.get("highs", [])]
        self._kt = [k[0] for k in self.kicks]
        self._ht = [h[0] for h in self.highs]
        self.seed = int(settings["seed"])
        self.drift = float(self.cfg.get("drift", 1.0))
        yy, xx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        self._xx, self._yy = xx, yy
        r2 = ((xx / self.lw - 0.5) ** 2 + (yy / self.lh - 0.5) ** 2) * 2
        self.vignette = (1 - 0.55 * r2)[..., None].astype(np.float32)
        rng = np.random.default_rng(self.seed + 77)
        # soft grain (drawn at half size): filmic, and far kinder to the video encoder than pixel noise
        self.grain = [cv2.resize(rng.standard_normal((self.H // 2, self.W // 2)).astype(np.float32), (self.W, self.H),
                                 interpolation=cv2.INTER_LINEAR) for _ in range(12)]
        self._init_light(np.random.default_rng(self.seed + 1))
        self._init_archive()
        self._init_sea(sheets or [], np.random.default_rng(self.seed + 3))

    # ------------------------------------------------------------------ audio envelopes
    def _env(self, times, hits, t, decay, window=1.0) -> float:
        hi = bisect.bisect_right(times, t)
        lo = bisect.bisect_left(times, t - window)
        return min(1.0, sum(hits[i][1] * math.exp(-(t - hits[i][0]) / decay) for i in range(lo, hi)))

    def kick(self, t: float) -> float:
        return self._env(self._kt, self.kicks, t, 0.16)

    def high(self, t: float) -> float:
        return self._env(self._ht, self.highs, t, 0.07, 0.5)

    # ------------------------------------------------------------------ scene weights
    def weights(self, t: float) -> dict[str, float]:
        """Scene mix at t: one scene inside a section, a crossfade around each boundary."""
        i = max(0, bisect.bisect_right(self._sec_t, t) - 1)
        cur = self.sections[i]["scene"]
        half = float(self.cfg["crossfade_beats"]) * self.period / 2
        if i + 1 < len(self.sections) and t > self.sections[i + 1]["t0"] - half:
            nxt, edge = self.sections[i + 1]["scene"], self.sections[i + 1]["t0"]
        elif i > 0 and t < self.sections[i]["t0"] + half:
            nxt, edge, cur = cur, self.sections[i]["t0"], self.sections[i - 1]["scene"]
        else:
            return {cur: 1.0}
        if nxt == cur:
            return {cur: 1.0}
        p = float(np.clip((t - (edge - half)) / (2 * half), 0, 1))
        p = p * p * (3 - 2 * p)
        return {cur: 1 - p, nxt: p}

    # ------------------------------------------------------------------ light
    def _init_light(self, rng) -> None:
        cols = [FLUORESCENT, CARBON, MANILA, TEAL, STAMP_RED, MANILA, CARBON, TEAL]
        self.blobs = []
        for c in cols:
            self.blobs.append({
                "c": np.array(c, np.float32), "r": float(rng.uniform(0.09, 0.2)) * self.lw,
                "a": float(rng.uniform(0.2, 0.34)),
                "w": rng.uniform(0.025, 0.07, 2), "p": rng.uniform(0, 2 * np.pi, 3),
                "ax": float(rng.uniform(0.25, 0.42)), "ay": float(rng.uniform(0.2, 0.35))})

    def scene_light(self, t: float) -> np.ndarray:
        t = t * self.drift
        img = np.empty((self.lh, self.lw, 3), np.float32)
        img[:] = DARK * 1.6
        for bl in self.blobs:
            cx = (0.5 + bl["ax"] * math.sin(bl["w"][0] * t + bl["p"][0])) * self.lw
            cy = (0.5 + bl["ay"] * math.sin(bl["w"][1] * t + bl["p"][1])) * self.lh
            breathe = 1 + 0.25 * math.sin(0.05 * t + bl["p"][2])
            g = np.exp(-((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2 * (bl["r"] * breathe) ** 2))
            img += (bl["a"] * g)[..., None] * bl["c"]
        return img

    # ------------------------------------------------------------------ archive
    def _init_archive(self) -> None:
        self._bays: dict = {}
        lw, lh = self.lw, self.lh
        base = np.empty((lh, lw, 3), np.float32)
        ys = np.linspace(0, 1, lh, dtype=np.float32)[:, None, None]
        base[:] = np.array([0.075, 0.085, 0.08], np.float32) * (1 - ys) + np.array([0.11, 0.11, 0.1], np.float32) * ys
        d = np.sqrt(((self._xx - lw / 2) / lw) ** 2 + ((self._yy - lh * 0.45) / lh) ** 2)
        base += (0.16 * np.exp(-(d / 0.12) ** 2))[..., None] * np.array([0.8, 0.9, 0.82], np.float32)
        self._arch_base = base

    def _bay(self, n: int, side: int) -> list:
        """Boxes on one shelving bay (deterministic per bay, so the aisle never ends)."""
        key = (n, side)
        if key not in self._bays:
            rng = np.random.default_rng((self.seed * 1000003 + n * 7919 + (side + 1) * 104729) % 2**32)
            boxes = []
            for j in range(1, 5):
                yb = -1.15 + 0.6 * j
                z = n * 1.2 + 0.05
                while z < (n + 1) * 1.2 - 0.15:
                    w = float(rng.uniform(0.18, 0.5))
                    if rng.random() < 0.85:
                        h = float(rng.uniform(0.3, 0.5))
                        col = [MANILA, MANILA, (0.7, 0.6, 0.42), (0.36, 0.45, 0.72), (0.72, 0.28, 0.22),
                               (0.3, 0.58, 0.45), (0.6, 0.58, 0.55)][int(rng.integers(7))]
                        tone = float(rng.uniform(0.55, 1.0))
                        boxes.append((yb - h, yb, z, min(z + w, (n + 1) * 1.2 - 0.04), np.array(col) * tone,
                                      bool(rng.random() < 0.6)))
                    z += w + float(rng.uniform(0.01, 0.05))
            self._bays[key] = boxes
            if len(self._bays) > 400:
                self._bays.pop(next(iter(self._bays)))
        return self._bays[key]

    def scene_archive(self, t: float) -> np.ndarray:
        lw, lh = self.lw, self.lh
        img = self._arch_base.copy()
        glow = np.zeros_like(img)
        cz = 0.32 * self.drift * t
        sx = 0.06 * math.sin(0.11 * t)
        sy = 0.03 * math.sin(0.07 * t + 1.0)
        f, cx, cy = 0.8 * lw, lw / 2, lh * 0.45
        fog_c = np.array([0.26, 0.31, 0.29])
        near, far = 0.45, 26.0

        def proj(x, y, z):
            zz = z - cz
            return cx + f * (x - sx) / zz, cy + f * (y - sy) / zz

        def fogged(col, z):
            k = math.exp(-(z - cz) / 9.0)
            return col * k + fog_c * (1 - k)

        def quad(canvas, pts, col):
            cv2.fillConvexPoly(canvas, np.round(np.array(pts) * 16).astype(np.int32), tuple(float(c) for c in col),
                               cv2.LINE_AA, shift=4)

        def wall_quad(canvas, x, y0, y1, z0, z1, col):
            z0, z1 = max(z0, cz + near), min(z1, cz + far)
            if z1 <= z0:
                return
            quad(canvas, [proj(x, y0, z0), proj(x, y0, z1), proj(x, y1, z1), proj(x, y1, z0)], fogged(col, (z0 + z1) / 2))

        first, last = int((cz + near) // 1.2), int((cz + far) // 1.2)
        tube_light = 1.0 - float(self.cfg["high_flicker"]) * self.high(t) * 0.6
        # far to near, so nearer things cover farther ones
        for n in range(last, first - 1, -1):
            for side in (-1, 1):
                x = side * 1.1
                for y0, y1, z0, z1, col, label in self._bay(n, side):
                    wall_quad(img, x, y0, y1, z0, z1, col * 0.75)
                    if label:
                        wall_quad(img, x, y0 + 0.08, y0 + 0.16, z0 + 0.04, z1 - 0.04, np.array([0.9, 0.88, 0.8]) * 0.5)
                for j in range(5):  # shelf edges catch the light
                    yb = -1.15 + 0.6 * j
                    wall_quad(img, x, yb, yb + 0.035, n * 1.2, (n + 1) * 1.2, np.array([0.5, 0.52, 0.5]))
                wall_quad(img, x, -1.3, 1.45, n * 1.2, n * 1.2 + 0.05, np.array([0.14, 0.15, 0.15]))
        for m in range(int((cz + far) // 2.4), int((cz + near) // 2.4) - 1, -1):
            zc = m * 2.4 + 1.2
            z0, z1 = max(zc - 0.7, cz + near), min(zc + 0.7, cz + far)
            if z1 <= z0:
                continue
            # each tube flickers on its own share of the high hits
            hi = bisect.bisect_right(self._ht, t)
            lo = bisect.bisect_left(self._ht, t - 0.4)
            dip = sum(self.highs[i][1] * math.exp(-(t - self.highs[i][0]) / 0.06)
                      for i in range(lo, hi) if (i * 7 + m * 13) % 5 == 0)
            level = tube_light * (1 - min(0.8, float(self.cfg["high_flicker"]) * 2.5 * dip))
            level *= 0.93 + 0.07 * math.sin(1.3 * t + m)
            col = np.array(FLUORESCENT) * level
            k = math.exp(-((z0 + z1) / 2 - cz) / 14.0)
            pts = [proj(-0.1, -1.6, z0), proj(0.1, -1.6, z0), proj(0.1, -1.6, z1), proj(-0.1, -1.6, z1)]
            quad(img, pts, col * k + fog_c * (1 - k))
            quad(glow, pts, col * k)
            pool = [proj(-0.7, 1.45, z0 - 0.3), proj(0.7, 1.45, z0 - 0.3), proj(0.7, 1.45, z1 + 0.3),
                    proj(-0.7, 1.45, z1 + 0.3)]
            quad(glow, pool, col * k * 0.18)
        img = cv2.GaussianBlur(img, (0, 0), max(0.8, 0.005 * lw))  # out of focus
        img += cv2.GaussianBlur(glow, (0, 0), 0.03 * lw) * 0.9 + cv2.GaussianBlur(glow, (0, 0), 0.008 * lw) * 0.5
        return img

    # ------------------------------------------------------------------ sea of forms
    def _init_sea(self, sheets: list[np.ndarray], rng) -> None:
        self.sheets = [s.astype(np.float32) / 255.0 if s.dtype == np.uint8 else s for s in sheets]
        self.layers = []
        if not self.sheets:
            return
        lh = self.lh
        # (count, height as share of frame height, brightness, blur, drift speed), deepest first
        spec = [(40, 0.17, 0.22, 0.010, 0.008), (24, 0.27, 0.31, 0.007, 0.013), (10, 0.42, 0.40, 0.005, 0.02)]
        for count, size, light, blur, speed in spec:
            items = []
            for _ in range(count):
                items.append({"tex": int(rng.integers(len(self.sheets))), "u": float(rng.random()),
                              "v": float(rng.random()), "a": float(rng.uniform(-25, 25)),
                              "spin": float(rng.uniform(-0.6, 0.6)), "s": size * float(rng.uniform(0.85, 1.15))})
            ang = float(rng.uniform(0, 2 * np.pi))
            self.layers.append({"items": items, "light": light, "blur": blur * self.lw,
                                "vel": (speed * math.cos(ang), speed * 0.6 * math.sin(ang))})
        big = self.sheets[int(rng.integers(len(self.sheets)))]
        g = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
        h = int(lh * 2.6)
        w = int(g.shape[1] * h / g.shape[0])
        self._type_big = cv2.GaussianBlur(cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 0.006 * self.lw)
        self._scaled: dict = {}

    def _tex(self, i: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        key = (i, height)
        if key not in self._scaled:
            src = self.sheets[i]
            w = max(2, int(src.shape[1] * height / src.shape[0]))
            self._scaled[key] = cv2.resize(src, (w, height), interpolation=cv2.INTER_AREA)
        return self._scaled[key]

    def scene_sea(self, t: float) -> np.ndarray:
        lw, lh = self.lw, self.lh
        out = np.empty((lh, lw, 3), np.float32)
        out[:] = DARK * 1.2
        # coloured light falling across the sea, slowly moving
        for c, ph in ((CARBON, 0.0), (MANILA, 2.1), (TEAL, 4.2)):
            cx = (0.5 + 0.4 * math.sin(0.021 * t * self.drift + ph)) * lw
            cy = (0.5 + 0.35 * math.sin(0.017 * t * self.drift + ph * 1.3)) * lh
            g = np.exp(-((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2 * (0.3 * lw) ** 2))
            out += (0.12 * g)[..., None] * np.array(c, np.float32)
        wash = out.copy()
        if not self.layers:
            return out
        td = t * self.drift
        tint = np.array([0.92, 0.96, 1.0], np.float32)
        for li, layer in enumerate(self.layers):
            rgb = np.zeros((lh, lw, 3), np.float32)
            alpha = np.zeros((lh, lw), np.float32)
            for it in layer["items"]:
                th = max(4, int(it["s"] * lh))
                tex = self._tex(it["tex"], th)
                h, w = tex.shape[:2]
                span_x, span_y = lw + 2 * w, lh + 2 * h
                x = ((it["u"] + layer["vel"][0] * td) % 1.0) * span_x - w
                y = ((it["v"] + layer["vel"][1] * td) % 1.0) * span_y - h
                angle = it["a"] + it["spin"] * td
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                c, s = abs(M[0, 0]), abs(M[0, 1])
                bw, bh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
                x0, y0 = int(x - bw / 2), int(y - bh / 2)
                M[0, 2] += bw / 2 - w / 2
                M[1, 2] += bh / 2 - h / 2
                X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(lw, x0 + bw), min(lh, y0 + bh)
                if X1 <= X0 or Y1 <= Y0:
                    continue
                M[0, 2] -= X0 - x0
                M[1, 2] -= Y0 - y0
                size = (X1 - X0, Y1 - Y0)
                a = cv2.warpAffine(np.ones((h, w), np.float32), M, size, flags=cv2.INTER_LINEAR)
                p = cv2.warpAffine(tex, M, size, flags=cv2.INTER_LINEAR)
                region, ra = rgb[Y0:Y1, X0:X1], alpha[Y0:Y1, X0:X1]
                region[:] = region * (1 - a[..., None]) + p * a[..., None]
                ra[:] = ra * (1 - a) + a
            k = max(0.5, layer["blur"])
            rgb = cv2.GaussianBlur(rgb, (0, 0), k)
            alpha = cv2.GaussianBlur(alpha, (0, 0), k)[..., None]
            depth_fade = layer["light"] * tint
            out = out * (1 - alpha) + rgb * depth_fade
        out = out * (0.75 + 0.6 * wash / (wash.max() + 1e-6))  # the coloured light tints the sheets too
        # huge, blurred form rules drifting across, like type seen far too close
        g = self._type_big
        gh, gw = g.shape
        ox = int((0.5 + 0.5 * math.sin(0.013 * td)) * max(0, gw - lw))
        oy = int((0.5 + 0.5 * math.sin(0.009 * td + 1.0)) * max(0, gh - lh))
        win = g[oy:oy + lh, ox:ox + lw]
        if win.shape == (lh, lw):
            out *= (1 - 0.3 * (1 - win))[..., None]
        return out

    # ------------------------------------------------------------------ frame
    def fade(self, t: float) -> float:
        """Up from black during the intro, down to black at the end of the outro."""
        f = 1.0
        if self.intro > 0:
            f = min(f, (t + self.intro) / max(0.1, 0.6 * self.intro))
        if self.outro > 0:
            f = min(f, (self.duration + self.outro - t) / max(0.1, 0.4 * self.outro))
        f = float(np.clip(f, 0, 1))
        return f * f * (3 - 2 * f)

    def frame(self, t: float) -> np.ndarray:
        """Background at music time t (negative during the intro, past the end in the outro)."""
        mix = self.weights(t)
        img = None
        for name, w in mix.items():
            if w <= 1e-3:
                continue
            part = getattr(self, f"scene_{name}")(t) * w
            img = part if img is None else img + part
        sat = float(self.cfg.get("saturation", 1.0))
        if sat != 1.0:
            grey = img.mean(axis=2, keepdims=True)
            img = np.clip(grey + (img - grey) * sat, 0, None)
        img *= float(self.cfg["brightness"]) * (1 + float(self.cfg["kick_pulse"]) * self.kick(t))
        # soft ceiling: highlights roll off below paper white, so the pile always stands out
        knee, top = 0.45, float(self.cfg.get("max_level", 0.72))
        over = np.maximum(img - knee, 0)
        img = np.minimum(img, knee) + (top - knee) * (1 - np.exp(-over / (top - knee)))
        img *= self.vignette * self.fade(t)
        if self.camera is not None:
            z, vx, vy = self.camera(t)
            p = float(self.cfg["parallax"])
            zb = 1 + (z - 1) * p
            sx = self.lw / self.cam_W
            ox = (vx - self.cam_W / 2) * p * sx
            oy = (vy - self.cam_H / 2) * p * sx
            cx, cy = self.lw / 2, self.lh / 2
            if zb != 1 or ox or oy:
                M = np.float32([[zb, 0, cx - zb * (cx + ox)], [0, zb, cy - zb * (cy + oy)]])
                img = cv2.warpAffine(img, M, (self.lw, self.lh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        up = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        g = float(self.cfg["grain"]) * (1 + 2.0 * float(self.cfg["high_flicker"]) * self.high(t)) * self.fade(t)
        noise = self.grain[int(round(t * 30)) % len(self.grain)]
        up += (noise * g)[..., None]
        return cv2.convertScaleAbs(up, alpha=255.0)


# ---------------------------------------------------------------------- pipeline
def render_background(project: Project, *, start: float = 0.0, duration: float | None = None,
                      output: Path | None = None, composite: bool = False, log: Log = print) -> Path:
    from .pipeline import sequence
    from .render import FrameRenderer, frame_times
    from .sinks import FFmpegSink, ffmpeg_exe

    settings = project.load_settings()
    tl, manifest = sequence(project, settings, log)
    beats = json.loads(project.beats_path.read_text(encoding="utf-8"))
    audio = project.audio_file(settings)
    sections = plan_sections(beats, audio, settings["background"])
    plan_path = project.build / "background_plan.json"
    plan_path.write_text(json.dumps(sections, indent=1), encoding="utf-8")
    log(f"background: {len(sections)} sections -> {project.show(plan_path)}")
    for s in sections:
        log(f"  {s['t0']:7.2f}-{s['t1']:7.2f}s  {s['scene']:8s} (level {s['level']} dB, kicks {s['kicks']}/s)")

    cam = FrameRenderer(project, settings, manifest, tl) if settings["background"].get("parallax", 0) > 0 else None
    small = FrameRenderer(project, settings, manifest, tl, scale=0.3)
    sheets = [small._frozen(i) for i in range(len(small.sheets) - 1)]  # every sheet that ends up covered
    bg = BackgroundRenderer(project, settings, beats, sections, camera=cam.camera if cam else None, sheets=sheets)

    fps = float(settings["video"]["fps"])
    intro = bg.intro
    total = intro + tl.duration + bg.outro
    end = total if duration is None else min(total, start + duration)
    out = output or project.out / "background.mp4"
    sink = FFmpegSink(out, audio, start=start, duration=end - start, preset=settings["video"]["preset"], crf=18,
                      lead=intro)
    times = frame_times(start, end, fps)
    log(f"rendering background: {len(times)} frames at {bg.W}x{bg.H} (scenes drawn at {bg.lw}x{bg.lh}) -> {out}")
    t_wall = time.time()
    sink.open(bg.W, bg.H, fps)
    step = max(1, len(times) // 10)
    try:
        for i, t in enumerate(times):
            sink.write(bg.frame(t - intro))
            if i % step == 0 and i:
                log(f"  {100 * i // len(times):3d}%  t={t:6.2f}s  ({i / (time.time() - t_wall):.1f} frames/s)")
    finally:
        sink.close()
    log(f"done in {time.time() - t_wall:.1f}s: {out}")

    if composite:
        import subprocess
        forms = project.out / "final.mov"
        if not forms.exists():
            log(f"no {forms.name} to composite (render with video.alpha: true first)")
            return out
        comp = out.with_name("composite.mp4")
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out), "-ss", f"{start:.4f}",
               "-i", str(forms), "-t", f"{end - start:.4f}", "-filter_complex", "[0:v][1:v]overlay=format=auto[v]",
               "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", settings["video"]["preset"],
               "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(comp)]
        log(f"compositing {forms.name} over {out.name} -> {comp}")
        subprocess.run(cmd, check=True)
        log(f"composite: {comp}")
    return out
