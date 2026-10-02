"""Turn beat analysis + fragments + templates into a deterministic timeline.

The timeline is plain data (JSON-serializable): *segments* (a fragment shown in a slot
rectangle of a template from t0 to t1, with an entry transition), *form changes* (the
underlying template changing, with phase times), and revision marks. Renderers only
interpret it; all choices and all randomness (a seeded numpy Generator) live here, so a
given audio analysis + manifest + settings + seed always yields the same timeline.

Musical logic
  * Form changes (major): planned from the audio analysis — downbeats where the music
    changes (novelty) and major onsets, at least form_changes.min_bars apart. Each change
    has a *drama* that picks how different the next template is, how many fragments are
    carried over, and how long the change takes. Phases: old cells erase -> page
    reshapes (frame/header/footer slide) -> new rules draw -> fragments arrive on beat
    subdivisions.
  * Revisions (minor), on the detected beats: a token bucket (events_per_bar) limits how
    often cells change; tokens are spent on downbeats or strong beats; each cell holds its
    fragment for min_hold_beats; swaps exchange two cells; strong off-beat onsets stamp.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .layout import Slot, fill_fraction
from .templates import Template, build_templates, distance, match_cells, page_size

TRANSITIONS = ("print", "stamp", "cut", "move")


@dataclass
class Segment:
    frag: str
    layout: str                                   # template the rect belongs to
    slot: str
    rect: tuple[float, float, float, float]
    t0: float
    t1: float = math.inf
    enter: str = "print"
    enter_dur: float = 0.25
    exit_dur: float = 0.0
    from_rect: tuple[float, float, float, float] | None = None
    center: bool = False
    from_layout: str | None = None                # template of from_rect (moves across a form change)


@dataclass
class Timeline:
    duration: float
    seed: int
    initial: str                                         # template shown at t=0
    segments: list[Segment] = field(default_factory=list)
    forms: list[dict] = field(default_factory=list)      # form changes with phase times
    marks: list[dict] = field(default_factory=list)      # revision marks {t, rect, layout, n, dur}
    events: list[dict] = field(default_factory=list)     # human-readable log
    templates: dict = field(default_factory=dict)        # name -> {aspect, slots: [{name, number, rect}]}
    stamps: list = field(default_factory=list)           # rubber stamps slammed on kicks {t, until, layout, ...}
    writings: list = field(default_factory=list)         # handwriting written on high hits {t, sheet, layout, ...}
    pulses: list = field(default_factory=list)           # [t, strength] of the hits the sheets jostle to

    def to_json(self) -> dict:
        d = asdict(self)
        for s in d["segments"]:
            s["t1"] = None if math.isinf(s["t1"]) else round(s["t1"], 4)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Timeline":
        segs = []
        for s in d["segments"]:
            s = dict(s)
            s["t1"] = math.inf if s["t1"] is None else s["t1"]
            s["rect"] = tuple(s["rect"])
            s["from_rect"] = tuple(s["from_rect"]) if s["from_rect"] else None
            segs.append(Segment(**s))
        return cls(d["duration"], d["seed"], d["initial"], segs, d["forms"], d["marks"], d["events"], d["templates"],
                   d.get("stamps", []), d.get("writings", []), d.get("pulses", []))


@dataclass(frozen=True)
class Frag:
    id: str
    kind: str
    w: int
    h: int
    page: str = ""
    x: int = 0
    y: int = 0
    ink: float = 0.05  # share of inked pixels; near-empty grids read as blank cells

    def overlaps(self, other: "Frag") -> bool:
        """True if both crops come from overlapping areas of the same source page."""
        return self.page == other.page and self.x < other.x + other.w and other.x < self.x + self.w \
            and self.y < other.y + other.h and other.y < self.y + self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h


class Sequencer:
    def __init__(self, beats: dict, manifest: dict, settings: dict, stamps: list[dict] | None = None,
                 writing: list[dict] | None = None):
        self.cfg = settings["sequencer"]
        self.fcfg = settings["form_changes"]
        self.settings = settings
        self.rng = np.random.default_rng(int(settings["seed"]))
        self.templates: list[Template] = build_templates(settings)
        self.tmap = {t.name: t for t in self.templates}
        self.frags = sorted(
            (Frag(f["id"], f["kind"], f["bbox"][2], f["bbox"][3], f["page"], f["bbox"][0], f["bbox"][1],
                  f.get("ink", 0.05))
             for f in manifest["fragments"] if f.get("enabled", True) and not f.get("excluded")),
            key=lambda f: f.id,
        )
        if not self.frags:
            raise ValueError("no usable fragments in the manifest")
        self.by_id = {f.id: f for f in self.frags}
        v = settings["video"]
        self.page_px = {t.name: page_size(t.aspect, v["width"], v["height"], settings["page"]["height_fraction"])
                        for t in self.templates}
        self.b = np.asarray(beats["beats"], float)
        self.strength = np.asarray(beats["strength"], float)
        self.loud = np.asarray(beats.get("strength_raw", beats["strength"]), float)
        self.novelty = np.asarray(beats["novelty"], float)
        self.bpb = int(beats["beats_per_bar"])
        self.phase = int(beats["downbeat_offset"])
        self.onsets = beats.get("onsets", [])
        self.kicks = beats.get("kicks", [])
        self.stamp_library = stamps or []
        self.highs = beats.get("highs", [])
        self.writing_library = writing or []
        self.duration = float(beats["duration"])

        self.current: dict[str, Segment] = {}
        self.last_change: dict[str, int] = {}
        self.last_used: dict[str, int] = {}
        self.revision = 0
        self.busy_until = -math.inf
        self.tl = Timeline(self.duration, int(settings["seed"]), self.templates[0].name)
        self.tl.templates = {
            t.name: {"aspect": t.aspect, "family": t.family,
                     "slots": [{"name": s.name, "number": s.number, "rect": list(s.rect)} for s in t.slots]}
            for t in self.templates
        }

    # ------------------------------------------------------------------ helpers
    def period(self, i: int) -> float:
        i = min(max(i, 0), len(self.b) - 2)
        return float(self.b[i + 1] - self.b[i])

    def slot_px(self, s: Slot) -> tuple[float, float]:
        pw, ph = self.page_px[s.layout]
        return s.rect[2] * pw, s.rect[3] * ph

    def scale(self, f: Frag, s: Slot) -> float:
        """Rendered pixels per source pixel if f is fitted into s."""
        sw, sh = self.slot_px(s)
        return min(sw / f.w, sh / f.h)

    def fits(self, f: Frag, s: Slot, min_fill: float) -> bool:
        return fill_fraction(f.aspect, s.aspect()) >= min_fill and 0.25 <= self.scale(f, s) <= 1.3

    def on_page(self) -> set[str]:
        return {seg.frag for seg in self.current.values()}

    def frag(self, fid: str) -> Frag:
        return self.by_id[fid]

    def subdivisions(self, t_from: float, n: int, per_beat: int = 2) -> list[float]:
        """The next n beat subdivisions (from the detected beats, so tempo drift is kept) at or after t_from."""
        out = []
        for k in range(len(self.b) - 1):
            for j in range(per_beat):
                t = self.b[k] + j * (self.b[k + 1] - self.b[k]) / per_beat
                if t >= t_from - 1e-6:
                    out.append(float(t))
                    if len(out) == n:
                        return out
        while len(out) < n:  # past the last beat: keep the final tempo
            out.append((out[-1] if out else t_from) + self.period(len(self.b) - 2) / per_beat)
        return out

    def score(self, f: Frag, s: Slot, beat: int, relax: bool = False) -> float:
        fill = fill_fraction(f.aspect, s.aspect())
        if fill < float(self.cfg["min_fill"]) * (0.5 if relax else 1.0):
            return 0.0
        if s.prefer:
            kind = 1.0 if f.kind == s.prefer[0] else 0.6 if f.kind in s.prefer else 0.12
        else:
            kind = 0.5
        if f.kind == "header" and s.prefer and s.prefer[0] != "header":
            kind *= 0.3  # headers read as headers; keep them out of body slots
        scale = self.scale(f, s)
        legible = float(np.clip((scale - 0.3) / 0.25, 0.02, 1.0))  # small print must stay readable
        # avoid visibly soft blow-ups: mild enlargement is fine, big ones read as a different document
        upscale = 1.0 if scale <= 1.3 else 0.12 if scale <= 2.2 else (0.0 if not relax else 0.02)
        age = beat - self.last_used.get(f.id, -10**6)
        cool = 1.0 if relax or age >= int(self.cfg["reuse_cooldown_beats"]) else 0.08
        # Variety: the same kind many times at once reads as repetition, and two crops of the
        # same source area on the page at once read as a duplicate.
        shown = [self.frag(seg.frag) for name, seg in self.current.items() if name != s.name]
        same_kind = sum(o.kind == f.kind for o in shown)
        lookalike = sum(o.kind == f.kind and abs(math.log(o.aspect / f.aspect)) < 0.25 for o in shown)
        dup = any(f.overlaps(o) for o in shown)
        variety = 0.3 ** same_kind * 0.25 ** lookalike * (0.02 if dup and not relax else 1.0)
        content = float(np.clip(f.ink / 0.05, 0.1, 1.0))
        return fill ** 1.5 * kind * legible * upscale * cool * variety * content

    def pick(self, s: Slot, beat: int, exclude: set[str]) -> Frag | None:
        for relax in (False, True):
            scored = [(self.score(f, s, beat, relax), f) for f in self.frags if f.id not in exclude]
            scored = [(sc, f) for sc, f in scored if sc > 0]
            if scored:
                scored.sort(key=lambda x: (-x[0], x[1].id))
                top = scored[:8]
                w = np.array([sc for sc, _ in top]) ** 3
                return top[int(self.rng.choice(len(top), p=w / w.sum()))][1]
        return None

    def place(self, s: Slot, f: Frag, t: float, beat: int, enter: str, dur: float, from_rect=None,
              exit_dur: float | None = None, mark: bool = True, from_layout: str | None = None) -> Segment:
        old = self.current.get(s.name)
        if old is not None:
            old.t1 = t
            if exit_dur is None:
                exit_dur = 0.0 if enter == "cut" else dur * 0.6
            old.exit_dur = exit_dur
        seg = Segment(f.id, s.layout, s.name, s.rect, round(t, 4), enter=enter, enter_dur=round(dur, 4),
                      from_rect=from_rect, center=f.kind == "stamp",
                      from_layout=(from_layout or s.layout) if from_rect else None)
        self.tl.segments.append(seg)
        self.current[s.name] = seg
        self.last_change[s.name] = beat
        self.last_used[f.id] = beat
        if mark:
            self.revision += 1
            self.tl.marks.append({"t": round(t, 4), "rect": list(s.rect), "layout": s.layout, "n": self.revision,
                                  "dur": round(4 * self.period(beat), 3)})
        return seg

    # ------------------------------------------------------------------ minor events
    def substitute(self, form: Template, t: float, beat: int, strong: bool, accent: bool = False) -> bool:
        hold = int(self.cfg["min_hold_beats"]) // (2 if accent else 1)
        eligible = [s for s in form.slots if beat - self.last_change.get(s.name, -10**6) >= hold]
        if accent:
            eligible = [s for s in eligible if "stamp" in s.prefer or "signature" in s.prefer]
        if not eligible:
            return False
        age = np.array([beat - self.last_change.get(s.name, beat - hold) for s in eligible], float) + 1
        s = eligible[int(self.rng.choice(len(eligible), p=age / age.sum()))]
        f = self.pick(s, beat, self.on_page())
        if f is None:
            return False
        enter = "stamp" if (accent or f.kind == "stamp") else ("cut" if strong and self.rng.random() < 0.3 else "print")
        dur = float(self.cfg["transition_beats"]) * self.period(beat)
        if enter == "stamp":
            dur *= 0.6
        self.place(s, f, t, beat, enter, dur)
        self.tl.events.append({"t": round(t, 4), "beat": beat, "type": "accent" if accent else "substitute",
                               "slot": s.name, "frag": f.id, "transition": enter})
        return True

    def swap(self, form: Template, t: float, beat: int) -> bool:
        hold = int(self.cfg["min_hold_beats"])
        eligible = [s for s in form.slots if beat - self.last_change.get(s.name, -10**6) >= hold and s.name in self.current]
        pairs = []
        for i, a in enumerate(eligible):
            for b in eligible[i + 1:]:
                fa, fb = self.frag(self.current[a.name].frag), self.frag(self.current[b.name].frag)
                need = float(self.cfg["min_fill"]) + 0.15
                if self.fits(fa, b, need) and self.fits(fb, a, need):
                    pairs.append((a, b, fa, fb))
        if not pairs:
            return False
        a, b, fa, fb = pairs[int(self.rng.integers(len(pairs)))]
        dur = float(self.cfg["move_beats"]) * self.period(beat)
        ra, rb = self.current[a.name].rect, self.current[b.name].rect
        self.place(b, fa, t, beat, "move", dur, from_rect=ra, exit_dur=0.0)
        self.place(a, fb, t, beat, "move", dur, from_rect=rb, exit_dur=0.0)
        self.tl.events.append({"t": round(t, 4), "beat": beat, "type": "swap", "slots": [a.name, b.name]})
        return True

    # ------------------------------------------------------------------ form changes
    def plan_form_changes(self, start: int) -> list[tuple[int, float, str]]:
        """Beats where the form itself changes: (beat index, magnitude 0..1, reason).

        Candidates are section changes (novelty at a downbeat) and major onsets (beats
        louder than a percentile). They are accepted strongest first, keeping min_bars
        apart; gaps longer than max_bars get the strongest downbeat inside them.
        """
        c = self.fcfg
        bars = int(c["min_bars"]) * self.bpb
        nov_thr = float(c["section_novelty"])
        downs = [k for k in range(start, len(self.b)) if (k - self.phase) % self.bpb == 0]
        cands: dict[int, tuple[float, str]] = {}
        for k in downs:
            if self.novelty[k] >= nov_thr:
                cands[k] = (float(self.novelty[k] / nov_thr), "section")
        pct = float(c["major_onset_percentile"])
        if pct < 100 and len(self.loud):
            thr = float(np.percentile(self.loud, pct))
            for k in range(start, len(self.b)):
                if self.loud[k] >= thr and self.loud[k] > 0:
                    mag = float(self.loud[k] / thr)
                    if k not in cands or cands[k][0] < mag:
                        cands[k] = (mag, "onset")
        accepted: list[tuple[int, float, str]] = []
        for k, (mag, why) in sorted(cands.items(), key=lambda kv: (-kv[1][0], kv[0])):
            if k - start >= bars // 2 and all(abs(k - a) >= bars for a, _, _ in accepted) \
                    and len(self.b) - k > self.bpb:
                accepted.append((k, mag, why))
        max_bars = int(c["max_bars"]) * self.bpb
        if max_bars > 0:
            changed = True
            while changed:
                changed = False
                marks = sorted([start] + [a for a, _, _ in accepted] + [len(self.b) - 1])
                for lo, hi in zip(marks[:-1], marks[1:]):
                    if hi - lo <= max_bars:
                        continue
                    inside = [k for k in downs if lo + bars <= k <= hi - bars]
                    if not inside:
                        continue
                    best = max(inside, key=lambda k: (self.novelty[k] + self.strength[k], -k))
                    accepted.append((best, 0.0, "interval"))
                    changed = True
                    break
        accepted.sort()
        if len(accepted) > 1:
            # magnitude as a rank (0 = mildest change in the track, 1 = biggest)
            order = np.argsort(np.argsort([m for _, m, _ in accepted], kind="stable"), kind="stable")
            accepted = [(k, float(r / (len(accepted) - 1)), why) for (k, _, why), r in zip(accepted, order)]
        elif accepted:
            accepted = [(accepted[0][0], 0.5, accepted[0][2])]
        return accepted

    def choose_template(self, cur: Template, drama: float) -> Template:
        """The template whose difference from the current form best matches the drama,
        preferring forms not seen recently or often, so the document keeps evolving."""
        others = [t for t in self.templates if t.name != cur.name]
        dists = np.array([distance(cur, t) for t in others])
        target = drama * dists.max()
        recent = [f["from"] for f in self.tl.forms[-2:]]
        seen = [f["to"] for f in self.tl.forms] + [self.tl.initial]
        cost = np.abs(dists - target) + self.rng.uniform(0, 0.08, len(others))
        cost += np.array([0.6 * (t.name in recent) + 0.15 * seen.count(t.name) for t in others])
        return others[int(np.argmin(cost))]

    def form_change(self, cur: Template, k: int, magnitude: float, reason: str) -> Template:
        c = self.fcfg
        drama = float(c["drama"])
        if c["drama_follows_music"]:
            drama = float(np.clip(drama + (magnitude - 0.5) * 0.6, 0.0, 1.0))
        new = self.choose_template(cur, drama)
        T = float(self.b[k])
        P = self.period(k)
        D = (float(c["beats_min"]) + drama * (float(c["beats_max"]) - float(c["beats_min"]))) * P
        if c.get("style", "stack") == "stack":
            return self._stack(cur, new, k, T, P, D, drama, reason)
        same_geometry = abs(cur.aspect - new.aspect) < 1e-6
        # erase unpaired cells -> morph paired cells (and the page) -> draw new cells -> arrivals
        erase, reshape, draw, arrive = (T, T + 0.3 * D), (T + 0.25 * D, T + 0.6 * D), (T + 0.5 * D, T + 0.85 * D), T + 0.8 * D

        old = dict(self.current)
        self.current = {}
        partner = {nb: na for na, nb in match_cells(cur, new)}
        kept: set[str] = set()
        carried: set[str] = set()
        arrivals: list[Slot] = []
        for s in new.slots:
            name = partner.get(s.name)
            seg = old.get(name) if name else None
            if seg is None:
                arrivals.append(s)
                continue
            if same_geometry and tuple(round(v, 5) for v in seg.rect) == tuple(round(v, 5) for v in s.rect):
                kept.add(name)  # identical cell in a sibling form: the fragment simply stays
                self.current[s.name] = seg
                continue
            f = self.frag(seg.frag)
            if self.fits(f, s, 0.45) and self.rng.random() > 0.8 * drama:
                # the fragment rides its cell while the cell glides to its new place
                carried.add(name)
                seg.t1, seg.exit_dur = reshape[0], 0.0
                self.place(s, f, reshape[0], k, "move", reshape[1] - reshape[0], from_rect=seg.rect,
                           mark=False, from_layout=seg.layout)
            else:
                arrivals.append(s)
        for name, seg in old.items():
            if name not in kept and name not in carried:
                seg.t1, seg.exit_dur = T, erase[1] - T  # fade out while the cells erase
        times = self.subdivisions(arrive, len(arrivals), max(1, int(c["arrival_subdivision"])))
        exclude = self.on_page() | {seg.frag for seg in old.values()}
        for s, ta in zip(arrivals, times):
            kb = int(np.clip(np.searchsorted(self.b, ta), 0, len(self.b) - 1))
            f = self.pick(s, kb, exclude)
            if f is not None:
                exclude.add(f.id)
                self.place(s, f, ta, kb, "print", float(self.cfg["transition_beats"]) * self.period(kb), mark=False)
        self.busy_until = (times[-1] if times else draw[1]) + P
        self.revision += 1
        form = {"t": round(T, 4), "beat": k, "from": cur.name, "to": new.name, "drama": round(drama, 3),
                "reason": reason, "dur": round(D, 4), "style": "morph", "same_geometry": same_geometry,
                "erase": [round(v, 4) for v in erase], "reshape": [round(v, 4) for v in reshape],
                "draw": [round(v, 4) for v in draw], "arrive": round(arrive, 4)}
        self.tl.forms.append(form)
        self.tl.events.append({"t": round(T, 4), "beat": k, "type": "form", "from": cur.name, "to": new.name,
                               "drama": round(drama, 3), "reason": reason})
        return new

    def _stack(self, cur: Template, new: Template, k: int, T: float, P: float, D: float, drama: float,
               reason: str) -> Template:
        """A fresh sheet carrying the new form lands on the pile; the old sheet stays underneath
        exactly as it was, and the new sheet's cells fill on beat subdivisions after landing."""
        c = self.fcfg
        messy = 0.4 + 0.6 * drama
        angle = float(self.rng.uniform(-1, 1) * float(c["pile_angle"]) * messy)
        po = float(c["pile_offset"])
        dx, dy = (self.rng.uniform(-1, 1, 2) * po * messy).tolist()
        prev_pose = self.tl.forms[-1]["pose"] if self.tl.forms else [0.0, 0.0, 0.0]
        if abs(cur.aspect - new.aspect) < 0.15:  # a similar shape would hide the sheet below
            vx, vy = dx - prev_pose[1], dy - prev_pose[2]
            dist = float(np.hypot(vx, vy))
            if dist < 0.6 * po:
                if dist < 1e-6:
                    a = float(self.rng.uniform(0, 2 * np.pi))
                    vx, vy, dist = np.cos(a), np.sin(a), 1.0
                dx = float(np.clip(prev_pose[1] + vx / dist * 0.6 * po, -1.2 * po, 1.2 * po))
                dy = float(np.clip(prev_pose[2] + vy / dist * 0.6 * po, -1.2 * po, 1.2 * po))
        entry = str(self.rng.choice(["left", "right", "top", "bottom", "drop"]))
        land = T + D
        for seg in self.current.values():
            seg.t1, seg.exit_dur = T, 0.0  # frozen on the covered sheet
        self.current = {}
        times = self.subdivisions(land, len(new.slots), max(1, int(c["arrival_subdivision"])))
        exclude: set[str] = set()
        for s, ta in zip(new.slots, times):
            kb = int(np.clip(np.searchsorted(self.b, ta), 0, len(self.b) - 1))
            f = self.pick(s, kb, exclude)
            if f is not None:
                exclude.add(f.id)
                self.place(s, f, ta, kb, "print", float(self.cfg["transition_beats"]) * self.period(kb), mark=False)
        self.busy_until = (times[-1] if times else land) + P
        self.revision += 1
        form = {"t": round(T, 4), "beat": k, "from": cur.name, "to": new.name, "drama": round(drama, 3),
                "reason": reason, "dur": round(D, 4), "style": "stack", "same_geometry": False,
                "erase": [round(T, 4)] * 2, "reshape": [round(T, 4), round(land, 4)],
                "draw": [round(land, 4)] * 2, "arrive": round(land, 4),
                "land": round(land, 4), "pose": [round(angle, 3), round(dx, 4), round(dy, 4)], "entry": entry}
        self.tl.forms.append(form)
        self.tl.events.append({"t": round(T, 4), "beat": k, "type": "form", "style": "stack", "from": cur.name,
                               "to": new.name, "drama": round(drama, 3), "reason": reason, "entry": entry})
        return new

    # ------------------------------------------------------------------ main
    def run(self) -> Timeline:
        cfg = self.cfg
        form = self.templates[0]

        # Intro: fill cells one by one in reading order from the first downbeat.
        step = float(cfg["intro_beats_per_slot"])
        per_beat = max(1, int(round(1 / step))) if step < 1 else 1
        stride = max(1, int(round(step))) if step >= 1 else 1
        times = self.subdivisions(float(self.b[self.phase]), len(form.slots) * stride, per_beat)[::stride]
        for s, ts in zip(form.slots, times):
            i = int(np.clip(np.searchsorted(self.b, ts + 1e-6) - 1, 0, len(self.b) - 1))
            f = self.pick(s, i, self.on_page())
            if f is not None:
                self.place(s, f, ts, i, "print", float(cfg["transition_beats"]) * self.period(i), mark=False)
        start = int(np.searchsorted(self.b, times[-1] + 1e-6)) if times else self.phase

        changes = {k: (m, why) for k, m, why in self.plan_form_changes(start)} if len(self.templates) > 1 else {}

        # Candidate moments: beats after the intro, plus strong off-beat onsets.
        moments = [(float(self.b[k]), "beat", k) for k in range(start, len(self.b))]
        if cfg["offbeat_accents"] and self.onsets:
            first = self.b[max(start - 1, 0)]
            off = []
            for o in self.onsets:
                k = int(np.clip(np.searchsorted(self.b, o["t"]), 1, len(self.b) - 1))
                nearest = min(abs(self.b[k - 1] - o["t"]), abs(self.b[k] - o["t"]))
                if nearest > 0.2 * self.period(k - 1):
                    off.append((o, k - 1))
            if off:
                typical = float(np.median([o["s"] for o, _ in off]))
                thr = max(0.3, float(self.settings["audio"]["accent_ratio"]) * typical)
                moments += [(float(o["t"]), "accent", k) for o, k in off if o["s"] >= thr and o["t"] > first]
        moments.sort()

        tokens = 0.0
        rate = float(cfg["events_per_bar"]) / self.bpb
        last_accent = -10**6
        last_beat_seen = start - 1
        for t, what, k in moments:
            if what == "beat":
                tokens = min(3.0, tokens + rate * (k - last_beat_seen))
                last_beat_seen = k
                if k in changes:
                    form = self.form_change(form, k, *changes[k])
                    tokens = 0.0
                    continue
            if t < self.busy_until:
                continue  # the form is being rebuilt
            down = (k - self.phase) % self.bpb == 0
            if what == "accent":
                # accents bypass the budget (but not a half hold); spacing keeps them rare
                if k - last_accent >= int(cfg["accent_min_bars"]) * self.bpb \
                        and self.substitute(form, t, k, strong=True, accent=True):
                    last_accent = k
                continue
            if tokens < 1.0:
                continue
            strong = self.strength[k] >= float(cfg["strength_threshold"])
            if not (down or strong):
                continue
            if self.rng.random() < float(cfg["swap_probability"]) and self.swap(form, t, k):
                tokens -= 1.0
                continue
            if self.substitute(form, t, k, strong=self.strength[k] > 0.85):
                tokens -= 1.0
                if down and tokens >= 1.0 and self.strength[k] > 0.85 and self.substitute(form, t, k, strong=True):
                    tokens -= 1.0

        self.tl.segments.sort(key=lambda s: (s.t0, s.slot))
        t_start = float(self.b[start]) if start < len(self.b) else self.duration
        self.plan_stamps(t_start)
        self.plan_writing(t_start)
        self.plan_pulses()
        return self.tl

    def plan_pulses(self) -> None:
        """The hits the pile jostles to: kicks (or beats, without kick data) above
        `motion.pulse_percentile`, at most one per 0.1 s."""
        c = self.settings.get("motion", {})
        hits = [(float(k["t"]), float(k["s"])) for k in self.kicks] or                [(float(t), float(s)) for t, s in zip(self.b, self.strength)]
        if not hits or not c.get("enabled", True):
            return
        thr = float(np.percentile([s for _, s in hits], float(c.get("pulse_percentile", 40))))
        last = -1.0
        for t, s in hits:
            if s >= thr and t - last >= 0.1:
                self.tl.pulses.append([round(t, 4), round(s, 3)])
                last = t

    def size_cap(self, item: dict, layout: str, cfg: dict) -> float:
        """Largest size (share of page height) at which a stamp/writing image is not enlarged
        beyond `max_upscale` times its own resolution in the final render (so it stays sharp)."""
        ph = self.page_px[layout][1]
        return float(cfg.get("max_upscale", 1.25)) * max(item["size"]) / ph

    def plan_writing(self, t_start: float) -> None:
        """Handwriting written on strong high-band hits (hats, cymbals, high notes).

        In stack mode it goes on the top sheet or, with probability `lower_sheets`, on the
        visible part of a sheet lower in the pile (never under a sheet covering it at that
        moment). A separate seeded stream keeps all other choices unchanged."""
        from . import pile
        c = self.settings["writing"]
        if not c.get("enabled", True) or not self.writing_library or not self.highs:
            return
        rng = np.random.default_rng(int(self.settings["seed"]) + 5151)
        v = self.settings["video"]
        W, H = int(v["width"]), int(v["height"])
        thr = float(np.percentile([h["s"] for h in self.highs], float(c["trigger_percentile"])))
        forms = self.tl.forms
        stack = self.fcfg.get("style", "stack") == "stack"
        depth = int(self.fcfg.get("pile_depth", 6))
        change_times = [f["t"] for f in forms]
        busy = [(f["t"], max(f.get("land", f["t"]), f["draw"][1])) for f in forms]
        per_sheet: dict[int, int] = {}
        last_t, last_id = -math.inf, None
        sheets = [(self.tl.initial, (0.0, 0.0, 0.0))] + [(f["to"], tuple(f.get("pose", (0, 0, 0)))) for f in forms]
        grid = 8
        for h in self.highs:
            t = float(h["t"])
            if h["s"] < thr or t < t_start or t >= self.duration - 1.0:
                continue
            beat = int(np.clip(np.searchsorted(self.b, t) - 1, 0, len(self.b) - 1))
            if t - last_t < float(c["min_beats"]) * self.period(beat) or any(a <= t < b for a, b in busy):
                continue
            k = int(np.searchsorted(change_times, t, side="right"))
            min_size = float(c.get("min_size", 0.1))
            top_layout = sheets[k][0]
            sharp = [w for w in self.writing_library if self.size_cap(w, top_layout, c) >= min_size]
            choices = [w for w in sharp if w["id"] != last_id] or sharp
            if not choices:
                continue
            wr = choices[int(rng.integers(len(choices)))]
            size = min(float(c["size"]) * float(rng.uniform(0.75, 1.25)), self.size_cap(wr, top_layout, c))
            angle = float(rng.uniform(-1, 1) * float(c["max_angle"]))
            target, x, y = k, None, None
            if stack and k > 0 and rng.random() < float(c["lower_sheets"]):
                lo = max(0, k - depth)
                geo = []
                for j in range(lo, k + 1):
                    pw, ph = self.page_px[sheets[j][0]]
                    a, cx, cy = pile.centre(sheets[j][1], W, H)
                    geo.append((pw, ph, pile.matrix(pw, ph, a, cx, cy)))
                masks = pile.visible_masks(geo, W, H, grid)
                options = []
                ww, wh = wr["size"]
                min_size = float(c.get("min_size", 0.1))
                for idx, j in enumerate(range(lo, k)):
                    if per_sheet.get(j, 0) >= int(c["max_per_sheet"]):
                        continue
                    pw, ph = geo[idx][:2]
                    # the writing may run under the sheets above, but at least `min_visible` of its
                    # actual (rotated) outline must show; it shrinks (down to min_size) until it fits
                    m = masks[idx].astype(np.float32)
                    sheet_angle = sheets[j][1][0]
                    use = size
                    while use >= min_size:
                        scale = use * ph / max(ww, wh) / grid
                        kern = pile.footprint(ww * scale, wh * scale, sheet_angle + angle)
                        seen = cv2.filter2D(m, -1, kern, borderType=cv2.BORDER_CONSTANT)
                        ys, xs = np.nonzero((m > 0) & (seen >= float(c.get("min_visible", 0.6))))
                        if len(xs):
                            options.append((j, idx, xs, ys, use, seen[ys, xs]))
                            break
                        use *= 0.85
                if options:
                    weights = np.array([len(o[2]) for o in options], float)
                    j, idx, xs, ys, use, seen = options[int(rng.choice(len(options), p=weights / weights.sum()))]
                    best = np.nonzero(seen >= np.quantile(seen, 0.8))[0]  # among the most exposed spots
                    n = int(best[int(rng.integers(len(best)))])
                    pw, ph, M = geo[idx]
                    px, py = pile.to_page(M, (xs[n] + 0.5) * grid, (ys[n] + 0.5) * grid)
                    target, x, y, size = j, px / pw, py / ph, use
            if x is None:  # the top sheet
                if per_sheet.get(k, 0) >= int(c["max_per_sheet"]):
                    continue
                target, x, y = k, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
            size = min(size, self.size_cap(wr, sheets[target][0], c))  # sharp on the sheet it lands on
            lines = max(1, int(wr.get("lines", 1)))
            self.tl.writings.append({
                "t": round(t, 4), "sheet": target, "layout": sheets[target][0], "writing": wr["id"],
                "x": round(x, 4), "y": round(y, 4), "angle": round(angle, 2), "size": round(size, 4),
                "dur": round(float(c["write_beats"]) * lines ** 0.7 * self.period(beat), 4),
                "on_top": target == k, "strength": h["s"],
            })
            self.tl.events.append({"t": round(t, 4), "beat": beat, "type": "writing", "writing": wr["id"],
                                   "sheet": target})
            per_sheet[target] = per_sheet.get(target, 0) + 1
            last_t, last_id = t, wr["id"]
        self.tl.events.sort(key=lambda e: e["t"])

    def plan_stamps(self, t_start: float) -> None:
        """Rubber stamps slam onto the current sheet on strong kicks (low-band onsets).

        A separate seeded stream is used so adding stamps never changes other choices."""
        c = self.settings["stamps"]
        if not c.get("enabled", True) or not self.stamp_library or not self.kicks:
            return
        rng = np.random.default_rng(int(self.settings["seed"]) + 4242)
        thr = float(np.percentile([k["s"] for k in self.kicks], float(c["trigger_percentile"])))
        forms = self.tl.forms
        change_times = [f["t"] for f in forms]
        busy = [(f["t"], max(f.get("land", f["t"]), f["draw"][1])) for f in forms]
        per_sheet: dict[int, int] = {}
        last_t, last_id = -math.inf, None
        for k in self.kicks:
            t = float(k["t"])
            if k["s"] < thr or t < t_start or t >= self.duration - 0.5:
                continue
            beat = int(np.clip(np.searchsorted(self.b, t) - 1, 0, len(self.b) - 1))
            if t - last_t < float(c["min_beats"]) * self.period(beat):
                continue
            if any(a <= t < b for a, b in busy):
                continue  # a sheet is still landing / the form is still changing
            sheet = int(np.searchsorted(change_times, t, side="right"))
            if per_sheet.get(sheet, 0) >= int(c["max_per_sheet"]):
                continue
            layout = forms[sheet - 1]["to"] if sheet > 0 else self.tl.initial
            until = change_times[sheet] if sheet < len(change_times) else math.inf
            sharp = [s for s in self.stamp_library
                     if self.size_cap(s, layout, c) >= float(c.get("min_size", 0.1))]
            choices = [s for s in sharp if s["id"] != last_id] or sharp
            if not choices:
                continue
            st = choices[int(rng.integers(len(choices)))]
            size = min(float(c["size"]) * float(rng.uniform(0.75, 1.25)), self.size_cap(st, layout, c))
            self.tl.stamps.append({
                "t": round(t, 4), "until": None if math.isinf(until) else round(until, 4), "sheet": sheet,
                "layout": layout, "stamp": st["id"],
                "x": round(float(rng.uniform(0.15, 0.85)), 4), "y": round(float(rng.uniform(0.18, 0.85)), 4),
                "angle": round(float(rng.uniform(-1, 1) * float(c["max_angle"])), 2),
                "size": round(size, 4),
                "opacity": round(float(c["opacity"]) * float(rng.uniform(0.85, 1.05)), 3),
                "dur": round(float(c["slam_beats"]) * self.period(beat), 4),
                "strength": k["s"],
            })
            self.tl.events.append({"t": round(t, 4), "beat": beat, "type": "stamp", "stamp": st["id"]})
            per_sheet[sheet] = per_sheet.get(sheet, 0) + 1
            last_t, last_id = t, st["id"]
        self.tl.events.sort(key=lambda e: e["t"])


def build_timeline(beats: dict, manifest: dict, settings: dict, stamps: list[dict] | None = None,
                   writing: list[dict] | None = None) -> Timeline:
    return Sequencer(beats, manifest, settings, stamps, writing).run()
