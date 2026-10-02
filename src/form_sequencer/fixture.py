"""Synthetic test fixture: a drifting-tempo track and a set of fake administrative forms.

Nothing here is copyrighted; everything is generated from a seed.
The track deliberately avoids 120 BPM, starts its first beat after t=0, drifts in
tempo, has audible section changes, and contains off-beat accents.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pymupdf as fitz
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

from .config import Project
from .fonts import font

# ---------------------------------------------------------------- audio


def _env(n: int, sr: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay * sr))


def synth_track(path: Path, seed: int = 1, duration: float = 40.0) -> dict:
    sr = 44100
    rng = np.random.default_rng(seed)
    out = np.zeros(int(duration * sr) + sr)
    bpm0, bpm1, first = 94.0, 101.0, 0.41

    beats = []
    t = first
    while t < duration - 0.6:
        beats.append(t)
        t += 60.0 / (bpm0 + (bpm1 - bpm0) * t / duration)
    beats = np.array(beats)
    bars = len(beats) // 4
    section_of_bar = lambda b: ["A", "B", "C", "A2"][min((b // 4) % 4, 3)]

    def add(sig: np.ndarray, at: float, gain: float) -> None:
        i = int(at * sr)
        j = min(len(out), i + len(sig))
        out[i:j] += gain * sig[: j - i]

    n = int(0.35 * sr)
    tt = np.arange(n) / sr
    kick = np.sin(2 * np.pi * (45 * tt + 75 * (1 - np.exp(-tt * 30)) / 30)) * _env(n, sr, 0.09)
    noise = rng.standard_normal(n)
    snare = (np.diff(noise, prepend=0) * 0.6 + np.sin(2 * np.pi * 190 * tt) * 0.5) * _env(n, sr, 0.06)
    hat = np.diff(np.diff(rng.standard_normal(int(0.06 * sr)), prepend=0), prepend=0) * _env(int(0.06 * sr), sr, 0.012) * 0.25
    crash_n = int(1.6 * sr)
    crash = np.diff(rng.standard_normal(crash_n), prepend=0) * _env(crash_n, sr, 0.5) * 0.5
    stab_n = int(0.25 * sr)
    st = np.arange(stab_n) / sr
    stab = sum(np.sign(np.sin(2 * np.pi * f * st)) for f in (220, 277.2, 329.6)) * _env(stab_n, sr, 0.07) * 0.18

    for i, bt in enumerate(beats):
        bar, pos = divmod(i, 4)
        sec = section_of_bar(bar)
        nxt = beats[i + 1] if i + 1 < len(beats) else bt + 60 / bpm1
        add(kick, bt, 1.0 if pos == 0 else 0.55)
        add(hat, (bt + nxt) / 2, 1.0)
        if sec in ("B", "C") and pos in (1, 3):
            add(snare, bt, 0.55)
        if pos == 0 and bar % 4 == 0 and bar > 0:
            add(crash, bt, 0.6)
        if sec == "C" and pos == 2 and bar % 2 == 1:
            add(stab, (bt + nxt) / 2, 1.0)  # syncopated off-beat accent
        if pos == 0 and sec in ("B", "C", "A2"):
            bar_len = (beats[min(i + 4, len(beats) - 1)] - bt) or 2.4
            m = int(bar_len * sr)
            bt_ = np.arange(m) / sr
            root = [55.0, 55.0, 49.0, 41.2][bar % 4]
            voice = np.sin(2 * np.pi * root * bt_) * 0.35
            if sec == "C":
                voice += sum(np.sin(2 * np.pi * root * k * bt_) for k in (4, 5, 6)) * 0.05
            add(voice * np.minimum(1, bt_ * 20) * np.exp(-bt_ * 0.8), bt, 1.0)

    out = out[: int(duration * sr)]
    out = 0.9 * out / np.abs(out).max()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out.astype(np.float32), sr, subtype="PCM_16")
    return {
        "beats": [round(float(b), 4) for b in beats],
        "downbeats": [round(float(b), 4) for b in beats[::4]],
        "sections": [round(float(beats[b * 4]), 4) for b in range(0, bars, 4)],
        "bpm_start": bpm0,
        "bpm_end": bpm1,
        "first_beat": first,
    }


# ---------------------------------------------------------------- documents

W, H, M = 1275, 1650, 95  # US letter at 150 dpi
INK = (28, 28, 34)
WORDS = (
    "applicant declarant schedule pursuant section subsection filing authority record "
    "certify accurate complete statement office registry annex provision attached "
    "hereby acknowledge receipt notice period renewal review department claim number "
    "reference submitted approved pending verification copy original enclosure item "
    "amended supplementary deadline obligation compliance form part herein thereof"
).split()


def _sentence(rng, n=None) -> str:
    n = n or int(rng.integers(6, 14))
    s = " ".join(rng.choice(WORDS, n))
    return s[0].upper() + s[1:] + "."


class FormPainter:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.img = Image.new("RGB", (W, H), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.y = M

    def gap(self, px: int) -> None:
        self.y += px

    def header(self, title: str, code: str) -> None:
        d, y = self.d, self.y
        d.text((M, y), "DEPARTMENT OF SYNTHETIC RECORDS", font=font("sans-bold", 22), fill=INK)
        d.text((M, y + 32), title, font=font("sans-bold", 32), fill=INK)
        d.text((M, y + 76), "Office of Continuous Revision", font=font("serif", 22), fill=INK)
        bx = W - M - 250
        d.rectangle([bx, y, W - M, y + 100], outline=INK, width=3)
        d.text((bx + 16, y + 12), f"Form {code}", font=font("sans-bold", 30), fill=INK)
        d.text((bx + 16, y + 56), f"Rev. {self.rng.integers(1, 12):02d}/19{self.rng.integers(50, 99)}", font=font("mono", 20), fill=INK)
        self.y = y + 118
        d.rectangle([M, self.y, W - M, self.y + 5], fill=INK)
        self.y += 30

    def band(self, text: str) -> None:
        self.d.rectangle([M, self.y, W - M, self.y + 34], fill=(60, 60, 66))
        self.d.text((M + 12, self.y + 4), text, font=font("sans-bold", 22), fill=(255, 255, 255))
        self.y += 50

    def paragraph(self, lines: int = 4) -> None:
        f = font("serif", 22)
        text = " ".join(_sentence(self.rng) for _ in range(lines * 2))
        words, line = text.split(), ""
        count = 0
        for w in words:
            trial = f"{line} {w}".strip()
            if f.getlength(trial) > W - 2 * M:
                self.d.text((M, self.y), line, font=f, fill=INK)
                self.y += 30
                line, count = w, count + 1
                if count >= lines:
                    break
            else:
                line = trial
        self.y += 20

    def fields(self, labels: list[str], columns: int = 2) -> None:
        f = font("sans", 20)
        mono = font("mono", 22)
        colw = (W - 2 * M) // columns
        for i, label in enumerate(labels):
            c = i % columns
            x = M + c * colw
            if c == 0 and i:
                self.y += 58
            self.d.text((x, self.y), label, font=f, fill=INK)
            ly = self.y + 44
            self.d.line([x, ly, x + colw - 30, ly], fill=INK, width=2)
            if self.rng.random() < 0.6:
                val = " ".join(self.rng.choice(WORDS, 2)).upper() if self.rng.random() < 0.5 else str(self.rng.integers(10000, 99999))
                self.d.text((x + 12, ly - 26), val, font=mono, fill=(40, 40, 90))
        self.y += 80

    def checkboxes(self, question: str, options: list[str]) -> None:
        f = font("sans", 20)
        self.d.text((M, self.y), question, font=f, fill=INK)
        self.y += 36
        x = M + 10
        for opt in options:
            self.d.rectangle([x, self.y, x + 22, self.y + 22], outline=INK, width=2)
            if self.rng.random() < 0.35:
                self.d.line([x + 4, self.y + 4, x + 18, self.y + 18], fill=(40, 40, 90), width=3)
                self.d.line([x + 18, self.y + 4, x + 4, self.y + 18], fill=(40, 40, 90), width=3)
            self.d.text((x + 32, self.y), opt, font=f, fill=INK)
            x += 60 + int(f.getlength(opt))
            if x > W - M - 200:
                x = M + 10
                self.y += 36
        self.y += 60

    def table(self, cols: int, rows: int) -> None:
        f, mono = font("sans-bold", 18), font("mono", 18)
        x0, x1 = M, W - M
        rh = 38
        colw = (x1 - x0) / cols
        top = self.y
        self.d.rectangle([x0, top, x1, top + rh], fill=(215, 215, 210))
        for r in range(rows + 2):
            self.d.line([x0, top + r * rh, x1, top + r * rh], fill=INK, width=2)
        for c in range(cols + 1):
            xx = int(x0 + c * colw)
            self.d.line([xx, top, xx, top + (rows + 1) * rh], fill=INK, width=2)
        heads = ["Item", "Code", "Period", "Amount", "Ref.", "Status", "Date"]
        for c in range(cols):
            self.d.text((x0 + c * colw + 8, top + 9), heads[c % len(heads)], font=f, fill=INK)
            for r in range(rows):
                if self.rng.random() < 0.75:
                    v = f"{self.rng.integers(0, 9999):04d}" if c % 2 else str(self.rng.choice(WORDS))[:9]
                    self.d.text((x0 + c * colw + 8, top + (r + 1) * rh + 9), v, font=mono, fill=(40, 40, 90))
        self.y = top + (rows + 1) * rh + 40

    def signature(self) -> None:
        x0, ly = M + 20, self.y + 90
        rng = self.rng
        pts, x, yy = [], x0 + 20, ly - 30
        for _ in range(9):
            for k in range(12):
                a = k / 12 * 2 * math.pi
                pts.append((x + k * 3.2, yy + math.sin(a * rng.uniform(1, 2.5)) * rng.uniform(10, 28)))
            x += rng.uniform(30, 48)
        self.d.line(pts, fill=(25, 45, 140), width=3, joint="curve")
        self.d.line([x0, ly, x0 + 480, ly], fill=INK, width=2)
        self.d.text((x0, ly + 8), "Signature of declarant", font=font("sans", 18), fill=INK)
        self.d.line([x0 + 560, ly, x0 + 820, ly], fill=INK, width=2)
        self.d.text((x0 + 560, ly + 8), "Date", font=font("sans", 18), fill=INK)
        self.y = ly + 60

    def email_line(self, address: str) -> tuple[int, int, int, int]:
        """A 'Contact e-mail' line; returns the address's bbox (must be excluded)."""
        f_label, f_val = font("sans", 20), font("mono", 22)
        self.d.text((M, self.y), "Contact e-mail:", font=f_label, fill=INK)
        x = M + int(f_label.getlength("Contact e-mail:")) + 16
        l, t, r, b = self.d.textbbox((x, self.y), address, font=f_val)
        self.d.text((x, self.y), address, font=f_val, fill=(40, 40, 90))
        self.y += 50
        return l, t, r - l, b - t

    def photo_box(self, x: int, y: int) -> tuple[int, int, int, int]:
        """'Affix photograph here' box holding a synthetic photo; returns the photo's bbox."""
        ph = _photo(self.rng)
        self.d.rectangle([x - 8, y - 8, x + ph.width + 8, y + ph.height + 8], outline=INK, width=2)
        self.d.text((x, y + ph.height + 14), "Affix recent photograph", font=font("sans", 16), fill=INK)
        self.img.paste(ph, (x, y))
        return x, y, ph.width, ph.height

    def qr(self, x: int, y: int) -> tuple[int, int, int, int]:
        q = _qr("https://records.example/verify/114")
        self.img.paste(q, (x, y))
        return x, y, q.width, q.height

    def emblem(self, x: int, y: int, size: int = 170) -> tuple[int, int, int, int]:
        e = _emblem(size)
        self.img.paste(e, (x, y))
        return x, y, e.width, e.height

    def diagram(self) -> None:
        """A line drawing with dimension lines (a printed technical illustration)."""
        d, x0, y0 = self.d, M + 20, self.y
        d.rectangle([x0, y0 + 40, x0 + 420, y0 + 260], outline=INK, width=3)
        d.ellipse([x0 + 60, y0 + 90, x0 + 180, y0 + 210], outline=INK, width=3)
        d.ellipse([x0 + 250, y0 + 90, x0 + 370, y0 + 210], outline=INK, width=3)
        for cx in (x0 + 120, x0 + 310):
            d.line([cx - 70, y0 + 150, cx + 70, y0 + 150], fill=INK, width=1)
            d.line([cx, y0 + 80, cx, y0 + 220], fill=INK, width=1)
        d.line([x0, y0 + 20, x0 + 420, y0 + 20], fill=INK, width=1)
        for xx in (x0, x0 + 420):
            d.line([xx, y0 + 10, xx, y0 + 30], fill=INK, width=1)
        d.text((x0 + 180, y0), "1420 mm", font=font("mono", 16), fill=INK)
        d.text((x0 + 440, y0 + 140), "Fig. 2  Arrangement of apparatus", font=font("serif", 20), fill=INK)
        self.y = y0 + 290

    def chart(self) -> None:
        """A bar chart drawn with rules and hatching (a printed chart)."""
        d, x0, y0 = self.d, M + 40, self.y
        d.line([x0, y0, x0, y0 + 220], fill=INK, width=2)
        d.line([x0, y0 + 220, x0 + 560, y0 + 220], fill=INK, width=2)
        for i, v in enumerate(self.rng.integers(40, 200, 8)):
            bx = x0 + 20 + i * 66
            d.rectangle([bx, y0 + 220 - int(v), bx + 40, y0 + 220], outline=INK, width=2)
            for k in range(y0 + 220 - int(v), y0 + 220, 8):
                d.line([bx, k, bx + 40, k], fill=INK, width=1)
            d.text((bx + 8, y0 + 228), f"{1950 + i * 5}", font=font("mono", 14), fill=INK)
        d.text((x0 + 600, y0 + 90), "Declarations filed,", font=font("serif", 20), fill=INK)
        d.text((x0 + 600, y0 + 116), "by quinquennium", font=font("serif", 20), fill=INK)
        self.y = y0 + 270

    def stamp(self, text: str, color=(190, 35, 35), at: tuple[int, int] | None = None) -> None:
        s = 250
        layer = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(layer)
        d.ellipse([6, 6, s - 6, s - 6], outline=255, width=7)
        d.ellipse([28, 28, s - 28, s - 28], outline=255, width=3)
        f = font("sans-bold", 34)
        d.text((s / 2, s / 2 - 16), text, font=f, fill=255, anchor="mm")
        d.text((s / 2, s / 2 + 28), f"{self.rng.integers(1, 28):02d} MAR 1987", font=font("mono-bold", 22), fill=255, anchor="mm")
        layer = layer.rotate(float(self.rng.uniform(-25, 25)), resample=Image.BICUBIC)
        mottled = np.asarray(layer, np.float32) * self.rng.uniform(0.55, 1.0, (s, s))
        layer = Image.fromarray(mottled.clip(0, 255).astype(np.uint8))
        x, y = at or (W - M - s, self.y)
        ink = Image.new("RGB", (s, s), color)
        self.img.paste(ink, (x, y), layer)
        if at is None:
            self.y += s + 20


def _photo(rng, w: int = 210, h: int = 260) -> Image.Image:
    """A procedural 'photograph': soft continuous tone, no hard edges (must be excluded)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 0.55 + 0.25 * yy / h + 0.05 * np.sin(xx / 17)
    head = np.exp(-(((xx - w * 0.5) / (w * 0.22)) ** 2 + ((yy - h * 0.38) / (h * 0.24)) ** 2) * 1.6)
    body = 1 / (1 + np.exp(-(yy - h * 0.72) / 8)) * np.exp(-((xx - w * 0.5) / (w * 0.42)) ** 4)
    eyes = sum(np.exp(-(((xx - w * (0.5 + dx)) / 7) ** 2 + ((yy - h * 0.36) / 4) ** 2)) for dx in (-0.08, 0.08))
    lum = bg * (1 - head) + (0.78 - 0.12 * yy / h) * head
    lum = lum * (1 - body) + 0.25 * body - 0.35 * eyes
    mottle = np.asarray(Image.fromarray((rng.random((h, w)) * 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(3)), np.float32) / 255 - 0.5
    lum += 0.25 * mottle + rng.normal(0, 0.015, (h, w))
    rgb = np.stack([lum * 1.05, lum * 0.92, lum * 0.82], -1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def _qr(text: str, size: int = 180) -> Image.Image:
    """A real QR code (OpenCV encoder), must be excluded."""
    import cv2
    m = cv2.QRCodeEncoder.create().encode(text)
    return Image.fromarray(m).convert("RGB").resize((size, size), Image.NEAREST)


def stamp_photo(rng, kind: int) -> tuple[Image.Image, dict]:
    """A photographed rubber-stamp imprint on creased, unevenly lit paper with a dark edge line.
    Returns the image and ground truth: a ring point (ink), a paper corner and a point on the line."""
    w, h = 420, 380
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    paper = 0.82 + 0.1 * (xx / w) - 0.06 * (yy / h) - 0.05 * np.exp(-((xx - yy * 0.8 - 60) / 25) ** 2)  # shading + crease
    img = np.dstack([paper * 0.98, paper * 0.97, paper * 0.95]) * 255
    pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    ink = (40, 70, 170) if kind == 0 else (95, 70, 175)
    cx, cy, r = 200, 185, 140
    if kind == 0:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink, width=9)
        d.line([cx - r + 25, cy - 45, cx + r - 25, cy - 45], fill=ink, width=5)
        d.line([cx - r + 25, cy + 50, cx + r - 25, cy + 50], fill=ink, width=5)
        d.text((cx, cy - 85), "RECEIVED", font=font("sans-bold", 34), fill=ink, anchor="mm")
        d.text((cx, cy + 3), "104. 06. 06", font=font("mono-bold", 30), fill=ink, anchor="mm")
        ring = (cx + r - 4, cy)
    else:
        d.rectangle([40, 120, 380, 250], outline=ink, width=8)
        d.text((210, 185), "FILED", font=font("sans-bold", 56), fill=ink, anchor="mm")
        ring = (44, 185)
    d.line([395, 0, 405, h], fill=(20, 20, 22), width=5)  # the paper's dark edge in the photo
    arr = np.asarray(pil, np.float32) + rng.normal(0, 3, (h, w, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), {"ink": ring, "paper": (8, h - 8), "line": (400, 190)}


def writing_photo(rng, lines: int = 2) -> Image.Image:
    """A photographed handwritten note: cursive-like ballpoint strokes on tinted, ruled paper."""
    w, h = 520, 70 + 60 * lines
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    paper = 0.9 - 0.06 * (xx / w) + 0.02 * np.sin(yy / 9)
    img = np.dstack([paper, paper * 0.97, paper * 0.86]) * 255  # yellowish note paper
    pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for li in range(lines + 1):
        d.line([0, 58 + 60 * li, w, 58 + 60 * li], fill=(170, 190, 215), width=1)  # printed ruling
    ink = (30, 45, 150)
    for li in range(lines):
        base = 45 + 60 * li
        x = 30.0
        pts = []
        while x < w - 40:
            for k in range(14):
                a = k / 14 * 2 * math.pi
                pts.append((x + k * 2.2, base - 12 - math.sin(a * rng.uniform(1, 2)) * rng.uniform(6, 16)))
            x += rng.uniform(34, 48)
            if rng.random() < 0.25:
                d.line(pts, fill=ink, width=3, joint="curve")
                pts = []
                x += 14
        if pts:
            d.line(pts, fill=ink, width=3, joint="curve")
    arr = np.asarray(pil, np.float32) + rng.normal(0, 2.5, (h, w, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _emblem(size: int = 170) -> Image.Image:
    """A flat-colour printed emblem (must stay eligible)."""
    im = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([4, 4, size - 4, size - 4], fill=(30, 70, 140))
    d.ellipse([22, 22, size - 22, size - 22], fill=(235, 190, 40))
    c = size / 2
    pts = [(c + (60 if k % 2 == 0 else 24) * math.sin(k * math.pi / 5),
            c - (60 if k % 2 == 0 else 24) * math.cos(k * math.pi / 5)) for k in range(10)]
    d.polygon(pts, fill=(180, 30, 30))
    return im


def _scanify(img: Image.Image, rng, rotate: float = 0.0) -> Image.Image:
    arr = np.asarray(img, np.float32)
    paper = np.array([248, 245, 236], np.float32) + rng.normal(0, 2.0, 3)
    arr = arr / 255.0 * paper
    arr += rng.normal(0, 4.0, arr.shape[:2])[..., None]
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    if rotate:
        out = out.rotate(rotate, resample=Image.BICUBIC, fillcolor=tuple(int(p) for p in paper))
    return out


def _form(kind: int, rng, marks: dict | None = None) -> Image.Image:
    p = FormPainter(rng)
    marks = marks if marks is not None else {}
    if kind == 0:
        p.header("APPLICATION FOR CONTINUED RECOGNITION", "SR-114")
        p.paragraph(3)
        p.band("PART I  —  APPLICANT")
        p.fields(["Full legal name", "Registry number", "Address of record", "Date of first filing", "Occupation", "Reference"])
        marks.setdefault("emails", []).append(p.email_line("clerk.registry@records-office.example"))
        p.checkboxes("Has the applicant previously been recognized?", ["Yes", "No", "Pending", "Not applicable"])
        p.band("PART II  —  SCHEDULE OF ITEMS")
        p.table(5, 6)
        p.signature()
        p.stamp("RECEIVED", at=(W - M - 280, p.y - 250))
    elif kind == 1:
        p.header("SCHEDULE OF INCIDENTAL DECLARATIONS", "SR-7B")
        p.band("SECTION A  —  DECLARATIONS")
        p.table(4, 9)
        p.paragraph(4)
        p.checkboxes("Tick all that apply:", ["Original", "Copy", "Amended", "Supplementary", "Annex", "Withdrawn"])
        p.fields(["Officer", "Station code", "Countersigned by"], columns=3)
        p.stamp("APPROVED", color=(35, 60, 160))
        marks.setdefault("eligible", []).append(p.emblem(M + 700, p.y - 250))
        p.signature()
        marks.setdefault("qr", []).append(p.qr(W - M - 200, p.y + 10))
    elif kind == 3:
        p.header("RECORD OF CORRESPONDENCE", "SR-9")
        marks.setdefault("logos", []).append(p.emblem(M + 700, M + 2, size=96))
        p.fields(["File", "Correspondent", "Received", "Answered", "Clerk", "Folio"], columns=3)
        p.band("ENTRIES")
        p.table(7, 5)
        p.checkboxes("Disposition", ["Filed", "Forwarded", "Destroyed", "Awaiting instruction"])
        p.paragraph(2)
        p.chart()
        p.stamp("FILED", color=(20, 110, 80))
        p.signature()
    else:
        p.header("NOTICE OF REVIEW PERIOD", "SR-40")
        marks.setdefault("photos", []).append(p.photo_box(W - M - 230, p.y + 10))
        p.paragraph(2)
        p.diagram()
        p.band("PART III  —  VERIFICATION")
        p.fields(["Claim number", "Period", "Enclosures", "Office"], columns=2)
        p.checkboxes("Verification method", ["In person", "By post", "Deferred"])
        p.table(6, 3)
        p.stamp("VOID", color=(170, 30, 60), at=(W - M - 300, M + 700))
        p.signature()
    return p.img


def _vector_pdf_page(doc: fitz.Document, rng) -> None:
    """A native (non-scanned) PDF page drawn with vector text and rules."""
    page = doc.new_page(width=612, height=792)
    m = 46
    page.insert_text((m, m + 18), "DEPARTMENT OF SYNTHETIC RECORDS", fontsize=11, fontname="hebo")
    page.insert_text((m, m + 44), "ANNEX C: STATEMENT OF PARTICULARS", fontsize=19, fontname="hebo")
    page.draw_rect(fitz.Rect(612 - m - 120, m, 612 - m, m + 50), width=1.5)
    page.insert_text((612 - m - 110, m + 30), "Form SR-C3", fontsize=14, fontname="hebo")
    page.draw_line((m, m + 64), (612 - m, m + 64), width=2.5)
    y = m + 100
    for i in range(6):
        page.insert_text((m, y), f"{i + 1}. {_sentence(rng, 9)}", fontsize=10, fontname="tiro")
        y += 16
    y += 8
    page.insert_text((m, y), "Correspondence: annex.desk@synthetic-records.example", fontsize=10, fontname="helv")
    y += 24
    for label in ("Surname", "Given names", "Station", "Reference no."):
        page.insert_text((m, y), label, fontsize=10, fontname="helv")
        page.draw_line((m + 110, y + 2), (m + 330, y + 2), width=0.8)
        y += 30
    y += 20
    cols, rows, cw, rh = 5, 7, (612 - 2 * m) / 5, 20
    for r in range(rows + 1):
        page.draw_line((m, y + r * rh), (612 - m, y + r * rh), width=0.8)
    for c in range(cols + 1):
        page.draw_line((m + c * cw, y), (m + c * cw, y + rows * rh), width=0.8)
    for c in range(cols):
        page.insert_text((m + c * cw + 4, y + 14), ["No.", "Particular", "Qty", "Unit", "Remarks"][c], fontsize=9, fontname="hebo")
        for r in range(1, rows):
            page.insert_text((m + c * cw + 4, y + r * rh + 14), str(rng.choice(WORDS))[:10], fontsize=9, fontname="cour")
    y += rows * rh + 40
    for opt_i, opt in enumerate(("Accepted", "Rejected", "Returned for correction")):
        page.draw_rect(fitz.Rect(m, y - 10, m + 11, y + 1), width=1)
        page.insert_text((m + 18, y), opt, fontsize=10, fontname="helv")
        y += 22


def make_fixture(project: Project, seed: int = 1) -> dict:
    project.init()
    rng = np.random.default_rng(seed)
    truth = synth_track(project.audio_dir / "fixture_track.wav", seed=seed)
    docs = project.root / "input" / "documents"
    docs.mkdir(parents=True, exist_ok=True)

    marks: dict[str, dict] = {"form_sr114_png_p1": {}, "form_sr7b_png_p1": {}, "notice_sr40_jpg_p1": {}}
    _scanify(_form(0, rng, marks["form_sr114_png_p1"]), rng).save(docs / "form_sr114.png")
    _scanify(_form(1, rng, marks["form_sr7b_png_p1"]), rng, rotate=0.35).save(docs / "form_sr7b.png")
    _scanify(_form(2, rng, marks["notice_sr40_jpg_p1"]), rng).save(docs / "notice_sr40.jpg", quality=88)
    truth["photos"] = {k: v["photos"] for k, v in marks.items() if v.get("photos")}
    truth["eligible_graphics"] = {k: v["eligible"] for k, v in marks.items() if v.get("eligible")}
    truth["emails"] = {k: v["emails"] for k, v in marks.items() if v.get("emails")}
    truth["pdf_email"] = "annex.desk@synthetic-records.example"
    truth["qr"] = {k: v["qr"] for k, v in marks.items() if v.get("qr")}
    truth["logos"] = {k: v["logos"] for k, v in marks.items() if v.get("logos")}

    pdf = fitz.open()
    _vector_pdf_page(pdf, rng)
    marks["annex_bundle_pdf_p2"] = {}
    scan = _scanify(_form(3, np.random.default_rng(seed + 10), marks["annex_bundle_pdf_p2"]), rng)
    tmp = docs / "_scan_tmp.png"
    scan.save(tmp)
    page = pdf.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(tmp))
    tmp.unlink()
    pdf.save(str(docs / "annex_bundle.pdf"))
    pdf.close()

    stamp_dir = project.root / "input" / "stamps"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    truth["stamps"] = {}
    for kind, name in enumerate(("ring_received", "box_filed")):
        img, pts = stamp_photo(rng, kind)
        img.save(stamp_dir / f"{name}.jpg", quality=92)
        truth["stamps"][name] = pts

    writing_dir = project.root / "input" / "writing"
    writing_dir.mkdir(parents=True, exist_ok=True)
    for n in (1, 2, 3):
        writing_photo(rng, n).save(writing_dir / f"note_{n}.jpg", quality=92)

    (project.root / "fixture_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")

    # Four-bar sections in a 40 s fixture: allow form changes every 4 bars.
    text = project.settings_path.read_text(encoding="utf-8")
    text = text.replace("  min_bars: 8                  #", "  min_bars: 4                  #")
    project.settings_path.write_text(text, encoding="utf-8")
    return truth
