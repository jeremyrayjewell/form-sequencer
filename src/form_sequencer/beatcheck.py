"""Tools to inspect and correct beat alignment.

``form-seq beats`` writes to build/beatcheck/:
  * click.wav        the track with clicks on every beat (high click = downbeat)
  * beats_plot.png   onset strength with beat lines, in 8-second strips
  * beats_labels.txt Audacity label track ("1.1", "1.2", ... bar.beat)

To correct: import the track and beats_labels.txt into Audacity (File > Import >
Labels), drag/add/delete labels, export them (File > Export > Export Labels) to
input/beats.txt and re-run `form-seq analyze`. Small global shifts can instead be
made with audio.beat_offset in settings.yaml; wrong bar phase with audio.downbeat_offset.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from .audio import SR, onset_envelope
from .fonts import font


def bar_labels(beats: dict) -> list[str]:
    bpb, phase = beats["beats_per_bar"], beats["downbeat_offset"]
    out = []
    for i in range(len(beats["beats"])):
        k = i - phase
        bar, pos = (k // bpb) + 1, (k % bpb) + 1
        out.append(f"{bar}.{pos}")
    return out


def write_beatcheck(beats: dict, audio_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    labels = bar_labels(beats)
    times = np.asarray(beats["beats"])

    # Click track mixed under the music.
    down = [t for t, lab in zip(times, labels) if lab.endswith(".1")]
    up = [t for t, lab in zip(times, labels) if not lab.endswith(".1")]
    clicks = librosa.clicks(times=up, sr=sr, length=len(y), click_freq=1000.0) * 0.5
    clicks += librosa.clicks(times=down, sr=sr, length=len(y), click_freq=2000.0) * 0.7
    mix = 0.7 * y / (np.abs(y).max() + 1e-9) + clicks
    mix /= max(1.0, np.abs(mix).max())
    click_path = out_dir / "click.wav"
    sf.write(str(click_path), mix.astype(np.float32), sr)

    labels_path = out_dir / "beats_labels.txt"
    labels_path.write_text("".join(f"{t:.4f}\t{t:.4f}\t{lab}\n" for t, lab in zip(times, labels)), encoding="utf-8")

    # Plot: onset strength with beats, 8 s per strip.
    env, _, env_t = onset_envelope(y, sr)
    env = env / (env.max() + 1e-9)
    strip, W, Hs = 8.0, 1600, 110
    n = int(np.ceil(beats["duration"] / strip))
    img = Image.new("RGB", (W, Hs * n + 30), (250, 250, 247))
    d = ImageDraw.Draw(img)
    f = font("sans", 12)
    d.text((8, 6), f"tempo ~{beats['tempo']} BPM (local {beats['tempo_min']}-{beats['tempo_max']}), "
                   f"{len(times)} beats, source: {beats['source']}.  red = downbeat, blue = beat, grey = onset strength",
           font=f, fill=(20, 20, 20))
    for r in range(n):
        t0 = r * strip
        top = 30 + r * Hs
        sel = (env_t >= t0) & (env_t < t0 + strip)
        xs = (env_t[sel] - t0) / strip * W
        ys = top + Hs - 12 - env[sel] * (Hs - 30)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=(120, 120, 120), width=1)
        for t, lab in zip(times, labels):
            if t0 <= t < t0 + strip:
                x = (t - t0) / strip * W
                col = (200, 30, 30) if lab.endswith(".1") else (40, 80, 200)
                d.line([x, top + 4, x, top + Hs - 12], fill=col, width=2 if lab.endswith(".1") else 1)
                if lab.endswith(".1"):
                    d.text((x + 3, top + 4), lab.split(".")[0], font=f, fill=col)
        d.text((4, top + Hs - 14), f"{t0:.0f}s", font=f, fill=(60, 60, 60))
        d.line([0, top + Hs - 1, W, top + Hs - 1], fill=(200, 200, 200))
    plot_path = out_dir / "beats_plot.png"
    img.save(plot_path)
    return [click_path, plot_path, labels_path]
