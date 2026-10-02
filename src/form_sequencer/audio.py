"""Audio analysis: tempo, beat times, beat strength, downbeat phase, novelty, onsets.

Everything downstream works from the *list of beat times* (not a fixed BPM grid),
so tempo drift and a non-zero first beat are handled naturally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np

SR = 22050
HOP = 256  # ~11.6 ms frames


def load_manual_beats(path: Path) -> np.ndarray | None:
    """Read beat times from an Audacity label export or a plain list (one time per line)."""
    if not path.exists():
        return None
    times = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "\\")):
            continue
        try:
            times.append(float(line.replace(",", " ").split()[0]))
        except ValueError:
            continue
    if len(times) < 2:
        raise ValueError(f"{path}: need at least two beat times")
    return np.unique(np.asarray(times, dtype=float))


def onset_envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-band plus low-band (kick/bass) onset strength, and the low band alone.

    Adding the low band keeps the tracker from locking onto off-beat hi-hats,
    which dominate a full-band spectral flux.
    """
    full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, n_mels=32, fmax=200)
    full = full / (np.percentile(full, 99) + 1e-9)
    low = low / (np.percentile(low, 99) + 1e-9)
    return full + low, low, librosa.times_like(full, sr=sr, hop_length=HOP)


def _estimate_tempo(env: np.ndarray, sr: int, lo: float, hi: float) -> float:
    bpm = float(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=HOP, max_tempo=hi)[0])
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def _refine(beats: np.ndarray, y: np.ndarray, sr: int, radius: float = 0.05) -> np.ndarray:
    """Snap each beat to the strongest attack within +/- radius on a ~3 ms onset envelope.

    The coarse envelope used for tracking peaks 20-30 ms after the actual attack;
    this puts visual events on the hit itself.
    """
    hop = 64
    fine = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=512)
    ft = librosa.times_like(fine, sr=sr, hop_length=hop)
    out = beats.copy()
    for i, t in enumerate(beats):
        idx = np.nonzero((ft > t - radius) & (ft < t + radius))[0]
        if idx.size and fine[idx].max() > 0:
            out[i] = ft[idx[np.argmax(fine[idx])]]
    return out


def _regularize(beats: np.ndarray, tol: float = 0.08, window: int = 17) -> np.ndarray:
    """Repair local tracking errors: extra, missing or displaced beats.

    Intervals that deviate from the local median by more than ``tol`` are grouped into
    runs; each run is re-spaced evenly between the steady beats on either side, with the
    beat count taken from the local median. Gradual tempo drift passes through untouched.
    """
    if len(beats) < window:
        return beats
    ibi = np.diff(beats)
    half = window // 2
    med = np.array([np.median(ibi[max(0, i - half): i + half + 1]) for i in range(len(ibi))])
    bad = np.abs(ibi / med - 1) > tol
    # one error often shows as two bad runs around a lucky normal interval: merge them
    idx = np.nonzero(bad)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 3:
            bad[a:b] = True
    out, i = [beats[0]], 0
    while i < len(ibi):
        if not bad[i]:
            out.append(beats[i + 1])
            i += 1
            continue
        j = i
        while j < len(ibi) and bad[j]:
            j += 1
        a, b = beats[i], beats[j]  # steady anchors around the run
        n = max(1, int(round((b - a) / float(np.median(med[i:j])))))
        out.extend(np.linspace(a, b, n + 1)[1:].tolist())
        i = j
    return np.asarray(out)


def _window_max(env: np.ndarray, times: np.ndarray, at: np.ndarray, width: float = 0.04) -> np.ndarray:
    step = times[1] - times[0]
    r = max(1, int(round(width / step)))
    idx = np.clip(np.round(at / step).astype(int), 0, len(env) - 1)
    return np.array([env[max(0, i - r): i + r + 1].max() for i in idx])


def _trim_silent(beats: np.ndarray, env: np.ndarray, times: np.ndarray, frac: float = 0.12) -> np.ndarray:
    """Drop leading/trailing beats that sit in silence (the tracker extrapolates into it)."""
    s = _window_max(env, times, beats)
    loud = np.nonzero(s > frac * np.percentile(s, 90))[0]
    return beats[loud[0]: loud[-1] + 1] if loud.size else beats


def analyze_audio(audio_path: Path, settings: dict, manual_beats: Path | None = None) -> dict[str, Any]:
    cfg = settings["audio"]
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    duration = len(y) / sr
    env, low, times = onset_envelope(y, sr)
    lo, hi = cfg["tempo_range"]

    manual = load_manual_beats(manual_beats) if manual_beats else None
    if manual is not None:
        beats = manual[(manual >= 0) & (manual <= duration)]
        source = "manual"
        tempo = float(60.0 / np.median(np.diff(beats)))
    else:
        tempo = float(cfg["tempo_hint"]) if cfg["tempo_hint"] else _estimate_tempo(env, sr, lo, hi)
        _, frames = librosa.beat.beat_track(
            onset_envelope=env, sr=sr, hop_length=HOP, bpm=tempo,
            tightness=float(cfg["tightness"]), trim=False, units="frames",
        )
        beats = _regularize(_refine(_trim_silent(times[frames], env, times), y, sr))
        source = "detected"
    beats = np.sort(beats + float(cfg["beat_offset"]))
    beats = beats[(beats >= 0) & (beats < duration)]
    if len(beats) < 4:
        raise RuntimeError("fewer than 4 beats found; set audio.tempo_hint or supply input/beats.txt")

    # Beat strength: onset peak near each beat, normalized to 0..1.
    raw = _window_max(env, times, beats)
    strength = np.clip(raw / (np.percentile(raw, 95) + 1e-9), 0, 1)

    # Novelty per beat: change in timbre/loudness between the two bars before and after.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    rms = np.log1p(20 * librosa.feature.rms(y=y, hop_length=HOP))
    feats = np.vstack([mfcc, rms * 10])
    bframes = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)
    sync = librosa.util.sync(feats, bframes, aggregate=np.mean)[:, 1:]  # segment starting at each beat
    sync = (sync - sync.mean(axis=1, keepdims=True)) / (sync.std(axis=1, keepdims=True) + 1e-9)
    w = 2 * int(cfg["beats_per_bar"])
    novelty = np.zeros(len(beats))
    for i in range(w, len(beats) - w + 1):
        novelty[i] = np.linalg.norm(sync[:, i:i + w].mean(1) - sync[:, i - w:i].mean(1))
    valid = novelty[w:len(beats) - w + 1]
    if valid.size and valid.std() > 0:
        novelty = (novelty - valid.mean()) / valid.std()
        novelty[:w] = 0
        novelty[len(beats) - w + 1:] = 0

    # Onsets (for off-beat accents).
    # Kicks: onsets of the low band (kick drum / bass hits), used for stamp slams.
    kframes = librosa.onset.onset_detect(onset_envelope=low, sr=sr, hop_length=HOP, units="frames")
    knorm = np.percentile(low[kframes], 95) + 1e-9 if len(kframes) else 1.0
    ktimes = _refine(times[kframes], y, sr, radius=0.035) + float(cfg["beat_offset"])
    kicks = [{"t": round(float(t), 4), "s": round(float(min(low[f] / knorm, 1.0)), 3)}
             for t, f in zip(ktimes, kframes)]

    # Highs: onsets above ~3 kHz (hi-hats, cymbals, high notes), used for handwriting.
    high = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, n_mels=32, fmin=3000, fmax=sr / 2)
    hframes = librosa.onset.onset_detect(onset_envelope=high, sr=sr, hop_length=HOP, units="frames")
    hnorm = np.percentile(high[hframes], 95) + 1e-9 if len(hframes) else 1.0
    htimes = _refine(times[hframes], y, sr, radius=0.035) + float(cfg["beat_offset"])
    highs = [{"t": round(float(t), 4), "s": round(float(min(high[f] / hnorm, 1.0)), 3)}
             for t, f in zip(htimes, hframes)]

    oframes = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="frames")
    onorm = np.percentile(env[oframes], 95) + 1e-9 if len(oframes) else 1.0
    otimes = _refine(times[oframes], y, sr, radius=0.035) + float(cfg["beat_offset"])
    onsets = [{"t": round(float(t), 4), "s": round(float(min(env[f] / onorm, 1.0)), 3)}
              for t, f in zip(otimes, oframes)]

    ibi = np.diff(beats)
    local_bpm = np.append(60.0 / ibi, 60.0 / ibi[-1])
    # Downbeat phase: the position in the bar with the most kick/bass and the most change.
    bpb = int(cfg["beats_per_bar"])
    if cfg["downbeat_offset"] is not None:
        phase = int(cfg["downbeat_offset"]) % bpb
        phase_scores = []
    else:
        low_at = _window_max(low, times, beats)
        low_at = low_at / (low_at.max() + 1e-9)
        score = low_at + 0.5 * strength + 0.25 * np.clip(novelty, 0, None)
        phase_scores = [float(np.mean(score[p::bpb])) for p in range(bpb)]
        phase = int(np.argmax(phase_scores))

    return {
        "audio": str(audio_path),
        "duration": round(duration, 4),
        "source": source,
        "tempo": round(tempo, 2),
        "tempo_min": round(float(local_bpm.min()), 2),
        "tempo_max": round(float(local_bpm.max()), 2),
        "beats_per_bar": bpb,
        "downbeat_offset": phase,
        "downbeat_scores": [round(s, 3) for s in phase_scores],
        "beats": [round(float(t), 4) for t in beats],
        "strength": [round(float(s), 3) for s in strength],
        "strength_raw": [round(float(v), 3) for v in raw / (np.median(raw) + 1e-9)],  # unclipped, for major onsets
        "local_bpm": [round(float(b), 2) for b in local_bpm],
        "novelty": [round(float(n), 3) for n in novelty],
        "onsets": onsets,
        "kicks": kicks,
        "highs": highs,
    }


def save_beats(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_beats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
