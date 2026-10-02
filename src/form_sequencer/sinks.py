"""Frame sinks: where rendered frames go.

The renderer produces RGB uint8 arrays; a sink consumes them. New outputs (a
TouchDesigner/Spout/NDI stream, a live window driven by an audio clock) only need
to implement ``open / write / close`` — the renderer and sequencer stay unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class FrameSink(Protocol):
    def open(self, width: int, height: int, fps: float) -> None: ...
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class FFmpegSink:
    """Pipe raw frames into ffmpeg and mux the original audio in sync.

    Default: RGB frames to H.264 (.mp4). With ``alpha=True``: RGBA frames (straight alpha)
    to ProRes 4444 with its alpha channel and PCM audio (.mov), which editors such as
    Premiere, Resolve, After Effects and Final Cut read with transparency."""

    def __init__(self, path: Path, audio: Path | None, start: float = 0.0, duration: float | None = None,
                 preset: str = "medium", crf: int = 18, alpha: bool = False, lead: float = 0.0):
        # start: video time of the first frame; lead: video time at which the audio begins
        # (an intro before the music); silence fills the video before and after the audio
        self.path, self.audio, self.start, self.duration = Path(path), audio, start, duration
        self.lead = lead
        self.preset, self.crf, self.alpha = preset, crf, alpha
        self.proc: subprocess.Popen | None = None

    def open(self, width: int, height: int, fps: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgba" if self.alpha else "rgb24", "-s", f"{width}x{height}",
               "-r", f"{fps}", "-i", "-"]
        if self.audio:
            at = self.start - self.lead  # audio time of the first frame
            cmd += ["-ss", f"{max(0.0, at):.4f}", "-i", str(self.audio), "-map", "0:v", "-map", "1:a"]
            delay = int(round(max(0.0, -at) * 1000))
            cmd += ["-af", (f"adelay={delay}:all=1," if delay else "") + "apad"]
            # padded audio never ends on its own: cut the output at the video's length
            cmd += ["-t", f"{self.duration:.4f}"] if self.duration else ["-shortest"]
            cmd += ["-c:a", "pcm_s16le"] if self.alpha else ["-c:a", "aac", "-b:a", "192k"]
        if self.alpha:
            cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-alpha_bits", "16",
                    "-vendor", "apl0", str(self.path)]
        else:
            cmd += ["-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")


class PNGSink:
    """Write numbered PNG frames (for inspection, or import into other tools)."""

    def __init__(self, folder: Path, every: int = 1):
        self.folder, self.every, self.i = Path(folder), max(1, every), 0

    def open(self, width: int, height: int, fps: float) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for old in self.folder.glob("frame_*.png"):
            old.unlink()

    def write(self, frame: np.ndarray) -> None:
        if self.i % self.every == 0:
            Image.fromarray(frame).save(self.folder / f"frame_{self.i:06d}.png", compress_level=1)
        self.i += 1

    def close(self) -> None:
        pass


class TeeSink:
    def __init__(self, *sinks: FrameSink):
        self.sinks = sinks

    def open(self, width: int, height: int, fps: float) -> None:
        for s in self.sinks:
            s.open(width, height, fps)

    def write(self, frame: np.ndarray) -> None:
        for s in self.sinks:
            s.write(frame)

    def close(self) -> None:
        for s in self.sinks:
            s.close()
