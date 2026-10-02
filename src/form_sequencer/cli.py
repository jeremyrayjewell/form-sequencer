"""Command line: form-seq <command> [PROJECT]."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import click

from .config import Project

PROJECT = click.argument("project", type=click.Path(file_okay=False, path_type=Path), default=".")


def _log(msg: str) -> None:
    click.echo(msg)


def _project(path: Path) -> Project:
    p = Project(path.resolve())
    if not p.settings_path.exists():
        raise click.ClickException(f"{p.settings_path} not found; run `form-seq init {path}` first")
    return p


@click.group()
def main() -> None:
    """Form Sequencer: an audio-driven, ever-revised bureaucratic form."""
    warnings.simplefilter("ignore", category=FutureWarning)
    for stream in (sys.stdout, sys.stderr):  # file names in any script, even when output is redirected
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


@main.command()
@PROJECT
def init(project: Path) -> None:
    """Create a project folder with settings.yaml and input folders."""
    p = Project(project.resolve())
    p.init()
    click.echo(f"project ready: {p.root}\n"
               f"  put one audio track in   {p.audio_dir}\n"
               f"  put scans / PDFs in      {p.root / 'input' / 'documents'}\n"
               f"  then run                 form-seq analyze {project}")


@main.command()
@PROJECT
@click.option("--seed", default=1, show_default=True, help="Seed for the synthetic track and forms.")
def fixture(project: Path, seed: int) -> None:
    """Create a project filled with a synthetic track and synthetic forms."""
    from .fixture import make_fixture

    p = Project(project.resolve())
    truth = make_fixture(p, seed=seed)
    click.echo(f"fixture written to {p.root}: {len(truth['beats'])} beats, "
               f"{truth['bpm_start']}->{truth['bpm_end']} BPM, first beat at {truth['first_beat']}s "
               f"(ground truth in fixture_truth.json)")


@main.command()
@PROJECT
@click.option("--only", type=click.Choice(["audio", "documents"]), help="Run just one half of the analysis.")
def analyze(project: Path, only: str | None) -> None:
    """Detect beats, extract fragments, and build the timeline."""
    from .pipeline import analyze_audio_step, analyze_documents_step, sequence

    p = _project(project)
    settings = p.load_settings()
    if only in (None, "audio"):
        analyze_audio_step(p, settings, _log)
    if only in (None, "documents"):
        analyze_documents_step(p, settings, _log)
    if only is None:
        sequence(p, settings, _log)
        click.echo(f"next: `form-seq beats {project}` to check timing, `form-seq preview {project}` to watch")


@main.command()
@PROJECT
def beats(project: Path) -> None:
    """Write a click track, beat plot and Audacity labels to check/correct beat alignment."""
    from .audio import load_beats
    from .beatcheck import write_beatcheck
    from .pipeline import load_inputs

    p = _project(project)
    settings = p.load_settings()
    load_inputs(p, settings, _log)
    paths = write_beatcheck(load_beats(p.beats_path), p.audio_file(settings), p.build / "beatcheck")
    for path in paths:
        click.echo(f"  {p.show(path)}")
    click.echo(
        "Listen to click.wav (high click = downbeat) and look at beats_plot.png.\n"
        "  all beats early/late by a constant  -> audio.beat_offset in settings.yaml\n"
        "  bars start on the wrong beat        -> audio.downbeat_offset (0..beats_per_bar-1)\n"
        "  half/double tempo                   -> audio.tempo_hint\n"
        f"  anything else: edit beats_labels.txt in Audacity and export to {settings['inputs']['beats_file']}"
    )


def _render_opts(f):
    f = click.option("--start", default=0.0, show_default=True, help="Start time in seconds.")(f)
    f = click.option("--duration", type=float, help="Render only this many seconds.")(f)
    f = click.option("--frames", "frames_dir", type=click.Path(file_okay=False, path_type=Path),
                     help="Also write every frame as PNG to this folder.")(f)
    f = click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Output video path.")(f)
    return f


@main.command()
@PROJECT
@_render_opts
@click.option("--stills", default=9, show_default=True, help="Also write a contact sheet of N moments (0 = off).")
def preview(project: Path, start: float, duration: float | None, frames_dir: Path | None, output: Path | None, stills: int) -> None:
    """Render a reduced-resolution preview with audio (out/preview.mp4)."""
    from .pipeline import render

    render(_project(project), preview=True, start=start, duration=duration, frames_dir=frames_dir,
           stills=stills, output=output, log=_log)


@main.command(name="render")
@PROJECT
@_render_opts
def render_cmd(project: Path, start: float, duration: float | None, frames_dir: Path | None, output: Path | None) -> None:
    """Render the final full-resolution video with the original audio (out/final.mp4)."""
    from .pipeline import render

    render(_project(project), preview=False, start=start, duration=duration, frames_dir=frames_dir,
           output=output, log=_log)


@main.command()
@PROJECT
@click.option("--start", default=0.0, show_default=True, help="Start time in seconds.")
@click.option("--duration", type=float, help="Render only this many seconds.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Output video path.")
@click.option("--composite", is_flag=True, help="Also lay out/final.mov over it (out/composite.mp4).")
def background(project: Path, start: float, duration: float | None, output: Path | None, composite: bool) -> None:
    """Render the background scene for post (out/background.mp4)."""
    from .background import render_background

    render_background(_project(project), start=start, duration=duration, output=output, composite=composite, log=_log)


@main.command()
@PROJECT
@click.option("-t", "--time", "t", type=float, required=True, help="Time in seconds.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="PNG path (default out/still_<t>.png).")
def still(project: Path, t: float, output: Path | None) -> None:
    """Render a single full-resolution frame."""
    from .pipeline import still as still_fn

    p = _project(project)
    out = still_fn(p, t, output or p.out / f"still_{t:07.3f}.png", log=_log)
    click.echo(out)


if __name__ == "__main__":
    main()
