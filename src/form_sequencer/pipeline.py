"""High-level steps used by the CLI: analyze, sequence, render."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .audio import analyze_audio, load_beats, save_beats
from .config import IMAGE_EXTS, PDF_EXTS, Project
from . import photos, stamps
from .documents import load_page, rasterize
from .fonts import font
from .fragments import extract, load_manifest, write_inspection
from .render import FrameRenderer, frame_times
from .sequencer import Timeline, build_timeline
from .sinks import FFmpegSink, FrameSink, PNGSink, TeeSink

Log = Callable[[str], None]


def analyze_audio_step(project: Project, settings: dict, log: Log) -> dict:
    audio = project.audio_file(settings)
    manual = project.beats_file(settings)
    log(f"audio: analyzing {os.path.relpath(audio, project.root)}"
        + (f" with manual beats from {os.path.relpath(manual, project.root)}" if manual.exists() else ""))
    beats = analyze_audio(audio, settings, manual)
    beats["config"] = settings["audio"]
    save_beats(beats, project.beats_path)
    log(f"  tempo ~{beats['tempo']} BPM (local {beats['tempo_min']}-{beats['tempo_max']}), "
        f"{len(beats['beats'])} beats ({beats['source']}), first beat {beats['beats'][0]:.3f}s, "
        f"downbeat offset {beats['downbeat_offset']}")
    return beats


def analyze_documents_step(project: Project, settings: dict, log: Log) -> dict:
    log(f"documents: rasterizing {os.path.relpath(project.documents_dir(settings), project.root)}")
    pages = rasterize(project, settings)
    manifest = extract(project, settings, pages, log)
    written = write_inspection(project, manifest)
    review = json.loads((project.build / "exclusions_review.json").read_text(encoding="utf-8"))
    sheets = photos.write_review(project.build / "exclusions", manifest["pages"], review,
                                 lambda page: load_page(project, page))
    kinds = [r["verdict"] for v in review.values() for r in v["detected"]["regions"]]
    n_scr = sum(len(v.get("scramble_final", [])) for v in review.values())
    n_rejected = sum(v["final"]["rejected"] for v in review.values())
    log(f"  excluded: {kinds.count('photo') + kinds.count('uncertain')} photo areas, {kinds.count('email')} email "
        f"addresses, {kinds.count('qr')} QR codes, {n_rejected} pages rejected; {n_scr} logo/title areas scrambled; "
        f"review sheets in "
        f"{project.show(sheets[0].parent)} (corrections: exclusions.yaml)")
    kinds: dict[str, int] = {}
    for f in manifest["fragments"]:
        if f.get("enabled", True):
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    log(f"  {len(pages)} pages, {sum(kinds.values())} fragments: "
        + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    log(f"  manifest: {project.show(project.manifest_path)}; overlays and contact sheet in "
        f"{project.show(written[0].parent)}")
    return manifest


def _newest(paths) -> float:
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def load_inputs(project: Project, settings: dict, log: Log) -> tuple[dict, dict]:
    """Load analysis results, re-running whatever is missing or older than its inputs."""
    audio = project.audio_file(settings)
    manual = project.beats_file(settings)
    beats = None
    if project.beats_path.exists():
        beats = load_beats(project.beats_path)
        stale = Path(beats["audio"]).resolve() != audio.resolve() or beats.get("config") != settings["audio"] \
            or "kicks" not in beats or "highs" not in beats \
            or _newest([audio, manual]) > project.beats_path.stat().st_mtime
        if stale:
            log("audio or beat settings changed since last analysis")
            beats = None
    if beats is None:
        beats = analyze_audio_step(project, settings, log)

    docs = project.documents_dir(settings)
    doc_files = [p for p in docs.rglob("*") if p.suffix.lower() in IMAGE_EXTS | PDF_EXTS]
    if not project.manifest_path.exists() or \
            _newest(doc_files + [project.exclusions_path]) > project.manifest_path.stat().st_mtime:
        manifest = analyze_documents_step(project, settings, log)
    else:
        manifest = load_manifest(project)
    for kind in ("stamps", "writing"):
        src = project.root / settings[kind]["source"]
        index = project.build / kind / "index.json"
        if src.exists():
            sources = [p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            if not index.exists() or _newest(sources + [src]) > index.stat().st_mtime:
                stamps.process(project, settings, log, kind)
    return beats, manifest


def sequence(project: Project, settings: dict, log: Log) -> tuple[Timeline, dict]:
    beats, manifest = load_inputs(project, settings, log)
    tl = build_timeline(beats, manifest, settings, stamps.load(project), stamps.load(project, "writing"))
    project.timeline_path.write_text(json.dumps(tl.to_json(), indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for e in tl.events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    log(f"timeline: {len(tl.segments)} placements, events: "
        + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none")
        + f" -> {project.show(project.timeline_path)}")
    return tl, manifest


def render(project: Project, *, preview: bool, start: float = 0.0, duration: float | None = None,
           frames_dir: Path | None = None, stills: int = 0, output: Path | None = None, log: Log = print) -> Path:
    settings = project.load_settings()
    tl, manifest = sequence(project, settings, log)
    mode = settings["preview"] if preview else settings["video"]
    scale = float(settings["preview"]["scale"]) if preview else 1.0
    fps = float(mode["fps"])
    intro, outro = float(settings["video"].get("intro", 0)), float(settings["video"].get("outro", 0))
    total = intro + tl.duration + outro  # video time: intro, then the music, then the outro
    end = total if duration is None else min(total, start + duration)
    if preview:
        settings["video"]["alpha"] = False  # previews are for viewing: always composited
    renderer = FrameRenderer(project, settings, manifest, tl, scale=scale)
    out = output or project.out / ("preview.mp4" if preview else "final.mp4")
    audio = project.audio_file(settings)
    sink: FrameSink = FFmpegSink(out, audio, start=start, duration=end - start,
                                 preset=mode["preset"], crf=int(mode["crf"]), lead=intro)
    if frames_dir:
        sink = TeeSink(sink, PNGSink(frames_dir))
    alpha_out = out.with_suffix(".mov") if renderer.alpha else None
    alpha_sink = FFmpegSink(alpha_out, audio, start=start, duration=end - start, alpha=True,
                            lead=intro) if alpha_out else None
    times = frame_times(start, end, fps)
    log(f"rendering {len(times)} frames at {renderer.W}x{renderer.H}, {fps:g} fps -> {out}"
        + (f" and {alpha_out.name} (with transparency)" if alpha_out else ""))
    t_wall = time.time()
    sink.open(renderer.W, renderer.H, fps)
    if alpha_sink:
        alpha_sink.open(renderer.W, renderer.H, fps)
    step = max(1, len(times) // 10)
    try:
        for i, t in enumerate(times):
            rgb, trans = renderer.layers(t - intro)
            sink.write(renderer.composite(rgb, trans))
            if alpha_sink:
                alpha_sink.write(renderer.rgba(rgb, trans) if trans is not None
                                 else np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]))
            if i % step == 0 and i:
                el = time.time() - t_wall
                log(f"  {100 * i // len(times):3d}%  t={t:6.2f}s  ({i / el:.1f} frames/s)")
    finally:
        sink.close()
        if alpha_sink:
            alpha_sink.close()
    log(f"done in {time.time() - t_wall:.1f}s: {out}" + (f" + {alpha_out}" if alpha_out else ""))
    cam = export_camera(renderer, times, fps, out)
    log(f"camera moves: {cam.with_suffix('.csv').name}, {cam.with_suffix('.jsx').name} (After Effects)")
    if stills:
        sheet = contact_sheet(renderer, tl, start, end, stills)
        path = out.with_name(out.stem + "_stills.jpg")
        sheet.save(path, quality=88)
        log(f"stills: {path}")
    return out


def export_camera(renderer: FrameRenderer, times: list[float], fps: float, out: Path) -> Path:
    """Write the camera move of every frame, so a background added in post can move with it.

    The renderer shows the scene as  out = (in - view) * zoom + centre.  A background layer
    (anchored at its centre, filling the frame) follows when given  scale = zoom * 100 %  and
    position = centre + zoom * (centre - view).  Written as CSV and as an After Effects script
    that keys the selected layer's Scale and Position."""
    W, H = renderer.W, renderer.H
    rows = []
    for i, t in enumerate(times):
        z, vx, vy = renderer.camera(t - renderer.intro)
        rows.append((i, round(t, 4), z, vx, vy, round(z * 100, 3),
                     round(W / 2 + z * (W / 2 - vx), 2), round(H / 2 + z * (H / 2 - vy), 2)))
    base = out.with_name(out.stem + "_camera")
    lines = ["frame,time,zoom,view_x,view_y,layer_scale_percent,layer_position_x,layer_position_y"]
    lines += [",".join(str(v) for v in r) for r in rows]
    base.with_suffix(".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    t0 = times[0] if times else 0.0  # the video starts at t0: keys are in video time
    keys = ",".join(f"[{round(r[1] - t0, 4)},{r[5]},{r[6]},{r[7]}]" for r in rows)
    jsx = f"""// Form Sequencer camera move ({len(rows)} frames, {fps:g} fps, {W}x{H}).
// Select the background layer in a {W}x{H} comp, then File > Scripts > Run Script File.
// Its Scale and Position are keyed so it moves with the forms' camera. For depth (parallax),
// scale the movement down: set PARALLAX below 1.
(function () {{
    var PARALLAX = 1.0;
    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem) || comp.selectedLayers.length !== 1) {{
        alert("Select exactly one layer in the comp first."); return;
    }}
    var layer = comp.selectedLayers[0];
    var k = [{keys}];
    var cx = comp.width / 2, cy = comp.height / 2, sx = comp.width / {W};
    app.beginUndoGroup("Form Sequencer camera");
    var sc = layer.property("Transform").property("Scale");
    var po = layer.property("Transform").property("Position");
    var times = [], scales = [], positions = [];
    for (var i = 0; i < k.length; i++) {{
        var s = 100 + (k[i][1] - 100) * PARALLAX;
        times.push(k[i][0]);
        scales.push([s, s]);
        positions.push([cx + (k[i][2] - {W / 2}) * sx * PARALLAX, cy + (k[i][3] - {H / 2}) * sx * PARALLAX]);
    }}
    sc.setValuesAtTimes(times, scales);
    po.setValuesAtTimes(times, positions);
    app.endUndoGroup();
}})();
"""
    base.with_suffix(".jsx").write_text(jsx, encoding="utf-8")
    return base


def contact_sheet(renderer: FrameRenderer, tl: Timeline, start: float, end: float, n: int) -> Image.Image:
    """Frames shortly after the busiest moments, for quick visual review."""
    moments = sorted({round(e["t"], 3) for e in tl.events if start <= e["t"] < end})
    if len(moments) > n:
        idx = np.linspace(0, len(moments) - 1, n).round().astype(int)
        moments = [moments[i] for i in idx]
    if not moments:
        moments = list(np.linspace(start, end, n + 2)[1:-1])
    # mid-transition and settled
    shots = []
    for t in moments:
        shots.append((t + 0.12, "mid"))
    cols = min(3, len(shots))
    rows = (len(shots) + cols - 1) // cols
    tw = 640
    th = int(tw * renderer.H / renderer.W)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    f = font("sans", 14)
    for k, (t, _) in enumerate(shots):
        img = Image.fromarray(renderer.frame(min(t, end - 1e-3))).resize((tw, th), Image.LANCZOS)
        x, y = (k % cols) * tw, (k // cols) * (th + 22)
        sheet.paste(img, (x, y))
        d.text((x + 6, y + th + 3), f"t = {t:.2f}s", font=f, fill=(230, 230, 230))
    return sheet


def still(project: Project, t: float, out: Path, preview: bool = False, log: Log = print) -> Path:
    settings = project.load_settings()
    tl, manifest = sequence(project, settings, log)
    scale = float(settings["preview"]["scale"]) if preview else 1.0
    r = FrameRenderer(project, settings, manifest, tl, scale=scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(r.frame(t)).save(out)
    return out
