import copy
import math
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pytest

from form_sequencer.audio import analyze_audio, load_beats
from form_sequencer.config import DEFAULTS
from form_sequencer.fragments import extract, load_manifest, save_manifest
from form_sequencer.documents import rasterize
from form_sequencer.layout import fit
from form_sequencer.pipeline import render
from form_sequencer.render import FrameRenderer
from form_sequencer.sequencer import Timeline, build_timeline
from form_sequencer.sinks import ffmpeg_exe
from form_sequencer.templates import build_template, build_templates, match_cells


def _truth(p):
    return json.loads((p.root / "fixture_truth.json").read_text())


# ---------------------------------------------------------------- audio

def test_beats_match_ground_truth(fixture_project):
    beats = load_beats(fixture_project.beats_path)
    truth = np.array(_truth(fixture_project)["beats"])
    found = np.array(beats["beats"])
    err = np.abs(truth[:, None] - found[None]).min(axis=1)
    assert (err < 0.03).mean() >= 0.95, "at least 95% of true beats found within 30 ms"
    assert found[0] > 0.3, "first beat is not assumed to be at t=0"
    assert not 115 < beats["tempo"] < 125, "tempo is detected, not assumed to be 120"
    assert beats["tempo_max"] - beats["tempo_min"] > 3, "tempo drift is preserved in beat times"


def test_downbeat_and_sections(fixture_project):
    beats = load_beats(fixture_project.beats_path)
    truth = _truth(fixture_project)
    b = np.array(beats["beats"])
    first_down = b[beats["downbeat_offset"]]
    assert min(abs(first_down - d) for d in truth["downbeats"]) < 0.03
    nov = np.array(beats["novelty"])
    peaks = b[nov > 1.0]
    for section in truth["sections"][2:]:
        assert np.min(np.abs(peaks - section)) < 0.8, f"section change near {section}s is detected"


def test_manual_beats_override(fixture_project, tmp_path):
    truth = _truth(fixture_project)["beats"]
    manual = tmp_path / "beats.txt"
    manual.write_text("".join(f"{t}\t{t}\t{i}\n" for i, t in enumerate(truth)))  # Audacity label format
    settings = fixture_project.load_settings()
    beats = analyze_audio(fixture_project.audio_file(settings), settings, manual)
    assert beats["source"] == "manual"
    assert np.allclose(beats["beats"], truth, atol=1e-3)


# ---------------------------------------------------------------- fragments

def test_fragments_cover_kinds(fixture_project):
    manifest = load_manifest(fixture_project)
    kinds = {f["kind"] for f in manifest["fragments"]}
    assert {"header", "table", "fields", "checkboxes", "stamp", "signature"} <= kinds
    pages = {p["key"]: p for p in manifest["pages"]}
    assert any(k.endswith("_pdf_p2") for k in pages), "multipage PDF is split into pages"
    for f in manifest["fragments"]:
        x, y, w, h = f["bbox"]
        page = pages[f["page"]]
        assert 0 <= x and 0 <= y and x + w <= page["width"] and y + h <= page["height"]
        assert f["source"].startswith("input/documents/")


def test_manifest_rerun_keeps_manual_and_disabled(fixture_project):
    settings = fixture_project.load_settings()
    manifest = load_manifest(fixture_project)
    original = copy.deepcopy(manifest)
    victim = manifest["fragments"][0]["id"]
    manifest["fragments"][0]["enabled"] = False
    manifest["fragments"].append({"id": "my_crop", "page": manifest["pages"][0]["key"], "source": "x", "source_page": 1,
                                  "bbox": [10, 10, 200, 100], "kind": "text", "enabled": True, "origin": "manual"})
    save_manifest(fixture_project, manifest)
    try:
        again = extract(fixture_project, settings, rasterize(fixture_project, settings))
        by_id = {f["id"]: f for f in again["fragments"]}
        assert "my_crop" in by_id
        assert by_id[victim]["enabled"] is False
    finally:
        save_manifest(fixture_project, original)


# ---------------------------------------------------------------- layout

def test_templates_reject_overlap():
    spec = copy.deepcopy(DEFAULTS["templates"][0])
    spec["slots"] = {"a": {"cell": [0, 0, 6, 4]}, "b": {"cell": [5, 2, 4, 4]}}
    with pytest.raises(ValueError, match="overlap"):
        build_template(spec, DEFAULTS)


def test_templates_are_varied_and_inside_their_grid():
    ts = build_templates(DEFAULTS)
    assert len({t.aspect for t in ts}) >= 4, "several page proportions"
    assert len({t.header_pos for t in ts}) == 3 and len({t.rules for t in ts}) >= 4
    assert len({t.columns for t in ts}) >= 3
    assert any(s.prefer and s.prefer[0] == "diagram" for t in ts for s in t.slots), "diagram areas"
    for t in ts:
        gx, gy, gw, gh = t.grid
        for s in t.slots:
            x, y, w, h = s.rect
            assert x >= gx - 1e-9 and y >= gy - 1e-9 and x + w <= gx + gw + 1e-9 and y + h <= gy + gh + 1e-9
        rects = [s.rect for s in t.slots]
        for i, r1 in enumerate(rects):
            for r2 in rects[i + 1:]:
                assert r1[0] + r1[2] <= r2[0] + 1e-9 or r2[0] + r2[2] <= r1[0] + 1e-9 \
                    or r1[1] + r1[3] <= r2[1] + 1e-9 or r2[1] + r2[3] <= r1[1] + 1e-9, t.name


def test_cell_matching_is_one_to_one():
    ts = build_templates(DEFAULTS)
    for a in ts:
        for b in ts:
            pairs = match_cells(a, b)
            assert len(pairs) == min(len(a.slots), len(b.slots))
            assert len({x for x, _ in pairs}) == len(pairs) == len({y for _, y in pairs})


def test_fit_preserves_aspect():
    x, y, w, h = fit(400, 100, (0, 0, 300, 300))
    assert (w, h) == (300, 75)
    x, y, w, h = fit(100, 400, (10, 10, 300, 200), center=True)
    assert h == 200 and w == 50 and x == 10 + 125


# ---------------------------------------------------------------- sequencing / rendering

def _timeline(p, seed=None):
    settings = p.load_settings()
    if seed is not None:
        settings["seed"] = seed
    return build_timeline(load_beats(p.beats_path), load_manifest(p), settings), settings


def test_timeline_deterministic(fixture_project):
    a, _ = _timeline(fixture_project)
    b, _ = _timeline(fixture_project)
    assert json.dumps(a.to_json()) == json.dumps(b.to_json())
    c, _ = _timeline(fixture_project, seed=99)
    assert json.dumps(a.to_json()) != json.dumps(c.to_json())
    assert Timeline.from_json(json.loads(json.dumps(a.to_json()))).to_json() == a.to_json()


def test_events_on_beats_or_onsets(fixture_project):
    tl, _ = _timeline(fixture_project)
    beats = load_beats(fixture_project.beats_path)
    anchors = np.array(beats["beats"] + [o["t"] for o in beats["onsets"]])
    for e in tl.events:
        assert np.min(np.abs(anchors - e["t"])) < 1e-3, e
    assert tl.events and min(e["t"] for e in tl.events) >= beats["beats"][beats["downbeat_offset"]]


def test_settled_fragments_never_overlap(fixture_project):
    tl, settings = _timeline(fixture_project)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.5)
    for t in np.linspace(0, tl.duration - 0.01, 240):
        cur, f, (pw, ph), on_page, slips = r.visible(t)
        boxes = []
        for s, st, rect in on_page:
            if st[1] < 1 or st[3] is not None or st[2] != 1.0 or st[0] < 1:
                continue  # in transition
            boxes.append((s.slot, r.placement(s, r.rect_px(rect, pw, ph), 1.0)))
        slots = [b[0] for b in boxes]
        assert len(slots) == len(set(slots)), f"two settled fragments share a slot at t={t:.2f}"
        for i, (_, a) in enumerate(boxes):
            for _, b in boxes[i + 1:]:
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                assert ox <= 0 or oy <= 0, f"fragments overlap at t={t:.2f}"


def test_frames_deterministic(fixture_project):
    def hashes():
        tl, settings = _timeline(fixture_project)
        r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
        return [hashlib.sha1(r.frame(t).tobytes()).hexdigest() for t in (0.0, 5.0, 12.43, 20.6, 33.3)]
    assert hashes() == hashes()


def test_preview_render_has_synced_audio(fixture_project):
    out = render(fixture_project, preview=True, start=10.0, duration=2.0, log=lambda m: None)
    probe = subprocess.run([ffmpeg_exe(), "-i", str(out)], capture_output=True, text=True).stderr
    assert "Video: h264" in probe and "Audio: aac" in probe
    assert "Duration: 00:00:02.0" in probe


def test_regularize_repairs_extra_and_displaced_beats():
    from form_sequencer.audio import _regularize

    grid = np.arange(0, 60, 0.5) + 0.017
    broken = np.sort(np.append(grid, 30.29))           # extra beat inside a bar
    broken[np.searchsorted(broken, 12.0)] += 0.08       # displaced beat
    fixed = _regularize(broken)
    assert len(fixed) == len(grid)
    assert np.abs(fixed - grid).max() < 0.01
    drift = np.cumsum(np.linspace(0.64, 0.59, 80))      # gradual tempo change is left alone
    assert np.allclose(_regularize(drift), drift)


# ---------------------------------------------------------------- photo exclusion

def test_photos_are_excluded_and_graphics_kept(fixture_project):
    truth = _truth(fixture_project)
    manifest = load_manifest(fixture_project)
    pages = {p["key"]: p for p in manifest["pages"]}
    from form_sequencer.photos import intersects
    for key, boxes in truth["photos"].items():
        for box in boxes:
            forbidden = pages[key]["forbidden"]
            x, y, w, h = box
            assert any(fx <= x and fy <= y and fx + fw >= x + w and fy + fh >= y + h
                       for fx, fy, fw, fh in forbidden), "the whole photo is forbidden"
    for f in manifest["fragments"]:
        if f.get("enabled", True) and not f.get("excluded"):
            assert not intersects(f["bbox"], pages[f["page"]]["forbidden"]), f["id"]
    for key, boxes in truth["eligible_graphics"].items():
        for box in boxes:
            assert not intersects(box, pages[key]["forbidden"]), "a flat printed emblem stays eligible"
    kinds = {f["kind"] for f in manifest["fragments"]}
    assert {"stamp", "signature", "diagram"} <= kinds
    assert list((fixture_project.build / "exclusions").glob("review_*.jpg")), "review sheet written"


def test_exclusion_overrides(fixture_project):
    from form_sequencer.photos import apply_overrides
    detected = {"regions": [{"id": "p1", "bbox": [10, 10, 50, 50], "verdict": "uncertain"}], "page": "ok", "reason": ""}
    assert apply_overrides("pg", 500, 500, detected, {})["forbidden"] == [[10, 10, 50, 50]]
    assert apply_overrides("pg", 500, 500, detected, {"pg": {"allow": ["p1"]}})["forbidden"] == []
    added = apply_overrides("pg", 500, 500, detected, {"pg": {"add": [[100, 100, 20, 20]]}})["forbidden"]
    assert [100, 100, 20, 20] in added
    rej = apply_overrides("pg", 500, 500, detected, {"pg": {"reject_page": True}})
    assert rej["rejected"] and rej["forbidden"] == [[0, 0, 500, 500]]
    auto = dict(detected, page="rejected", reason="x")
    assert not apply_overrides("pg", 500, 500, auto, {"pg": {"allow_page": True}})["rejected"]


# ---------------------------------------------------------------- form changes

def test_form_changes_follow_the_music(fixture_project):
    tl, settings = _timeline(fixture_project)
    beats = load_beats(fixture_project.beats_path)
    truth = _truth(fixture_project)
    b = np.array(beats["beats"])
    assert len(tl.forms) >= 2
    min_gap = settings["form_changes"]["min_bars"] * beats["beats_per_bar"]
    for f in tl.forms:
        assert np.min(np.abs(b - f["t"])) < 1e-3, "form changes happen on detected beats"
        assert f["erase"][0] <= f["reshape"][0] <= f["draw"][0] <= f["arrive"], "erase -> reshape -> draw -> arrive"
        assert f["from"] != f["to"]
    ks = [f["beat"] for f in tl.forms]
    assert all(b2 - b1 >= min_gap for b1, b2 in zip(ks, ks[1:]))
    sections = [f["t"] for f in tl.forms if f["reason"] == "section"]
    for s_t in truth["sections"][2:]:
        assert min(abs(s_t - t) for t in sections) < 0.8, f"section change near {s_t}s changes the form"


def test_form_change_controls(fixture_project):
    beats = load_beats(fixture_project.beats_path)
    manifest = load_manifest(fixture_project)
    settings = fixture_project.load_settings()
    fewer = copy.deepcopy(settings)
    fewer["form_changes"]["min_bars"] = 40
    fewer["form_changes"]["max_bars"] = 0
    assert len(build_timeline(beats, manifest, fewer).forms) < len(build_timeline(beats, manifest, settings).forms)
    calm = copy.deepcopy(settings)
    calm["form_changes"].update(drama=0.0, drama_follows_music=False)
    wild = copy.deepcopy(settings)
    wild["form_changes"].update(drama=1.0, drama_follows_music=False)
    tmap = {t.name: t for t in build_templates(settings)}
    from form_sequencer.templates import distance
    d_calm = [distance(tmap[f["from"]], tmap[f["to"]]) for f in build_timeline(beats, manifest, calm).forms]
    d_wild = [distance(tmap[f["from"]], tmap[f["to"]]) for f in build_timeline(beats, manifest, wild).forms]
    assert np.mean(d_calm) < np.mean(d_wild)


def test_form_stays_readable_during_changes(fixture_project):
    base = fixture_project.load_settings()
    beats, manifest = load_beats(fixture_project.beats_path), load_manifest(fixture_project)
    for style in ("stack", "morph"):
        settings = copy.deepcopy(base)
        settings["form_changes"]["style"] = style
        tl = build_timeline(beats, manifest, settings)
        r = FrameRenderer(fixture_project, settings, manifest, tl, scale=0.25)
        assert tl.forms and all(f["style"] == style for f in tl.forms)
        for i, f in enumerate(tl.forms):
            for t in np.linspace(f["t"], f["draw"][1] + 0.01, 10):
                if style == "stack":
                    # the covered sheet stays on the pile exactly as it was; the new sheet arrives printed
                    k = r.top_sheet(t)
                    assert k == i + 1 and r.sheets[k - 1]["freeze"] == f["t"]
                    assert k - 1 >= max(0, k - r.pile_depth)
                    assert r.frame(t).shape == (r.H, r.W, 3)
                    continue
                items = r.transition_items(f, t)
                frame_lines = [p for p, q, _, _ in items if p.role.startswith("chrome:frame") and q >= 0.99]
                cell_lines = [p for p, q, _, _ in items if p.kind == "line" and p.role.startswith("cell:") and q >= 0.5]
                assert len(frame_lines) == 4, f"page frame always drawn (t={t:.2f})"
                assert len(cell_lines) >= 8, f"a ruled grid is visible throughout (t={t:.2f}: {len(cell_lines)})"


def test_stacked_sheets_keep_their_content(fixture_project):
    tl, settings = _timeline(fixture_project)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    f = tl.forms[0]
    before, _ = r.page_at(f["t"] - 1e-3, f["from"])
    frozen = r._frozen(0).astype(np.float32) / 255
    assert np.abs(before - frozen).mean() < 0.01, "a covered sheet looks exactly as it did before it was covered"


# ---------------------------------------------------------------- email exclusion

def test_emails_are_excluded(fixture_project):
    import pymupdf
    from form_sequencer.photos import intersects
    from form_sequencer.emails import looks_like_email
    truth = _truth(fixture_project)
    manifest = load_manifest(fixture_project)
    pages = {p["key"]: p for p in manifest["pages"]}
    boxes = {k: list(v) for k, v in truth["emails"].items()}  # typed on a scan: found by OCR
    pdf = fixture_project.root / "input" / "documents" / "annex_bundle.pdf"
    with pymupdf.open(pdf) as doc:  # in a vector PDF: found in the text layer
        page = doc[0]
        scale = pages["annex_bundle_pdf_p1"]["width"] / page.rect.width
        for r in page.search_for(truth["pdf_email"]):
            boxes.setdefault("annex_bundle_pdf_p1", []).append(
                [int(r.x0 * scale), int(r.y0 * scale), int(r.width * scale), int(r.height * scale)])
    assert sum(len(v) for v in boxes.values()) >= 2
    for key, bs in boxes.items():
        for b in bs:
            assert intersects(b, pages[key]["forbidden"]), f"email on {key} is forbidden"
            for f in manifest["fragments"]:
                if f["page"] == key and f.get("enabled", True) and not f.get("excluded"):
                    assert not intersects(f["bbox"], [b]), f"fragment {f['id']} shows an email address"
    assert looks_like_email("contact: name.surname@mail.example.org")
    assert looks_like_email("jdoe © gmail . com")
    assert not looks_like_email("E-mail address of applicant")


# ---------------------------------------------------------------- QR codes and scrambling

def test_qr_codes_are_excluded(fixture_project):
    from form_sequencer.photos import intersects
    truth = _truth(fixture_project)
    manifest = load_manifest(fixture_project)
    pages = {p["key"]: p for p in manifest["pages"]}
    assert truth["qr"]
    for key, boxes in truth["qr"].items():
        for x, y, w, h in boxes:
            assert any(fx <= x and fy <= y and fx + fw >= x + w and fy + fh >= y + h
                       for fx, fy, fw, fh in pages[key]["forbidden"]), "the whole QR code is forbidden"
            for f in manifest["fragments"]:
                if f["page"] == key and f.get("enabled", True) and not f.get("excluded"):
                    assert not intersects(f["bbox"], [[x, y, w, h]]), f"fragment {f['id']} shows a QR code"


def test_logos_and_institution_titles_are_scrambled(fixture_project):
    from form_sequencer.documents import load_page
    from form_sequencer.fragments import crop
    from form_sequencer.photos import intersects
    truth = _truth(fixture_project)
    manifest = load_manifest(fixture_project)
    pages = {p["key"]: p for p in manifest["pages"]}
    for key, boxes in truth["logos"].items():
        regions = [r["bbox"] for r in pages[key]["scramble"] if r["verdict"] == "logo"]
        for b in boxes:
            assert intersects(b, regions), "the letterhead crest is scrambled"
    for key, page in pages.items():
        titles = [r for r in page["scramble"] if r["verdict"] == "title"]
        assert titles, f"the institution name on {key} is scrambled"
        zone = manifest and fixture_project.load_settings()["scramble"]["letterhead_zone"] * page["height"]
        assert all(r["bbox"][1] < zone for r in page["scramble"] if r["verdict"] == "logo")
    for key, boxes in truth["eligible_graphics"].items():  # an emblem lower on the page is not a logo
        assert not intersects(boxes[0], [r["bbox"] for r in pages[key]["scramble"]])
    # crops really are scrambled where they overlap a scramble region, and untouched elsewhere
    key = "form_sr114_png_p1"
    x, y, w, h = pages[key]["scramble"][0]["bbox"]
    frag = {"page": key, "bbox": [x, y, w, h]}
    raw = load_page(fixture_project, pages[key])[y:y + h, x:x + w]
    assert np.abs(crop(fixture_project, manifest, frag).astype(int) - raw.astype(int)).mean() > 3
    frag = {"page": key, "bbox": [x, y + h + 400, w, h]}
    raw = load_page(fixture_project, pages[key])[y + h + 400:y + 2 * h + 400, x:x + w]
    assert np.array_equal(crop(fixture_project, manifest, frag), raw)


# ---------------------------------------------------------------- rubber stamps

def test_stamps_are_isolated_with_transparent_background(fixture_project):
    from form_sequencer import stamps
    from form_sequencer.fixture import stamp_photo
    truth = _truth(fixture_project)
    settings = fixture_project.load_settings()
    index = stamps.load(fixture_project)
    assert {s["id"] for s in index} == set(truth["stamps"])
    for kind, name in enumerate(("ring_received", "box_filed")):
        img, pts = stamp_photo(np.random.default_rng(5), kind)
        full = np.asarray(img.convert("RGB"))
        rgba, info = stamps.extract_stamp(full, settings["stamps"])
        assert 200 < info["ink_hue"] < 270, "blue/violet ink hue found"
        # map source points into the cropped result via the ink bounding box
        a = rgba[..., 3]
        assert a.max() > 200
        # the paper corner and the dark edge line are cropped away or transparent
        assert rgba.shape[1] < full.shape[1] - 10, "the dark paper edge is not part of the stamp"
        corners = [a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]
        assert max(corners) < 40, "background is transparent"
        assert (a > 128).mean() > 0.03, "the imprint itself is opaque"


def test_stamps_slam_on_kicks(fixture_project):
    tl, settings = _timeline(fixture_project)
    tl2 = build_timeline(load_beats(fixture_project.beats_path), load_manifest(fixture_project), settings,
                         __import__("form_sequencer.stamps", fromlist=["load"]).load(fixture_project))
    beats = load_beats(fixture_project.beats_path)
    kicks = np.array([k["t"] for k in beats["kicks"]])
    assert tl2.stamps, "stamps are placed"
    c = settings["stamps"]
    per_sheet = {}
    last = -1e9
    for st in tl2.stamps:
        assert np.min(np.abs(kicks - st["t"])) < 1e-3, "every stamp lands on a kick"
        assert st["t"] - last >= c["min_beats"] * 0.5, "stamps are spaced"
        last = st["t"]
        per_sheet[st["sheet"]] = per_sheet.get(st["sheet"], 0) + 1
        for f in tl2.forms:
            assert not (f["t"] <= st["t"] < max(f.get("land", f["t"]), f["draw"][1])), "no stamps while a sheet lands"
    assert max(per_sheet.values()) <= c["max_per_sheet"]
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl2, scale=0.25)
    st = tl2.stamps[0]
    before, _ = r.page_at(st["t"] - 0.01, st["layout"])
    after, _ = r.page_at(st["t"] + st["dur"] + 0.01, st["layout"])
    ph, pw = after.shape[:2]
    y, x = int(st["y"] * ph), int(st["x"] * pw)
    rad = int(st["size"] * ph / 2)
    region = (slice(max(0, y - rad), y + rad), slice(max(0, x - rad), x + rad))
    assert np.abs(after[region] - before[region]).mean() > 0.005, "the stamp is inked onto the sheet"


# ---------------------------------------------------------------- handwriting

def _writing_timeline(p, **overrides):
    from form_sequencer import stamps
    settings = p.load_settings()
    settings["writing"].update(overrides)
    tl = build_timeline(load_beats(p.beats_path), load_manifest(p), settings, stamps.load(p),
                        stamps.load(p, "writing"))
    return tl, settings


def test_writing_is_isolated_in_one_piece(fixture_project):
    from form_sequencer import stamps
    index = stamps.load(fixture_project, "writing")
    assert len(index) == 3, "each handwriting image gives exactly one writing"
    for it in index:
        rgba = np.asarray(__import__("PIL.Image", fromlist=["open"]).open(fixture_project.root / it["file"]))
        a = rgba[..., 3]
        assert max(a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]) < 40, "paper and ruling are transparent"
        assert (a > 128).mean() > 0.02
    assert sorted(it["lines"] for it in index) == [1, 2, 3], "written lines are counted"


def test_writing_follows_high_hits_and_reveals(fixture_project):
    tl, settings = _writing_timeline(fixture_project)
    beats = load_beats(fixture_project.beats_path)
    highs = np.array([h["t"] for h in beats["highs"]])
    assert tl.writings, "writings are placed"
    for w in tl.writings:
        assert np.min(np.abs(highs - w["t"])) < 1e-3, "every writing starts on a high-band hit"
        for f in tl.forms:
            assert not (f["t"] <= w["t"] < max(f.get("land", f["t"]), f["draw"][1]))
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    w = next(x for x in tl.writings if x["on_top"])
    half, _ = r.page_at(w["t"] + w["dur"] * 0.4, w["layout"], sheet=w["sheet"])
    full, _ = r.page_at(w["t"] + w["dur"] + 0.01, w["layout"], sheet=w["sheet"])
    before, _ = r.page_at(w["t"] - 0.01, w["layout"], sheet=w["sheet"])
    d_half, d_full = np.abs(half - before).sum(), np.abs(full - before).sum()
    assert 0 < d_half < d_full, "the writing appears progressively"


def test_writing_on_visible_parts_of_lower_sheets(fixture_project):
    from form_sequencer import pile
    tl, settings = _writing_timeline(fixture_project, lower_sheets=1.0, min_size=0.05, min_visible=0.5)
    lower = [w for w in tl.writings if not w["on_top"]]
    assert lower, "some writing goes on sheets under the top one"
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=1.0)
    for w in lower:
        k = r.top_sheet(w["t"])
        assert w["sheet"] < k
        # the writing's centre is on its sheet and not covered by any sheet above at that time
        geo = []
        for j in range(w["sheet"], k + 1):
            pw, ph = r.geo[r.sheets[j]["tpl"]]
            a, cx, cy = r._sheet_pos(r.sheets[j]["pose"])
            geo.append((pw, ph, pile.matrix(pw, ph, a, cx, cy)))
        pw, ph, M = geo[0]
        fx, fy = M @ np.array([w["x"] * pw, w["y"] * ph, 1.0])
        masks = pile.visible_masks(geo, r.W, r.H, grid=4)
        assert masks[0][int(fy // 4), int(fx // 4)], f"writing at t={w['t']} is visible"
    # and it is drawn into the pile under the top sheet
    w = lower[0]
    k = r.top_sheet(w["t"])
    before = r.pile(k, w["t"] - 0.001)
    after = r.pile(k, w["t"] + w["dur"] + 0.01)
    assert np.abs(after.astype(int) - before.astype(int)).mean() > 0.01


def test_listed_names_are_scrambled_when_large():
    from form_sequencer.scramble import find_titles
    cfg = copy.deepcopy(DEFAULTS["scramble"])
    cfg["names"] = ["Argentina", "United States", "China", "Hong Kong"]
    H = 1650
    body = 14.0

    def line(text, y, size=body):
        return {"text": text, "bbox": [100, y, 900, int(size)], "size": size}

    lines = [line("filler text of normal size " + str(i), 900 + 30 * i) for i in range(10)] + [
        line("Application Form for Employment with the Government of the Hong Kong Special "
             "Administrative Region of the People's Republic of China", 700, 24),   # large, long
        line("U.S. Individual Income Tax Return", 1200, 26),                      # large
        line("中华人民共和国出入境管理局", 1300, 28),                                   # large, Chinese
        line("REPÚBLICA ARGENTINA", 1400, 14),                                     # capitals
        line("Country of birth: United States / China / other", 1000, 14),       # small body text
        line("Please describe the usage of the bus.", 1050, 24),                  # 'usa'/'u.s.' inside words
    ]
    hits = {tuple(b) for b in find_titles(lines, H, cfg)}
    assert tuple(lines[10]["bbox"]) in hits and tuple(lines[11]["bbox"]) in hits
    assert tuple(lines[12]["bbox"]) in hits and tuple(lines[13]["bbox"]) in hits
    assert tuple(lines[14]["bbox"]) not in hits, "small mentions in body text stay readable"
    assert tuple(lines[15]["bbox"]) not in hits, "names only match as whole words"


def test_vertical_writing_is_revealed_bottom_to_top():
    from form_sequencer.stamps import writing_order
    # two horizontal lines of 'writing' (blocks of ink), then the same turned 90 degrees CCW
    a = np.zeros((60, 300), np.float32)
    a[10:25, 20:280] = 1
    a[35:50, 20:200] = 1
    order_h, n_h = writing_order(a, 2.0)
    assert n_h == 2
    assert order_h[15, 30] < order_h[15, 270] < order_h[40, 30], "left to right, then the next line"
    v = np.rot90(a, k=1).copy()  # reads bottom to top; the first line is now the left column
    order_v, n_v = writing_order(v, 2.0)
    assert n_v == 2
    h = v.shape[0]
    first_col, second_col = 17, 42
    assert order_v[h - 30, first_col] < order_v[h - 270, first_col], "bottom to top"
    assert order_v[h - 270, first_col] < order_v[h - 30, second_col], "then the next column"
    order_nv, _ = writing_order(v, None)
    assert not np.allclose(order_nv, order_v), "the ratio setting switches it on"


def test_stamps_and_writing_are_never_enlarged_beyond_their_resolution(fixture_project):
    from form_sequencer import stamps
    from form_sequencer.templates import build_templates, page_size
    settings = fixture_project.load_settings()
    settings["stamps"]["size"] = 0.6      # ask for sizes far beyond what the images can give
    settings["writing"]["size"] = 0.8
    lib_s, lib_w = stamps.load(fixture_project), stamps.load(fixture_project, "writing")
    tl = build_timeline(load_beats(fixture_project.beats_path), load_manifest(fixture_project), settings,
                        lib_s, lib_w)
    v = settings["video"]
    tm = {t.name: t for t in build_templates(settings)}
    ph = {n: page_size(t.aspect, v["width"], v["height"], settings["page"]["height_fraction"])[1]
          for n, t in tm.items()}
    src = {it["id"]: max(it["size"]) for it in lib_s + lib_w}
    assert tl.stamps or tl.writings
    for item, key, cfg in [(s, "stamp", settings["stamps"]) for s in tl.stamps] + \
                          [(w, "writing", settings["writing"]) for w in tl.writings]:
        drawn = item["size"] * ph[item["layout"]]
        assert drawn <= cfg["max_upscale"] * src[item[key]] + 1, f"{item[key]} enlarged to {drawn:.0f}px"


def test_chroma_key_background_stays_flat(fixture_project):
    tl, settings = _timeline(fixture_project)
    settings["video"].update(background=[0, 177, 64], vignette=0, desk_shadows=False)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    green = np.array([0, 177, 64])
    for t in (tl.forms[0]["t"] - 0.5, tl.forms[0]["t"] + 0.3, tl.duration - 1):
        f = r.frame(t).astype(int)
        # every pixel is either clean key green, or paper/ink (never a darkened green from a shadow)
        is_green = np.all(np.abs(f - green) <= 2, axis=2)
        greenish = (f[..., 1] > f[..., 0] + 60) & (f[..., 1] > f[..., 2] + 60)
        assert is_green.mean() > 0.05, "the background is visible"
        # off-key green is only allowed right at paper edges (anti-aliasing) or on the paper
        # itself (green ink); a shadow or vignette would put it out in the open background
        import cv2
        near_paper = cv2.dilate((~greenish).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        assert (greenish & ~is_green & ~near_paper).mean() < 0.0005, "no shadowed or vignetted green"


def test_non_latin_file_names_get_distinct_pages(tmp_path):
    from PIL import Image
    from form_sequencer.config import Project, safe_key
    from form_sequencer.documents import rasterize, load_page
    names = ["(양식) 출생신고서.png", "[여권] 여권발급신청서.png", "申請書表格式.png", "こわれもの.png", "Form B-91.png"]
    assert len({safe_key(n) for n in names}) == len(names)
    assert safe_key("Form B-91.png") == "Form_B_91_png", "Latin names keep their readable key"
    p = Project(tmp_path)
    p.init()
    docs = tmp_path / "input" / "documents"
    for i, n in enumerate(names):  # each page a different grey, so mix-ups are visible
        Image.new("RGB", (850, 1100), (40 * i, 40 * i, 40 * i)).save(docs / n)
    settings = p.load_settings()
    refs = rasterize(p, settings)
    assert len({r.key for r in refs}) == len(names), "every file gets its own page cache"
    by_source = {Path(r.source).name: load_page(p, r.asdict())[5, 5, 0] for r in refs}
    for i, n in enumerate(names):
        assert by_source[n] == 40 * i, f"{n} shows its own page"


def test_keywords_match_without_accents_or_case():
    from form_sequencer.scramble import find_titles
    cfg = copy.deepcopy(DEFAULTS["scramble"])
    lines = [{"text": t, "bbox": [100, 50 + 30 * i, 600, 20], "size": 20.0} for i, t in enumerate([
        "ΥΠΟΥΡΓΕΙΟ ΟΙΚΟΝΟΜΙΚΩΝ",           # Greek capitals have no accents
        "Ανεξάρτητη Αρχή Δημοσίων Εσόδων",
        "الجمهورية اللبنانية - وزارة المالية",
        "JAMHURI YA MUUNGANO WA TANZANIA",
    ])]
    hits = {tuple(b) for b in find_titles(lines, 1650, cfg)}
    assert len(hits) == 4


def test_unreadable_letterhead_lines_are_scrambled():
    from form_sequencer.scramble import find_unreadable_letterhead
    cfg = copy.deepcopy(DEFAULTS["scramble"])
    lines = [
        {"text": "yig igi -igicis", "bbox": [400, 60, 500, 30], "size": 30.0, "source": "ocr", "conf": 0.52},
        {"text": "Application form", "bbox": [400, 120, 400, 30], "size": 30.0, "source": "ocr", "conf": 0.97},
        {"text": "s - 2hie", "bbox": [100, 900, 300, 20], "size": 20.0, "source": "ocr", "conf": 0.5},
        {"text": "ministry", "bbox": [100, 60, 200, 30], "size": 30.0, "source": "text layer"},
    ]
    hits = find_unreadable_letterhead(lines, 1650, cfg)
    assert hits == [[400, 60, 500, 30]], "only unreadable OCR lines in the letterhead zone"
    cfg["unreadable_letterhead"] = False
    assert find_unreadable_letterhead(lines, 1650, cfg) == []


def test_kicks_jostle_and_camera_follows_stamps(fixture_project):
    tl, settings = _timeline(fixture_project)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    assert tl.pulses, "kicks become jostle pulses"
    t0 = tl.pulses[len(tl.pulses) // 2][0]
    moved = r.jostle(t0 + 0.07)
    assert abs(moved[0]) + abs(moved[1]) > 0 and moved[3] > 0, "a kick pushes the sheet and bumps the zoom"
    assert r.jostle(t0 + 0.07) == r.jostle(t0 + 0.07), "deterministic"
    if r.focus:
        t, fx, fy = r.focus[0]
        z, vx, vy = r.camera(t + 1.5)
        assert z > 1.02, "the view zooms in on a new stamp/writing"
        before = math.hypot(fx - r.W / 2, fy - r.H / 2)
        assert math.hypot(fx - vx, fy - vy) <= before + 1e-6, "and moves toward it"
    settings["motion"]["enabled"] = False
    still = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    assert still.camera(5.0) == (1.0, still.W / 2, still.H / 2)


def test_alpha_layers_are_transparent_off_the_paper(fixture_project):
    tl, settings = _timeline(fixture_project)
    settings["video"]["alpha"] = True
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    t = tl.forms[1]["t"] + 3.0 if len(tl.forms) > 1 else 20.0
    rgba = r.frame_rgba(t)
    assert rgba.shape == (r.H, r.W, 4)
    a = rgba[..., 3]
    assert (a == 0).mean() > 0.02, "the desk is transparent"
    assert a[r.H // 2, r.W // 2] == 255, "paper is opaque"
    assert ((a > 0) & (a < 255)).mean() > 0.005, "shadows are see-through"
    # straight RGBA over the background colour gives the composited viewing frame
    rgb = rgba[..., :3].astype(np.float32)
    k = a[..., None].astype(np.float32) / 255
    over = rgb * k + r._backdrop * (1 - k)
    assert np.abs(over - r.frame(t).astype(np.float32)).mean() < 1.5


def test_camera_export(fixture_project, tmp_path):
    from form_sequencer.pipeline import export_camera
    tl, settings = _timeline(fixture_project)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    base = export_camera(r, [i / 10 for i in range(50)], 10, tmp_path / "final.mp4")
    rows = base.with_suffix(".csv").read_text().splitlines()
    assert len(rows) == 51 and rows[0].startswith("frame,time,zoom")
    assert "setValuesAtTimes" in base.with_suffix(".jsx").read_text()



def test_background_scenes(fixture_project):
    import json
    from form_sequencer.background import BackgroundRenderer, plan_sections
    tl, settings = _timeline(fixture_project)
    beats = json.loads(fixture_project.beats_path.read_text(encoding="utf-8"))
    cfg = settings["background"]
    cfg["min_section_bars"] = 2
    secs = plan_sections(beats, fixture_project.audio_file(settings), cfg)
    assert secs[0]["t0"] == 0 and abs(secs[-1]["t1"] - beats["duration"]) < 1e-3
    assert all(a["t1"] == b["t0"] for a, b in zip(secs, secs[1:])), "sections tile the track"
    assert {s["scene"] for s in secs} <= {"light", "archive", "sea"}
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    sheets = [r._frozen(i) for i in range(len(r.sheets) - 1)]
    bg = BackgroundRenderer(fixture_project, settings, beats, secs, camera=r.camera, sheets=sheets, scale=0.25)
    for name in ("light", "archive", "sea"):
        img = getattr(bg, f"scene_{name}")(7.0)
        assert img.shape == (bg.lh, bg.lw, 3) and 0.01 < img.mean() < 0.5, f"{name} is a dim, non-empty scene"
    for t in (1.0, secs[-1]["t0"] if len(secs) > 1 else 5.0, 20.0):
        w = bg.weights(t)
        assert abs(sum(w.values()) - 1) < 1e-6
        f = bg.frame(t)
        assert f.shape == (bg.H, bg.W, 3) and np.array_equal(f, bg.frame(t)), "deterministic"


def test_sheet_shadow_does_not_wrap_around(fixture_project):
    tl, settings = _timeline(fixture_project)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.5)
    frame = np.full((r.H, r.W, 3), 200, np.uint8)
    h, w = int(r.H * 0.6), int(r.W * 0.5)
    top = int(r.H * 0.6)  # the sheet runs off the bottom of the frame
    placed = r._placement(frame, h, w, 0.0, r.W / 2, top + h / 2)
    y0 = placed[0][1]
    # the top rows of the shadow area lie well above the sheet: only the faint tail of its blur
    # reaches them (a wrapped-around bottom edge would darken them by up to half)
    assert frame[y0:y0 + 3].min() >= 195, "no shadow band above a sheet clipped at the bottom"


def test_intro_and_outro(fixture_project):
    tl, settings = _timeline(fixture_project)
    settings["video"].update(alpha=True, intro=4.0, outro=6.0)
    r = FrameRenderer(fixture_project, settings, load_manifest(fixture_project), tl, scale=0.25)
    rgb, trans = r.layers(-2.0)
    assert trans.min() == 1.0, "intro: nothing on the desk before the music"
    _, trans = r.layers(0.3)
    assert (trans < 0.01).mean() > 0.05, "the first sheet slides in with the music"
    _, trans = r.layers(tl.duration + 0.1)
    assert (trans < 0.01).mean() > 0.2, "the pile is still there as the music ends"
    _, trans = r.layers(tl.duration + 5.0)
    assert (trans > 0.99).mean() > 0.99, "outro: every sheet has been cleared off the desk"


def test_audio_starts_after_the_intro(fixture_project, tmp_path):
    import subprocess
    from form_sequencer.sinks import FFmpegSink, ffmpeg_exe
    settings = fixture_project.load_settings()
    out = tmp_path / "lead.mp4"
    sink = FFmpegSink(out, fixture_project.audio_file(settings), start=0.0, duration=3.0, preset="ultrafast", lead=1.5)
    sink.open(64, 36, 10)
    for _ in range(30):
        sink.write(np.zeros((36, 64, 3), np.uint8))
    sink.close()
    raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(out), "-f", "s16le", "-ac", "1", "-ar", "8000", "-"],
                         capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, np.int16).astype(float)
    assert abs(len(pcm) / 8000 - 3.0) < 0.1, "audio runs exactly as long as the video"
    assert np.abs(pcm[:int(1.3 * 8000)]).max() < 50, "silence during the intro"
    assert np.abs(pcm[int(1.6 * 8000):]).max() > 500, "then the music"
