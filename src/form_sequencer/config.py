"""Project layout and settings.

The commented YAML template below is both the file written by ``form-seq init``
and the source of default values, so the two can never drift apart.
"""

from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SETTINGS_TEMPLATE = """\
# Form Sequencer settings. Paths are relative to this project folder.
# Everything here is safe to edit; re-run `form-seq preview` to see the effect.

seed: 7                        # change for a different (but reproducible) arrangement

paths:
  build: null                  # caches (default: build/ in this folder); may be another drive, e.g. C:/fs-work/build
  out: null                    # rendered videos (default: out/ in this folder)

inputs:
  audio: null                  # null = first audio file found in input/audio/
  documents: input/documents   # folder of scans (png/jpg/tif/bmp) and PDFs
  beats_file: input/beats.txt  # if this file exists it REPLACES detected beats
                               # (Audacity label export, or one time in seconds per line)

audio:
  tempo_hint: null             # BPM prior if detection picks half/double tempo, e.g. 96
  tempo_range: [60, 190]       # plausible tempo range searched during detection
  tightness: 60                # lower = beat tracker follows tempo drift more freely (librosa default 100)
  beat_offset: 0.0             # seconds added to every beat (nudge if visuals feel early/late)
  beats_per_bar: 4
  downbeat_offset: null        # which beat (0..beats_per_bar-1) is the first downbeat; null = auto
  accent_ratio: 1.6            # off-beat onsets this many times stronger than typical ones are accents

documents:
  dpi: 150                     # rasterization resolution for PDFs
  assume_page_inches: 11       # longest side of scanned images, used to normalize their scale
  min_region_inches: 0.3       # ignore regions smaller than this on either side
  max_region_fraction: 0.45    # ignore regions covering more than this fraction of a page

photos:
  enabled: true                # detect photographs and never extract fragments that touch them
  tile_inches: 0.2             # size of the analysis tiles
  texture_max: 0.15            # max edge sharpness of continuous tone (text and line art are sharper)
  entropy_min: 4.8             # min tonal richness (bits) of an excluded region
  photo_entropy: 5.4           # above this, with soft edges, a region is a certain photo
  min_inches: 0.35             # ignore continuous-tone areas smaller than this square
  margin_inches: 0.1           # safety margin added around each excluded area
  reject_page_tone: 0.45       # reject a whole page if this share of it is continuous tone
  reject_page_area: 0.5        # reject a whole page if this share of it is forbidden

emails:
  enabled: true                # find email addresses and never extract fragments that touch them
  ocr: true                    # read scans and embedded images with OCR (PDF text layers are always read);
                               # the text found is also used to find institutional titles
  workers: 0                   # parallel OCR processes (0 = number of CPUs - 1)
  margin_inches: 0.05          # safety margin around each address

qr:
  enabled: true                # find QR codes and never extract fragments that touch them
  margin_inches: 0.08

scramble:
  enabled: true                # scramble logos and institutional titles (stamps are left alone)
  letterhead_zone: 0.16        # top share of a page treated as letterhead (logos only searched here)
  max_title_words: 10          # longer lines are body text, never scrambled as titles
  unreadable_letterhead: true  # scramble letterhead lines OCR cannot read (e.g. Arabic/Greek scans)
  unreadable_confidence: 0.7   # OCR confidence below which a line counts as unreadable
  min_logo_inches: 0.3         # smallest logo, longest side
  max_logo_inches: 2.8         # largest logo, longest side
  keywords: []                 # institutional words to match in text lines; [] = built-in multilingual list
  names: [Argentina, United States, China, Hong Kong]   # large references are always scrambled
                               # (in any language/abbreviation: U.S., USA, 美国, 中华人民共和国, 香港, ...)

stamps:
  enabled: true                # slam rubber stamps onto the top sheet on kick-drum hits
  source: input/stamps         # folder of photos/scans of stamp imprints (isolated automatically)
  direction_tolerance: 0.9     # how closely a pixel must match the ink's colour signature (0..1)
  paper_margin: 3.0            # how far above the paper's own tint ink must be (higher = cleaner, fainter)
  trigger_percentile: 75       # kicks louder than this percentile of all kicks can stamp
  min_beats: 2                 # at least this many beats between two stamps
  max_per_sheet: 6             # a sheet takes at most this many stamps
  size: 0.26                   # stamp size, fraction of page height (varies +-25%)
  max_upscale: 1.25            # never drawn larger than this x the image's own resolution (stays sharp)
  min_size: 0.1                # images too small to reach this size sharply are not used
  max_angle: 25                # stamps land rotated up to this many degrees
  opacity: 0.92                # ink strength (varies a little per stamp)
  slam_beats: 0.25             # duration of the slam (drop and settle), in beats

writing:
  enabled: true                # handwriting written onto sheets on high-pitched hits (hats, cymbals, high notes)
  source: input/writing         # folder of photos/scans of handwriting (isolated like stamps)
  direction_tolerance: 0.9
  paper_margin: 3.0
  split: false                 # one image = one piece of writing
  vertical_ratio: 2.0          # taller than this x wider: written at 90 degrees, revealed bottom to top
  trigger_percentile: 80       # high-band hits louder than this percentile of all of them can write
  min_beats: 1                 # at least this many beats between two writings
  max_per_sheet: 8             # a sheet takes at most this many writings
  size: 0.3                    # longest side of a writing, fraction of page height (varies +-25%)
  min_size: 0.1                # on a lower sheet a writing shrinks to fit the visible part, down to this
  max_upscale: 1.25            # never drawn larger than this x the image's own resolution (stays sharp)
  min_visible: 0.7             # on a lower sheet at least this share of a writing must be visible
  max_angle: 8                 # writings sit rotated up to this many degrees on their sheet
  write_beats: 1.5             # time to write one line, in beats
  lower_sheets: 0.5            # share of writings that go on visible parts of sheets under the top one

video:
  width: 1920
  height: 1080
  fps: 30
  preset: medium               # x264 preset for the final render
  crf: 18
  background: [36, 35, 33]     # desk colour around the page; for chroma key use e.g. [0, 177, 64]
  vignette: 0.35               # darkening toward the frame edges (0 = flat colour, best for keying)
  desk_shadows: true           # sheets cast shadows on the background (false keeps a key colour clean)
  intro: 0.0                   # seconds before the music starts (background only; the first sheet slides in)
  outro: 0.0                   # seconds after it ends (the pile is cleared, then the background fades out)
  alpha: false                 # also write a .mov (ProRes 4444) with a transparent background and
                               # see-through shadows, for compositing over a background in post

# Background for compositing under the forms in post: `form-seq background` writes
# out/background.mp4 (and with --composite, out/composite.mp4 with final.mov on top).
# Each section of the track gets a scene by its energy; the forms' camera is baked in.
background:
  scale: 0.25                  # scenes are drawn at this fraction of the video size (they are out of focus)
  parallax: 0.5                # follows the forms' camera this much (lower = further away, 0 = still)
  min_section_bars: 8          # shortest stretch of one scene
  novelty_threshold: 0.8       # how distinct a change in the music must be to start a new section
  crossfade_beats: 4           # scene changes blend over this many beats
  scenes: {quiet: light, middle: archive, loud: sea}   # scene per energy (light, archive or sea)
  brightness: 1.3              # overall level (keep it darker than the paper so the pile stands out)
  saturation: 1.5              # colour strength (1 = as drawn, 0 = grey)
  max_level: 0.72              # brightest the background gets (paper is ~0.94): highlights roll off below it
  kick_pulse: 0.07             # light lifts this much on a full-strength kick
  high_flicker: 0.25           # fluorescent flicker and grain flare on high hits
  grain: 0.015                 # film grain strength
  drift: 1.0                   # speed of the slow movement

# Movement on top of the form's own changes: the sheets jostle on kicks, and a slow camera
# drifts and zooms toward each new stamp or piece of writing.
motion:
  enabled: true
  pulse_percentile: 40         # kicks louder than this percentile of all kicks jostle the sheets
  jostle_px: 9                 # top sheet nudge on a full-strength kick (1/1000 of page height)
  jostle_deg: 0.6              # top sheet twist on a full-strength kick
  pile_jostle: 0.5             # the sheets below move this much as far (0 = still)
  pulse_zoom: 0.01             # whole-picture zoom bump on a full-strength kick
  camera: true                 # drift toward new stamps and writing
  camera_zoom: 0.07            # how far the view zooms in on them
  camera_pan: 0.4              # how far it travels toward them (share of their offset from centre)
  camera_ease: 0.8             # seconds to settle on a new target
  camera_hold: 2.5             # seconds it lingers before drifting back to the whole pile

preview:
  scale: 0.5                   # fraction of final resolution
  fps: 24
  preset: veryfast
  crf: 26

page:
  height_fraction: 0.94        # page height as a fraction of frame height
  paper: [243, 240, 230]
  form_number: "FS-01"         # default form number for templates that do not set one

# Form templates. Each is a complete, coherent form: page proportion (aspect = width /
# height), margins (fractions of the page), a grid of columns x rows, a header band
# (top, bottom title block, or left strip), a rule style (single, double, heavy, dotted,
# boxed) and slots. Slot cell = [column, row, width, height] in grid units; slots must not
# overlap. `prefer` lists fragment kinds that suit a slot (header, table, fields, checkboxes,
# stamp, signature, text, diagram); a slot preferring `diagram` first is drawn as a figure
# area. Templates of one `family` share geometry, so changing between them redraws only
# the sections that differ. The first template opens the video.
templates:
  - name: declaration
    family: declaration
    title: "COMPOSITE DECLARATION"
    form_number: "FS-01"
    aspect: 1.294
    margins: {top: 0.05, bottom: 0.07, left: 0.04, right: 0.04}
    columns: 12
    rows: 10
    gutter: 0.012
    header: {position: top, size: 0.075}
    rules: single
    slots:
      s1: {cell: [0, 0, 12, 2], prefer: [header]}
      s2: {cell: [0, 2, 7, 4], prefer: [table, fields, text]}
      s3: {cell: [7, 2, 5, 2], prefer: [fields, checkboxes]}
      s4: {cell: [7, 4, 5, 2], prefer: [checkboxes, fields, text]}
      s5: {cell: [0, 6, 4, 4], prefer: [text, fields]}
      s6: {cell: [4, 6, 5, 4], prefer: [diagram, table, fields]}
      s7: {cell: [9, 6, 3, 2], prefer: [stamp, checkboxes]}
      s8: {cell: [9, 8, 3, 2], prefer: [signature, fields]}
  - name: declaration-revised
    family: declaration
    title: "COMPOSITE DECLARATION (REVISED)"
    form_number: "FS-01R"
    aspect: 1.294
    margins: {top: 0.05, bottom: 0.07, left: 0.04, right: 0.04}
    columns: 12
    rows: 10
    gutter: 0.012
    header: {position: top, size: 0.075}
    rules: single
    slots:
      s1: {cell: [0, 0, 12, 2], prefer: [header]}
      s2: {cell: [0, 2, 7, 4], prefer: [table, fields, text]}
      s3: {cell: [7, 2, 5, 2], prefer: [fields, checkboxes]}
      s4: {cell: [7, 4, 5, 2], prefer: [checkboxes, fields, text]}
      s5: {cell: [0, 6, 6, 2], prefer: [fields, text]}
      s6: {cell: [6, 6, 6, 2], prefer: [checkboxes, fields]}
      s7: {cell: [0, 8, 8, 2], prefer: [signature, fields, text]}
      s8: {cell: [8, 8, 4, 2], prefer: [stamp, fields]}
  - name: ledger
    family: ledger
    title: "GENERAL LEDGER OF RECORDS"
    form_number: "FL-7"
    aspect: 1.6
    margins: {top: 0.045, bottom: 0.075, left: 0.03, right: 0.03}
    columns: 16
    rows: 8
    gutter: 0.01
    header: {position: left, size: 0.05}
    rules: double
    slots:
      s1: {cell: [0, 0, 11, 2], prefer: [header, fields]}
      s2: {cell: [11, 0, 5, 2], prefer: [stamp, checkboxes]}
      s3: {cell: [0, 2, 16, 3], prefer: [table]}
      s4: {cell: [0, 5, 5, 3], prefer: [fields, text]}
      s5: {cell: [5, 5, 6, 3], prefer: [diagram, table]}
      s6: {cell: [11, 5, 5, 3], prefer: [signature, checkboxes]}
  - name: ledger-annex
    family: ledger
    title: "LEDGER: ANNEX OF FIGURES"
    form_number: "FL-7A"
    aspect: 1.6
    margins: {top: 0.045, bottom: 0.075, left: 0.03, right: 0.03}
    columns: 16
    rows: 8
    gutter: 0.01
    header: {position: left, size: 0.05}
    rules: double
    slots:
      s1: {cell: [0, 0, 11, 2], prefer: [header, fields]}
      s2: {cell: [11, 0, 5, 2], prefer: [stamp, checkboxes]}
      s3: {cell: [0, 2, 7, 6], prefer: [diagram, table]}
      s4: {cell: [7, 2, 9, 3], prefer: [table, fields]}
      s5: {cell: [7, 5, 5, 3], prefer: [fields, text]}
      s6: {cell: [12, 5, 4, 3], prefer: [signature, stamp]}
  - name: dossier
    family: dossier
    title: "PERSONAL DOSSIER"
    form_number: "PD-3"
    aspect: 0.773
    margins: {top: 0.045, bottom: 0.065, left: 0.06, right: 0.06}
    columns: 8
    rows: 12
    gutter: 0.016
    header: {position: top, size: 0.06}
    rules: heavy
    slots:
      s1: {cell: [0, 0, 8, 2], prefer: [header]}
      s2: {cell: [0, 2, 5, 4], prefer: [fields, text]}
      s3: {cell: [5, 2, 3, 4], prefer: [stamp, checkboxes]}
      s4: {cell: [0, 6, 8, 3], prefer: [diagram, table]}
      s5: {cell: [0, 9, 4, 3], prefer: [text, fields]}
      s6: {cell: [4, 9, 4, 3], prefer: [signature, fields]}
  - name: register
    family: register
    title: "REGISTER OF ENTRIES"
    form_number: "RG-12"
    aspect: 1.0
    margins: {top: 0.055, bottom: 0.08, left: 0.05, right: 0.05}
    columns: 10
    rows: 10
    gutter: 0.014
    header: {position: left, size: 0.07}
    rules: boxed
    slots:
      s1: {cell: [0, 0, 10, 2], prefer: [header, fields]}
      s2: {cell: [0, 2, 4, 5], prefer: [checkboxes, fields]}
      s3: {cell: [4, 2, 6, 5], prefer: [diagram, table]}
      s4: {cell: [0, 7, 6, 3], prefer: [table, fields]}
      s5: {cell: [6, 7, 4, 3], prefer: [stamp, signature]}
  - name: schedule
    family: schedule
    title: "SCHEDULE OF APPARATUS"
    form_number: "SA-40"
    aspect: 1.414
    margins: {top: 0.04, bottom: 0.065, left: 0.035, right: 0.035}
    columns: 12
    rows: 8
    gutter: 0.012
    header: {position: bottom, size: 0.09}
    rules: dotted
    slots:
      s1: {cell: [0, 0, 8, 5], prefer: [diagram, table]}
      s2: {cell: [8, 0, 4, 3], prefer: [text, fields]}
      s3: {cell: [8, 3, 4, 2], prefer: [checkboxes, fields]}
      s4: {cell: [0, 5, 5, 3], prefer: [table, fields]}
      s5: {cell: [5, 5, 4, 3], prefer: [fields, text]}
      s6: {cell: [9, 5, 3, 3], prefer: [stamp, signature]}

# Major changes of the form itself (style: stack piles a fresh sheet on top; morph reshapes in place), timed from the audio analysis: section changes
# (novelty at a downbeat) and major onsets. During a change the old cells erase, the page
# reshapes (frame, header and footer slide to their new places), new rules draw in, and
# only then do new fragments arrive, on beat subdivisions.
form_changes:
  style: stack                 # stack = a new sheet lands on a growing pile; morph = the form reshapes in place
  pile_depth: 6                # covered sheets that stay visible under the top one
  pile_angle: 3.0              # max rotation of a sheet on the pile, degrees (scaled by drama)
  pile_offset: 0.035           # max offset of a sheet on the pile, fraction of frame height (scaled by drama)
  min_bars: 8                  # at least this many bars between form changes (how often)
  max_bars: 32                 # if nothing happened for this long, change at the strongest downbeat (0 = never)
  section_novelty: 1.0         # novelty z-score at a downbeat that counts as a section change
  major_onset_percentile: 98   # beats louder than this percentile also count (100 = off)
  drama: 0.6                   # 0 = subtle (sibling template, fragments stay), 1 = total reorganization
  drama_follows_music: true    # bigger musical changes get more dramatic form changes
  beats_min: 2                 # duration of a change at drama 0, in beats
  beats_max: 8                 # duration of a change at drama 1, in beats
  arrival_subdivision: 2       # new fragments arrive this many per beat after a change

sequencer:
  intro_beats_per_slot: 1      # beats between cells filling at the start (0.5 = two per beat)
  events_per_bar: 1.0          # average slot changes per bar (continuity vs. activity)
  min_hold_beats: 8            # a slot keeps its fragment at least this many beats
  strength_threshold: 0.45     # beats weaker than this (0..1) never trigger changes, except downbeats
  swap_probability: 0.2        # chance an event exchanges two fragments instead of substituting
  transition_beats: 0.5        # transition length as a fraction of the local beat period
  move_beats: 1.0              # length of swaps, in beats
  offbeat_accents: true        # strong off-beat onsets may trigger a 'stamp' substitution
  accent_min_bars: 4           # at most one off-beat accent per this many bars
  reuse_cooldown_beats: 24     # avoid bringing a fragment back within this many beats
  min_fill: 0.35               # a fragment must cover at least this share of a slot to be placed there
"""

DEFAULTS: dict[str, Any] = yaml.safe_load(SETTINGS_TEMPLATE)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}


def safe_key(name: str, max_len: int = 80) -> str:
    """A filesystem-safe, unique-per-name key for a file name.

    Latin names keep a readable key ("Form B-91.pdf" -> "Form_B_91_pdf"). Names with other
    scripts would lose most of their characters ("(양식) 출생신고서.pdf" -> "pdf") and collide,
    so they get a short fingerprint of the full name appended."""
    import hashlib
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if name.isascii() and slug:
        return slug[:max_len]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return (slug[:max_len - 7] + "_" + digest) if slug else digest


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        where = f"{path}{key}"
        if key not in base:
            warnings.warn(f"settings: unknown key '{where}' (ignored by defaults, kept as-is)")
            out[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict) and key not in ("layouts", "templates"):
            out[key] = _merge(base[key], value, where + ".")
        else:
            out[key] = value
    return out


@dataclass
class Project:
    """Paths of a project folder. All generated files live under build/ and out/."""

    root: Path

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.yaml"

    @property
    def audio_dir(self) -> Path:
        return self.root / "input" / "audio"

    def _dir(self, which: str) -> Path:
        """build/ or out/, or wherever settings.yaml `paths:` puts them (e.g. another drive)."""
        where = None
        if self.settings_path.exists():
            with open(self.settings_path, encoding="utf-8") as fh:
                where = ((yaml.safe_load(fh) or {}).get("paths") or {}).get(which)
        return (self.root / where).resolve() if where else self.root / which

    @property
    def build(self) -> Path:
        return self._dir("build")

    @property
    def out(self) -> Path:
        return self._dir("out")

    def resolve(self, recorded: str) -> Path:
        """A path recorded in a manifest or index: `build/...` is relative to the build folder
        (wherever it lives), anything else to the project folder."""
        if recorded.startswith("build/"):
            return self.build / recorded[len("build/"):]
        return self.root / recorded

    def show(self, path: Path) -> str:
        """A path for log messages: relative to the project when possible."""
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return str(path)

    @property
    def beats_path(self) -> Path:
        return self.build / "beats.json"

    @property
    def manifest_path(self) -> Path:
        return self.build / "manifest.yaml"

    @property
    def timeline_path(self) -> Path:
        return self.build / "timeline.json"

    @property
    def exclusions_path(self) -> Path:
        """User-edited corrections to photo exclusion (survives re-analysis)."""
        return self.root / "exclusions.yaml"

    @property
    def pages_dir(self) -> Path:
        return self.build / "pages"

    def load_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return copy.deepcopy(DEFAULTS)
        with open(self.settings_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        return _merge(DEFAULTS, user)

    def documents_dir(self, settings: dict) -> Path:
        return self.root / settings["inputs"]["documents"]

    def beats_file(self, settings: dict) -> Path:
        return self.root / settings["inputs"]["beats_file"]

    def audio_file(self, settings: dict) -> Path:
        explicit = settings["inputs"].get("audio")
        if explicit:
            path = self.root / explicit
            if not path.exists():
                raise FileNotFoundError(f"audio file not found: {path}")
            return path
        candidates = sorted(p for p in self.audio_dir.glob("*") if p.suffix.lower() in AUDIO_EXTS)
        if not candidates:
            raise FileNotFoundError(
                f"no audio found. Put one track in {self.audio_dir} "
                f"(or set inputs.audio in {self.settings_path})"
            )
        return candidates[0]

    def init(self) -> None:
        for d in (self.audio_dir, self.root / "input" / "documents", self.build, self.out):
            d.mkdir(parents=True, exist_ok=True)
        if not self.settings_path.exists():
            self.settings_path.write_text(SETTINGS_TEMPLATE, encoding="utf-8")
