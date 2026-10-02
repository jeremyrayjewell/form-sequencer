# Form Sequencer

An audio-driven generative music video. It assembles one strange but legible
administrative form from fragments of real documents and keeps revising it in time
with the music.

```
audio ──► beats.json ─┐
                      ├─► sequencer ──► timeline.json ──► renderer ──► sink (ffmpeg mp4 / PNG / …)
docs  ──► manifest ───┘        ▲                              ▲
                          settings.yaml                  settings.yaml
```

## Install

```
pip install -e .            # Python 3.10+; ffmpeg on PATH (or the bundled imageio-ffmpeg is used)
pip install -e .[dev]       # + pytest
```

This installs a `form-seq` command. If your Scripts folder isn't on PATH,
`python -m form_sequencer …` does the same thing.

## Commands

| Command | What it does |
|---|---|
| `form-seq init PROJECT` | Creates `PROJECT/settings.yaml` and the `input/` folders |
| `form-seq fixture PROJECT` | Same as `init`, then fills the project with a synthetic track and forms (no copyrighted material) |
| **`form-seq analyze PROJECT`** | Detects beats, extracts fragments, and builds the timeline |
| `form-seq beats PROJECT` | Writes a click track, beat plot and Audacity labels to check or correct timing |
| **`form-seq preview PROJECT`** | Renders a reduced-resolution video with audio to `out/preview.mp4`, plus `out/preview_stills.jpg` |
| **`form-seq render PROJECT`** | Renders the final full-resolution video with the original audio to `out/final.mp4` |
| `form-seq still PROJECT -t 12.5` | Renders one full-resolution frame |

`preview` and `render` accept `--start S --duration D` to render one part of the
track, and `--frames DIR` to also write PNG frames. Both commands rebuild the
timeline from the current settings and manifest, so after editing either you only
need to re-run `preview`. Audio or documents that changed since the last analysis
are re-analyzed automatically.

## Project layout

```
PROJECT/
  settings.yaml                 all tunable settings, with comments
  input/audio/                  one track (wav, mp3, flac, ogg, m4a, aiff)
  input/documents/              scans (png, jpg, tif, bmp) and PDFs, including multipage
  input/stamps/                 photos/scans of rubber-stamp imprints
  input/writing/                photos/scans of handwriting
  input/beats.txt               optional: manual beat times; replaces detection when present
  build/beats.json              detected beats, strengths, local tempo, downbeat, novelty, onsets
  exclusions.yaml               your corrections to photo exclusion (kept across re-analysis)
  build/manifest.yaml           fragments: source, page, bbox, kind, enabled, locked (editable);
                                pages with their forbidden (photo) areas
  build/exclusions/             review_NN.jpg: every page with its excluded areas hatched
  build/exclusions.yaml         what photo detection found (a record, not for editing)
  build/pages/                  rasterized page cache (bbox coordinates refer to these images)
  build/fragments/              overlay_<page>.jpg, contact_sheet.jpg, crops/
  build/beatcheck/              click.wav, beats_plot.png, beats_labels.txt
  build/timeline.json           every placement and transition (renderer input)
  out/                          preview.mp4, final.mp4 (+ final.mov, camera files), stills
```

`build/` and `out/` can live elsewhere (e.g. a faster drive) via `paths:` in settings.
Your own material (scans, stamps, handwriting, audio) stays in your project folder, which is not
part of this repository: `projects/` is ignored by git. `form-seq fixture` builds a synthetic
project to try everything without any real material.

## How it works

**Audio** ([audio.py](src/form_sequencer/audio.py))
- Tempo is estimated within `audio.tempo_range`, with no 120 BPM assumption.
- Beats are tracked with librosa on a full-band plus low-band (kick/bass) onset
  envelope. The low band stops the tracker from locking onto off-beat hi-hats.
- Leading and trailing beats that fall in silence are trimmed.
- Each beat is snapped to the actual attack on a ~3 ms envelope. The coarse
  envelope peaks 20–30 ms late.
- Everything downstream uses the list of beat times, so tempo drift and a late
  first beat just work.
- Also computed:
  - per-beat strength
  - the downbeat phase (the bar position with the most kick/bass and change)
  - MFCC/RMS novelty, used to find section changes
  - onsets

**Photo exclusion** ([photos.py](src/form_sequencer/photos.py))
- Runs on every rendered page before any fragment is extracted.
- Lighting is flattened first, so aged paper and uneven scans read as paper.
- A morphological opening then removes everything made of thin strokes: text,
  handwriting, rules, line diagrams, charts, stamps and signatures.
- Whatever survives is checked in small tiles for continuous tone: soft edges,
  varied tones and little bare paper.
- Tiles are merged into regions and grown to the whole photo:
  - **photo:** a face is found, or the tones are clearly photographic
  - **uncertain:** otherwise; also forbidden, since doubt means exclude
- A page that is mostly continuous tone or mostly forbidden is rejected whole.
- No fragment may touch a forbidden area. Manual or locked manifest entries that
  do are kept but marked `excluded`.
- Flat printed emblems, hatching, QR codes, charts and bold type stay eligible.
- Gradient-shaded emblems usually come out *uncertain* (excluded); allow them by id.

**Email exclusion** ([emails.py](src/form_sequencer/emails.py), `emails:` in settings)
- Every page is searched for email addresses. PDF text layers give exact word
  boxes instantly.
- Raster content is read with local OCR (RapidOCR, nothing leaves the machine):
  image files, scanned PDF pages, and images embedded in vector PDFs such as
  letterheads.
- Matching is generous. It catches:
  - normal addresses
  - "@" or its OCR misreading "©" next to a domain
  - "@" between letters (truncated addresses)
  - common mail-provider names
- The whole text line of a hit is forbidden, in the same way as photos. Hits appear
  in blue on the review sheets and can be allowed by id (`e1`, `e2`…) in
  `exclusions.yaml`.
- OCR results are cached per page image in `build/ocr/`. The first run over many
  scanned pages is slow (about 10 s per page per core); later runs are instant.

**QR codes** ([qr.py](src/form_sequencer/qr.py), `qr:` in settings)
- Detected with OpenCV's Aruco-based detector, which finds codes even when they
  can't be decoded.
- As a fallback, three nested-square finder marks at a right angle also count.
  Candidates must be densely inked and small, which rules out the nested boxes of
  form grids.
- Each code is forbidden (ids `q1`, `q2`…).

**Scrambled logos and institutional titles** ([scramble.py](src/form_sequencer/scramble.py), `scramble:` in settings)
- Not forbidden: fragments containing these areas are still used, but the pixels
  are scrambled before the crop is taken.
- **Titles:** short text lines containing an institutional keyword (ministry,
  department, bank, university, prefeitura, وزارة, 政府, सरकार, Wizara…; the list
  is editable via `keywords`).
  - In the letterhead zone any such line counts.
  - Elsewhere only heading-like lines count, so body text mentioning an office
    stays readable.
- **Logos:** marks in the letterhead zone (the top `letterhead_zone` of the page)
  that are coloured, oversized wordmark letters, or detailed grey crests.
  - A logo grows sideways into its lockup, so the institution's name set beside
    the crest is scrambled with it, in any script.
  - Hollow coloured rings (stamps and seals) and anything outside the letterhead
    zone are left alone.
- Scrambled areas appear in purple on the review sheets. `keep: [s2]` in
  `exclusions.yaml` leaves one readable; `scramble: [[x, y, w, h]]` adds one.

**Rubber stamps on kicks** ([stamps.py](src/form_sequencer/stamps.py), `stamps:` in settings)
- `stamps.source` is a folder of photos or scans of stamp imprints. Each is
  isolated automatically:
  - paper shading and creases are flattened out
  - ink is kept only close to the stamp's own ink hue, which rejects grey shading,
    black paper edges and faint security patterns
  - the stamp's frame is kept with every ink shape inside it, and outside specks
    are dropped
  - the ink is un-mixed from the paper, so the rubber texture stays
- Results go to `build/stamps/*.png` (transparent) with `review.jpg` (on a
  checkerboard and on paper).
- Audio analysis finds *kicks* (onsets of the low band). Strong kicks slam a stamp
  onto the top sheet:
  - it drops from 150% and inks in over `slam_beats`, jolting the sheet
  - it stays on that sheet, and in the pile once the sheet is covered
- Controls: `trigger_percentile`, `min_beats` spacing and `max_per_sheet`.
- Which stamp, where, the angle, size and ink strength come from the seed.

**Handwriting on high hits** (`writing:` in settings, [pile.py](src/form_sequencer/pile.py))
- `writing.source` is a folder of photos or scans of handwriting. Each is isolated
  like a stamp, but always as one piece, and the number of written lines is counted.
  Results go to `build/writing/` with `review.jpg`.
- Audio analysis finds *highs* (onsets above about 3 kHz: hats, cymbals, high notes).
  Strong ones write a piece of handwriting. It appears as if written, line by line
  and left to right with a soft pen front, over `write_beats` per line.
- A writing more than `vertical_ratio` times taller than wide is taken to be written
  at 90°. It is revealed bottom to top, one column after another from the left.
  Line gaps are found as low-ink valleys, so slanted handwriting still counts its
  lines correctly.
- Placement:
  - It goes on the top sheet, or, with probability `lower_sheets`, on a sheet lower
    in the pile.
  - For a lower sheet, the sequencer uses the same geometry as the renderer to work
    out what is visible at that moment. It then picks a spot where at least
    `min_visible` of the writing shows, shrinking it down to `min_size` if needed.
  - The writing is drawn onto that sheet, so sheets above cover the rest of it.
- `pile_offset` and `pile_angle` in `form_changes` set how much of the lower sheets
  shows.

**Jostle and camera** (`motion:` in settings)
- Kicks above `pulse_percentile` jostle the pile. The top sheet gets a quick push and
  twist with a small rebound (`jostle_px`, `jostle_deg`), each hit in its own direction.
  The sheets below follow a moment later, `pile_jostle` as far. The whole picture also
  gets a tiny zoom bump (`pulse_zoom`).
- A slow camera eases toward each new stamp or writing: it zooms in by up to
  `camera_zoom` and travels `camera_pan` of the way toward it, over `camera_ease`
  seconds. It lingers for `camera_hold` seconds, then drifts back to the whole pile.
- Both are functions of time only, so any frame renders the same on its own.
  `enabled: false` turns all of this off.

**Background in post** (`video.alpha`)
- With `alpha: true`, `render` also writes `final.mov`: ProRes 4444 with a transparent
  background, see-through shadows and the audio. Premiere, Resolve, After Effects and
  Final Cut read it with transparency. It is large: about 50 MB per second at 1080p.
- `final.mp4` is the same render composited over `video.background`, for viewing.
- `final_camera.csv` holds the camera move of every frame. A background layer that fills
  the frame and is anchored at its centre moves with the forms when given
  `layer_scale_percent` and `layer_position_x/y`. `final_camera.jsx` does this in After
  Effects: select the background layer and run it via File > Scripts > Run Script File.
  Set `PARALLAX` in the script below 1 to move the background less, which adds depth.

**Intro and outro** (`video.intro`, `video.outro`, in seconds)
- The video starts `intro` seconds before the music: the desk is empty (only the background
  shows), and the first sheet slides in as the music starts.
- After the music ends, `outro` seconds follow: the sheets slide off the desk one by one,
  top first, then the background fades to black.
- The audio is delayed by `intro` and padded with silence, so it stays in sync. The `.mov`,
  camera files, background and composite all share this timeline.

**Background scene** (`form-seq background`, `background:` in settings, [background.py](src/form_sequencer/background.py))
- Renders `out/background.mp4` to lay under `final.mov`. With `--composite` it also
  writes `out/composite.mp4`, which is `final.mov` over it, to watch the result.
- The track is split into sections where the music changes most (novelty peaks on
  downbeats, at least `min_section_bars` apart). Each section is ranked by loudness and
  kick density, and gets a scene:
  - quietest third: **light**, soft drifting fields of office-coloured light;
  - middle: **archive**, a dim, out-of-focus aisle of shelved boxes under fluorescent tubes,
    slowly dollying forward;
  - loudest third: **sea**, a deep field of out-of-focus forms (the video's own sheets) drifting
    far below the pile.
- Scenes crossfade over `crossfade_beats`. Kicks lift the light (`kick_pulse`); high hits
  flicker individual tubes and flare the grain (`high_flicker`).
- The forms' camera move is already applied, at `parallax` strength, so the background
  moves less than the pile and reads as further away. No keyframing is needed.
- The plan is saved to `build/background_plan.json`. Scenes are drawn at `scale` of the
  video size, since they are out of focus, then upscaled, and grain is added.

**Fragments** ([fragments.py](src/form_sequencer/fragments.py))
- Pages are rasterized to a common DPI; scans are normalized to a common physical size.
- OpenCV finds ink, ruling lines and coloured ink.
- A recursive XY-cut splits each page at its widest whitespace first. Every node,
  and every run of 2–4 adjacent siblings, becomes a candidate region.
- Coloured ink (stamps, signatures) is segmented separately so it can't glue
  blocks together.
- Candidates are classified as header, table, fields, checkboxes, stamp,
  signature, diagram (mostly non-text ink: drawings, charts, figures) or text.
- Crop edges are pulled back so they never slice through neighbouring text.

**Templates** ([templates.py](src/form_sequencer/templates.py), `templates:` in settings)
- Seven forms in five families. Each has its own:
  - page proportion (portrait 0.77, square 1.0, landscape 1.29, A-landscape 1.41, wide 1.6)
  - margins and column/row grid
  - header: top band, bottom title block, or left strip with vertical title
  - rule style: single, double, heavy, dotted, or boxed with shaded bands
  - figure areas (cells preferring `diagram`, captioned FIG. n)
- Cells never overlap; this is checked when templates are built.
- Sibling templates (e.g. `declaration` and `declaration-revised`) share geometry
  and part of their cells, so changing between them replaces a section rather
  than the whole page.

**Sequencer** ([sequencer.py](src/form_sequencer/sequencer.py))
- Nothing happens before the first downbeat. Then the cells fill one per beat.
- **Form changes (major)** come from the audio analysis:
  - candidates are downbeats with high novelty (section changes) and major onsets
    (beats above `major_onset_percentile`)
  - they are accepted strongest first, at least `min_bars` apart; a gap longer than
    `max_bars` gets its strongest downbeat
  - each change has a *drama* (`drama`, optionally scaled by how big the musical
    change ranks). Drama sets how different the next template is, how many fragments
    are carried, and how long the change lasts (`beats_min`–`beats_max`)
  - templates seen recently or often are avoided, so the document keeps evolving
  - `style: stack` (default): a fresh sheet carrying the new form slides in from a
    desk edge or drops from above, lands askew, and its cells fill on beat subdivisions
    (`arrival_subdivision`). Covered sheets stay on the pile exactly as they were,
    each at its own rotation and offset (`pile_depth`, `pile_angle`, `pile_offset`,
    scaled by drama)
  - `style: morph`: fragments not carried fade and unpaired cells erase → paired cells
    glide and resize into the new form while the page reshapes (carried fragments
    ride along) → new cells draw in → new fragments arrive on beat subdivisions
- **Revisions (minor)** fall on the detected beats:
  - a token budget (`events_per_bar`) limits how often cells change
  - changes happen only on downbeats or strong beats
  - each cell holds its fragment for at least `min_hold_beats`
  - swaps exchange two cells
  - strong off-beat onsets trigger a "stamp" substitution
- Fragments are chosen by:
  - aspect fit
  - slot kind preference
  - legibility (no heavy shrinking or blow-ups)
  - a reuse cooldown
  - variety: the same kind is penalized, and crops of the same source area are
    never shown twice at once
- All randomness comes from `seed`.

**Renderer** ([render.py](src/form_sequencer/render.py))
- `frame(t)` is a pure function of its inputs; the tests check this.
- Each crop's own paper tone is normalized away and the crop is multiplied onto
  one sheet of paper, so pieces from different sources read as ink on the same
  document.
- The form is drawn from template primitives:
  - page frame, header rule and title
  - ruled cells with section numbers
  - a footer with a `REV.` counter
  - red `AMENDED Rnnn` tags on changed cells
- During a form change:
  - the frame, header and footer slide rather than disappear
  - paired cells morph
  - rules that change orientation fade instead of rotating

  Tests check that the page frame and a ruled grid are on the page at every moment
  of a change.
- Transitions:
  - `print`: top-down reveal
  - `stamp`: drop-in
  - `cut`
  - `move`: an opaque paper slip with a shadow, so moving pieces never smear
    ink over other cells

**Sinks** ([sinks.py](src/form_sequencer/sinks.py))
- ffmpeg (H.264 + AAC, muxed with the original audio at the same start offset),
  PNG frames, and a tee.

## Correcting beat alignment

Run `form-seq beats PROJECT`. Listen to `build/beatcheck/click.wav` (a high click
marks the downbeat) and look at `beats_plot.png`.

- **All beats early or late by the same amount:** set `audio.beat_offset` (seconds).
- **Bars start on the wrong beat:** set `audio.downbeat_offset` (0 to beats_per_bar−1).
- **Half or double tempo:** set `audio.tempo_hint`.
- **Anything else:**
  1. In Audacity, import the track and `beats_labels.txt` (File → Import → Labels).
  2. Move, add or delete labels.
  3. Export them (File → Export → Export Labels) to `input/beats.txt`.
  4. Re-run `analyze`.

  A plain text file with one time in seconds per line also works.

## Correcting photo exclusion

Open `build/exclusions/review_*.jpg`:
- red hatching marks a photo
- orange marks an uncertain region
- blue marks an email address
- teal marks a QR code
- a purple outline marks an area to scramble
- a red cross marks a rejected page

Each area is labelled with its id (`p1`, `p2`…). Corrections go in
`exclusions.yaml` in the project folder:

```yaml
AppFSample_pdf_p1:
  add: [[80, 90, 240, 320]]     # extra forbidden rectangle (pixels of build/pages/<page>.png)
PDA_Dental_Chart_pdf_p1:
  allow: [p1]                   # this region is a printed seal, not a photo
Some_scan_pdf_p3:
  reject_page: true             # never use this page
```

The next `analyze` or `preview` re-applies them, and the corrections survive
re-analysis.

## Adjusting fragments

- Open `build/fragments/overlay_*.jpg` to see every detected box and its id.
- In `build/manifest.yaml` you can:
  - set `enabled: false` to drop a fragment; this is remembered when analysis is re-run
  - fix a `kind`
  - edit a `bbox` and set `locked: true` to keep it
  - add your own entry with `origin: manual` and a unique id
- Changes take effect on the next `preview`; crops are cut at render time.

## Tests

`pytest` runs the whole pipeline on the synthetic fixture (about 30 s). The tests check:
- beat accuracy against ground truth: at least 95 % of beats within 30 ms, including
  a late first beat and tempo drift
- downbeat and section detection
- manual beat override
- fragment kinds, bounds and manifest merge
- photo exclusion: the fixture's synthetic photo is forbidden in full, no fragment
  touches it, a flat printed emblem stays eligible, and manual overrides work
- templates: the overlap validator, variety of proportions, headers and rules,
  and one-to-one cell matching
- form changes: they land on detected beats at section changes, keep the minimum
  spacing, and respond to the frequency and drama controls; the page frame and a
  ruled grid stay visible throughout a change
- aspect-preserving fit
- deterministic timelines and frames
- that settled fragments never overlap
- that the preview has video and audio of the right length

## Extending

The timeline is plain JSON with normalized page coordinates, and crops are
addressable from the manifest. A TouchDesigner patch can read `timeline.json`
and `build/fragments/crops/` directly. Alternatively, a live mode can drive
`FrameRenderer.frame(t)` from an audio clock and write to a new `FrameSink`
(Spout, NDI, a window) without touching analysis or sequencing.
