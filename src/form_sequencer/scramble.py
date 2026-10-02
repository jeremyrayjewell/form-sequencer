"""Scramble logos and institutional titles (stamps are left alone).

Unlike photos, emails and QR codes, these areas are not forbidden: fragments that contain
them are still used, but those pixels are scrambled before a crop reaches the video, so
the form keeps its shape while the institution can no longer be read.

Detection
  * titles — text lines (text layer or OCR) that contain an institutional keyword
    (ministry, department, bank, university, prefeitura, 政府, सरकार, …). In the
    letterhead zone any such line counts; elsewhere only heading-like lines (large or
    mostly upper-case), so body text mentioning an office is left readable.
  * logos — graphic marks in the letterhead zone: coloured or solid shapes and oversized
    wordmark letters. Round, hollow, coloured marks are treated as stamps and skipped,
    and nothing outside the letterhead zone is treated as a logo.

Scrambling (deterministic per page)
  * title lines: ink clusters (letters/words) are shuffled and some mirrored, so the line
    still looks like printed type but reads as nonsense;
  * logos: cut into tiles that are permuted and flipped, keeping the palette.
"""

from __future__ import annotations

import re
import unicodedata
import zlib

import cv2
import numpy as np

DEFAULT_KEYWORDS = [
    # English and romance/germanic languages
    "ministry", "minister", "ministère", "ministerio", "ministério", "ministero", "ministerie", "ministerium",
    "department of", "departamento", "département", "government", "gouvernement", "gobierno", "governo",
    "republic", "république", "república", "repubblica", "kingdom", "royaume", "reino", "federal", "national",
    "nacional", "nationale", "nazionale", "united nations", "embassy", "consulate", "ambassade", "council",
    "commission", "committee", "authority", "agency", "agence", "bureau", "board", "office of", "secretariat",
    "secretaria", "secretaría", "prefeitura", "prefecture", "préfecture", "municipality", "municipal",
    "city of", "county of", "state of", "province of", "district", "bank", "banco", "banque", "university",
    "universidad", "université", "universidade", "universität", "institute", "instituto", "institut",
    "academy", "association", "corporation", "inspectorate", "police", "polícia", "policía", "court",
    "tribunal", "administration", "administração", "insurance", "service", "centre for", "center for",
    # Swahili, Polish, Ukrainian, Russian
    "wizara", "serikali", "jamhuri", "mamlaka", "halmashauri", "urząd", "ministerstwo", "rzeczpospolita",
    "міністерство", "україни", "министерство", "федерац",
    # South and South-East Asian
    "सरकार", "मंत्रालय", "विभाग", "संस्थान", "कार्यालय", "भारत सरकार", "निदेशालय", "sansthan",
    "ក្រសួង", "ព្រះរាជាណាចក្រ", "ກະຊວງ", "ສາທາລະນະລັດ", "กระทรวง", "สำนักงาน", "ဝန်ကြီးဌာန", "அமைச்சு", "திணைக்களம்",
    # Greek (matched without accents, so capitals like ΥΠΟΥΡΓΕΙΟ match too)
    "υπουργείο", "ελληνική δημοκρατία", "κυβέρνηση", "ανεξάρτητη αρχή", "αρχή δημοσίων εσόδων", "ααδε",
    "δου", "εφορία", "δήμος", "δήμου", "περιφέρεια", "περιφέρειας", "τράπεζα", "πανεπιστήμιο", "ίδρυμα",
    "οργανισμός", "επιτροπή", "πρεσβεία", "προξενείο", "αστυνομία", "δικαστήριο", "υπηρεσία",
    # Arabic
    "وزارة", "حكومة", "جمهورية", "مملكة", "سلطنة", "دولة", "هيئة", "جامعة", "بنك", "مصرف", "إدارة", "مديرية",
    "شرطة", "محكمة", "بلدية", "مجلس",
    # East Asian
    "政府", "人民政府", "委员会", "工业和信息化局", "局", "厅", "部", "省", "県", "庁", "役所", "市役所",
    # Korean: whole institutional words (single syllables like 청/부 also occur in 신청서 etc.)
    "대한민국", "정부", "위원회", "법무부", "외교부", "행정안전부", "출입국", "외국인정책본부", "경찰청",
    "시청", "구청", "도청", "주민센터", "대사관", "영사관", "법원",
]


# Names whose large references are always scrambled (settings: scramble.names). Each name
# expands to its common spellings, abbreviations and translations; other names in the list
# are matched as written.
NAME_VARIANTS = {
    "argentina": ["argentina", "argentine", "república argentina", "republica argentina", "阿根廷"],
    "united states": ["united states", "u.s.a.", "u.s.a", "u.s.", "usa", "estados unidos", "états-unis",
                      "etats-unis", "vereinigte staaten", "美国", "美國", "アメリカ合衆国", "米国", "미국", "미합중국"],
    "china": ["china", "chine", "prc", "p.r.c.", "people's republic", "peoples republic", "中国", "中國",
              "中华", "中華", "中华人民共和国", "中華人民共和國", "중국"],
    "hong kong": ["hong kong", "hongkong", "hksar", "h.k.s.a.r.", "香港", "홍콩"],
}


def _names_re(names: list[str]) -> re.Pattern | None:
    variants: list[str] = []
    for n in names or []:
        variants += NAME_VARIANTS.get(n.strip().lower(), [n])
    if not variants:
        return None
    parts = []
    for v in sorted(set(variants), key=len, reverse=True):
        body = r"\s*".join(re.escape(w) for w in v.split())
        # Latin names must stand as words ("usa" not in "usage", "u.s." not in "bus."), and OCR
        # may drop spaces ("HONGKONG", "UNITEDSTATES"), hence the optional whitespace
        parts.append(rf"(?<![A-Za-z]){body}(?![A-Za-z])" if v.isascii() else body)
    return re.compile("|".join(parts), re.I)


def fold(text: str) -> str:
    """Case- and accent-insensitive form for matching: Υπουργείο, ΥΠΟΥΡΓΕΙΟ and υπουργειο
    all fold to the same string (likewise Arabic with or without vowel marks)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c))).casefold()


def _keyword_re(words: list[str]) -> re.Pattern:
    """Match keywords regardless of spacing (OCR often drops spaces: 'DEPARTMENTOF'), case and
    accents. Search it in fold(text)."""
    folded = sorted({fold(w) for w in words}, key=len, reverse=True)
    parts = [r"\s*".join(re.escape(w) for w in kw.split()) for kw in folded]
    return re.compile("|".join(parts), re.I)


def _heading_like(line: dict, median_size: float) -> bool:
    letters = [c for c in line["text"] if c.isalpha()]
    upper = sum(c.isupper() for c in letters) / max(1, len(letters))
    return line.get("size", 0) >= 1.3 * median_size or (upper >= 0.7 and len(letters) >= 4)


def _whole_line(box: list[int], lines: list[dict]) -> list[int]:
    """Grow a line box over its neighbours on the same baseline (PDF text layers and OCR often
    split one printed line into several pieces)."""
    x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
    grew = True
    while grew:
        grew = False
        for ln in lines:
            lx, ly, lw, lh = ln["bbox"]
            overlap = min(y1, ly + lh) - max(y0, ly)
            near = lx - x1 < 2.5 * (y1 - y0) and x0 - (lx + lw) < 2.5 * (y1 - y0)
            if overlap > 0.6 * min(y1 - y0, lh) and near and (lx < x0 or lx + lw > x1):
                x0, x1 = min(x0, lx), max(x1, lx + lw)
                y0, y1 = min(y0, ly), max(y1, ly + lh)
                grew = True
    return [x0, y0, x1 - x0, y1 - y0]


def find_unreadable_letterhead(lines: list[dict], height: int, cfg: dict) -> list[list[int]]:
    """Letterhead lines OCR could not read (low confidence: a script it does not know, such as
    Arabic or Greek on a scan). On official forms these are the institution's header, so they
    are scrambled even though no keyword can be matched."""
    if not cfg.get("unreadable_letterhead", True):
        return []
    zone = 0.75 * float(cfg["letterhead_zone"]) * height  # the header itself, not the first fields
    floor = float(cfg.get("unreadable_confidence", 0.7))
    return [_whole_line(ln["bbox"], lines) for ln in lines
            if ln.get("source") == "ocr" and ln["bbox"][1] < zone and ln.get("conf", 1.0) < floor]


def page_unreadable(lines: list[dict], cfg: dict) -> bool:
    """True when OCR could not read this page's script (most OCR lines have low confidence)."""
    confs = [ln.get("conf", 1.0) for ln in lines if ln.get("source") == "ocr"]
    return len(confs) >= 3 and float(np.median(confs)) < float(cfg.get("unreadable_confidence", 0.7))


def letterhead_side_lines(rgb: np.ndarray, dpi: int, cfg: dict) -> list[list[int]]:
    """Printed text lines in the letterhead zone that sit toward the left or right edge, found
    from the image alone. On an official form whose script OCR cannot read, these are the
    institution's block (republic, ministry, directorate); centred lines are usually the form's
    title and are left readable."""
    H, W = rgb.shape[:2]
    zone = int(0.75 * float(cfg["letterhead_zone"]) * H)  # the header itself, not the first fields below
    gray = cv2.cvtColor(rgb[:zone], cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, int(dpi / 5) | 1, 18)
    # remove ruling lines and band borders first, so text touching them is found on its own
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.6 * dpi), 1)))
    rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.4 * dpi))))
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(rules, np.ones((3, 3), np.uint8))))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.12 * dpi)), max(1, int(0.02 * dpi))))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, k), 8)
    out = []
    for x, y, w, h, _ in stats[1:]:
        if not (0.05 * dpi <= h <= 0.4 * dpi and w >= 0.25 * dpi and w <= 0.6 * W):
            continue  # text-line sized only (not rules, boxes or the whole band)
        centre = (x + w / 2) / W
        if centre < 0.36 or centre > 0.64:
            out.append([int(x), int(y), int(w), int(h)])
    return out


def find_names(lines: list[dict], height: int, cfg: dict) -> list[list[int]]:
    """Whole lines carrying a large reference to one of `cfg["names"]`."""
    names = _names_re(cfg.get("names", []))
    if names is None:
        return []
    zone = float(cfg["letterhead_zone"]) * height
    sizes = [ln.get("size", 0) for ln in lines if ln.get("size")]
    median = float(np.median(sizes)) if sizes else 0.0
    return [_whole_line(ln["bbox"], lines) for ln in lines
            if names.search(ln["text"]) and (ln["bbox"][1] < zone or _heading_like(ln, median))]


def find_titles(lines: list[dict], height: int, cfg: dict) -> list[list[int]]:
    words = cfg.get("keywords") or DEFAULT_KEYWORDS
    kw = _keyword_re(words)
    zone = float(cfg["letterhead_zone"]) * height
    sizes = [ln.get("size", 0) for ln in lines if ln.get("size")]
    median = float(np.median(sizes)) if sizes else 0.0
    out = find_names(lines, height, cfg)  # large references to listed names, however long the line
    for ln in lines:
        x, y, w, h = ln["bbox"]
        if not kw.search(fold(ln["text"])):
            continue
        if len(ln["text"].split()) > int(cfg.get("max_title_words", 10)):
            continue  # a sentence mentioning an office is body text, not a title
        if y < zone or _heading_like(ln, median):
            out.append([x, y, w, h])
    return out


def find_logos(rgb: np.ndarray, dpi: int, cfg: dict, lines: list[dict] | None = None,
               forbidden: list[list[int]] | None = None) -> list[list[int]]:
    H, W = rgb.shape[:2]
    zone = int(float(cfg["letterhead_zone"]) * H)
    top = rgb[:zone]
    gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    colour = (hsv[..., 1] > 70) & (hsv[..., 2] < 245) & (hsv[..., 2] > 40)
    ink = (gray < 170) | colour
    k = max(3, int(dpi * 0.03))
    joined = cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    lo, hi = float(cfg["min_logo_inches"]) * dpi, float(cfg["max_logo_inches"]) * dpi
    out = []
    for i in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if not (lo <= max(w, h) <= hi and min(w, h) >= lo * 0.5):
            continue
        if any(_overlap((x, y, w, h), f) > 0.3 for f in forbidden or []):
            continue  # QR codes and photos are excluded, not treated as logos
        m = lab[y:y + h, x:x + w] == i
        ink_m = ink[y:y + h, x:x + w] & m
        fill = ink_m.sum() / float(w * h)
        col = (colour[y:y + h, x:x + w] & m).sum() / max(1, ink_m.sum())
        round_ = 0.75 <= w / h <= 1.33
        if col > 0.3 and fill < 0.25 and round_:
            continue  # a hollow coloured ring: a stamp or seal, left alone
        wordmark = h >= 0.35 * dpi and fill > 0.2  # oversized letters: a logo, not body type
        if not wordmark:
            # printed words are handled as titles: skip shapes that are text
            if any(_overlap((x, y, w, h), ln["bbox"]) > 0.3 for ln in lines or []):
                continue
            parts, _, pstats, _ = cv2.connectedComponentsWithStats(ink_m.astype(np.uint8), 8)
            glyphs = sum(1 for st in pstats[1:] if dpi * 0.03 <= st[3] <= dpi * 0.3 and st[4] > 6)
            if glyphs > 3:
                continue
        detailed = False
        if col <= 0.25 and fill > 0.35 and not wordmark:
            edges = cv2.Canny(gray[y:y + h, x:x + w], 60, 160)
            detailed = edges.mean() / 255 > 0.12  # a crest has fine inner detail; a grey bar does not
        if col > 0.25 or wordmark or detailed:
            out.append([x, y, w, h])
    return [_lockup(b, stats, dpi, W) for b in out]


def _lockup(logo, stats, dpi: int, page_w: int) -> list[int]:
    """Grow a logo to its letterhead lockup: marks on the same band chained within ~0.3 in
    (the institution's name set beside its crest), capped at 45% of the page width."""
    x0, y0, x1, y1 = logo[0], logo[1], logo[0] + logo[2], logo[1] + logo[3]
    ly0, ly1 = y0, y1  # grow sideways only: the lockup stays level with the logo
    gap, band = 0.3 * dpi, 0.1 * dpi
    grew = True
    while grew:
        grew = False
        for sx, sy, sw, sh, _ in stats[1:]:
            sx1, sy1 = sx + sw, sy + sh
            if sx >= x0 and sx1 <= x1 and sy >= y0 and sy1 <= y1:
                continue
            level = sy >= ly0 - band and sy1 <= ly1 + band
            close = sx - x1 < gap and x0 - sx1 < gap
            nx0, nx1 = min(x0, sx), max(x1, sx1)
            if level and close and nx1 - nx0 <= 0.45 * page_w:
                x0, y0, x1, y1 = nx0, min(y0, sy), nx1, max(y1, sy1)
                grew = True
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _overlap(a, b) -> float:
    """Share of box a covered by box b."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, a[2] * a[3])


def detect(rgb: np.ndarray, lines: list[dict], dpi: int, cfg: dict,
           forbidden: list[list[int]] | None = None) -> list[dict]:
    if not cfg.get("enabled", True):
        return []
    H, W = rgb.shape[:2]
    name_boxes = find_names(lines, H, cfg)
    regions = [{"bbox": b, "verdict": "name"} for b in name_boxes]
    regions += [{"bbox": b, "verdict": "title"} for b in find_titles(lines, H, cfg) if b not in name_boxes]
    seen = [r["bbox"] for r in regions]
    regions += [{"bbox": b, "verdict": "title"} for b in find_unreadable_letterhead(lines, H, cfg) if b not in seen]
    if cfg.get("unreadable_letterhead", True) and page_unreadable(lines, cfg):
        regions += [{"bbox": b, "verdict": "title"} for b in letterhead_side_lines(rgb, dpi, cfg)]
    logos = find_logos(rgb, dpi, cfg, lines, forbidden)
    regions += [{"bbox": b, "verdict": "logo"} for b in logos]
    near = 0.6 * dpi
    for ln in lines:
        x, y, w, h = ln["bbox"]
        if len(ln["text"].split()) > int(cfg.get("max_title_words", 10)):
            continue
        for lx, ly, lw, lh in logos:
            beside = x - (lx + lw) < near and lx - (x + w) < near       # horizontally close
            level = y < ly + lh + 0.25 * dpi and ly - 0.25 * dpi < y + h  # vertically alongside
            if beside and level and not any(r["bbox"] == [x, y, w, h] for r in regions):
                regions.append({"bbox": [x, y, w, h], "verdict": "title"})
                break
    pad = int(0.02 * dpi)
    for i, r in enumerate(sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0])), start=1):
        x, y, w, h = r["bbox"]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        r["bbox"] = [x0, y0, min(W, x + w + pad) - x0, min(H, y + h + pad) - y0]
        r["id"] = f"s{i}"
    return sorted(regions, key=lambda r: r["id"])


def apply_overrides(key: str, regions: list[dict], overrides: dict) -> list[dict]:
    """exclusions.yaml: `keep: [s2]` leaves a detected area readable, `scramble: [[x, y, w, h]]` adds one."""
    o = overrides.get(key, {}) or {}
    keep = set(o.get("keep", []) or [])
    out = [r for r in regions if r["id"] not in keep]
    out += [{"id": f"m{i}", "bbox": list(map(int, b)), "verdict": "logo"}
            for i, b in enumerate(o.get("scramble", []) or [], start=1)]
    return out


# ------------------------------------------------------------------ applying

def _scramble_line(img: np.ndarray, rng: np.random.Generator, thorough: bool = False) -> np.ndarray:
    """Shuffle (and partly mirror) the ink clusters of a text line, keeping its gaps."""
    gray = img.mean(axis=2) if img.ndim == 3 else img
    ink_cols = (gray < gray.max() - 40).any(axis=0)
    segs, start = [], None
    for x, on in enumerate(np.append(ink_cols, False)):
        if on and start is None:
            start = x
        elif not on and start is not None:
            segs.append((start, x))
            start = None
    # cut long ink runs (whole words) into letter-sized pieces so no word survives intact
    piece_w = max(4, int(img.shape[0] * (0.4 if thorough else 0.55)))
    fine = []
    for a, b in segs:
        n = max(1, round((b - a) / piece_w))
        edges = np.linspace(a, b, n + 1).round().astype(int)
        fine += [(int(edges[i]), int(edges[i + 1])) for i in range(n) if edges[i + 1] > edges[i]]
    segs = fine
    if len(segs) < 3:
        return _scramble_tiles(img, rng, (1, 6))
    pieces = [img[:, a:b].copy() for a, b in segs]
    order = rng.permutation(len(pieces))
    if thorough and len(pieces) > 1:
        # a derangement: no piece stays in (or next to) its own place
        for _ in range(50):
            if np.all(np.abs(order - np.arange(len(pieces))) > 1) or len(pieces) < 4:
                break
            order = rng.permutation(len(pieces))
    if np.all(order == np.arange(len(pieces))):
        order = order[::-1]
    out = img.copy()
    paper = np.median(img[:, ~ink_cols], axis=(0, 1)) if (~ink_cols).any() else img.max(axis=(0, 1))
    out[:, segs[0][0]:segs[-1][1]] = paper  # clear the line, then lay the clusters back down
    x = segs[0][0]
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)] + [0]
    for gi, idx in enumerate(order):
        piece = pieces[idx]
        if rng.random() < (0.7 if thorough else 0.5):
            piece = piece[:, ::-1]
        if rng.random() < (0.4 if thorough else 0.25):
            piece = piece[::-1]
        w = piece.shape[1]
        if x + w > out.shape[1]:
            break
        out[:, x:x + w] = piece
        x += w + gaps[gi]
    return out


def _scramble_tiles(img: np.ndarray, rng: np.random.Generator, grid=(4, 4)) -> np.ndarray:
    """Cut into tiles, permute and flip them: an abstract collage in the same colours."""
    h, w = img.shape[:2]
    gy, gx = grid
    th, tw = max(1, h // gy), max(1, w // gx)
    out = img.copy()
    cells = [(r, c) for r in range(gy) for c in range(gx)]
    order = rng.permutation(len(cells))
    for (r, c), k in zip(cells, order):
        sr, sc = cells[k]
        tile = img[sr * th:(sr + 1) * th, sc * tw:(sc + 1) * tw]
        if rng.random() < 0.5:
            tile = tile[::-1]
        if rng.random() < 0.5:
            tile = tile[:, ::-1]
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    return out


def apply(page: np.ndarray, regions: list[dict], key: str, seed: int = 0) -> np.ndarray:
    """Scramble the given regions of a page image (deterministic for page key + seed)."""
    if not regions:
        return page
    out = page.copy()
    for r in regions:
        x, y, w, h = r["bbox"]
        rng = np.random.default_rng(zlib.crc32(f"{key}|{seed}|{x},{y},{w},{h}".encode()))
        area = out[y:y + h, x:x + w]
        if area.size == 0:
            continue
        if r["verdict"] in ("title", "name") and w > 2 * h:
            out[y:y + h, x:x + w] = _scramble_line(area, rng, thorough=r["verdict"] == "name")
        else:
            out[y:y + h, x:x + w] = _scramble_tiles(area, rng, (4, 4) if w < 3 * h else (3, 8))
    return out
