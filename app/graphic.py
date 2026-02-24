# ============================================================
# MERGED FILE = (graphic1.py) + (graphic - Copy (4).py)
# Uses RELATIVE paths - works on any computer!
# ============================================================

# ============================================================
# FILE A: DOCX -> Extract 7 Figures (FROM graphic1.py)
# ============================================================

import os, io
import numpy as np
from docx import Document
from PIL import Image, ImageDraw
from pathlib import Path

# =========================
# AUTO-DETECT PATHS (works on any computer)
# =========================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# All paths are relative to script location
UPLOAD_DIR = SCRIPT_DIR / "data" / "uploads"
EXTRACT_DOCX_PATH = str(UPLOAD_DIR / "dwr-25-45-1.docx")

EXTRACT_OUT_DIR = str(SCRIPT_DIR / "only_7_figures_output")
EXTRACT_DEBUG_DIR = os.path.join(EXTRACT_OUT_DIR, "_debug_candidates")
EXTRACT_FINAL_DIR = os.path.join(EXTRACT_OUT_DIR, "final_7")

# ❗ Put the debug candidate numbers you DON'T want here
# Example: {3} means skip cand_003_*.png
EXTRACT_EXCLUDE_DEBUG_CANDIDATES = set(
)  # Don't skip any - we need all 7 for proper order mapping

# =============================
# FIGURE ORDER MAPPING (to match document order)
# =============================
# Maps: extracted_index -> correct_figure_number
# Example: If extracted Figure_1 should be Figure 2 in document, use {1: 2}
# Set to None to keep original extraction order (sorted by size)
# Set to a dict to reorder: {old_idx: new_idx, ...}
FIGURE_ORDER_MAP = {
    1: 3,  # Extracted 1 (line chart) -> Figure 3
    2: 1,  # Extracted 2 (infographic) -> Figure 1 
    3: 4,  # Extracted 3 -> Figure 4
    4: 6,  # Extracted 4 -> Figure 6
    5: 5,  # Extracted 5 -> Figure 5
    6: 7,  # Extracted 6 -> Figure 7
    7: 2,  # Extracted 7 -> Figure 2
}
# To reorder: Change the values. Example: {1: 2, 2: 1} swaps Fig1 and Fig2

# =========================
# FILTER SETTINGS (SAFE)
# =========================
EXTRACT_MIN_AREA = 100_000  # small icons get removed
EXTRACT_MIN_W = 280
EXTRACT_MIN_H = 210
EXTRACT_MAX_ASPECT = 10.0  # stripe-like super long images removed

# Split stacked charts settings
EXTRACT_EMPTY_ROW_DENSITY = 0.010  # how "empty" a row should be to count as gap
EXTRACT_MIN_GAP_RATIO = 0.08  # big empty gap needed to split

# Crop settings (lower tol + higher pad = preserve more of the figure)
EXTRACT_CROP_TOL = 10  # lower tolerance to avoid treating light-gray panels as background
EXTRACT_CROP_PAD = 24  # more padding to preserve rounded corners and margins


# =========================
# HELPERS
# =========================
def extract_light_crop(img: Image.Image,
                       pad=EXTRACT_CROP_PAD,
                       tol=EXTRACT_CROP_TOL):
    """
    Trim only pure white margins from edges.
    Preserves figures with colored backgrounds (light blue, gray, etc.)
    """
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.int16)
    h, w, _ = arr.shape

    if h < 20 or w < 20:
        return img

    # Only crop pure white (RGB close to 255,255,255)
    white_ref = np.array([255, 255, 255], dtype=np.int16)
    diff = np.abs(arr - white_ref).sum(axis=2)
    is_white = diff <= 15  # Very strict - only pure white

    # Trim only full-white rows/cols from edges
    top = 0
    while top < h and is_white[top].all():
        top += 1
    bottom = h - 1
    while bottom >= 0 and is_white[bottom].all():
        bottom -= 1
    left = 0
    while left < w and is_white[:, left].all():
        left += 1
    right = w - 1
    while right >= 0 and is_white[:, right].all():
        right -= 1

    if top >= bottom or left >= right:
        return img

    x0 = max(left - pad, 0)
    y0 = max(top - pad, 0)
    x1 = min(right + pad + 1, w)
    y1 = min(bottom + pad + 1, h)
    return img.crop((x0, y0, x1, y1))


def extract_split_if_stacked(img: Image.Image):
    """
    Split image into parts if there are big empty horizontal gaps.
    For tall full-page screenshots, removes header and footer sections.
    """
    w_img, h_img = img.size

    # Only apply header/footer removal to TALL images (likely full-page screenshots)
    # Regular images just get split at gaps as before
    is_tall_screenshot = h_img > 1500 and (h_img / w_img) > 1.2

    g = np.array(img.convert("L"), dtype=np.int16)
    h, w = g.shape

    # --- edge map (simple gradient magnitude threshold) ---
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    dx = np.pad(dx, ((0, 0), (0, 1)), mode="constant")
    dy = np.pad(dy, ((0, 1), (0, 0)), mode="constant")
    grad = dx + dy

    EDGE_THR = 40
    edges = grad > EDGE_THR

    # Row "content" = edge density per row
    row_density = edges.mean(axis=1)

    # Adaptive empty threshold
    q = float(np.quantile(row_density, 0.08)) if h > 20 else 0.0
    empty_thr = max(0.002, min(EXTRACT_EMPTY_ROW_DENSITY, q * 1.2))

    empty = row_density < empty_thr

    # Find continuous empty gaps
    gaps = []
    start = None
    for y in range(h):
        if empty[y] and start is None:
            start = y
        if (not empty[y]) and start is not None:
            gaps.append((start, y - 1))
            start = None
    if start is not None:
        gaps.append((start, h - 1))

    # Keep only "big" gaps
    big = [(a, b) for a, b in gaps if (b - a) > int(h * EXTRACT_MIN_GAP_RATIO)]

    if not big:
        return [img]

    # For tall screenshots, extract content between header and footer gaps
    if is_tall_screenshot and len(big) >= 2:
        big.sort(key=lambda ab: ab[0])

        # Content is between gaps: starts after first gap ends, ends before second gap starts
        first_gap = big[0]
        second_gap = big[1] if len(big) > 1 else None

        # Content starts after the first gap ends (with small padding)
        content_start = first_gap[1] + 1 - 20  # Slight padding above
        content_start = max(0, content_start)

        # Content ends before the second gap starts (with small padding)
        if second_gap and second_gap[0] > h * 0.5:
            content_end = second_gap[0] + 20  # Slight padding below
            content_end = min(h, content_end)
        else:
            content_end = h

        # Only crop if we're removing significant portions
        if content_start > h * 0.1 or content_end < h * 0.9:
            return [img.crop((0, content_start, w, content_end))]

    # Original behavior: split at gap closest to center
    center = h // 2
    big.sort(key=lambda ab: abs(((ab[0] + ab[1]) // 2) - center))
    a, b = big[0]
    cut = (a + b) // 2

    top = img.crop((0, 0, w, cut))
    bottom = img.crop((0, cut, w, h))

    return [top, bottom]


def extract_is_symbol_or_stripe(img: Image.Image):
    """Remove tiny icons and stripe-like images."""
    w, h = img.size
    area = w * h

    if area < EXTRACT_MIN_AREA:
        return True
    if w < EXTRACT_MIN_W or h < EXTRACT_MIN_H:
        return True
    if max(w / h, h / w) > EXTRACT_MAX_ASPECT:
        return True

    return False


# =========================
# FIGURE 6 (plain grid): generate a clean grid image with NO TEXT
# =========================
def generate_plain_grid_figure6(out_png_path: str,
                                width: int = 700,
                                height: int = 450,
                                rows: int = 4,
                                cols: int = 3,
                                bg_color=(230, 240, 242),
                                grid_color=(20, 20, 20),
                                grid_width: int = 2):
    """
    Creates a plain grid (like the chart background) with zero labels/text.
    Saved directly to out_png_path.
    """
    # Add padding around the grid
    padding_left = 30
    padding_top = 25
    padding_right = 20
    padding_bottom = 30
    
    grid_width_px = width - padding_left - padding_right
    grid_height_px = height - padding_top - padding_bottom
    
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    tick_extend = 10  # Tick length

    # vertical lines
    for i in range(cols + 1):
        x = padding_left + int(i * grid_width_px / cols)
        draw.line([(x, padding_top), (x, height - padding_bottom)], fill=grid_color, width=grid_width)
        # Bottom axis tick for first column (i=0) and second column (i=1)
        if i == 0 or i == 1:
            draw.line([(x, height - padding_bottom), (x, height - padding_bottom + tick_extend)], fill=grid_color, width=grid_width)

    # horizontal lines
    for j in range(rows + 1):
        y = padding_top + int(j * grid_height_px / rows)
        draw.line([(padding_left, y), (width - padding_right, y)], fill=grid_color, width=grid_width)
        # Left axis tick only for rows 2 and 3 (j=2 and j=3)
        if j == 2 or j == 3:
            draw.line([(padding_left - tick_extend, y), (padding_left, y)], fill=grid_color, width=grid_width)

    img.save(out_png_path)
    return img


# =========================
# FIGURE 5 (special): render page and crop Figure 5 panel
# =========================
def extract_figure5_via_render(docx_path: str,
                               out_png_path: str,
                               dpi: int = 200):
    """
    Figure 5 is not embedded as a standalone image in the DOCX (it's part of the page layout),
    so we render the DOCX -> PDF -> page image and crop the Figure 5 panel.
    """
    import subprocess
    import tempfile
    from pdf2image import convert_from_path

    docx_path = str(docx_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert DOCX -> PDF (LibreOffice)
        subprocess.run(
            [
                "soffice", "--headless", "--convert-to", "pdf", "--outdir",
                tmpdir, docx_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Find the produced PDF (LibreOffice keeps the original basename)
        pdf_files = [p for p in Path(tmpdir).glob("*.pdf")]
        if not pdf_files:
            raise RuntimeError("DOCX->PDF conversion failed: no PDF produced.")
        pdf_path = str(pdf_files[0])

        # Figure 5 appears on page 6 in this report (1-indexed)
        page_img = convert_from_path(pdf_path,
                                     dpi=dpi,
                                     first_page=6,
                                     last_page=6)[0]

        W, H = page_img.size

        # Crop ratios tuned to the provided document layout (left panel on page 6)
        x0, y0, x1, y1 = 0.045, 0.10, 0.545, 0.575
        crop = page_img.crop(
            (int(W * x0), int(H * y0), int(W * x1), int(H * y1)))

        crop.save(out_png_path)
        return crop


# =========================
# MAIN
# =========================
def main_docx_extract():
    os.makedirs(EXTRACT_OUT_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DEBUG_DIR, exist_ok=True)
    os.makedirs(EXTRACT_FINAL_DIR, exist_ok=True)

    doc = Document(EXTRACT_DOCX_PATH)

    raw_imgs = []
    for rel in doc.part.related_parts.values():
        if "image" in rel.content_type:
            try:
                raw_imgs.append(Image.open(io.BytesIO(rel.blob)).copy())
            except Exception:
                pass

    print("Total embedded images found:", len(raw_imgs))

    candidates = []
    for im in raw_imgs:
        im = extract_light_crop(im)

        parts = extract_split_if_stacked(im)
        for p in parts:
            p = extract_light_crop(p)
            if not extract_is_symbol_or_stripe(p):
                candidates.append(p)

    # Sort candidates by size (largest first)
    candidates.sort(key=lambda im: im.size[0] * im.size[1], reverse=True)

    print("Candidates after filtering:", len(candidates))

    # Save all candidates to debug folder
    debug_files = []
    for idx, im in enumerate(candidates, start=1):
        w, h = im.size
        fname = f"cand_{idx:03d}_{w}x{h}.png"
        path = os.path.join(EXTRACT_DEBUG_DIR, fname)
        im.save(path)
        debug_files.append((idx, path))

    print("Saved debug candidates in:", EXTRACT_DEBUG_DIR)

    # Build final list excluding specified debug candidate numbers
    final_imgs = []
    for idx, path in debug_files:
        if idx in EXTRACT_EXCLUDE_DEBUG_CANDIDATES:
            print("Skipping debug candidate:", os.path.basename(path))
            continue

        final_imgs.append(Image.open(path))

        if len(final_imgs) == 7:
            break

    # Save final 7 figures (with optional reordering)
    for i, im in enumerate(final_imgs, start=1):
        # Apply order mapping if defined
        new_idx = FIGURE_ORDER_MAP.get(i, i) if FIGURE_ORDER_MAP else i

        # Special cropping for Figure 1 - trim extra left/right
        if new_idx == 1:
            w, h = im.size
            im = im.crop((50, 0, w - 50, h))
            print(f"  -> Cropped Figure 1 to {im.size}")

        # Special cropping for Figure 3 - remove rainbow lines and margins
        if new_idx == 3:
            w, h = im.size
            # Crop: left 45px (rainbow), top 120px, bottom at 500px
            im = im.crop((45, 120, w, 500))
            print(f"  -> Cropped Figure 3 to {im.size}")

        # Special cropping for Figure 4 - remove bottom portion with legend
        if new_idx == 4:
            w, h = im.size
            im = im.crop((0, 0, w, 450))
            print(f"  -> Cropped Figure 4 to {im.size}")

        # Special handling for Figure 5 - render page 6 and crop the Figure 5 panel
        if new_idx == 5:
            # Figure 5 is not a clean standalone embedded image in this DOCX,
            # so we render the page and crop it.
            fig5_out = os.path.join(EXTRACT_FINAL_DIR, f"Figure_{new_idx}.png")
            try:
                im = extract_figure5_via_render(EXTRACT_DOCX_PATH,
                                                fig5_out,
                                                dpi=200)
                print(f"  -> Figure 5: extracted via render, size {im.size}")
                # Skip the normal save below since we already saved it
                continue
            except Exception as e:
                print(
                    f"  -> Figure 5: render-crop failed ({e}); falling back to embedded candidate crop"
                )
                w, h = im.size
                im = im.crop((0, 50, w - 45, 430))
                print(f"  -> Cropped Figure 5 (fallback) to {im.size}")
        # Special handling for Figure 6 - GENERATE PLAIN GRID (NO TEXT)
        if new_idx == 6:
            out = os.path.join(EXTRACT_FINAL_DIR, "Figure_6.png")
            generate_plain_grid_figure6(out)
            print(f"  -> Figure 6: generated plain grid at {out}")
            continue

        out = os.path.join(EXTRACT_FINAL_DIR, f"Figure_{new_idx}.png")
        im.save(out)
        print(
            f"Saved final: {out} (extracted as #{i}, saved as Figure_{new_idx})"
        )

    print("\nDONE ✅")
    print("Final 7 saved in:", EXTRACT_FINAL_DIR)


# ============================================================
# FILE B: Images -> OCR -> JSON (FROM graphic - Copy (4).py)
# ============================================================

import json
from pathlib import Path

import cv2
import pytesseract

# =============================
# CONFIG (AUTO-DETECT PATHS - works on any computer)
# =============================
# Use SCRIPT_DIR from Part A (already defined above)
IMAGES = [str(UPLOAD_DIR / f"figure_{i}.png") for i in range(7)]

OUT_DIR = SCRIPT_DIR / "out_json_fixed_fig0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detect Tesseract path (Windows vs Linux/Mac)
import platform
if platform.system() == "Windows":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
# On Linux/Mac, tesseract is usually in PATH - no need to set

UPSCALE = 3  # Higher = better OCR quality

MIN_BOX_W = 8
MIN_BOX_H = 8
MIN_CONF = 0.30  # Lower to catch more text

# Line clustering (build one line-box per visual line)
LINE_Y_MERGE_MULT = 0.35  # 0.30–0.45

# Paragraph merge (general)
PARA_Y_GAP_MULT = 1.60
LEFT_ALIGN_TOL = 120
X_OVERLAP_MIN = 0.45
HEIGHT_SIMILAR_TOL = 0.60
MAX_PARAGRAPH_WIDTH_JUMP = 3.5

# 🔥 HARD FIX FOR FIG 0 TITLE:
# Merge ALL text lines in top region into ONE title textbox.
FIG0_TITLE_TOP_REGION = 0.30  # top 30% of image
FIG0_TITLE_MAX_BOTTOM = 0.36  # allow title to extend slightly lower
FIG0_MIN_TITLE_LINES = 2  # only merge if at least 2 lines found


# =============================
# Helpers
# =============================
def median(vals):
    vals = sorted(vals)
    if not vals:
        return 10
    return vals[len(vals) // 2]


def clean_token(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = t.replace("|", "")
    return " ".join(t.split())


def is_junk_token(t: str) -> bool:
    import re
    if not t:
        return True
    # Filter single characters (except numbers and meaningful letters)
    if len(t) == 1 and t.lower() not in "0123456789":
        return True
    # Filter 2-char noise patterns
    if len(t) == 2 and t.lower() in {
            "xx", "rr", "aa", "xy", "yx", ";/", "/;", "//", "\\\\", ".,", ",."
    }:
        return True
    # Filter pure punctuation/symbol noise
    if re.match(r'^[;/,.\-_=+*#@!?\'"\\|]+$', t):
        return True
    return False


def union_bbox(items):
    x1 = min(a["x"] for a in items)
    y1 = min(a["y"] for a in items)
    x2 = max(a["x"] + a["w"] for a in items)
    y2 = max(a["y"] + a["h"] for a in items)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def join_words_as_line(words_sorted):
    parts = []
    for w in words_sorted:
        txt = w["text"]
        if not txt:
            continue
        if not parts:
            parts.append(txt)
        else:
            if parts[-1].endswith("-"):
                parts[-1] = parts[-1][:-1] + txt
            else:
                parts.append(" " + txt)
    return " ".join("".join(parts).split())


def join_lines(lines):
    out = []
    for ln in lines:
        t = ln["text"].strip()
        if not t:
            continue
        if out and out[-1].endswith("-"):
            out[-1] = out[-1][:-1] + t
        else:
            out.append(t)
    result = " ".join(" ".join(out).split())
    # Clean OCR artifacts from final merged text
    result = result.replace("\\", "").replace("  ", " ")
    result = result.replace("'\\,", "").replace("\\'", "'")
    return result


def x_overlap_ratio(a, b):
    a1, a2 = a["x"], a["x"] + a["w"]
    b1, b2 = b["x"], b["x"] + b["w"]
    inter = max(0, min(a2, b2) - max(a1, b1))
    denom = max(1, min(a["w"], b["w"]))
    return inter / denom


def height_ratio_ok(a, b):
    ha, hb = max(1, a["h"]), max(1, b["h"])
    r = min(ha, hb) / max(ha, hb)
    return r >= HEIGHT_SIMILAR_TOL


def width_jump_ok(a, b):
    wa, wb = max(1, a["w"]), max(1, b["w"])
    bigger = max(wa, wb) / min(wa, wb)
    return bigger <= MAX_PARAGRAPH_WIDTH_JUMP


def can_be_same_paragraph(a, b):
    gap = b["y"] - (a["y"] + a["h"])
    max_gap = int(min(a["h"], b["h"]) * PARA_Y_GAP_MULT)
    if gap < 0 or gap > max(3, max_gap):
        return False

    ov = x_overlap_ratio(a, b)
    if ov < X_OVERLAP_MIN:
        return False

    # allow some left drift if overlap is strong
    left_ok = abs(a["x"] - b["x"]) <= LEFT_ALIGN_TOL
    if not left_ok and ov < 0.80:
        return False

    if not height_ratio_ok(a, b):
        return False

    if not width_jump_ok(a, b) and ov < 0.85:
        return False

    if a["text"].rstrip().endswith((".", "!", "?")):
        return False

    return True


# =============================
# OCR ENGINE SELECTION
# =============================
# Set to "google" for best accuracy, "easyocr" for colored text, or "tesseract" for compatibility
OCR_ENGINE = "easyocr"  # Options: "google", "easyocr", or "tesseract"

# Google Vision API Key
GOOGLE_VISION_API_KEY = "AIzaSyCkHPaenlrEL1Dty5vavK7okp8u8Sq7qPk"

# EasyOCR reader (initialized lazily)
_easyocr_reader = None


def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except ImportError:
            print("EasyOCR not installed. Install with: pip install easyocr")
            return None
    return _easyocr_reader


# =============================
# OCR words (EasyOCR version)
# =============================
def ocr_words_easyocr(image_path: str):
    reader = get_easyocr_reader()
    if reader is None:
        raise RuntimeError("EasyOCR not available")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h0, w0 = img.shape[:2]
    img = cv2.resize(img, (w0 * UPSCALE, h0 * UPSCALE),
                     interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]

    # EasyOCR works directly on color images - no preprocessing needed!
    results = reader.readtext(img)

    words = []
    for (bbox, text, conf) in results:
        txt = clean_token(text)
        if is_junk_token(txt):
            continue
        if conf < MIN_CONF:
            continue

        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        x1 = int(min(p[0] for p in bbox))
        y1 = int(min(p[1] for p in bbox))
        x2 = int(max(p[0] for p in bbox))
        y2 = int(max(p[1] for p in bbox))
        w = x2 - x1
        h = y2 - y1

        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue

        words.append({
            "text": txt,
            "x": x1,
            "y": y1,
            "w": w,
            "h": h,
            "conf": conf,
            "yc": y1 + h / 2.0
        })

    return words, W, H


# =============================
# OCR words (Google Vision version)
# =============================
def ocr_words_google(image_path: str):
    import requests
    import base64

    if not GOOGLE_VISION_API_KEY:
        raise RuntimeError(
            "GOOGLE_VISION_API_KEY not set. Set it as environment variable.")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h0, w0 = img.shape[:2]
    # Scale up for better OCR
    img = cv2.resize(img, (w0 * UPSCALE, h0 * UPSCALE),
                     interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]

    # Encode image to base64
    _, buffer = cv2.imencode('.png', img)
    image_content = base64.b64encode(buffer).decode('utf-8')

    # Call Google Vision API
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [{
            "image": {
                "content": image_content
            },
            "features": [{
                "type": "TEXT_DETECTION"
            }]
        }]
    }

    response = requests.post(url, json=payload)
    result = response.json()

    if 'error' in result:
        raise RuntimeError(f"Google Vision API error: {result['error']}")

    words = []
    responses = result.get('responses', [])
    if not responses or 'textAnnotations' not in responses[0]:
        return words, W, H

    # Skip first annotation (full text), process individual words
    annotations = responses[0]['textAnnotations'][1:]  # Skip first (full text)

    for idx, ann in enumerate(annotations):
        txt = clean_token(ann.get('description', ''))
        if is_junk_token(txt):
            continue

        # Get bounding box vertices
        vertices = ann.get('boundingPoly', {}).get('vertices', [])
        if len(vertices) < 4:
            continue

        x1 = min(v.get('x', 0) for v in vertices)
        y1 = min(v.get('y', 0) for v in vertices)
        x2 = max(v.get('x', 0) for v in vertices)
        y2 = max(v.get('y', 0) for v in vertices)
        w = x2 - x1
        h = y2 - y1

        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue

        words.append({
            "text": txt,
            "x": x1,
            "y": y1,
            "w": w,
            "h": h,
            "conf": 0.95,  # Google doesn't return confidence per word
            "yc": y1 + h / 2.0
        })

    return words, W, H


# =============================
# OCR words (Tesseract version)
# =============================
def ocr_words_tesseract(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h0, w0 = img.shape[:2]
    # Higher upscale for better text recognition
    scale = max(UPSCALE, 3)
    img = cv2.resize(img, (w0 * scale, h0 * scale),
                     interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhanced preprocessing for chart images
    gray = cv2.bilateralFilter(gray, 11, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=10)
    # Adaptive threshold works better for varying backgrounds
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 8)

    # Use PSM 11 (sparse text) for chart images - better for scattered labels
    data = pytesseract.image_to_data(thr,
                                     config="--oem 3 --psm 11",
                                     output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        txt = clean_token(data["text"][i])
        if is_junk_token(txt):
            continue

        try:
            conf_raw = float(data["conf"][i])
            if conf_raw < 0:
                continue
            conf = conf_raw / 100.0
        except:
            conf = None

        if conf is not None and conf < MIN_CONF:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue

        words.append({
            "text": txt,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
            "yc": y + h / 2.0
        })

    return words, W, H


# =============================
# OCR words (auto-select engine)
# =============================
def ocr_words(image_path: str):
    if OCR_ENGINE == "google":
        try:
            return ocr_words_google(image_path)
        except Exception as e:
            print(
                f"Google Vision failed: {e}, falling back to EasyOCR/Tesseract"
            )
            try:
                return ocr_words_easyocr(image_path)
            except:
                return ocr_words_tesseract(image_path)
    elif OCR_ENGINE == "easyocr":
        try:
            return ocr_words_easyocr(image_path)
        except Exception as e:
            print(f"EasyOCR failed: {e}, falling back to Tesseract")
            return ocr_words_tesseract(image_path)
    else:
        return ocr_words_tesseract(image_path)


# =============================
# Build line boxes
# =============================
def words_to_line_boxes(words):
    if not words:
        return []

    med_h = median([w["h"] for w in words])
    y_thresh = max(5, int(med_h * LINE_Y_MERGE_MULT))

    words_sorted = sorted(words, key=lambda z: (z["yc"], z["x"]))

    lines = []
    for w in words_sorted:
        placed = False
        for line in lines:
            if abs(w["yc"] - line["yc"]) <= y_thresh:
                line["words"].append(w)
                line["yc"] = (line["yc"] * 0.9) + (w["yc"] * 0.1)
                placed = True
                break
        if not placed:
            lines.append({"yc": w["yc"], "words": [w]})

    line_boxes = []
    for ln in lines:
        ws = sorted(ln["words"], key=lambda z: z["x"])
        text = join_words_as_line(ws)
        if not text:
            continue

        x, y, w, h = union_bbox(ws)
        confs = [a["conf"] for a in ws if a["conf"] is not None]
        avg_conf = sum(confs) / len(confs) if confs else None

        line_boxes.append({
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": avg_conf
        })

    line_boxes.sort(key=lambda t: (t["y"], t["x"]))
    return line_boxes


# =============================
# Merge paragraphs (general)
# =============================
def merge_paragraphs(line_boxes):
    if not line_boxes:
        return []

    merged = []
    i = 0
    while i < len(line_boxes):
        group = [line_boxes[i]]
        j = i + 1

        while j < len(line_boxes) and can_be_same_paragraph(
                group[-1], line_boxes[j]):
            group.append(line_boxes[j])
            j += 1

        para_text = join_lines(group)
        x, y, w, h = union_bbox(group)
        confs = [b["conf"] for b in group if b["conf"] is not None]
        avg_conf = sum(confs) / len(confs) if confs else None

        merged.append({
            "text": para_text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": avg_conf
        })

        i = j

    merged.sort(key=lambda t: (t["y"], t["x"]))
    return merged


# =============================
# HARD FIX: merge ALL title lines in Fig0 into ONE textbox
# =============================
def force_merge_title_block_if_fig0(textboxes, W, H, fig_index_zero_based):
    if fig_index_zero_based != 0:
        return textboxes

    top_limit = int(FIG0_TITLE_TOP_REGION * H)
    bottom_limit = int(FIG0_TITLE_MAX_BOTTOM * H)

    title_lines = []
    rest = []

    for tb in textboxes:
        # take lines that live in the header zone
        if tb["y"] <= bottom_limit and (
                tb["y"] + tb["h"]) <= bottom_limit and tb["y"] <= top_limit:
            title_lines.append(tb)
        else:
            rest.append(tb)

    # If we didn't find enough lines, do nothing
    if len(title_lines) < FIG0_MIN_TITLE_LINES:
        return textboxes

    # Merge ALL title lines into ONE
    title_lines.sort(key=lambda t: (t["y"], t["x"]))
    merged_text = join_lines(title_lines)
    x, y, w, h = union_bbox(title_lines)
    confs = [b["conf"] for b in title_lines if b["conf"] is not None]
    avg_conf = sum(confs) / len(confs) if confs else None

    merged_title = {
        "text": merged_text,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "conf": avg_conf
    }

    # Put merged title back + rest
    out = [merged_title] + rest
    out.sort(key=lambda t: (t["y"], t["x"]))
    return out


def add_ids(textboxes):
    for k, tb in enumerate(textboxes, start=1):
        tb["id"] = f"tb_{k:05d}"
    return textboxes


# =============================
# SKIP IF JSON EXISTS (preserves manual edits)
# =============================
SKIP_IF_JSON_EXISTS = True  # Set to False to always regenerate


# =============================
# MAIN
# =============================
def main_json_ocr():
    combined = {"figures": []}

    for idx, img_path in enumerate(IMAGES, start=1):
        out_file = OUT_DIR / f"fig{idx}.json"

        # Load existing JSON if present (silent - no skip message)
        if SKIP_IF_JSON_EXISTS and out_file.exists():
            try:
                existing_fig = json.loads(out_file.read_text(encoding="utf-8"))
                combined["figures"].append(existing_fig)
                print(
                    f"[OK] Fig{idx}: loaded {len(existing_fig.get('textboxes', []))} textboxes -> {out_file.name}"
                )
            except:
                pass
            continue

        words, W, H = ocr_words(img_path)
        line_boxes = words_to_line_boxes(words)
        merged_boxes = merge_paragraphs(line_boxes)

        # 🔥 force Fig0 title block into ONE textbox
        merged_boxes = force_merge_title_block_if_fig0(
            merged_boxes, W, H, fig_index_zero_based=(idx - 1))

        merged_boxes.sort(key=lambda t: (t["y"], t["x"]))
        merged_boxes = add_ids(merged_boxes)

        fig = {
            "figure_id": f"Figure {idx}",
            "image_path": img_path,
            "width": W,
            "height": H,
            "textboxes": merged_boxes
        }

        out_file.write_text(json.dumps(fig, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        combined["figures"].append(fig)

        print(
            f"[OK] Fig{idx}: final_textboxes={len(merged_boxes)} -> {out_file.name}"
        )

    all_file = OUT_DIR / "all_figures.json"
    all_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n✅ Saved combined: {all_file.resolve()}")


# ============================================================
# RUN BOTH (one file)
# ============================================================
if __name__ == "__main__":
    # 1) Extract images from DOCX -> only_7_figures_output/final_7/
    main_docx_extract()

    # 2) OCR/JSON from the IMAGES list -> out_json_fixed_fig0/
    main_json_ocr()
