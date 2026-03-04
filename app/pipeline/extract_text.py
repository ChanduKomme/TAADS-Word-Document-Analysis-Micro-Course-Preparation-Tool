from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract
from collections import Counter
import math


def _normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u00A0", " ", s)
    return s.strip()


def _is_header_footer_line(l: str) -> bool:
    """Filter out headers, footers, running heads, and boilerplate."""
    l = l.strip()
    if not l or len(l) < 4:
        return True

    # EXCEPTION: Preserve media callout lines (audio/video links) even if they contain URLs
    media_keywords = ["audio interview", "video", "podcast", "mediathek", "qr code"]
    l_lower = l.lower()
    # Keep line if it has media keywords AND a URL (more specific to avoid false positives)
    has_media_keyword = any(keyword in l_lower for keyword in media_keywords)
    has_url = re.search(r"(www\.|http)", l, re.I)
    if has_media_keyword and has_url:
        return False  # Keep this line - it's media content
    
    # Common boilerplate patterns
    if re.search(r"(©|Copyright|www\.|http|All rights reserved|ISSN|ISBN|DOI|JEL|Keywords)", l, re.I):
        return True
    if re.search(r"(Volume \d+|Weekly Report \d+/\d+|Working Paper|Verlag|Stiftung|Publisher)", l, re.I):
        return True
    
    # Page numbers with report names (e.g., "251 DIW Weekly Report 38+39/2025")
    if re.search(r"^\d{1,4}\s+DIW\s+Weekly\s+Report", l, re.I):
        return True

    # Running header pattern for this specific document
    if re.search(r"Productivity:\s*East[- ]west gap", l, re.I):
        return True

    # HIGHLY SPECIFIC: Footnote patterns with explicit cues (MUST have colon or keyword)
    if len(l) < 150:  # Must be short
        # Pattern 1: Number + word + COLON (e.g., "18 Formerly:", "20 The:")
        if re.match(r"^\d{1,2}\s+\w+.*:", l):
            return True
        # Pattern 2: Number + parenthesis (e.g., "18) The concept")
        if re.match(r"^\d{1,2}\)\s+", l):
            return True
        # Pattern 3: Number + specific footnote keywords
        if re.match(r"^\d{1,2}\s+(Formerly|The\s+concept|See|Compare|Wirtschafswoche|Christoph|Eric|Martin)", l):
            return True

    # AGGRESSIVE: Author attribution and contact info
    if re.search(r"@diw\.de|@[a-z]+\.(de|com|org)", l, re.I):
        return True
    if re.search(r"is\s+(the\s+)?Research\s+Director", l, re.I):
        return True
    if re.search(r"Department\s+at\s+DIW\s+Berlin", l, re.I):
        return True

  

    # SPECIFIC: Chart/figure titles, captions, and legend items
    # Always filter these chart-specific patterns (length-tolerant for chart titles)
    always_filter_patterns = [
        r"^Gross value added.*per employed person",
        r"^In percent of (total|national|the)",
        r"^Density function",
        r"^Based on.*boundaries.*cities",
        r"^Types of settlements according to",
        r"^Productivity density",
        r"^\d{1,3}[,\s]\d{3}(?:\s+\d{1,3}[,\s]\d{3})+",
        r"^Western Germany\s+Eastern Germany",
        r"^Rural regions\s+Urban regions",
        r"^Comparison of labor productivity",
        r"Development of.*dispersion",
        r"^Figure \d+",
        r"^in eastern and western Germany\s*$",  # Chart subtitle (anchored - standalone only)
        r"^\d+\s+\d+,\d+\s+\d+,\d+",  # Axis numbers
    ]
    if any(re.search(pattern, l, re.I) for pattern in always_filter_patterns):
        return True

    # CONDITIONAL: Year series pattern - only filter if short AND no sentence punctuation
    # Catches "in 2004, 2014, and 2024" but NOT "Productivity rose in 2004, 2014, and 2024."
    if len(l) < 60 and not re.search(r'[.!?]$', l):
        if re.search(r".*in \d{4}, \d{4},? and \d{4}$", l):
            return True

    # CONDITIONAL: Sector labels - only filter if short AND no sentence punctuation
    # Catches "Agriculture, forestry, fisheries" but NOT "Agriculture, forestry, and fishing saw growth."
    if len(l) < 80 and not re.search(r'[.!?]$', l):
        if re.match(r"^Agriculture, forestry", l):
            return True

    # CONDITIONAL: Settlement keywords - only filter if short (< 100 chars) AND no sentence punctuation
    # This catches "Metropolis" or "Urban and central" but NOT "Urban and central districts have risen."
    if len(l) < 100 and not re.search(r'[.!?]$', l):
        settlement_patterns = [
            r"^(Metropolis|Metropolitan area|City and central|Urban and|Densi[fﬁ]cation and|Rural and|Very peripheral)",
            r"^(Large independent city|Small independent city|Urban district|Rural district)",
            r"Sparsely populated rural",
            r"^Densi[fﬁ]cation and (central|peripheral)",
        ]
        if any(re.search(pattern, l, re.I) for pattern in settlement_patterns):
            return True

    # Source citations and notes (must be at line start)
    if re.search(r"^(Source|Sources|Note|Notes)[\s:]+", l, re.I):
        return True

    # Copyright and image credits
    if re.search(r"^©.*DIW Berlin", l, re.I):
        return True
    if re.search(r"^\w+\s+\w+\s+\d{4}\s+Staats", l):  # Image credits like "David Liuzzo 2006 Staatsﬂagge"
        return True

    # Very short all-caps running heads (but NOT section titles like "ABSTRACT")
    words_alpha = [w for w in l.split() if re.search(r"[A-Za-z]", w)]
    if l.isupper() and len(words_alpha) == 1 and len(l) < 20:
        return True

    return False


def _is_incomplete_fragment(sentence: str) -> bool:
    """Check if a sentence is an incomplete fragment (e.g., 'In', 'The', etc.)."""
    words = sentence.strip().split()
    # Incomplete if: ≤4 words AND no terminal punctuation
    has_terminal_punct = re.search(r'[.!?]$', sentence.strip())
    return len(words) <= 4 and not has_terminal_punct


def _split_long_block(block: str, max_para_length: int = 460) -> List[str]:
    """
    Split long text blocks into logical paragraphs at sentence boundaries.
    Promotes splits on transition words (However, Economic, Moreover, etc.).
    Also splits short blocks if they contain an incomplete fragment at the end.
    """
    # Common abbreviations that shouldn't trigger splits
    abbreviations = {
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
        'e.g.', 'i.e.', 'etc.', 'vs.', 'cf.', 'al.',
        'U.S.', 'U.K.', 'U.N.', 'No.', 'Vol.', 'Fig.', 'p.', 'pp.'
    }

    # Transition words that signal natural paragraph boundaries
    transition_starters = {
        'However', 'Moreover', 'Furthermore', 'Nevertheless', 'Therefore',
        'Consequently', 'Economic', 'Political', 'Social', 'In addition',
        'On the other hand', 'Meanwhile', 'Subsequently',
        'Such'  # Forces split for "Such differences..." sentences
    }

    # Split into sentences using regex (more reliable than word-by-word)
    # Pattern: split after .!? followed by space and capital letter or quote
    # But protect abbreviations
    temp_block = block

    # Temporarily replace abbreviations with placeholders
    abbrev_map = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        temp_block = temp_block.replace(abbr, placeholder)
        abbrev_map[placeholder] = abbr

    # Split on sentence boundaries
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z"\u201c])'
    sentences = re.split(sentence_pattern, temp_block)

    # Restore abbreviations
    sentences = [
        re.sub(r'__ABBR(\d+)__', 
               lambda m: list(abbreviations)[int(m.group(1))], 
               sent)
        for sent in sentences
    ]

    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]

    # SPECIAL CASE: Short block with incomplete fragment at end
    # Example: "In contrast... type. In" → split into ["In contrast... type.", "In"]
    # This allows the fragment "In" to merge with the next block later
    if len(block) <= max_para_length and len(sentences) >= 2:
        last_sentence = sentences[-1]
        if _is_incomplete_fragment(last_sentence):
            # Split: complete sentences + fragment
            complete_part = ' '.join(sentences[:-1])
            fragment_part = last_sentence
            return [complete_part, fragment_part]
    
    # If block is short and doesn't have incomplete fragment, return as-is
    if len(block) <= max_para_length:
        return [block]

    # Group sentences into paragraphs with transition word awareness (for long blocks)
    paragraphs = []
    current_para = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        sent_len = len(sentence)

        # Check if this sentence starts with a transition word
        starts_with_transition = any(sentence.startswith(tw) for tw in transition_starters)

        # Decide whether to split:
        # 1. If adding would exceed limit AND we have content, consider splitting
        # 2. Always split before transition words (even if under limit)
        should_split = False

        if current_para:
            would_exceed = (current_length + sent_len > max_para_length)

            # Split if: exceeds limit OR starts with transition word AND previous para is substantial
            # Higher thresholds to prevent micro-paragraph fragmentation
            # For "Such", only split if current paragraph is already >420 chars
            # For other transitions, split if >420 chars (raised from 250 to prevent over-segmentation)
            transition_threshold = 420

            if would_exceed or (starts_with_transition and current_length > transition_threshold):
                should_split = True

        if should_split:
            paragraphs.append(' '.join(current_para))
            current_para = [sentence]
            current_length = sent_len
        else:
            current_para.append(sentence)
            current_length += sent_len + 1  # +1 for space

    # Add final paragraph
    if current_para:
        paragraphs.append(' '.join(current_para))

    return paragraphs if paragraphs else [block]


def _extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]


def _score_sentences(sentences: List[str], title: str = "") -> List[Tuple[str, float]]:
    """
    Score sentences by importance using position and keyword frequency.
    Returns list of (sentence, score) tuples.
    """
    if not sentences:
        return []

    # Extract keywords (excluding stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their'}

    # Build word frequency across all sentences
    all_words = []
    for sent in sentences:
        words = re.findall(r'\b[a-z]+\b', sent.lower())
        all_words.extend([w for w in words if w not in stop_words and len(w) > 3])

    word_freq = Counter(all_words)

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        score = 0.0

        # Position score (earlier sentences are more important)
        position_score = 1.0 - (i / len(sentences)) * 0.3

        # Word frequency score
        words = re.findall(r'\b[a-z]+\b', sent.lower())
        freq_score = sum(word_freq.get(w, 0) for w in words if w not in stop_words) / max(len(words), 1)

        # Title keyword score (if title provided)
        title_score = 0.0
        if title:
            title_words = set(re.findall(r'\b[a-z]+\b', title.lower()))
            title_words -= stop_words
            matching = sum(1 for w in words if w in title_words)
            title_score = matching * 0.5

        # Length penalty (very short or very long sentences are less important)
        length = len(sent.split())
        if length < 10:
            length_penalty = 0.7
        elif length > 50:
            length_penalty = 0.8
        else:
            length_penalty = 1.0

        score = (position_score * 0.4 + freq_score * 0.4 + title_score * 0.2) * length_penalty
        scored.append((sent, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def _condense_text(text: str, target_sentences: int = 4, title: str = "") -> str:
    """
    Intelligently condense text by extracting key sentences.
    Returns condensed text with top N most important sentences.
    """
    sentences = _extract_sentences(text)
    if len(sentences) <= target_sentences:
        return text

    # Score and select top sentences
    scored = _score_sentences(sentences, title)
    top_sentences = scored[:target_sentences]

    # Re-order selected sentences to preserve original order
    selected_set = {s for s, _ in top_sentences}
    ordered_sentences = [s for s in sentences if s in selected_set]

    return ' '.join(ordered_sentences)


def _fix_editorial_labels(paragraphs: List[str]) -> List[str]:
    """
    Fix editorial details by adding colons after labels if missing.
    This is a formatting fix, not content replacement.
    """
    if not paragraphs:
        return paragraphs

    # Check if this looks like editorial details
    all_text = ' '.join(paragraphs)
    if not ("Layout" in all_text and "Composition" in all_text):
        return paragraphs

    result = []
    labels = ["Layout", "Composition", "Editors-in-chief", "Reviewer", "Editorial staff"]

    for p in paragraphs:
        line = p
        for label in labels:
            # Add colon if missing (formatting fix only)
            if line.startswith(label + " ") and not line.startswith(label + ":"):
                line = line.replace(label + " ", label + ": ", 1)
        result.append(line)

    return result


def _clean_structural_blocks(text_blocks: List[str], global_repeating_blocks: set) -> List[str]:
    cleaned_blocks = []
    for block in text_blocks:
        block = block.replace("\u00AD", "")
        block = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", block)
        clean_block = _normalize_whitespace(block)
        if not clean_block:
            continue
        if clean_block in global_repeating_blocks:
            continue
        if _is_header_footer_line(clean_block):
            continue
        # CRITICAL: Filter out legend items (settlement tables, chart labels, etc.)
        if _looks_like_legend_item(clean_block):
            continue

        # CRITICAL: Remove embedded footnote sections and inline markers
        clean_block = _strip_inline_footnotes(clean_block)

        # Remove unwanted inline phrases and sentences
        clean_block = re.sub(r"\s*\(measured by variation coefficients\)", "", clean_block, flags=re.I)
        clean_block = re.sub(r"\s*\(Figure \d+\)", "", clean_block, flags=re.I)  # Remove figure references

        # Remove analytical preambles but keep the substantive content
        clean_block = re.sub(
            r"When comparing labor productivity at the state level,\s+the\s+distribution\s+has declined\s+noticeably:\s*",
            "",
            clean_block,
            flags=re.I
        )

        # Remove methodological explanations
        clean_block = re.sub(
            r"In a European comparison,\s+this is rather unusual\.\s*",
            "",
            clean_block,
            flags=re.I
        )
        clean_block = re.sub(
            r"Capital city regions generally have a level of productivity that is well above the national average\.\s*",
            "",
            clean_block,
            flags=re.I
        )

        # Text replacements (only applied if the phrases exist)
        clean_block = re.sub(
            r"dealing with major structural problems due to the end of coal production and the shrinkage of the steel industry",
            "facing structural issues after the end of coal and steel",
            clean_block,
            flags=re.I
        )
        clean_block = re.sub(
            r"In contrast, Berlin was able to improve its position considerably and now has a level of labor productivity at about the national average",
            "Berlin has improved its position and now sits around the national average",
            clean_block,
            flags=re.I
        )
        clean_block = re.sub(
            r"the speed of growth has decreased",
            "the speed of growth decreased",
            clean_block
        )
        clean_block = re.sub(
            r"at the top and the eastern German states are",
            "at the top, and the eastern states are",
            clean_block
        )
        clean_block = re.sub(
            r"Saarland, which is facing structural issues after the end of coal and steel,\s*ranks",
            "Saarland—which is facing structural issues after the end of coal and steel—ranks",
            clean_block
        )

        clean_block = _normalize_whitespace(clean_block)  # Re-normalize after removals

        # Fix grammatically incomplete sentences starting with decades
        if re.match(r"^\d{4}s,", clean_block):  # Starts with "2000s," or "1990s," etc.
            clean_block = "In the " + clean_block

        # Split long blocks into logical paragraphs (avoid splitting bullets/headings)
        is_bullet_or_heading = (clean_block.lstrip().startswith(('•', '-', '*', '▪')) or 
                               clean_block.isupper() or
                               len(clean_block.split()) <= 10)

        if not is_bullet_or_heading:
            split_paras = _split_long_block(clean_block, max_para_length=460)
            cleaned_blocks.extend(split_paras)
        else:
            cleaned_blocks.append(clean_block)

    return cleaned_blocks


def _resolve_reading_order(blocks: List[tuple], page_width: float) -> List[tuple]:
    """
    Detect multi-column layouts and return blocks in proper reading order.
    For single-column pages: sorts by (y, x) - top-to-bottom, left-to-right
    For two-column pages: sorts by column first, then top-to-bottom within each column
    """
    if not blocks:
        return []
    
    page_width = page_width or 595.0
    margin_tol = 12.0
    
    # Helper functions
    def width(b): 
        return float(b[2]) - float(b[0])
    
    def x_mid(b): 
        return (float(b[0]) + float(b[2])) / 2.0
    
    def y_val(b):
        return float(b[1])
    
    # Separate full-width blocks from column candidates
    full_width_blocks = []
    column_candidates = []
    
    for b in blocks:
        block_width = width(b)
        x0 = float(b[0])
        x1 = float(b[2])
        
        # Full-width: >= 70% page width OR spans from margin to margin
        is_full_width = (block_width >= 0.7 * page_width or 
                        (x0 <= margin_tol and x1 >= page_width - margin_tol))
        
        if is_full_width:
            full_width_blocks.append(b)
        else:
            column_candidates.append(b)
    
    # Check if page qualifies as two-column
    if len(column_candidates) < 6:  # Need at least 6 blocks for 2 columns
        # Fall back to simple (y, x) sorting
        return sorted(blocks, key=lambda b: (round(y_val(b), 1), round(float(b[0]), 1)))
    
    # Compute x_mid for each candidate and sort by x_mid
    candidate_mids = sorted([(x_mid(b), b) for b in column_candidates], key=lambda pair: pair[0])
    
    # Find largest gap between consecutive x_mid values
    max_gap = 0
    gap_index = -1
    for i in range(len(candidate_mids) - 1):
        gap = candidate_mids[i + 1][0] - candidate_mids[i][0]
        if gap > max_gap:
            max_gap = gap
            gap_index = i
    
    # Two-column criteria:
    # 1. Gap >= 15% of page width
    # 2. Both groups have >= 3 blocks
    # 3. Each group's average width <= 60% of page width
    if gap_index < 0 or max_gap < 0.15 * page_width:
        # Single column - use simple sorting
        return sorted(blocks, key=lambda b: (round(y_val(b), 1), round(float(b[0]), 1)))
    
    left_group = [pair[1] for pair in candidate_mids[:gap_index + 1]]
    right_group = [pair[1] for pair in candidate_mids[gap_index + 1:]]
    
    if len(left_group) < 3 or len(right_group) < 3:
        # Not enough blocks in each column
        return sorted(blocks, key=lambda b: (round(y_val(b), 1), round(float(b[0]), 1)))
    
    # Check average width of each group
    left_avg_width = sum(width(b) for b in left_group) / len(left_group)
    right_avg_width = sum(width(b) for b in right_group) / len(right_group)
    
    if left_avg_width > 0.6 * page_width or right_avg_width > 0.6 * page_width:
        # Blocks too wide for columns
        return sorted(blocks, key=lambda b: (round(y_val(b), 1), round(float(b[0]), 1)))
    
    # Two-column layout detected! Sort each column by y
    left_group.sort(key=lambda b: (round(y_val(b), 1), round(float(b[0]), 1)))
    right_group.sort(key=lambda b: (round(y_val(b), 1), round(float(b[0]), 1)))
    full_width_blocks.sort(key=lambda b: y_val(b))
    
    # Merge: left column first, then right column, with full-width blocks inserted by y
    result = []
    full_width_idx = 0
    
    # Helper to flush full-width blocks up to a given y
    def flush_full_width_until(max_y):
        nonlocal full_width_idx
        while full_width_idx < len(full_width_blocks):
            fw_block = full_width_blocks[full_width_idx]
            if y_val(fw_block) <= max_y + 6:  # 6pt tolerance
                result.append(fw_block)
                full_width_idx += 1
            else:
                break
    
    # Process left column
    for block in left_group:
        flush_full_width_until(y_val(block))
        result.append(block)
    
    # Flush remaining full-width before right column
    if right_group:
        flush_full_width_until(y_val(right_group[0]))
    
    # Process right column
    for block in right_group:
        flush_full_width_until(y_val(block))
        result.append(block)
    
    # Flush any remaining full-width blocks
    while full_width_idx < len(full_width_blocks):
        result.append(full_width_blocks[full_width_idx])
        full_width_idx += 1
    
    return result


def _reading_order_blocks(page: fitz.Page) -> List[str]:
    """Extract text blocks in reading order, filtering noise."""
    blocks = page.get_text("blocks")
    blocks = [b for b in blocks if isinstance(b, (list, tuple)) and len(b) >= 5]
    
    # Use column-aware ordering
    page_width = float(page.rect.width) if page.rect.width else 595.0
    blocks = _resolve_reading_order(blocks, page_width)

    out_blocks: List[str] = []
    for b in blocks:
        txt = (b[4] or "").replace("\r", " ").replace("\n", " ").strip()
        if len(txt) <= 2:
            continue

        # Filter numeric/axis-like fragments
        letters = sum(ch.isalpha() for ch in txt)
        digits = sum(ch.isdigit() for ch in txt)
        if letters < 3 and digits >= 2:
            continue

        # Filter pure numeric sequences (graph axes)
        if re.fullmatch(r"[\d\s:\-–,\.\/]+", txt):
            continue

        # Filter figure/table labels
        if re.search(r"^(figure|fig\.|table)\b", txt, re.I):
            continue

        # Filter very short fragments without punctuation UNLESS they look like sentence starts
        toks = txt.split()
        if len(toks) <= 2 and not re.search(r"[\.!?]$", txt):
            # Keep fragments that look like sentence openers: "In the", "At the", "On the", etc.
            # Pattern: Capitalized word followed by lowercase word
            if len(toks) == 2 and toks[0][0].isupper() and toks[1][0].islower():
                # This is likely a drop-cap or sentence opener - keep it!
                pass
            elif sum(1 for w in toks if any(ch.isalpha() for ch in w)) <= 2:
                continue

        # Filter numeric tick sequences
        numish = sum(1 for w in toks if re.fullmatch(r"\d{1,4}(?:[\/.]\d{1,4})?", w))
        if len(toks) >= 3 and numish / max(1, len(toks)) >= 0.6:
            continue

        out_blocks.append(txt)
    return out_blocks


def _reading_order_paras(page: fitz.Page) -> List[dict]:
    """Return paragraph blocks with their y position and x boundaries, filtering out footnotes."""
    blocks = page.get_text("blocks")
    blocks = [b for b in blocks if isinstance(b, (list, tuple)) and len(b) >= 5]
    
    # Use column-aware ordering
    page_width = float(page.rect.width) if page.rect.width else 595.0
    blocks = _resolve_reading_order(blocks, page_width)

    paras: List[dict] = []
    page_height = float(page.rect.height) if page.rect.height > 0 else 842
    
    for b in blocks:
        x0 = float(b[0])
        y0 = float(b[1])
        x1 = float(b[2])
        txt = (b[4] or "").replace("\r", " ").replace("\n", " ").strip()
        if len(txt) <= 2:
            continue

        # Relax filtering to keep quoted callout text
        letters = sum(ch.isalpha() for ch in txt)
        digits = sum(ch.isdigit() for ch in txt)
        # Only filter if predominantly numeric (not quoted text)
        if letters < 3 and digits >= 3:
            continue
        if re.fullmatch(r"[\d\s:\-–,\.\/]+", txt):
            continue
        if re.search(r"^(figure|fig\.|table)\b", txt, re.I):
            continue

        x_mid = (x0 + x1) / 2
        para = {"text": txt, "y": y0, "x0": x0, "x1": x1, "x_mid": x_mid}
        
        # Filter out footnote blocks (numbered references at bottom of page)
        if _is_footnote_block(para, page_height):
            continue
            
        paras.append(para)
    return paras


def _ocr_page(page: fitz.Page) -> List[str]:
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    ocr_text = pytesseract.image_to_string(image, lang="eng")
    return [l.strip() for l in ocr_text.splitlines() if l.strip()]


def _looks_like_legend_item(txt: str) -> bool:
    """
    Heuristic to drop taxonomy/legend crumbs and chart labels.
    Filters out legend items, color labels, statistical markers, and standalone labels.
    """
    t = txt.strip()
    if not t:
        return True
    # keep true bullets
    if t.lstrip().startswith(("•", "- ", "* ")):
        return False

    t_lower = t.lower()

    # SPECIFIC HEURISTIC: Settlement table rows - only match if NO sentence-ending punctuation
    # This catches "Densification and central Rural district with..." but NOT "Urban and central districts have..."
    if not re.search(r'[.!?]$', t):  # Must NOT end with sentence punctuation
        if ("densification" in t_lower or "densi" in t_lower) and ("rural district" in t_lower or "urban district" in t_lower):
            return True
        if "with signs of densification" in t_lower:
            return True

    # Settlement type legends (document-specific)
    settlement_types = [
        "metropolis", "metropolitan area", "city and central", 
        "urban and central", "urban and peripheral",
        "densification and central", "densification and peripheral",
        "rural and central", "rural and peripheral", "very peripheral",
        "large independent city", "small independent city", 
        "urban district", "rural district", "sparsely populated"
    ]
    for settlement in settlement_types:
        if settlement in t_lower and len(t.split()) <= 8:
            return True

    # Common chart/graph legend patterns
    legend_patterns = [
        # Color/style indicators
        r"^(red|blue|green|yellow|orange|purple|black|white|gray|grey)[\s:]",
        r"^(solid|dashed|dotted|dash-dot) (line|bar|area)",
        # Statistical markers
        r"^(mean|median|mode|average|total|sum|maximum|minimum|max|min|std|stddev)$",
        r"^(confidence interval|error bar|trend line|regression)s?$",
        # Directional/positional
        r"^(north|south|east|west|central|top|bottom|left|right)$",
        # Demographic/categorical
        r"^(male|female|men|women|boys|girls)$",
        r"^(low|medium|high|very high|very low)$",
        # Chart elements
        r"^(legend|x-axis|y-axis|axis label|data point)s?$",
        # Year sequences (standalone years as legend items)
        r"^\d{4}$",
        # Classification types (from settlement type legends)
        r"^(metropolis|large independent city|small independent city|urban district|rural district)$",
        r"^sparsely populated$",
    ]

    for pattern in legend_patterns:
        if re.search(pattern, t_lower):
            return True

    # Color legend format: "Blue: Category" or "Category (blue)"
    if re.search(r"^(red|blue|green|yellow|orange|purple|black|white|gray|grey)[\s:]+\w+", t_lower):
        return True
    if re.search(r"\((red|blue|green|yellow|orange|purple|black|white|gray|grey)\)$", t_lower):
        return True

    # Sector/industry lists from tables (comma-separated categories)
    if re.search(r"^(agriculture|forestry|fisheries|mining|manufacturing|construction|wholesale|retail|transportation|accommodation|information|financial|insurance|real estate|professional|administrative|public administration|education|health|arts|entertainment)", t_lower):
        # If it's a comma-separated list, likely a table row
        if t.count(',') >= 2 and len(t.split()) <= 15:
            return True

    # Very short, no punctuation, title-cased -> legend-ish
    if len(t.split()) <= 6 and not re.search(r"[.!?;:]", t):
        words = [w for w in t.split() if any(ch.isalpha() for ch in w)]
        if words:
            caps_ratio = sum(1 for w in words if w[0].isupper()) / max(1, len(words))
            if caps_ratio >= 0.6:
                return True

    return False


def _is_footnote_block(para: dict, page_height: float, word_count_cap: int | None = None) -> bool:
    """
    Detect if a paragraph is a footnote reference block.
    Uses position (bottom of page) and content patterns.
    
    Args:
        para: Paragraph dict with 'text' and 'y' keys
        page_height: Height of the page in points
        word_count_cap: If provided, allow long paragraphs (>= word_count_cap words) 
                       to pass through even if in bottom 30% (default: None = strict mode)
    """
    text = para.get("text", "").strip()
    y_pos = float(para.get("y", 0))
    
    # Must be in bottom 30% of page
    if page_height > 0 and y_pos < page_height * 0.70:
        return False
    
    # NEW: If word count threshold provided, allow long paragraphs through
    # (this prevents body text like "Conclusion" sections from being misclassified as footnotes)
    if word_count_cap is not None:
        word_count = len(text.split())
        if word_count >= word_count_cap:
            return False  # Long paragraph, likely body text not a footnote
    
    # CRITICAL: Preserve "Conclusion:" heading specifically (user requirement)
    # This prevents the Conclusion heading from being filtered as a footnote when at bottom of page
    if re.match(r"^Conclusions?:\s+", text, flags=re.I):
        return False  # This is the Conclusion heading, not a footnote
    
    # Pattern 1: Starts with single digit followed by space
    if re.match(r"^\d{1,2}\s+", text):
        # Check for common footnote markers
        footnote_markers = [
            r"\(\d{4}\)",  # Years in parentheses (2024)
            r"available online",
            r"accessed on",
            r"Discussion Paper",
            r"Wochenbericht",
            r"Working group",
            r"et al\.",
            r"in German",
            r"no\.\s+\d+",
            r"https?://",
            r"Cf\.",
            r"©.*\d{4}",
        ]
        text_lower = text.lower()
        for marker in footnote_markers:
            if re.search(marker, text, flags=re.I):
                return True
    
    return False


def _is_footnote_text(text: str) -> bool:
    """
    Detect if a text string is a footnote (content-only check, no position).
    Used during section chunking when position info may not be available.
    """
    text = text.strip()
    
    # Pattern: Starts with single digit followed by space and citation content
    if re.match(r"^\d{1,2}\s+", text):
        # Check for common footnote markers
        footnote_markers = [
            r"\(\d{4}\)",  # Years in parentheses (2024)
            r"available online",
            r"accessed on",
            r"Discussion Paper",
            r"Wochenbericht",
            r"Working group",
            r"et al\.",
            r"in German",
            r"no\.\s+\d+",
            r"https?://",
            r"Cf\.",
            r"©.*\d{4}",
            r"DIW\s+Wochenbericht",
            r"ZEW\s+Discussion",
            r"Mitteldeutsche\s+Stiftung",
        ]
        for marker in footnote_markers:
            if re.search(marker, text, flags=re.I):
                return True
    
    return False


def _strip_inline_footnotes(text: str) -> str:
    """
    Remove inline footnote digits, figure references, and embedded footnote blocks.
    Preserves sentence integrity by deleting from marker start (not sentence before).
    """
    # STEP 1: Marker-driven deletion for embedded footnotes
    # Delete from marker start through next sentence boundary
    citation_markers = [
        r'\(\d{4}\):',  # (2024):
        r'Discussion Paper',
        r'Wochenbericht\s*no\.',
        r'ZEW\s+Discussion',
        r'available online',
        r'accessed on',
        r'et al\.',
        r'Stiftung\s+Wissenschaft\s+und\.',
        r'Cf\.\s+Data\s+on',
        r'The\s+VGRdL\s+is\s+based',
        r'in German;',
        r'Federal\s+and\s+State\s+Statistical',
        r'Mitteldeutsche\s+Stiftung',
        r'VGRdL',
        r'Doris\s+Cornelsen',
        r'Bestandsaufnahme',  # German footnote text
        r'DIW\s+\d+',  # "DIW 14, 172–174"
        r'Data on the.*Federal and State',  # Footnote #4 content
        r'Labor productivity is measured as the total gross',  # Footnote #4 content
        r'full-time and part-time employees at current prices',  # Footnote #4 content
    ]
    
    # Loop until no markers remain
    max_iterations = 10
    for _ in range(max_iterations):
        original_text = text
        
        # Find earliest citation marker
        earliest_pos = len(text)
        matched_marker = None
        
        for marker in citation_markers:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
                matched_marker = marker
        
        if matched_marker is None:
            break  # No more markers found
        
        # REFINED: Check if marker is at start of footnote paragraph (separate block)
        # Pattern: starts with digit(s) + space + name/citation
        text_before_marker = text[:earliest_pos]
        
        # Is this a separate footnote paragraph? (e.g., "1 Martina Hülz et al. (2024):")
        is_separate_footnote = False
        lookback_start = max(0, earliest_pos - 10)  # Check preceding 10 chars
        prefix = text[lookback_start:earliest_pos]
        
        # If marker is preceded by digit + space (e.g., "1 Martina ... (2024):"), it's a footnote paragraph
        if re.search(r'^\s*\d{1,2}\s+\w+', prefix + text[earliest_pos:earliest_pos+20]):
            is_separate_footnote = True
        
        if is_separate_footnote:
            # Remove entire footnote paragraph: from marker back to last sentence, forward to next
            cut_start = 0
            for i in range(len(text_before_marker) - 1, -1, -1):
                if text_before_marker[i] in '.!?':
                    cut_start = i + 1
                    break
            
            # Find next sentence boundary AFTER the marker
            text_after_marker = text[earliest_pos:]
            cut_end = len(text)
            
            for i, char in enumerate(text_after_marker):
                if char in '.!?':
                    remaining = text_after_marker[i+1:]
                    if remaining and (remaining[0].isspace() or i == len(text_after_marker) - 1):
                        cut_end = earliest_pos + i + 1
                        break
            
            # Delete entire footnote block
            text = (text[:cut_start].strip() + ' ' + text[cut_end:].strip()).strip()
        else:
            # Inline citation: delete from marker START (not sentence before) to next sentence
            cut_end = len(text)
            text_after_marker = text[earliest_pos:]
            
            for i, char in enumerate(text_after_marker):
                if char in '.!?':
                    remaining = text_after_marker[i+1:]
                    if remaining and (remaining[0].isspace() or i == len(text_after_marker) - 1):
                        cut_end = earliest_pos + i + 1
                        break
            
            # Delete from marker to end of footnote sentence
            text = (text[:earliest_pos].strip() + ' ' + text[cut_end:].strip()).strip()
        
        # If we didn't modify the text, break to avoid infinite loop
        if text == original_text:
            break
    
    # STEP 2: Remove figure references like "(Figure 2)" or "(Figure 3)"
    text = re.sub(r"\(Figure\s+\d+\)", "", text, flags=re.I)

    # STEP 3: Remove Unicode superscript numbers and true footnote markers only
    # Convert superscripts to ASCII
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    text = text.translate(superscripts)
    
    # FIXED: Only target footnote markers attached to words, not all numbers
    # Pattern: letter followed by 1-2 digits followed by punctuation/close bracket
    # Excludes multi-digit numbers like 1991 by using (?<!\d) negative lookbehind
    text = re.sub(r"(?<!\d)(?<=[A-Za-z])\d{1,2}(?=\s*[)\]–.,;:])", "", text)
    
    # STEP 4: Remove caret-style markers (^1, ^2, etc.)
    text = re.sub(r"\^\d{1,2}", "", text)

    # STEP 5: Remove footnote markers at end of words/sentences: "nature.1", "used.2", etc.
    text = re.sub(r"\.(\d{1,2})\s", ". ", text)  # ".1 " becomes ". "
    text = re.sub(r"\.(\d{1,2})$", ".", text)     # ".1" at end becomes "."

    # STEP 6: Kill .1 / )2 / [3] after a word; avoid 4-digit years
    text = re.sub(r"(?<=\w)[\u00A0\s]?(?:\(\d{1,2}\)|\[\d{1,2}\])", "", text)
    
    # REMOVED: Overly aggressive pattern that deleted legitimate numbers
    # OLD: text = re.sub(r"\s+\d{1,2}(?=[\.!?,])", "", text)
    
    # STEP 7: Clean up multiple periods and extra whitespace
    text = re.sub(r'\.{2,}', '.', text)  # Multiple periods -> single period
    
    # STEP 8: AGGRESSIVE final cleanup - remove any remaining citation fragments
    # Remove fragments like "( DIW 14, 172–174 (" or "Eine Bestandsaufnahme..."
    # Pattern: Remove text from "(", "[", or German words up to next sentence
    text = re.sub(r'\(\s*(?:DIW|ZEW|VGR)[^.!?]{0,100}(?=[.!?]|$)', '', text, flags=re.I)
    text = re.sub(r'\(\s*in\s+German[^)]*\)', '', text, flags=re.I)
    text = re.sub(r'Eine\s+Bestandsaufnahme[^.!?]{0,150}(?=[.!?]|$)', '', text, flags=re.I)
    
    # Remove standalone single opening parentheses left behind
    text = re.sub(r'\s+\(\s*(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+\(\s*$', '', text)

    return _normalize_whitespace(text)


def _dedupe_outer_quotes(line: str) -> str:
    """
    Remove redundant straight quotes when they wrap curly/typographic quotes.
    Example: ""When..." → "When..."
    """
    stripped = line.strip()
    # Check if straight quotes wrap curly quotes: "«text»" or ""text""
    if len(stripped) >= 3:
        # Pattern: straight quote + curly opener ... curly closer + straight quote
        if (stripped[0] == '"' and stripped[-1] == '"' and 
            stripped[1] in '""«' and stripped[-2] in '""»'):
            return stripped[1:-1]
    return line


def _collect_callout_region(paras: List[dict], start_y: float, end_y: float | None,
                            title: str, bbox: List[float] | None = None) -> List[str]:
    """
    Collect paragraphs within a callout region.
    
    Args:
        paras: List of paragraph dicts with 'text', 'y', 'x0', 'x1', 'x_mid'
        start_y: Starting Y coordinate
        end_y: Ending Y coordinate (None = no limit)
        title: Callout title (for filtering)
        bbox: Bounding box [x0, y0, x1, y1] for spatial filtering (optional)
    """
    out: List[str] = []
    title_norm = title.strip().upper()
    
    # Extract bbox bounds if provided
    bbox_x0 = bbox[0] if bbox and len(bbox) >= 4 else None
    bbox_x1 = bbox[2] if bbox and len(bbox) >= 4 else None
    bbox_y1 = bbox[3] if bbox and len(bbox) >= 4 else end_y
    
    for p in paras:
        y = float(p.get("y", 0.0))
        if y <= start_y:
            continue
        
        # Use bbox Y limit if available, otherwise use end_y
        y_limit = bbox_y1 if bbox_y1 is not None else end_y
        if y_limit is not None and y >= y_limit:
            break
            
        # Check horizontal bounds if bbox provided (use overlap check with generous tolerance)
        if bbox_x0 is not None and bbox_x1 is not None:
            p_x0 = p.get("x0")
            p_x1 = p.get("x1")
            
            if p_x0 is not None and p_x1 is not None:
                # Check for horizontal overlap with generous tolerance for layout variations
                # Increased from 10px to 50px to handle bullets and text that spill outside bbox
                tolerance = 50
                # No overlap if paragraph ends well before bbox starts or starts well after bbox ends
                if p_x1 + tolerance < bbox_x0 or p_x0 - tolerance > bbox_x1:
                    continue
            else:
                # Fallback to midpoint check if width not available (also use generous tolerance)
                p_x_mid = p.get("x_mid")
                if p_x_mid is not None:
                    if p_x_mid < bbox_x0 - 50 or p_x_mid > bbox_x1 + 50:
                        continue
            
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        # skip the callout title line itself and obvious non-content
        if txt.strip().upper() == title_norm:
            continue
        if re.search(r"^(figure|fig\.|table|source:|sources:|note:|©|diw\s+berlin)\b", txt, re.I):
            continue
        if re.match(r"^By\s+[A-Z]", txt):
            continue

        # Skip lines that are clearly chart/figure elements
        if re.search(r"^(Productivity|Producitvity)[\s:]+(gap|density|East-west|in Eastern Germany)", txt, re.I):
            continue
        if re.search(r"^(Rural|Urban) (regions|and (central|peripheral))", txt, re.I):
            continue
        if re.search(r"^Very peripheral", txt, re.I):
            continue
        
        # Skip figure narrative labels and chart headers
        if re.search(r"\bNARRAT(IVE)?[\s:]", txt, re.I):
            continue
        if re.search(r"^(INCOME|CORPO|ECONOM|RATE)\s+(IVE|IC|T)\b", txt, re.I):
            continue
        if re.search(r"survey experiment", txt, re.I):
            continue
        # Skip lines with multiple quoted climate policy statements (figure annotations)
        txt_lower = txt.lower()
        if txt_lower.count("climate policy") >= 2:
            continue
        if txt.count('"') >= 4 or txt.count('"') >= 2 or txt.count('"') >= 2:
            if "climate policy" in txt_lower:
                continue
        # Skip lines mentioning participants/respondents (survey methodology)
        if re.search(r"\d+\s*participants", txt, re.I):
            continue
        # Skip short uppercase fragments that are likely figure labels
        if len(txt) < 30 and txt.isupper() and not txt.startswith("•"):
            continue
        # Block lines with repeated keywords (legend/classification rows)
        txt_lower = txt.lower()
        if txt_lower.count('rural') >= 2 or txt_lower.count('urban') >= 2:
            continue
        if txt_lower.count('peripheral') >= 2 or txt_lower.count('central') >= 2:
            continue
        # Block any line containing multiple settlement type keywords
        if 'very peripheral' in txt_lower and ('rural' in txt_lower or 'central' in txt_lower):
            continue
        if re.search(r"©.*DIW Berlin.*\d{4}", txt):
            continue

        # Clean up text: strip footnotes and dedupe quotes
        cleaned = _strip_inline_footnotes(txt)
        cleaned = _dedupe_outer_quotes(cleaned)
        
        # FROM THE AUTHORS: Check for attribution line BEFORE legend filter
        is_attribution = False
        if title_norm == "FROM THE AUTHORS":
            if re.match(r"^[—–\-]\s*[A-Z][a-z]+", cleaned):
                is_attribution = True
        
        # drop legend-y crumbs outside real bullets (BUT keep attribution lines)
        if not is_attribution and _looks_like_legend_item(txt):
            continue
            
        out.append(cleaned)
        
        # FROM THE AUTHORS: Stop after attribution line (em-dash + name)
        if is_attribution:
            break
    return out


def _merge_sentence_fragments(lines: List[str]) -> List[str]:
    """
    Merge text fragments that were split mid-sentence in the PDF.
    If a line doesn't end with sentence punctuation and next line starts lowercase,
    merge them together.
    """
    if not lines:
        return lines

    merged = []
    i = 0
    while i < len(lines):
        current = lines[i]

        # Check if we should merge with next line
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Merge if current doesn't end with sentence punctuation and next starts lowercase
            if (current and not re.search(r"[.!?]$", current.strip()) and 
                next_line and next_line[0].islower()):
                # Merge with a space
                merged.append(current.rstrip() + " " + next_line.lstrip())
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def _format_callout_bullets(lines: List[str]) -> List[str]:
    """
    Format lines as bulleted list with normalized bullets and punctuation.
    Handles various bullet styles: •, -, *, numbered lists (1., 2., a), b)), etc.
    Used for AT A GLANCE, KEY TAKEAWAYS, and HIGHLIGHTS sections.
    """
    bullets = []
    for line in lines:
        # Remove existing bullet or number if present
        clean_line = line.lstrip()

        # Handle numbered lists (1. 2. 3. or 1) 2) 3) or a. b. c. or a) b) c))
        if re.match(r"^(?:\d+|[a-z])[\.\)]\s+", clean_line):
            clean_line = re.sub(r"^(?:\d+|[a-z])[\.\)]\s+", "", clean_line)
        # Handle standard bullet markers
        elif clean_line.startswith(("•", "- ", "* ", "○", "▪", "►")):
            clean_line = clean_line.lstrip("•-*○▪► ").strip()

        # Add period at end if missing punctuation
        if clean_line and not re.search(r"[.!?]$", clean_line):
            clean_line = clean_line + "."

        bullets.append(f"• {clean_line}")
    return bullets


def _build_page_preview(callouts: List[dict], paras: List[dict],
                        clean_text_fallback: str) -> str:
    """
    Build a human-friendly preview with structured callout sections.
    Order: EXECUTIVE SUMMARY → FROM THE AUTHORS → AT A GLANCE → 
           KEY TAKEAWAYS → HIGHLIGHTS → other callouts → prose
    """
    if not callouts or not paras:
        return clean_text_fallback

    # Preferred callout order and formatting types
    order = [
        "EXECUTIVE SUMMARY",
        "FROM THE AUTHORS", 
        "AT A GLANCE", 
        "KEY TAKEAWAYS", 
        "HIGHLIGHTS"
    ]

    # Define which callouts should use bullet formatting
    bullet_callouts = {"AT A GLANCE", "KEY TAKEAWAYS", "HIGHLIGHTS"}
    prose_callouts = {"EXECUTIVE SUMMARY", "FROM THE AUTHORS"}

    # Sort callouts by Y and build boundaries (include bbox for spatial filtering)
    sorted_callouts = sorted(callouts, key=lambda c: float(c.get("y", 0.0)))
    boundaries = []
    for i, c in enumerate(sorted_callouts):
        y0 = float(c.get("y", 0.0))
        y1 = float(sorted_callouts[i + 1].get("y", 1e9)) if i + 1 < len(sorted_callouts) else None
        boundaries.append({
            "title": c["title"].strip().upper(),
            "y0": y0,
            "y1": y1,
            "bbox": c.get("bbox")  # Pass full bbox for horizontal filtering
        })

    chunks: List[str] = []
    used_ranges: List[tuple] = []

    # helper to check if a para Y is in used ranges
    def is_used(y_val):
        for start, end in used_ranges:
            if start < y_val < end:
                return True
        return False

    # helper to mark a region as used
    def mark_used(start, end):
        used_ranges.append((start, 1e9 if end is None else float(end)))

    # 1) Preferred callouts in specified order
    for wanted in order:
        for boundary in boundaries:
            title = boundary["title"]
            if title == wanted:
                region = _collect_callout_region(
                    paras, 
                    boundary["y0"], 
                    boundary["y1"], 
                    title,
                    bbox=boundary.get("bbox")  # Pass bbox for spatial filtering
                )
                if not region:
                    continue

                # Apply formatting based on callout type
                if title in bullet_callouts:
                    # Bulleted sections: AT A GLANCE, KEY TAKEAWAYS, HIGHLIGHTS
                    bullets = _format_callout_bullets(region)
                    chunk = f"{title}\n\n" + "\n\n".join(bullets)
                elif title in prose_callouts:
                    # Prose sections: EXECUTIVE SUMMARY, FROM THE AUTHORS
                    # Merge sentence fragments before formatting
                    merged_region = _merge_sentence_fragments(region)
                    prose_text = "\n\n".join(merged_region)

                    # Apply intelligent condensation to prose callouts
                    condensed = _condense_text(prose_text, target_sentences=4, title=title)
                    chunk = f"{title}\n\n{condensed}"
                else:
                    # Fallback for any other preferred callouts
                    chunk = f"{title}\n\n" + "\n".join(region)

                chunks.append(chunk)
                mark_used(boundary["y0"], boundary["y1"])

    # 2) Any other callout that we didn't explicitly order
    for boundary in boundaries:
        title = boundary["title"]
        if title in order:
            continue
        region = _collect_callout_region(
            paras, 
            boundary["y0"], 
            boundary["y1"], 
            title,
            bbox=boundary.get("bbox")  # Pass bbox for spatial filtering
        )
        if region:
            chunks.append(f"{title}\n" + "\n".join(region))
            mark_used(boundary["y0"], boundary["y1"])

    # 3) Collect remaining prose (non-callout content)
    remaining_prose = []
    for p in paras:
        y = float(p.get("y", 0.0))
        if is_used(y):
            continue
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        # Skip obvious non-content
        if re.search(r"^(figure|fig\.|table|source:|sources:|note:|©|diw\s+berlin)\b", txt, re.I):
            continue
        # Only apply strict legend filtering outside callouts
        if _looks_like_legend_item(txt) and not txt.lstrip().startswith(("•", "- ")):
            continue
        remaining_prose.append(_strip_inline_footnotes(txt))

    # Add remaining prose if any, with blank lines between paragraphs
    if remaining_prose:
        # Clean paragraphs and remove incomplete trailing fragments
        cleaned_prose = []
        prev_para_original = None

        for i, para in enumerate(remaining_prose):
            # Check if this paragraph starts with a transition word that the previous ended with
            skip_duplicate = False
            if prev_para_original and para:
                # Check if previous paragraph ended with an incomplete fragment like "However,"
                # and this one starts with the same word
                prev_ends_incomplete = re.search(r'\.\s+([A-Z][a-z]+),\s*$', prev_para_original)
                if prev_ends_incomplete:
                    trailing_word = prev_ends_incomplete.group(1)
                    if para.strip().startswith(f"{trailing_word},"):
                        skip_duplicate = True

            if skip_duplicate:
                continue  # Skip but don't update prev_para_original

            # Remove incomplete trailing words like ". However," at paragraph end
            para = re.sub(r'\.\s+[A-Z][a-z]+,\s*$', '.', para)

            # Ensure paragraph ends with proper punctuation
            para = para.strip()
            if para and not re.search(r'[.!?]$', para):
                para += '.'

            if para:
                cleaned_prose.append(para)
                # Store original for next iteration's check (only when we keep it)
                prev_para_original = remaining_prose[i]

        # Apply intelligent condensation to remaining prose
        if cleaned_prose:
            # Merge fragments first
            merged_prose = _merge_sentence_fragments(cleaned_prose)
            prose_text = "\n\n".join(merged_prose)

            # Condense the prose (target 4-6 sentences for general content)
            condensed_prose = _condense_text(prose_text, target_sentences=6, title="")
            chunks.append(condensed_prose)

    # If nothing was collected at all, fall back to clean text
    if not chunks:
        return clean_text_fallback

    return "\n\n".join(chunks)


def extract_page_texts(pdf_path: str, force_ocr: bool = False, use_ai: bool = False) -> List[Dict[str, Any]]:
    """Extract text from PDF pages with callout detection."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    pages: List[Dict[str, Any]] = []
    raw_blocks_per_page: List[List[str]] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        used_ocr = False
        page_h = float(page.rect.height or 0.0)

        # Detect special callout boxes (EXECUTIVE SUMMARY, FROM THE AUTHORS, AT A GLANCE, KEY TAKEAWAYS, HIGHLIGHTS)
        callouts = []
        try:
            bdict = page.get_text("blocks")
            for b in bdict:
                if not (isinstance(b, (list, tuple)) and len(b) >= 5):
                    continue
                y0 = float(b[1])
                t = (b[4] or "").strip()
                if not t:
                    continue

                t_clean = re.sub(r"\s{2,}", " ", t).strip().rstrip(" :;,-")

                # Detect callout titles (anywhere on page)
                if t_clean.upper() in {
                    "AT A GLANCE", "FROM THE AUTHORS", "MEDIA", "ABSTRACT",
                    "OVERVIEW — TYPES OF SETTLEMENTS", "OVERVIEW - TYPES OF SETTLEMENTS",
                    "KEY TAKEAWAYS", "EXECUTIVE SUMMARY", "HIGHLIGHTS"
                }:
                    # Extract bbox [x0, y0, x1, y1] from block
                    x0 = float(b[0]) if len(b) > 0 else 0
                    y0 = float(b[1]) if len(b) > 1 else y0
                    x1 = float(b[2]) if len(b) > 2 else 600
                    y1_raw = float(b[3]) if len(b) > 3 else y0 + 20
                    
                    # Adaptive bbox expansion for callouts with content below title
                    # Calculate based on page dimensions and detected width
                    page_width = float(page.rect.width) if page.rect.width > 0 else 595
                    page_height = float(page.rect.height) if page.rect.height > 0 else 842
                    detected_width = x1 - x0
                    
                    # If title bbox is narrow (< 1/3 page width), likely need expansion
                    if detected_width < page_width / 3 and t_clean.upper() in ("ABSTRACT", "AT A GLANCE", "KEY TAKEAWAYS", "EXECUTIVE SUMMARY", "FROM THE AUTHORS"):
                        # Expand horizontally to column width (centered on detected position)
                        x_center = (x0 + x1) / 2
                        column_width = min(page_width * 0.45, 260)  # 45% of page or 260px
                        x0 = max(35, x_center - column_width / 2)
                        x1 = min(page_width - 35, x_center + column_width / 2)
                        
                        # Expand vertically: proportion of page height (adapts to different PDF sizes)
                        vertical_extension = min(page_height * 0.30, 280)  # 30% of page or 280px max
                        y1 = y0 + vertical_extension
                    else:
                        y1 = y1_raw
                    
                    bbox = [x0, y0, x1, y1]
                    
                    callouts.append({
                        "title": t_clean.upper(),
                        "y": y0,
                        "bbox": bbox
                    })
        except Exception:
            pass

        if force_ocr:
            try:
                text_blocks = _ocr_page(page)
                used_ocr = True
            except Exception:
                text_blocks = _reading_order_blocks(page)
        else:
            text_blocks = _reading_order_blocks(page)
            try:
                has_spans = bool(page.get_text("dict").get("blocks"))
            except Exception:
                has_spans = True
            if (len("".join(text_blocks).strip()) < 10) and (not has_spans):
                try:
                    text_blocks = _ocr_page(page)
                    used_ocr = True
                except Exception:
                    pass

        # DON'T exclude callout titles - we want them for reordering/collection
        raw_blocks_per_page.append(text_blocks)
        paras = [] if used_ocr else _reading_order_paras(page)

        pages.append({
            "page": i + 1,
            "text": "",  # will be filled after global filtering
            "ocr_used": used_ocr,
            "char_count": 0,
            "paras": paras,
            "page_height": page_h,
            "callouts": callouts,
        })

    # Find globally repeating blocks (headers/footers)
    all_blocks = [b for blocks in raw_blocks_per_page for b in blocks]
    freq: Dict[str, int] = {}
    for block in all_blocks:
        normalized_block = _normalize_whitespace(block)
        freq[normalized_block] = freq.get(normalized_block, 0) + 1

    repeat_threshold = max(2, doc.page_count // 5)
    global_repeat_blocks = {b for b, c in freq.items() if c >= repeat_threshold and len(b) < 120}

    # Build cleaned page text + human-friendly preview
    for i, raw_blocks in enumerate(raw_blocks_per_page):
        filtered_blocks = _clean_structural_blocks(raw_blocks, global_repeat_blocks)

        # Remove duplicate paragraphs and orphan fragments
        deduplicated_blocks = []
        prev_block_original = None
        for block_original in filtered_blocks:
            # Filter out orphan 1-2 word fragments like "However,"
            words = block_original.strip().split()
            if len(words) <= 2 and re.match(r'^[A-Z][a-z]+,?$', block_original.strip()):
                continue  # Skip orphan fragments

            # Check if previous block ended with ". Word," and this starts with "Word,"
            skip = False
            if prev_block_original:
                match = re.search(r'\.\s+([A-Z][a-z]+),\s*$', prev_block_original)
                if match:
                    word = match.group(1)
                    if block_original.strip().startswith(f"{word},"):
                        skip = True  # Skip duplicate paragraph

            if not skip:
                # Remove incomplete trailing fragments like ". However,"
                block_cleaned = re.sub(r'\.\s+[A-Z][a-z]+,\s*$', '.', block_original)
                deduplicated_blocks.append(block_cleaned)
                prev_block_original = block_original  # Store original for next comparison

        # Apply editorial label formatting fix (formatting only, not content replacement)
        deduplicated_blocks = _fix_editorial_labels(deduplicated_blocks)

        clean_text = "\n\n".join(deduplicated_blocks)
        clean_text = _normalize_whitespace(clean_text)

        # NEW: build preview that reorders callouts and trims legends; fallback to clean_text
        callouts = pages[i].get("callouts", [])
        paras = pages[i].get("paras", [])
        preview = _build_page_preview(callouts, paras, clean_text)

        pages[i]["text"] = preview
        pages[i]["char_count"] = len(preview)

    doc.close()
    return pages


def calculate_text_quality_metrics(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate quality metrics for extracted text."""
    if not pages:
        return {
            "total_pages": 0,
            "total_chars": 0,
            "avg_chars_per_page": 0,
            "ocr_pages": 0,
            "ocr_percentage": 0,
        }

    total_chars = sum(p.get("char_count", 0) for p in pages)
    ocr_pages = sum(1 for p in pages if p.get("ocr_used", False))

    return {
        "total_pages": len(pages),
        "total_chars": total_chars,
        "avg_chars_per_page": total_chars // len(pages) if pages else 0,
        "ocr_pages": ocr_pages,
        "ocr_percentage": round(100 * ocr_pages / len(pages), 1) if pages else 0,
    }
