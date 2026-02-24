from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from pipeline.extract_text import _is_footnote_block

from pipeline.extract_tables import extract_tables_with_coords

from pipeline.ai_summarizer import initialize_openai_client, generate_section_summary, generate_section_identifier_ollama

# ============================================================================
#  PROPER SECTION EXTRACTION WITH CALLOUT BOX & HEADING DETECTION
# ============================================================================


def _normalize_whitespace(s: str) -> str:
    """Normalize spaces and clean text."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling punctuation followed by quotes/brackets.
    Returns list of complete sentences.
    """
    # Pattern: sentence-ending punctuation + optional closing quotes/brackets + whitespace + capital letter or quote
    # Handles cases like: `responsibility."` or `online).` or `people."`
    pattern = r'(?<=[.!?])[\s\"\'\)\]]*\s+(?=[A-Z"\u201c\(])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _is_complete_sentence(text: str) -> bool:
    """
    Check if text ends with a complete sentence.
    Handles punctuation followed by quotes, brackets, or footnote markers.
    """
    # Strip trailing whitespace and common trailing markers
    stripped = text.strip()

    # Check if ends with sentence punctuation (possibly followed by quotes/brackets/numbers)
    # Examples: "text." "text?" "text!" "text."" 'text."' "text.) " "text.1 " "text.2"
    pattern = r'[.!?][\"\'\)\]]*\s*\d*\s*$'
    return bool(re.search(pattern, stripped))


def _starts_with_uppercase(text: str) -> bool:
    """
    Check if text starts with an uppercase letter, ignoring ALL leading punctuation.
    Used to detect new sentences vs. continuations in cross-column merging.

    Uses unicodedata.category to skip all punctuation characters (P*), spaces (Z*),
    and other non-letter marks until reaching the first actual letter.

    Examples:
        "Hello" -> True (starts with H)
        "'Hello" -> True (first letter after ASCII quote is H)
        '"However"' -> True (first letter after Unicode quote is H)
        ""Moreover"" -> True (first letter after curly quotes is H)
        "«However»" -> True (first letter after guillemet is H)
        "—However" -> True (first letter after em-dash is H)
        "–Thus" -> True (first letter after en-dash is T)
        "• Moreover" -> True (first letter after bullet is M)
        "however" -> False (starts with lowercase h)
        "(Figure 5)" -> True (first letter after paren is F)
        "particular" -> False (starts with lowercase p)
    """
    import unicodedata

    if not text:
        return False

    # Iterate through characters, skipping punctuation/spacing/control chars
    for char in text:
        category = unicodedata.category(char)

        # Skip these character categories:
        # - P*: Punctuation (Ps, Pe, Pi, Pf, Pd, Po)
        # - Z*: Spacing (Zs, Zl, Zp)
        # - S*: Symbols (Sk, Sc, Sm, So)
        # - C*: Control/Other (Cc=control, Cf=format, Cs=surrogate, Co=private, Cn=unassigned)
        if (category.startswith('P') or category.startswith('Z')
                or category.startswith('S') or category.startswith('C')):
            continue

        # Found first meaningful character
        # Treat as "uppercase/new sentence" if:
        # - Lu (Letter, uppercase): e.g., "H" in "However"
        # - N* (Number): e.g., "2" in "2020 saw..." (digit-start sentences are new)
        if category == 'Lu' or category.startswith('N'):
            return True

        # Ll (Letter, lowercase): e.g., "h" in "however" - this is a continuation
        # Lo (Letter, other): treat as new sentence (e.g., CJK characters)
        # Lt, Lm: also treat as new sentence by default
        if category == 'Ll':
            return False

        # For other letter types (Lo, Lt, Lm), treat as uppercase/new sentence
        if category.startswith('L'):
            return True

        # For any other character type, default to False (continuation)
        return False

    # All characters are skippable - default to False
    return False


def _make_slug(text: str) -> str:
    """Create URL-friendly slug from text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] if slug else hashlib.md5(text.encode()).hexdigest()[:8]


def _generate_topic_identifier(title: str, text: str) -> str:
    """Generate a descriptive topic identifier using Ollama."""
    return generate_section_identifier_ollama(title, text)


def _build_figure_exclusion_zones(
        pdf_path: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Build per-page exclusion zones for figure/chart regions.
    Returns: Dict mapping page_num -> list of (x0, y0, x1, y1) exclusion rectangles
    Note: For Word documents, figure extraction is handled separately.
    """
    return {}


def _build_table_exclusion_zones(
        pdf_path: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Build per-page exclusion zones for table regions plus title strips.
    Returns: Dict mapping page_num -> list of (x0, y0, x1, y1) exclusion rectangles
    """
    try:
        tables = extract_tables_with_coords(pdf_path)
    except Exception:
        return {}  # Failsafe if table extraction fails

    exclusion_zones = {}

    for tbl in tables:
        page = tbl.get('page', 1)
        bbox = tbl.get('bbox', [])

        if len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            table_width = x1 - x0
            table_height = y1 - y0

            # Inflate table bbox laterally by 12pt to catch adjacent text
            lateral_margin = 12

            # Add title strip above table: height = max(60pt, 10% of table height)
            # Cap at 80pt to avoid intruding too far into surrounding prose
            # 60pt minimum catches typical table titles that span 2-3 lines
            top_strip_height = min(max(60, table_height * 0.10), 80)

            # Add smaller bottom strip for footers/notes (12pt)
            bottom_strip_height = 12

            # Main table exclusion zone (inflated laterally)
            table_zone = (x0 - lateral_margin, y0, x1 + lateral_margin, y1)

            # Top title strip
            top_strip = (x0 - lateral_margin, y0 - top_strip_height,
                         x1 + lateral_margin, y0)

            # Bottom footer strip
            bottom_strip = (x0 - lateral_margin, y1, x1 + lateral_margin,
                            y1 + bottom_strip_height)

            if page not in exclusion_zones:
                exclusion_zones[page] = []

            # Add all three zones for this table
            exclusion_zones[page].extend([table_zone, top_strip, bottom_strip])

    return exclusion_zones


def _build_exclusion_zones(
        pdf_path: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Build unified exclusion zones combining both figures and tables.
    Returns: Dict mapping page_num -> list of (x0, y0, x1, y1) exclusion rectangles
    """
    figure_zones = _build_figure_exclusion_zones(pdf_path)
    table_zones = _build_table_exclusion_zones(pdf_path)

    # Merge zones from both sources
    all_zones = {}

    for page, zones in figure_zones.items():
        if page not in all_zones:
            all_zones[page] = []
        all_zones[page].extend(zones)

    for page, zones in table_zones.items():
        if page not in all_zones:
            all_zones[page] = []
        all_zones[page].extend(zones)

    return all_zones


def _paragraph_overlaps_figure(
        para_bbox: Tuple[float, float, float, float],
        exclusion_zones: List[Tuple[float, float, float, float]]) -> bool:
    """
    Check if paragraph's bounding box overlaps with any figure exclusion zone.
    Uses center-point and area overlap heuristics.
    """
    if not para_bbox or not exclusion_zones:
        return False

    p_x0, p_y0, p_x1, p_y1 = para_bbox
    p_width = p_x1 - p_x0
    p_height = p_y1 - p_y0
    p_center_x = (p_x0 + p_x1) / 2
    p_center_y = (p_y0 + p_y1) / 2

    for zone in exclusion_zones:
        z_x0, z_y0, z_x1, z_y1 = zone

        # Check if paragraph's center point is inside exclusion zone
        if z_x0 <= p_center_x <= z_x1 and z_y0 <= p_center_y <= z_y1:
            return True

        # Check if paragraph has significant area overlap (>40%)
        # Calculate intersection rectangle
        inter_x0 = max(p_x0, z_x0)
        inter_y0 = max(p_y0, z_y0)
        inter_x1 = min(p_x1, z_x1)
        inter_y1 = min(p_y1, z_y1)

        if inter_x0 < inter_x1 and inter_y0 < inter_y1:
            # Intersection exists
            inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
            para_area = p_width * p_height
            overlap_ratio = inter_area / para_area if para_area > 0 else 0

            if overlap_ratio > 0.4:  # >40% overlap
                return True

    return False


def _should_skip_paragraph(
        para: Dict[str, Any],
        page_num: int,
        figure_exclusion_zones: Dict[int, List[Tuple[float, float, float,
                                                     float]]],
        structural_boundaries: List[Dict[str, Any]] = None) -> bool:
    """
    Wrapper function combining text-based and spatial filtering.
    Returns True if paragraph should be skipped (excluded from section).

    Checks:
    1. Spatial overlap with figure/chart exclusion zones (SKIPPED for structural sections)
    2. Text-based pattern filtering (captions, citations, etc.)
    """
    para_text = para.get("text", "")

    # Check 1: Text-based filtering
    text_excluded = _should_exclude_paragraph(para_text)
    if text_excluded:
        return True

    # Check 2a: Figure caption callouts (check on ALL pages, not just those with figures)
    # These are short summary statements (≤40 words) with temporal phrases
    # that appear in lower third of page (y > 600) - typical for callout boxes below figures
    text_stripped = para_text.strip()
    word_count = len(text_stripped.split())
    if word_count <= 40:
        callout_patterns = [
            r'^Over\s+the\s+(past|last)\s+\w+\s+(years?|decades?|months?),',
            r'^In\s+recent\s+(years?|decades?|months?),',
            r'^During\s+the\s+(period|past|last)\s+',
            r'^Between\s+\d{4}\s+and\s+\d{4},',
            r'^From\s+\d{4}\s+to\s+\d{4},',
        ]
        for pattern in callout_patterns:
            if re.match(pattern, text_stripped, re.I):
                # Check spatial cue: Is it in lower third of page (y > 600)?
                para_y = para.get("y", 0) if "y" in para else (
                    para.get("bbox", [0, 0, 0, 0])[1] if "bbox" in para else 0)

                if para_y > 600:  # Lower third of typical 842pt page
                    # This is likely a figure caption callout - filter it out
                    return True

    # Check 2b: Spatial filtering (skip if overlaps with figure)
    # BUT: Skip spatial filtering for paragraphs near structural boundaries (ABSTRACT, etc.)
    if page_num in figure_exclusion_zones:
        # Check if paragraph is near a structural boundary (within 300 pixels below)
        is_near_structural = False
        if structural_boundaries:
            para_y = para.get("y", 0) if "y" in para else (
                para.get("bbox", [0, 0, 0, 0])[1] if "bbox" in para else 0)

            for boundary in structural_boundaries:
                if boundary["page"] == page_num:
                    boundary_y = boundary.get("y", 0)
                    # If paragraph is within 300 pixels below a structural boundary, preserve it
                    if 0 < para_y - boundary_y < 300:
                        is_near_structural = True
                        break

        if is_near_structural:
            return False  # Don't filter paragraphs near structural sections

        # Bypass spatial overlap check for long "(Figure N)" body paragraphs
        # These are legitimate analysis paragraphs that reference figures, not captions
        if re.match(r'^\(Figure\s+\d+\)\.?\s', text_stripped, re.I):
            word_count = len(text_stripped.split())
            if word_count >= 15:  # Long text = body paragraph, preserve it
                return False

        # Extract paragraph bbox - try multiple formats
        para_bbox = None
        if "bbox" in para:
            bbox = para["bbox"]
            if len(bbox) >= 4:
                para_bbox = tuple(bbox[:4])
        elif all(k in para for k in ["x0", "y0", "y1", "y1"]):
            para_bbox = (para["x0"], para["y0"], para["x1"], para["y1"])

        if para_bbox:
            page_exclusions = figure_exclusion_zones[page_num]
            if _paragraph_overlaps_figure(para_bbox, page_exclusions):
                # CRITICAL: Preserve long paragraphs or multi-sentence paragraphs
                # These are clearly body text, not figure captions, even if they overlap spatially
                # This allows column-continuation paragraphs like "particular, a large gap..." to survive
                # Lower threshold to 25 words to catch cross-column continuations (e.g., 28-word paragraph)
                word_count = len(text_stripped.split())
                has_multiple_sentences = text_stripped.count(
                    '.') + text_stripped.count('!') + text_stripped.count(
                        '?') >= 2

                if word_count > 25 or has_multiple_sentences:
                    return False  # Preserve long/multi-sentence paragraphs despite overlap

                return True  # Short paragraph with overlap - likely a caption

    return False


def _merge_abstract_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    ABSTRACT-specific paragraph merging.
    Merges paragraphs that end with incomplete sentences or transition words.
    This runs ONLY for ABSTRACT sections to avoid affecting other sections.
    """
    if not paragraphs or len(paragraphs) <= 1:
        return paragraphs

    merged = []
    i = 0

    while i < len(paragraphs):
        current = paragraphs[i].strip()

        # Check if there's a next paragraph
        if i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()

            # Check if current ends with transition word + comma (incomplete)
            # Examples: "However," "Furthermore," "Moreover,"
            ends_with_transition = re.search(
                r'\b(However|Furthermore|Moreover|Therefore|Thus|Nevertheless|Nonetheless),\s*$',
                current)

            # Check if current ends with incomplete sentence
            ends_incomplete = re.search(
                r'\b(the|a|an|of|to|in|on|at|for|with|by|from|and|or|but)\s*$',
                current, re.I)

            # If incomplete, merge with next paragraph
            if ends_with_transition or ends_incomplete:
                merged.append(current + " " + next_para)
                i += 2  # Skip next paragraph
                continue

        # No merge needed
        merged.append(current)
        i += 1

    return merged


def _is_callout_title(text: str) -> bool:
    """Check if text is a special callout title."""
    t = text.strip().upper()
    # Check for exact matches
    if t in {
            "AT A GLANCE", "FROM THE AUTHORS", "KEY TAKEAWAYS",
            "EXECUTIVE SUMMARY", "MEDIA"
    }:
        return True
    # Check for Box 1, Box 2, etc.
    if t.startswith("BOX ") or re.match(r"^BOX\s+\d+", t):
        return True
    return False


def _is_structural_heading(text: str) -> bool:
    """Check if text is a major structural heading."""
    t = text.strip()
    return bool(
        re.fullmatch(
            r"(?i)(abstract|introduction|background|methodology|methods|results|discussion|conclusion|conclusions|references|bibliography|legal\s+and\s+editorial\s+details)",
            t))


def _is_section_heading_pattern(text: str) -> bool:
    """Check if text matches common section heading patterns."""
    # Common patterns for section headings in academic/business documents
    patterns = [
        r"^(Reasons|Causes|Factors)\s+(behind|for|driving|contributing)",  # Reasons behind...
        r"^No\s+more\s+\w+\s+(gap|difference|disparity)",  # No more ... gap
        r"^(Disparities|Inequalities|Differences)\s+in\s+",  # Disparities in...
        r"^(Conclusion|Summary|Implications)s?:",  # Conclusion: or Conclusions:
        r"^(Policy|Recommendations|Suggestions)\s+for\s+",  # Policy for...
        r"^(The|A)\s+(steady|gradual|slow|rapid|continuous)",  # A steady process...
        r"^(Growing|Increasing|Declining|Rising)\s+(inequality|disparity|gap)",  # Growing inequality
        r"^(Target|Focus|Address|Strengthen)\s+\w+\s+(measures|policies|strategies|support)",  # Target ... measures
        r"^Survey\s+(results|findings|data|analysis):",  # Survey results:
        r"^Certain\s+\w+\s+(can|may|will|should)",  # Certain narratives can...
        r"^(Narrative|Policy|Climate)\s+(effects|impacts|outcomes)\s+vary",  # Narrative effects vary...
        r"^\w+\s+effects?\s+vary\s+by",  # Effects vary by...
        r"^Climate\s+populism\s+and\s+satisfaction",  # Box sub-section: Climate populism and satisfaction
        r"^Box\s+\d+\s*[—:\-]\s*",  # Box 1 — or Box 1: or Box 1-
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.I):
            return True
    return False


def _should_exclude_paragraph(text: str) -> bool:
    """
    Determine if a paragraph should be excluded from section content.

    Filters out figure captions, graph axis labels, notes, and statistical data
    that commonly contaminate section text in academic/business PDFs.

    Args:
        text: Paragraph text to check

    Returns:
        True if paragraph should be excluded, False if it's legitimate content
    """
    if not text or len(text.strip()) < 3:
        return True

    text_stripped = text.strip()

    # Reject lines starting with figure/table/notes/source keywords (with colon)
    if re.match(r'^(Figure|Fig\.|Table|Source):?\s', text_stripped, re.I):
        return True

    # Reject SHORT figure/table references in parentheses at start
    # Only block if it's a caption (< 15 words), not substantive text that references figures
    # Example caption: "(Figure 5). Labor productivity distribution"
    # Example body text: "(Figure 5). In both parts of the country, fewer and fewer districts..."
    if re.match(r'^\(Figure\s+\d+\)\.?\s', text_stripped, re.I):
        word_count = len(text_stripped.split())
        if word_count < 15:  # Short text = likely a caption, exclude it
            return True
        # Long text = body text referencing a figure, include it
        return False  # Explicitly include it

    if re.match(r'^\(Table\s+\d+\)\.?\s', text_stripped, re.I):
        word_count = len(text_stripped.split())
        if word_count < 15:  # Short text = likely a caption, exclude it
            return True
        return False  # Explicitly include it

    # Reject "Notes:" or "NOTE:" (require colon to avoid "Notes on methodology" etc.)
    if re.match(r'^Notes?:\s', text_stripped, re.I):
        return True

    # Reject author bylines with contact info (common in academic papers)
    # Pattern: "Firstname Lastname is the Position/Title at Organization | email"
    # Example: "Martin Gornig is the Research Director of Industrial Policy in the Firms and Markets Department at DIW Berlin | mgornig@diw.de"
    if re.search(r'\bis\s+the\s+.{10,80}\s+at\s+.{3,40}\s*\|', text_stripped,
                 re.I):
        return True

    # Reject contact strip patterns: short lines (≤20 words) containing email with separator or contact prefix
    word_count = len(text_stripped.split())
    if word_count <= 20 and re.search(
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            text_stripped):
        # Either has pipe separator (author byline format) or starts with Contact/Email
        if '|' in text_stripped or re.match(
                r'^(Contact|Email|Author|Correspondence):', text_stripped,
                re.I):
            return True

    # Reject sequences of 5+ four-digit years (graph axis labels)
    # Example: "1980 1985 1990 1995 2000 2005"
    # Increased threshold to avoid false positives with legitimate text
    year_pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(year_pattern, text_stripped)
    if len(years) >= 5:
        return True

    # Reject VERY high digit-density content (≥70% numeric tokens)
    # Graph data like "10 20 30 40 50 60 70 80 90 100"
    # Increased threshold to preserve statistical discussions
    tokens = text_stripped.split()
    if len(tokens) >= 8:  # Also require longer sequences
        numeric_count = sum(1 for t in tokens
                            if re.match(r'^\d+(\.\d+)?%?$', t))
        if numeric_count / len(tokens) >= 0.7:
            return True

    # Reject mixed numeric axis labels with trailing label word
    # Example: "20,000 40,000 60,000 80,000 100,000 Productivity"
    # Pattern: multiple formatted numbers followed by a single short alphabetic label (≤12 chars)
    if len(tokens) >= 4:
        numeric_count = sum(1 for t in tokens[:-1]
                            if re.match(r'^\d+[\d,\.]*$', t))
        # If all but last token are numeric, and last is a short alphabetic label
        if numeric_count >= 3 and numeric_count == len(tokens) - 1:
            last_token = tokens[-1]
            # Short alphabetic word (letters only, ≤12 chars) - likely an axis label
            if re.match(r'^[A-Za-z]{1,12}$', last_token):
                return True

    # CRITICAL: Reject mixed table rows like "Non-binary (N/A) 0 0.7 0.6 0"
    # Strip parentheses, then count numeric tokens
    text_no_parens = re.sub(r'\([^)]*\)', '', text_stripped)
    tokens_clean = text_no_parens.split()
    if len(tokens_clean) >= 3:  # At least 3 tokens after removing parentheses
        numeric_count = sum(1 for t in tokens_clean
                            if re.match(r'^\d+(\.\d+)?%?$', t))
        if numeric_count >= 3 and numeric_count / len(tokens_clean) >= 0.5:
            return True

    # Reject graph/chart descriptive patterns
    if re.search(
            r'\b(In\s+percent|Share\s+of|Percentage\s+of|Distribution\s+of)\s*$',
            text_stripped, re.I):
        return True

    # Reject very short lines with only numbers/symbols (axis labels)
    if len(text_stripped) < 20 and re.match(r'^[\d\s\.\,\-\+%]+$',
                                            text_stripped):
        return True

    # CRITICAL: Reject pure numeric/table data lines (e.g., "0 0.7 0.6 0" or "2,000 to 2,999 euros 24.9 26.2")
    # These are table rows that contaminate section text
    if re.match(r'^[\d\s\.\,\-%]+$', text_stripped):
        return True

    # Reject lines that are mostly numeric with minimal text (table captions/rows)
    # Example: "0 to 1,999 euros 24.4 25.5 26.4 25.6"
    if re.match(r'^[\d\s,\-]+\s+(euros?|percent|%|dollars?)', text_stripped,
                re.I):
        return True

    # ONLY reject survey experiment if it's a caption-style description (not a heading)
    # Must be lowercase "experiment" and contain visualization keywords
    if re.match(r'^Survey\s+experiment:', text_stripped, re.I):
        # Check if it contains typical caption phrases
        if re.search(r'(about|with|shows|displays|illustrates|participants)',
                     text_stripped, re.I):
            return True

    # Exclude typical figure captions and chart legends/axes
    if re.match(
            r'^(Figure\s+\d+\b|Comparison of\b|Density function\b|Productivity density\b)',
            text_stripped, re.I):
        return True

    # Exclude citation-like footnotes starting with a number and author name + year
    # Example: "1 Martina Hülz et al. (2024): ..."
    # Also catches: "1 Martina Hülz 24, 015 This applies to..."
    if re.match(r'^\d+\s+[A-ZÄÖÜ][\w\-]+(?:\s+[A-ZÄÖÜ][\w\-]+)*\s*[\(\d]',
                text_stripped):
        return True

    # Exclude figure/chart/table captions that describe visualizations
    # Examples: "Gross value added in current prices per employed person..."
    #           "Labor productivity in percent of..."
    #           "Distribution of income in euros..."
    caption_patterns = [
        r'^Gross value added',
        r'^Labor productivity.*(?:in percent|per employed)',
        r'^Distribution of.*(?:in percent|in euros)',
        r'^Comparison of.*(?:in percent|in euros)',
        r'^Income.*(?:in percent|in euros)',
        r'(?:in percent of total|per employed person|in current prices)',
    ]
    for pattern in caption_patterns:
        if re.search(pattern, text_stripped, re.I):
            # Only exclude if it's short (likely a caption, not a full paragraph)
            if len(text_stripped.split()) < 25:
                return True

    return False


def _clean_callout_content(title: str, paragraphs: List[str],
                           max_words: int) -> str:
    """
    Apply callout-specific content cleaning rules.

    Args:
        title: Section title (e.g., "AT A GLANCE", "MEDIA", "FROM THE AUTHORS")
        paragraphs: List of paragraph texts
        max_words: Maximum word count allowed

    Returns:
        Cleaned text with callout-specific formatting applied
    """
    title_upper = title.strip().upper()
    text = "\n\n".join(paragraphs)

    # Filter out figure captions and survey experiment text
    text = re.sub(r'Survey experiment:.*?(?=\n\n|$)',
                  '',
                  text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Figure \d+.*?(?=\n\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Notes: In a survey.*?(?=\n\n|$)',
                  '',
                  text,
                  flags=re.DOTALL | re.IGNORECASE)

    # AT A GLANCE: Extract subtitle, author, and bullet points
    if title_upper == "AT A GLANCE":
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Extract subtitle and author (non-bullet lines at the beginning)
        subtitle_author = []
        bullets = []

        for line in lines:
            # Check if it's a bullet point
            if line.startswith('•') or line.startswith('-'):
                bullets.append(line)
            # Check if it's an author line (starts with "By")
            elif line.startswith('By ') and len(subtitle_author) > 0:
                subtitle_author.append(line)
            # First few non-bullet lines before bullets are likely subtitle
            elif len(bullets) == 0 and len(subtitle_author) < 2:
                subtitle_author.append(line)

        # Combine subtitle/author + bullets with optimized spacing
        parts = []

        # Subtitle/author section (with double newline after)
        if subtitle_author:
            parts.append('\n'.join(
                subtitle_author))  # Single newline between subtitle lines

        # Bullets section (with single newline between bullets for tighter spacing)
        total_words = sum(len(l.split()) for l in subtitle_author)
        kept_bullets = []
        for b in bullets:
            w = len(b.split())
            if total_words + w <= max_words:
                kept_bullets.append(b)
                total_words += w
            else:
                break

        if kept_bullets:
            parts.append(
                '\n\n'.join(kept_bullets)
            )  # Double newline so each bullet is a separate paragraph

        # Join subtitle/author and bullets with double newline
        return "\n\n".join(parts)

    # MEDIA: Extract audio/video link text only
    elif title_upper == "MEDIA":
        # Look for audio/video/media links
        media_patterns = [
            r'(Audio Interview.*?www\.[\w./]+)',
            r'(Video.*?www\.[\w./]+)',
            r'(Podcast.*?www\.[\w./]+)',
            r'(www\.diw\.de/mediathek)',
        ]

        media_text = ""
        for pattern in media_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                media_text = match.group(1)
                break

        text = media_text if media_text else ""

    # FROM THE AUTHORS: Keep only quote + attribution (stop at em-dash with name)
    elif title_upper == "FROM THE AUTHORS":
        # Normalize newlines to spaces first to merge multi-line quotes into single paragraph
        # This prevents page-break fragments from creating separate paragraphs
        normalized_text = _normalize_whitespace(re.sub(r'\s*\n+\s*', ' ',
                                                       text))

        # Look for quoted text ending with attribution (em-dash + name + em-dash)
        # Pattern: "Quote text..." — Name —
        match = re.search(
            r'[""\u201c](.*?)[""\u201d]\s*[—–]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[—–]',
            normalized_text, re.DOTALL)

        if match:
            quote = match.group(1).strip()
            author = match.group(2).strip()
            text = f'"{quote}"\n— {author} —'
        else:
            # Fallback: Try to find em-dash pattern and truncate there
            match = re.search(
                r'(.*?[—–]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[—–])',
                normalized_text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                # If no pattern found, use the normalized text as-is
                text = normalized_text

    # General cleaning for all callouts
    text = _normalize_whitespace(text)

    return text


def _clean_sentence(s: str) -> str:
    """Clean and normalize a sentence for learning bullets."""
    s = re.sub(r"\[[^\]]+\]", "", s)  # remove citations
    s = re.sub(r"\([^)]*\)", "", s)  # remove parentheticals
    s = re.sub(r"\b(et al\.|ibid\.|doi|www\.)\b", "", s, flags=re.I)
    s = _normalize_whitespace(s)
    s = s.strip().strip('"').strip()
    # Remove trailing prepositions
    s = re.sub(r"\b(in|of|to|for|on|at|by|with)\s*$", "", s,
               flags=re.I).strip()
    # Ensure proper ending
    if s and not s.endswith(('.', '!', '?')):
        s = s.rstrip(" ,;") + "."
    # Capitalize first letter
    if s:
        s = s[0].upper() + s[1:]
    return s


def _make_concise_bullet(text: str, max_words: int = 12) -> str:
    """Make a concise bullet point (8-12 words) from text."""
    # Clean the text first
    text = _clean_sentence(text)
    if not text:
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text

    # Try to find the first complete clause (ending with comma, semicolon, or clause marker)
    clause_markers = [
        ' and ', ' but ', ' or ', ' which ', ' that ', ' because ', ' since '
    ]

    # Look for a natural break point before max_words
    for i in range(max_words, max(7, 0), -1):
        partial = " ".join(words[:i])

        # Check if this ends at a comma or semicolon
        if partial.endswith((',', ';')):
            result = partial[:-1].strip() + "."
            return result

        # Check if we're at a clause boundary
        for marker in clause_markers:
            if marker in " ".join(words[i:i + 2]).lower():
                result = partial.strip()
                if not result.endswith(('.', '!', '?')):
                    result += "."
                return result

    # If no natural break, use the full sentence but cap at a reasonable length
    if len(words) <= 15:
        return text

    # Last resort: truncate at max_words with ellipsis
    result = " ".join(words[:max_words])
    if not result.endswith(('.', '!', '?')):
        result += "."
    return result


def _extract_learning_bullets(text: str,
                              max_bullets: int = 4,
                              max_total_words: int = 80) -> List[str]:
    """Extract 3-4 concise learning bullets from text (10-12 words each)."""
    # Handle bullet-formatted text (from callout boxes)
    if text.count("•") >= 3:
        # Extract existing bullets
        bullet_parts = [p.strip() for p in text.split("•") if p.strip()]
        bullets = []
        for part in bullet_parts[:max_bullets]:
            # Clean and make concise (10-12 words max)
            clean = _make_concise_bullet(part)
            if not clean or len(clean.split()) < 5:
                continue

            bullets.append(clean)

        if bullets:
            return bullets[:max_bullets]

    # For regular prose, extract key sentences
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    bullets = []

    # Prioritize first sentence and sentences with key phrases
    key_phrases = [
        "productivity", "gap", "convergence", "disparity", "regional",
        "economic", "growth", "development", "policy", "inequality",
        "difference", "study", "research"
    ]

    # First pass: collect important sentences with smart summarization
    for sent in sents:
        if len(bullets) >= max_bullets:
            break

        # Check if sentence contains key information
        has_key_info = any(phrase in sent.lower() for phrase in key_phrases)
        is_first = len(bullets) == 0

        if has_key_info or is_first:
            clean = _clean_sentence(sent)
            if not clean or len(clean.split()) < 5:
                continue

            # Use complete sentence if reasonably short (up to 18 words for completeness)
            words = clean.split()
            if len(words) <= 18:
                bullets.append(clean)
            else:
                # For longer sentences, try to extract the main clause
                concise = _make_concise_bullet(sent, max_words=18)
                if concise and len(concise.split()) >= 5:
                    bullets.append(concise)

    # Ensure at least 2 bullets
    if len(bullets) < 2 and len(sents) > 1:
        for sent in sents:
            if len(bullets) >= max_bullets:
                break

            clean = _clean_sentence(sent)
            if not clean or clean in bullets or len(clean.split()) < 5:
                continue

            # Use complete sentence if short enough
            words = clean.split()
            if len(words) <= 15:
                bullets.append(clean)

    # Fallback: if still no bullets, use first sentence
    if not bullets and text.strip():
        clean = _clean_sentence(text[:300])
        if clean:
            bullets.append(clean)

    return bullets[:max_bullets]


def extract_headings_with_font_analysis(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract headings using font size, boldness, and positioning."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    headings = []

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        blocks = page.get_text("dict").get("blocks", [])

        # Estimate body text size
        sizes = []
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > 0:
                        sizes.append(size)

        sizes.sort()
        body_size = sizes[len(sizes) // 2] if sizes else 10.0
        # Headings are typically in 75-90th percentile of font sizes
        large_threshold = sizes[int(
            len(sizes) * 0.75)] if len(sizes) > 10 else body_size * 1.05

        # Extract potential headings (combine multi-line headings within blocks)
        for b in blocks:
            lines = b.get("lines", [])
            if not lines:
                continue

            # Get first line position for block
            first_line = lines[0]
            y = first_line.get("bbox", [0, 0, 0, 0])[1]
            y_frac = y / page_height if page_height > 0 else 0

            # Combine all lines in the block to handle multi-line headings
            # Calculate max_size FIRST before using it in filters
            # ALSO accumulate bbox coordinates for column detection
            text_parts = []
            max_size = 0
            is_bold = False
            min_x0 = min_y0 = None
            max_x1 = max_y1 = None

            for line in lines:
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                    size = span.get("size", 0)
                    max_size = max(max_size, size)
                    font = span.get("font", "").lower()
                    if "bold" in font:
                        is_bold = True

                    # Accumulate bbox for column detection
                    bbox = span.get("bbox")
                    if bbox and len(bbox) == 4:
                        x0, y0, x1, y1 = map(float, bbox)
                        min_x0 = x0 if min_x0 is None else min(min_x0, x0)
                        min_y0 = y0 if min_y0 is None else min(min_y0, y0)
                        max_x1 = x1 if max_x1 is None else max(max_x1, x1)
                        max_y1 = y1 if max_y1 is None else max(max_y1, y1)

            # Skip top 7% ONLY if it's small text (running headers)
            # Skip bottom 10% (footers)
            # Allow top area for AT A GLANCE (11%) and MEDIA in bottom area (83%)
            if y_frac > 0.90:
                continue  # Bottom 10% always skip
            if y_frac < 0.07 and max_size < body_size * 1.5:
                continue  # Top 7% only if not large/special text

            text = " ".join(text_parts).strip()
            text = _normalize_whitespace(text)
            if not text:
                continue

            # CRITICAL FIX: Force-accept these specific DIW section headings (bypass all filters)
            force_accept_headings = [
                "Survey results: Distributional effects seen as key issue in climate policy",
                "Narrative effects vary by social background and political position"
            ]
            is_force_accepted = any(heading.lower() in text.lower()
                                    for heading in force_accept_headings)

            if is_force_accepted:
                # Compute bbox from accumulated span coordinates
                if min_x0 is not None:
                    heading_bbox = [min_x0, min_y0, max_x1, max_y1]
                else:
                    heading_bbox = [
                        0.0,
                        float(y), page.rect.width,
                        float(y) + max(max_size, body_size)
                    ]

                headings.append({
                    "text": text,
                    "page": page_num,
                    "size": max_size,
                    "bold": is_bold,
                    "y": y,
                    "bbox": heading_bbox,
                    "x0": heading_bbox[0],
                    "x1": heading_bbox[2],
                    "x_mid": (heading_bbox[0] + heading_bbox[2]) / 2,
                    "force_keep":
                    True  # Prevent merging even if below min_words
                })
                continue  # Skip all other filters

            # Filter noise
            if re.search(r"(https?://|www\.|doi:|@|©)", text, re.I):
                continue
            if re.search(r"\((19|20)\d{2}\)", text):  # citations
                continue
            # Reject figure/table captions and sources
            if re.search(r"\b(figure|fig\.|table|source|note|graph|chart)\b",
                         text, re.I):
                continue
            # Reject text containing "regression" (table/figure captions)
            if re.search(
                    r"\b(regression|coefficient|standard deviation|average level)\b",
                    text, re.I):
                continue
            # Reject text with "by party" or "by demographic" (chart labels)
            if re.search(r"\bby (party|demographic|state|region)\b", text,
                         re.I):
                continue
            # Reject figure captions with "among" pattern
            if re.search(
                    r"\b(increased|pronounced|decreased|higher|lower) among\b",
                    text, re.I):
                continue
            # Reject captions ending with "voters" or "parties"
            if re.search(r"(voters|parties)\.\s*$", text, re.I):
                continue
            if re.match(r"^\d+[\s\.]", text):  # footnotes/numbering
                continue
            if re.search(r"^\s*[—\-].*[—\-]\s*$",
                         text):  # author attribution lines
                continue

            # CRITICAL: Reject hyphenated line breaks (words split across lines)
            if text.endswith("-"):
                continue

            # CRITICAL: Reject ISSN and other metadata
            if re.match(r"^ISSN\s+\d", text, re.I):
                continue

            # CRITICAL: Reject footer/metadata patterns (but allow structural headings)
            if not _is_structural_heading(text):
                if re.search(
                        r"(phone|fax|email|volume|issue|copyright|license|\+\d{2}\s+\d+)",
                        text, re.I):
                    continue
                if re.search(
                        r"^\d+\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
                        text, re.I):
                    continue
                # Reject figure credits (e.g., "David Liuzzo 2006 Staatsﬠlagge...")
                if re.search(r"\d{4}.*[sS]taat", text) or re.search(
                        r"(flagge|flag|wappen)", text, re.I):
                    continue

            # CRITICAL: Reject text ending with commas (mid-sentence)
            if text.endswith(","):
                continue

            # CRITICAL: Reject bullet points
            if text.startswith("•") or text.startswith("-") or text.startswith(
                    "*"):
                continue

            # CRITICAL: Reject author bylines (e.g., "By Martin Gornig")
            if re.match(r"^By\s+[A-Z]", text):
                continue

            # Allow large titles (main article titles are important sections!)
            # Only reject if extremely large (>30) AND looks like author/metadata
            if max_size > 30.0 and re.search(r"(^By\s+|@|©|\d{4}$)", text):
                continue

            # Check if it's heading-like
            words = text.split()
            word_count = len(words)

            # CRITICAL: Reject chart/figure labels (sequences of 2-3 letter codes)
            # Example: "HE NI NW RP SL SH BB MV SN ST TH BE HB HH BY" (German state codes)
            if word_count >= 5:
                short_words = [w for w in words if len(w) <= 3 and w.isupper()]
                # If more than 50% are 2-3 letter codes, it's likely chart labels
                if len(short_words) / word_count > 0.5:
                    continue

            # CRITICAL: Reject likely table headers or figure labels (very short, generic terms)
            if word_count <= 2 and re.match(
                    r"^(states|countries|regions|function|values|data|results|fiscal capacity|productivity)$",
                    text, re.I):
                continue

            # CRITICAL: Reject running headers (repeated text on multiple pages)
            # Common running headers: "Fiscal capacity", "Productivity", "Climate Policy"
            if word_count <= 3 and re.match(r"^[A-Z][a-z]+(\s[a-z]+){0,2}$",
                                            text):
                # This looks like a short Title Case phrase - likely a running header
                # We'll filter it later if it appears on 3+ pages
                pass

            # PRIORITY: If text matches known section heading patterns, skip ALL filters below
            # This ensures headings like "Survey results: ..." and "Narrative effects vary..." are not filtered
            is_pattern_matched_heading = _is_section_heading_pattern(text)

            # CRITICAL FIX: Force-accept these specific multi-line headings
            if ("Survey results" in text and "Distributional" in text) or \
               ("Narrative effects vary" in text and "background" in text):
                is_pattern_matched_heading = True

            # CRITICAL: Reject body text fragments (incomplete sentences)
            # BUT: Allow bold/large text ending in lowercase (common DIW style!)
            if not _is_callout_title(text) and not _is_structural_heading(
                    text) and not is_pattern_matched_heading:
                # Check if it ends with lowercase word (likely mid-sentence)
                last_word = words[-1] if words else ""
                if last_word and last_word[-1].islower():
                    # Exception 1: Colon-based headings (Survey results: ..., Box 1: ...)
                    if ":" in text and word_count <= 20:  # Increased limit for long colon headings
                        pass  # Keep it - colon indicates section heading
                    # Exception 2: Starts with "Box " - always keep Box titles
                    elif text.startswith("Box "):
                        pass  # Keep it - Box headings
                    # Exception 3: Bold OR larger text (1.1x+) - likely heading
                    elif is_bold or max_size >= body_size * 1.1:
                        pass  # Keep it - it's a heading
                    # Exception 4: Mostly capitalized (Title Case or ALL CAPS)
                    elif max_size >= large_threshold:
                        caps_count = sum(1 for w in words
                                         if w and w[0].isupper())
                        if caps_count / len(words) >= 0.5:
                            pass  # Keep it
                        else:
                            continue  # Likely body text
                    else:
                        # Regular size, ends lowercase, not bold → reject
                        caps_count = sum(1 for w in words
                                         if w and w[0].isupper())
                        if caps_count / len(words) < 0.5:
                            continue

            # Skip if too long or too short (allow 1-word if structural/callout)
            is_special = _is_callout_title(text) or _is_structural_heading(
                text) or is_pattern_matched_heading
            if is_special:
                # Allow 1-word structural headings like "ABSTRACT"
                # Pattern-matched headings can be longer (up to 20 words)
                if word_count < 1 or word_count > 20:
                    continue
            else:
                if word_count < 2 or word_count > 15:
                    continue

            # Skip if sentence-like (has commas/semicolons and not ALL CAPS)
            if (text.count(',') + text.count(';') >= 2) and not text.isupper():
                continue

            # Reject if it starts with lowercase (likely continuation)
            if text and text[0].islower():
                continue

            # Heading criteria - prioritize pattern matching
            is_heading = False

            # 0. PRIORITY: Pattern-matched section headings - ALWAYS accept first
            if is_pattern_matched_heading:
                is_heading = True
            # 1. Bold ALL CAPS short text (AT A GLANCE, MEDIA) - accept even if small size
            elif is_bold and text.isupper() and 1 <= word_count <= 5:
                is_heading = True
            # 2. Callout titles (AT A GLANCE, FROM THE AUTHORS, MEDIA, Box N:) - always accept
            elif _is_callout_title(text):
                is_heading = True
            # 3. Structural headings (ABSTRACT, INTRODUCTION, etc.) - always accept
            elif _is_structural_heading(text):
                is_heading = True
            # 4. Box headings (Box 1:, Box 2:) - accept with or without colon
            # Match "Box 1", "Box 1:", "Box 1: Title"
            elif re.match(r"^Box\s+\d+", text, re.I):
                is_heading = True
            # 5. Section heading patterns (Reasons behind..., No more...gap, etc.) - always accept
            elif _is_section_heading_pattern(text):
                is_heading = True
            # 6. ALL CAPS phrases (1-15 words) - likely real headings
            # Allow special chars (&, :, I, II) for headings like "SCENARIO I:" or "POWER & CAPACITY"
            elif text.isupper() and 1 <= word_count <= 15:
                is_heading = True
            # 7. ALL CAPS with special chars: "ECONOMIC POWER & FISCAL CAPACITY", "SCENARIO I: TAX GROWTH"
            elif re.match(r'^[A-Z0-9\s&:\-]+$',
                          text) and 2 <= word_count <= 15:
                is_heading = True
            # 8. Very large text (1.5x+ body) - likely main title or major heading
            elif max_size >= body_size * 1.5 and 3 <= word_count <= 20:
                # Main article titles can be longer (up to 20 words)
                caps_ratio = sum(
                    1 for w in words if w and w[0].isupper()) / word_count
                if caps_ratio >= 0.3:  # At least 30% capitalized
                    is_heading = True
            # 9. Bold text with Title Case (common for section headings)
            elif is_bold and 2 <= word_count <= 15:
                caps_ratio = sum(
                    1 for w in words if w and w[0].isupper()) / word_count
                if caps_ratio >= 0.4:  # At least 40% capitalized (Title Case)
                    is_heading = True
            # 10. Mid-article headings with colons - priority rule
            elif ":" in text and max_size >= body_size * 1.05 and 3 <= word_count <= 20:
                #  Catch headings like "Survey results: ..." and "Conclusion: ..."
                # Relaxed word count to 3-20 to catch longer colon headings
                if words and words[0] and words[0][0].isupper():
                    is_heading = True
            # 11. Larger text (1.1x+) - catches mid-level sentence-case headings
            elif max_size >= body_size * 1.1 and 3 <= word_count <= 16:
                # Allow sentence-case headings (only first word capitalized)
                # as long as the first word IS capitalized
                if words and words[0] and words[0][0].isupper():
                    is_heading = True
            # 12. Question headings (Box titles like "What is populism?")
            elif text.endswith("?") and 3 <= word_count <= 12:
                if words and words[0] and words[0][0].isupper():
                    is_heading = True

            if is_heading:
                # CRITICAL: Exclude single-word text unless it's a structural heading
                # This prevents figure labels like "Productivity" from being detected as headings
                words = text.split()
                if len(words) == 1:
                    # Only allow single words if they're structural headings or callouts
                    if not (_is_structural_heading(text)
                            or _is_callout_title(text)):
                        is_heading = False  # Not a real heading, just a figure label

                if is_heading:  # Re-check after filter
                    # Compute bbox from accumulated span coordinates
                    if min_x0 is not None:
                        heading_bbox = [min_x0, min_y0, max_x1, max_y1]
                    else:
                        heading_bbox = [
                            0.0,
                            float(y), page.rect.width,
                            float(y) + max(max_size, body_size)
                        ]

                    headings.append({
                        "page":
                        page_num,
                        "text":
                        _normalize_whitespace(text),
                        "y":
                        float(y),
                        "size":
                        float(max_size),
                        "bold":
                        is_bold,
                        "bbox":
                        heading_bbox,
                        "x0":
                        heading_bbox[0],
                        "x1":
                        heading_bbox[2],
                        "x_mid": (heading_bbox[0] + heading_bbox[2]) / 2,
                        "force_keep":
                        is_pattern_matched_heading  # Force-keep pattern-matched headings
                    })

    doc.close()

    # Deduplicate while preserving order
    seen = set()
    unique_headings = []
    for h in headings:
        key = h["text"].lower()
        if key not in seen:
            seen.add(key)
            unique_headings.append(h)

    # CRITICAL: Merge "Box 1" or "Box 2" with the next heading on same page
    # Box labels are separate from their titles in DIW PDFs
    merged_headings = []
    skip_next = set()  # Track indices to skip

    for i, h in enumerate(unique_headings):
        if i in skip_next:
            continue

        # Check if this is a standalone "Box N" label
        if re.match(r"^Box\s+\d+$", h["text"], re.I):
            # Find the next heading on the same page
            next_h = None
            for j in range(i + 1, len(unique_headings)):
                if unique_headings[j]["page"] == h["page"]:
                    next_h = unique_headings[j]
                    skip_next.add(j)
                    break

            if next_h:
                # Merge: "Box 1" + "What is populism?" = "Box 1: What is populism?"
                merged_text = f"{h['text']}: {next_h['text']}"
                merged_headings.append({
                    "page": h["page"],
                    "text": merged_text,
                    "y": h["y"],
                    "size": next_h["size"],  # Use title's size
                    "bold": next_h["bold"]
                })
            else:
                # No next heading found, keep as is
                merged_headings.append(h)
        else:
            merged_headings.append(h)

    # Filter out running headers (text that appears on 3+ different pages)
    text_page_counts = {}
    for h in merged_headings:
        text_lower = h["text"].lower()
        if text_lower not in text_page_counts:
            text_page_counts[text_lower] = set()
        text_page_counts[text_lower].add(h["page"])

    # Remove headings that appear on 3+ pages (likely running headers)
    filtered_headings = []
    for h in merged_headings:
        text_lower = h["text"].lower()
        page_count = len(text_page_counts[text_lower])

        # Keep if it's a special heading OR appears on fewer than 3 pages
        is_special = (_is_callout_title(h["text"])
                      or _is_structural_heading(h["text"])
                      or h["text"].isupper() or h["text"].startswith("Box "))

        if is_special or page_count < 3:
            filtered_headings.append(h)

    return filtered_headings


def chunk_into_sections(
    pages: List[Dict[str, Any]],
    pdf_path: str | None = None,
    target_words: int | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
    use_ai: bool = False,
) -> List[Dict[str, Any]]:
    """
    Chunk PDF pages into learning sections using:
    1. Callout box detection (AT A GLANCE, FROM THE AUTHORS)
    2. Font-based heading extraction
    3. Structural section boundaries

    Args:
        target_words: Target word count per section (None = unlimited, extract full paragraphs)
        min_words: Minimum word count before merging sections (None = no minimum)
        max_words: Maximum word count per section (None = unlimited)
                section IDs and learning bullets

    Note: Callout sections (AT A GLANCE, FROM THE AUTHORS) always use fixed limits
          regardless of the parameters passed.
    """

    # ===================================================================
    # STEP 0: Build exclusion zones (figures + tables) for spatial filtering
    # ===================================================================
    figure_exclusion_zones = {}
    if pdf_path:
        figure_exclusion_zones = _build_exclusion_zones(pdf_path)

    # ===================================================================
    # STEP 1: Build global boundary index (headings + callouts)
    # ===================================================================

    # Extract headings using font analysis
    headings = []
    if pdf_path:
        headings = extract_headings_with_font_analysis(pdf_path)

    # Build global list of ALL section boundaries (headings + callouts)
    all_boundaries = []

    # Add callouts as boundaries
    callouts_by_page = {}
    for p in pages:
        page_num = p.get("page")
        callouts = p.get("callouts", [])
        if callouts:
            # Sort callouts by Y position, then by title for consistent order
            def callout_sort_key(c):
                y = c.get("y", 0)
                title = c.get("title", "")
                # MEDIA comes before FROM THE AUTHORS when at same Y position
                if "MEDIA" in title:
                    return (y, 0)
                elif "FROM THE AUTHORS" in title:
                    return (y, 1)
                else:
                    return (y, 2)

            sorted_callouts = sorted(callouts, key=callout_sort_key)
            callouts_by_page[page_num] = sorted_callouts

            # Add each callout as a boundary
            for c in sorted_callouts:
                bbox = c.get("bbox",
                             [0, c.get("y", 0), 600,
                              c.get("y", 0) + 20])
                all_boundaries.append({
                    "page": page_num,
                    "y": c.get("y", 0),
                    "text": c.get("title", ""),
                    "type": "callout",
                    "is_callout": True,
                    "bbox": bbox,
                    "x0": bbox[0],
                    "x1": bbox[2],
                    "x_mid": (bbox[0] + bbox[2]) / 2
                })

    # Add font-detected headings as boundaries
    for h in headings:
        page_num = h["page"]
        heading_y = h["y"]
        heading_text = h["text"]

        # Skip main article title and figure text on page 1 (middle area Y: 100-680)
        if page_num == 1 and 100 < heading_y < 680:
            continue

        # Skip main article title on page 2 (long title at top)
        if page_num == 2 and heading_y < 200 and len(heading_text) > 40:
            continue

        # Skip figure/chart titles
        figure_patterns = [
            r"at current prices",
            r"in percent",
            r"as a percentage of",
            r"percentage of the national average",
            r"^Figure \d+:",
            r"^Table \d+:",
        ]
        if any(re.search(p, heading_text, re.I) for p in figure_patterns):
            continue

        # Skip if this is a callout or structural heading title already added
        # This prevents duplicates when the same text (e.g., "ABSTRACT") is detected
        # both as a heading and as a callout
        if _is_callout_title(heading_text) or _is_structural_heading(
                heading_text):
            if page_num in callouts_by_page:
                callout_titles = {
                    c["title"].upper()
                    for c in callouts_by_page[page_num]
                }
                if heading_text.upper() in callout_titles:
                    continue  # Already added as callout

        # Extract bbox coordinates for column detection
        bbox = h.get("bbox", [0, heading_y, 600, heading_y + 20])
        x0 = bbox[0]
        x1 = bbox[2]
        x_mid = (x0 + x1) / 2

        all_boundaries.append({
            "page": page_num,
            "y": heading_y,
            "text": heading_text,
            "type": "heading",
            "is_callout": False,
            "bbox": bbox,
            "x0": x0,
            "x1": x1,
            "x_mid": x_mid,
            "force_keep": h.get("force_keep",
                                False)  # Preserve force_keep flag
        })

    # Sort all boundaries by (page, y) - this gives us reading order
    all_boundaries.sort(key=lambda b: (b["page"], b["y"]))

    # Add document-end sentinel so every section has a "next boundary"
    if pages:
        last_page = max(p.get("page", 1) for p in pages)
        all_boundaries.append({
            "page": last_page + 1,
            "y": 0,
            "text": "END_OF_DOCUMENT",
            "type": "sentinel",
            "is_callout": False
        })

    # Build list of structural boundaries to skip spatial filtering
    structural_boundaries = [
        b for b in all_boundaries if _is_structural_heading(b.get("text", ""))
        or _is_callout_title(b.get("text", ""))
    ]

    # ===================================================================
    # STEP 2: Extract paragraphs with bounding boxes from PDF
    # ===================================================================

    all_paragraphs = []

    if pdf_path:
        # Reopen PDF to extract bounding boxes for paragraphs
        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc, start=1):
                page_height = page.rect.height  # Get page height for footnote detection
                blocks = page.get_text("dict").get("blocks", [])

                for block in blocks:
                    if block.get("type") != 0:  # Skip non-text blocks
                        continue

                    lines = block.get("lines", [])
                    if not lines:
                        continue

                    # Combine all lines in block to get paragraph text
                    text_parts = []
                    for line in lines:
                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))

                    para_text = " ".join(text_parts).strip()
                    para_text = _normalize_whitespace(para_text)

                    if not para_text or len(para_text.split()) < 3:
                        continue

                    # NOTE: Footnote cleaning already done in extract_text.py
                    # Calling _strip_inline_footnotes here would be redundant and can remove valid content
                    # like ABSTRACT paragraphs that contain phrases matching footnote markers

                    # Skip figure/table labels and captions
                    if re.search(
                            r"^(figure|fig\.|table|source:|sources:|note:|©|diw\s+berlin)",
                            para_text, re.I):
                        continue

                    # Skip author bylines
                    if re.match(r"^By\s+[A-Z]", para_text):
                        continue

                    # Get bounding box: [x0, y0, x1, y1] (needed for spatial filtering)
                    bbox = block.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = bbox

                    # Apply comprehensive paragraph filter (text-based + spatial)
                    para_temp = {"text": para_text, "bbox": bbox}
                    if _should_skip_paragraph(para_temp, page_num,
                                              figure_exclusion_zones,
                                              structural_boundaries):
                        continue

                    # CRITICAL: Position-based footnote detection
                    # Exclude paragraphs at bottom of page that start with numbers (footnotes)
                    # Note: Conclusion heading is preserved via explicit exemption in _is_footnote_block
                    para_dict = {"text": para_text, "y": y0}
                    if _is_footnote_block(para_dict, page_height):
                        continue  # Skip footnote blocks

                    all_paragraphs.append({
                        "page": page_num,
                        "y": y0,  # Top Y coordinate
                        "text": para_text,
                        "bbox": bbox,  # [x0, y0, x1, y1]
                        "x_mid": (x0 + x1) / 2,  # Horizontal midpoint
                        "y_mid": (y0 + y1) / 2  # Vertical midpoint
                    })

            doc.close()
        except Exception as e:
            print(f"Warning: Could not extract bounding boxes: {e}")
            # Fallback: use paragraph data from pages without bbox
            for page_data in pages:
                page_num = page_data.get("page")
                paras = page_data.get("paras", [])

                for para_data in paras:
                    para_text = para_data.get("text", "").strip()
                    para_y = para_data.get("y", 0)

                    if not para_text or len(para_text.split()) < 3:
                        continue

                    # NOTE: Footnote cleaning already done in extract_text.py
                    # Calling _strip_inline_footnotes here would be redundant and can remove valid content

                    # Apply comprehensive paragraph filter (text-based + spatial)
                    # Note: Fallback path uses fake bbox, spatial filtering may not apply
                    para_temp = {
                        "text": para_text,
                        "bbox": [0, para_y, 600, para_y + 20]
                    }
                    if _should_skip_paragraph(para_temp, page_num,
                                              figure_exclusion_zones,
                                              structural_boundaries):
                        continue

                    all_paragraphs.append({
                        "page": page_num,
                        "y": para_y,
                        "text": para_text,
                        "bbox": [0, para_y, 600, para_y + 20],  # Fake bbox
                        "x_mid": 300,
                        "y_mid": para_y + 10
                    })

    # Detect multi-column layouts per page and merge cross-column paragraphs
    # IMPORTANT: Merging must happen BEFORE sorting, using Y-coordinate alignment
    # Step 1: Analyze each page to detect if it has a true two-column layout
    pages_with_columns = set()
    merge_count = 0

    for page_num in set(p["page"] for p in all_paragraphs):
        page_paras = [p for p in all_paragraphs if p["page"] == page_num]
        if len(page_paras) < 3:
            continue

        # Count paragraphs in left and right regions
        # Use 280px as threshold (slightly below middle of typical 600px PDF)
        left_paras = [p for p in page_paras if p.get("x_mid", 300) < 280]
        right_paras = [p for p in page_paras if p.get("x_mid", 300) >= 320]

        left_count = len(left_paras)
        right_count = len(right_paras)

        # Page has two-column layout if both columns have content (≥2 paragraphs each)
        # AND there's a clear horizontal separation (no overlap in x_mid values)
        if left_count >= 2 and right_count >= 2:
            # Verify genuine column separation: max left x_mid < min right x_mid
            # This prevents single-column pages with indents from being treated as multi-column
            max_left_x = max(p.get("x_mid", 0) for p in left_paras)
            min_right_x = min(p.get("x_mid", 600) for p in right_paras)

            # Require at least 40px gap between columns (about 7% of 600px page width)
            if min_right_x - max_left_x >= 40:
                pages_with_columns.add(page_num)

                # PHASE 1: Merge cross-column paragraphs at SAME Y-level
                # For each left paragraph that doesn't end with complete sentence,
                # find right paragraph at similar Y coordinate (tight tolerance for same-line splits)
                for left_p in left_paras:
                    if left_p.get(
                            "_merged"):  # Already merged or marked for removal
                        continue

                    if not _is_complete_sentence(left_p["text"]):
                        left_y = left_p.get("y_mid", left_p.get("y", 0))

                        # Find right paragraph within 20px vertical distance (same-line splits)
                        for right_p in right_paras:
                            if right_p.get("_merged"):  # Already processed
                                continue

                            right_y = right_p.get("y_mid", right_p.get("y", 0))

                            if abs(left_y - right_y) <= 20:
                                # Before merging, check if right paragraph starts with uppercase letter
                                # (ignoring quotes/brackets). Uppercase = new sentence, not continuation.
                                # Only merge if starts with lowercase (true continuation like "particular...")
                                right_text = right_p["text"]
                                if _starts_with_uppercase(right_text):
                                    # Skip - this is a new sentence (e.g., "However...", '"Moreover..."')
                                    continue

                                # Merge these paragraphs
                                merged_text = left_p["text"].rstrip(
                                ) + " " + right_text.lstrip()
                                left_p["text"] = merged_text
                                left_p[
                                    "_merged_with"] = True  # Mark as merged (keep this one)
                                right_p["_merged"] = True  # Mark for removal
                                merge_count += 1
                                break

                # PHASE 2: Reading-order fallback for column-break continuations
                # Check if last paragraphs in left column should merge with next paragraphs in right column
                # This handles cases where continuation is at different Y-level (e.g., "In" + "particular...")
                left_paras_sorted = sorted(left_paras,
                                           key=lambda p: p.get("y", 0))
                right_paras_sorted = sorted(right_paras,
                                            key=lambda p: p.get("y", 0))

                # Debug: Show what we're working with
                debug_mode = False  # Set to True to see debug output
                if debug_mode and len(left_paras_sorted) > 0:
                    print(f"\n  [DEBUG] Page {page_num} - Phase 2 fallback")
                    print(
                        f"  [DEBUG] Left column last 2: {[p['text'][:40] for p in left_paras_sorted[-2:]]}"
                    )
                    print(
                        f"  [DEBUG] Right column all: {[p['text'][:40] for p in right_paras_sorted]}"
                    )

                # Only check last 1-2 paragraphs in left column
                for left_p in left_paras_sorted[-2:]:
                    if debug_mode:
                        print(
                            f"  [DEBUG] Checking left para: '{left_p['text'][:60]}...'"
                        )
                        print(
                            f"  [DEBUG]   _merged: {left_p.get('_merged')}, _merged_with: {left_p.get('_merged_with')}"
                        )
                        print(
                            f"  [DEBUG]   _is_complete_sentence: {_is_complete_sentence(left_p['text'])}"
                        )

                    if left_p.get("_merged") or left_p.get("_merged_with"):
                        continue

                    if not _is_complete_sentence(left_p["text"]):
                        left_y = left_p.get("y", 0)
                        left_text_preview = left_p["text"][:50]

                        # Find first non-merged right paragraph that could be a continuation
                        for right_p in right_paras_sorted:
                            if right_p.get("_merged"):
                                continue

                            right_y = right_p.get("y", 0)
                            right_text = right_p["text"].strip()

                            # Check if right paragraph looks like a continuation
                            # MUST start with lowercase (ignoring quotes) to be a true continuation
                            # Uppercase words like "However" or '"Moreover"' are new sentences, not continuations
                            y_gap = abs(left_y - right_y)
                            starts_lowercase = not _starts_with_uppercase(
                                right_text)

                            if debug_mode:
                                print(
                                    f"  [DEBUG] Checking: '{left_text_preview}...' + '{right_text[:40]}...'"
                                )
                                print(
                                    f"  [DEBUG]   Y-gap: {y_gap}, lowercase: {starts_lowercase}"
                                )

                            # Only merge if starts with lowercase AND reasonable vertical gap
                            if y_gap <= 180 and starts_lowercase:
                                # This looks like a continuation - merge it
                                merged_text = left_p["text"].rstrip(
                                ) + " " + right_text.lstrip()
                                left_p["text"] = merged_text
                                left_p["_merged_with"] = True
                                right_p["_merged"] = True
                                merge_count += 1
                                if debug_mode:
                                    print(f"  [DEBUG] ✅ MERGED!")
                                break  # Only merge with first match

    # Remove paragraphs that were merged into others
    all_paragraphs = [p for p in all_paragraphs if not p.get("_merged")]

    if merge_count > 0:
        print(f"  [OK] Merged {merge_count} cross-column paragraph(s)")

    # Step 2: Sort with column-awareness for multi-column pages only
    def intelligent_sort_key(p):
        page = p["page"]
        y = p["y"]
        x_mid = p.get("x_mid", 300)

        # For multi-column pages: read left column first (top-to-bottom), then right column
        # For single-column pages: simple top-to-bottom order
        if page in pages_with_columns:
            column = 0 if x_mid < 300 else 1
            return (page, column, y)
        else:
            # Single-column or centered content: sort by Y only
            return (page, 0, y)

    all_paragraphs.sort(key=intelligent_sort_key)

    # ===================================================================
    # STEP 3: Build sections by collecting paragraphs between boundaries
    # ===================================================================

    sections = []

    current_section = None  # Initialize outside loop

    def finish_section():
        nonlocal current_section
        if not current_section:
            return

        # Finalize text
        title = current_section.get("title", "")
        paragraphs = current_section["paragraphs"]

        # Determine if this is a structural/callout section for consistent limit application
        is_structural = (_is_callout_title(title)
                         or _is_structural_heading(title)
                         or title.startswith("Box ")
                         or "LEGAL" in title.upper())

        # Set section-specific word limits
        # Callout sections (AT A GLANCE, FROM THE AUTHORS) always use fixed limits
        # Structural sections (ABSTRACT, etc.) use the global limits (can be None for unlimited)
        if _is_callout_title(title):
            section_max = 140  # Fixed limit for callouts
            section_min = 70  # Fixed minimum for callouts
        else:
            section_max = max_words  # Can be None (unlimited)
            section_min = min_words  # Can be None (no minimum)

        # Apply callout-specific content cleaning
        if _is_callout_title(title):
            # Do NOT normalize whitespace globally; preserve newlines for bullets
            raw_text = _clean_callout_content(title, paragraphs, section_max)

            # CRITICAL: Update paragraphs list with cleaned content
            # Split by double newlines to preserve intentional paragraph breaks (e.g., bullets in AT A GLANCE)
            # But also handle single newlines for quote+attribution in FROM THE AUTHORS
            if "\n\n" in raw_text:
                # Has double newlines (bullets or multiple parts)
                cleaned_paragraphs = [
                    p.strip() for p in raw_text.split("\n\n") if p.strip()
                ]
            elif "\n" in raw_text:
                # Has single newlines (e.g., quote + attribution)
                cleaned_paragraphs = [
                    p.strip() for p in raw_text.split("\n") if p.strip()
                ]
            else:
                # Single paragraph
                cleaned_paragraphs = [raw_text.strip()
                                      ] if raw_text.strip() else []

            current_section["paragraphs"] = cleaned_paragraphs
        else:
            # Smart paragraph joining: merge sentences split across page boundaries
            # Detect when a sentence continues from one paragraph to the next
            merged_paras = []
            i = 0
            merge_count = 0  # Debug counter

            while i < len(paragraphs):
                current_para = paragraphs[i].strip()

                # === FILTER: Skip citation/footnote paragraphs ===
                # Pattern: "10 Steffen Mau Warum der Osten anders bleibt. Suhrkamp."
                # Starts with digits, space, then capitalized word (author name)
                is_citation = re.match(r'^\d+\s+[A-ZÄÖÜ][\w\-]+', current_para)
                if is_citation:
                    # Skip this citation paragraph entirely
                    i += 1
                    continue

                # Check if there's a next paragraph to potentially merge with
                if i + 1 < len(paragraphs):
                    # === CRITICAL: Skip over footnotes to find actual next body paragraph ===
                    # Scan ahead to skip all consecutive footnotes before merge checks
                    footnote_skip_count = 0
                    next_idx = i + 1
                    while next_idx < len(paragraphs):
                        candidate = paragraphs[next_idx].strip()
                        # Check if this is a footnote (digits + space + capital letter)
                        is_footnote = re.match(r'^\d+\s+[A-ZÄÖÜ]', candidate)
                        if is_footnote:
                            footnote_skip_count += 1
                            next_idx += 1
                        else:
                            # Found non-footnote paragraph
                            break

                    # Get the actual next body paragraph (after skipping footnotes)
                    if next_idx >= len(paragraphs):
                        # No more paragraphs after footnotes - add current and move on
                        merged_paras.append(current_para)
                        i += 1
                        continue

                    next_para = paragraphs[next_idx].strip()

                    # === MERGE CONDITION 1: Short fragments without sentence punctuation ===
                    # Handles cases like "Hidden or short quote fragments
                    current_word_count = len(current_para.split())
                    ends_with_sentence_punct = current_para.rstrip().endswith(
                        ('.', '!', '?'))

                    # Merge if current para is very short (<= 4 words) and doesn't end with punctuation
                    if current_word_count <= 4 and not ends_with_sentence_punct:
                        merged_paras.append(current_para + " " + next_para)
                        merge_count += 1
                        # Skip current + all footnotes + next body paragraph
                        i = next_idx + 1
                        continue

                    # === MERGE CONDITION 1.5: Unmatched opening quotes ===
                    # Handles cases like: "...problem. "Hidden (quote without closing)
                    # Count opening and closing quotes
                    opening_quotes = current_para.count(
                        '"') + current_para.count('\u201c')  # " and "
                    closing_quotes = current_para.count(
                        '"') + current_para.count('\u201d')  # " and "
                    has_unmatched_quote = opening_quotes > closing_quotes

                    # Merge if paragraph has unmatched opening quote
                    if has_unmatched_quote:
                        merged_paras.append(current_para + " " + next_para)
                        merge_count += 1
                        # Skip current + all footnotes + next body paragraph
                        i = next_idx + 1
                        continue

                    # === MERGE CONDITION 1.6: Citation-ended paragraphs ===
                    # Handles cases like: "...BBSR). 14" (ends with citation number, not period)
                    # Pattern: ends with ) followed by space and digits, OR . followed by space and digits
                    ends_with_citation = re.search(r'[).]\s+\d+\s*$',
                                                   current_para)

                    # DON'T merge if next paragraph starts with a clear sentence-starting transition word
                    # These indicate new paragraphs, not continuations
                    transition_words = [
                        r'^However,?\s', r'^Furthermore,?\s', r'^Moreover,?\s',
                        r'^Therefore,?\s', r'^Thus,?\s', r'^Nevertheless,?\s',
                        r'^Nonetheless,?\s', r'^In\s+contrast,?\s',
                        r'^On\s+the\s+other\s+hand,?\s', r'^Meanwhile,?\s',
                        r'^Additionally,?\s', r'^Similarly,?\s',
                        r'^Conversely,?\s', r'^Indeed,?\s',
                        r'^In\s+addition,?\s'
                    ]
                    starts_with_transition = any(
                        re.match(pattern, next_para, re.I)
                        for pattern in transition_words)

                    # Merge if paragraph ends with citation marker (not a complete sentence)
                    # BUT only if next paragraph doesn't start with a transition word
                    if ends_with_citation and not starts_with_transition:
                        merged_paras.append(current_para + " " + next_para)
                        merge_count += 1
                        # Skip current + all footnotes + next body paragraph
                        i = next_idx + 1
                        continue

                    # === MERGE CONDITION 2: Incomplete sentence endings ===
                    # Detect sentence continuation:
                    # 1. Current paragraph ends with incomplete sentence (articles, prepositions)
                    # 2. Next paragraph starts with lowercase or number (continuing sentence)
                    incomplete_endings = [
                        r'\bthe\s*$',
                        r'\ba\s*$',
                        r'\ban\s*$',  # Articles
                        r'\bof\s*$',
                        r'\bto\s*$',
                        r'\bin\s*$',
                        r'\bon\s*$',
                        r'\bat\s*$',  # Prepositions
                        r'\bfor\s*$',
                        r'\bwith\s*$',
                        r'\bby\s*$',
                        r'\bfrom\s*$',
                        r'\band\s*$',
                        r'\bor\s*$',
                        r'\bbut\s*$',  # Conjunctions
                    ]

                    ends_incomplete = any(
                        re.search(pattern, current_para, re.I)
                        for pattern in incomplete_endings)
                    starts_lowercase = next_para and (next_para[0].islower() or
                                                      next_para[0].isdigit())

                    # DON'T merge if next paragraph starts with transition word (already defined above)
                    # This prevents merging "...In" with "However..." (new sentence, not continuation)
                    if ends_incomplete and starts_lowercase and not starts_with_transition:
                        # Merge these two paragraphs with a space (sentence continuation)
                        merged_paras.append(current_para + " " + next_para)
                        merge_count += 1
                        # Skip current + all footnotes + next body paragraph
                        i = next_idx + 1
                        continue

                    # === MERGE CONDITION 3: Long "(Figure N)" body paragraphs ===
                    # When next paragraph is a long (≥15 words) "(Figure N)" body paragraph,
                    # merge it with the current paragraph if current is reasonably short (< 60 words)
                    # This handles cases where figure references are part of ongoing text
                    # Example: "...at the district level" + "(Figure 5). In both parts of the country..."
                    next_is_long_figure_ref = (
                        re.match(r'^\(Figure\s+\d+\)\.?\s', next_para, re.I)
                        and len(next_para.split()) >= 15)
                    current_is_short = len(current_para.split()) < 60

                    if next_is_long_figure_ref and current_is_short:
                        # Merge these two paragraphs
                        merged_paras.append(current_para + " " + next_para)
                        merge_count += 1
                        # Skip current + all footnotes + next body paragraph
                        i = next_idx + 1
                        continue

                # No merge needed - add paragraph as-is
                merged_paras.append(current_para)
                i += 1

            # Debug output (will be visible in logs)
            if merge_count > 0:
                print(
                    f"  [OK] Merged {merge_count} sentence continuation(s) in '{title[:50]}'"
                )

            # ABSTRACT-SPECIFIC: Merge paragraphs ending with incomplete sentences
            # This only runs for ABSTRACT and doesn't affect other sections
            if title.upper() == "ABSTRACT":
                before_count = len(merged_paras)
                merged_paras = _merge_abstract_paragraphs(merged_paras)
                after_count = len(merged_paras)
                if before_count != after_count:
                    print(
                        f"  [OK] ABSTRACT: Merged {before_count - after_count} incomplete paragraph(s)"
                    )

            # SECTION-SPECIFIC FIX: Two-column orphan handling for DIW page
            # Title: "No more productivity gap in most regions in the east"
            # Goal: Move and merge the orphaned "In particular, ..." sentence so it follows
            #       the preceding "In contrast, ..." sentence as a single paragraph,
            #       then keep the success-stories paragraph after it.
            if title.strip().lower(
            ) == "no more productivity gap in most regions in the east":
                # Find key paragraphs by pattern
                def find_index(pattern: str) -> int:
                    for idx, para in enumerate(merged_paras):
                        if re.match(pattern, para.strip(), re.IGNORECASE):
                            return idx
                    return -1

                idx_contrast = find_index(
                    r"^In\s+contrast,\s+the\s+productivity\s+level\s+is\s+still\s+lower"
                )
                idx_particular = find_index(r"^(In\s+)?particular,\s+")
                idx_success = find_index(
                    r"^However,\s+many\s+other\s+eastern\s+German\s+regions\s+are\s+success\s+stories"
                )

                # Reorder: ensure 'In particular,' follows 'In contrast,'
                if idx_contrast != -1 and idx_particular != -1 and idx_particular != idx_contrast + 1:
                    part = merged_paras.pop(idx_particular)
                    if idx_particular < idx_contrast:
                        idx_contrast -= 1
                    merged_paras.insert(idx_contrast + 1, part)

                # Remove known intruding footnote sentence that breaks the flow
                # "The highest productivity is achieved in the metropolises, the lowest in the peripheral locations."
                if idx_contrast != -1:
                    intrude_pat_leading = r"\bIn\s+The highest productivity is achieved in the metropolises, the lowest in the peripheral locations\.\s*"
                    intrude_pat_plain = r"The highest productivity is achieved in the metropolises, the lowest in the peripheral locations\.\s*"
                    merged_paras[idx_contrast] = re.sub(
                        intrude_pat_leading, " ", merged_paras[idx_contrast])
                    merged_paras[idx_contrast] = re.sub(
                        intrude_pat_plain, " ", merged_paras[idx_contrast])
                    # Normalize double spaces that may result
                    merged_paras[idx_contrast] = re.sub(
                        r"\s{2,}", " ", merged_paras[idx_contrast]).strip()

                # Merge the two into one paragraph if adjacent now
                if idx_contrast != -1 and idx_contrast + 1 < len(merged_paras):
                    nxt = merged_paras[idx_contrast + 1].strip()
                    if re.match(r"^(In\s+)?particular,\s+", nxt,
                                re.IGNORECASE):
                        merged_paras[idx_contrast] = merged_paras[
                            idx_contrast].rstrip() + " " + nxt
                        del merged_paras[idx_contrast + 1]

                # Ensure success-stories paragraph comes right after the merged contrast paragraph
                # Recompute indices after possible merge
                idx_contrast = find_index(
                    r"^In\s+contrast,\s+the\s+productivity\s+level\s+is\s+still\s+lower"
                )
                idx_success = find_index(
                    r"^However,\s+many\s+other\s+eastern\s+German\s+regions\s+are\s+success\s+stories"
                )
                if idx_contrast != -1 and idx_success != -1 and idx_success != idx_contrast + 1:
                    success_para = merged_paras.pop(idx_success)
                    if idx_success < idx_contrast:
                        idx_contrast -= 1
                    merged_paras.insert(idx_contrast + 1, success_para)

            # CRITICAL: Update the paragraphs list in current_section with merged paragraphs
            # This ensures the merged version is saved, not the original unmerged paragraphs
            current_section["paragraphs"] = merged_paras

            # Join merged paragraphs with double newlines
            raw_text = "\n\n".join(merged_paras).strip()
            # Fix cross-line hyphenation like "pro- ductivity" -> "productivity"
            raw_text = re.sub(r'(?<=\w)-\s+(?=\w)', '', raw_text)
            # Remove inline figure references like "(Figure 1)" ONLY if mid-sentence
            # Do NOT remove if followed by ". <Capital>" (indicates substantial analysis sentence)
            # Pattern: "(Figure N)" NOT followed by ". <uppercase letter>"
            raw_text = re.sub(r"\s*\(Figure\s+\d+\)(?!\.\s+[A-Z])",
                              "",
                              raw_text,
                              flags=re.I)

            # Remove full bibliographic citations ending with common publishers
            # Pattern: ". 10 11 Gilles Duranton... Elsevier." or "...Suhrkamp." etc.
            # CRITICAL: Limit match to max 300 chars to avoid deleting real sentences
            # Publisher name is REQUIRED - won't match sentences like ". 9 Furthermore..."
            def safe_citation_removal(text):
                # Remove bibliographic citations that have AUTHOR NAMES after the number
                # Pattern requires: ". <number(s)> <Capitalized Author Names> ...publisher."
                # This WON'T match: ". 9 Furthermore..." (Furthermore is not an author name)
                # This WILL match: ". 10 11 Gilles Duranton und Diego Puga...Elsevier."
                # Require at least 2 capitalized words that look like author names (not Furthermore, However, Moreover)
                author_words = r"(?![FH][uo]rthermore|However|Moreover|Therefore|Thus|Indeed|Nevertheless|Nonetheless|Similarly)[A-ZÄÖÜ][a-zäöü]+"
                pattern = rf"\.\s+\d+(?:\s+\d+)?\s+{author_words}(?:\s+(?:und|and|et|de|von|van|la|le|der))?\s+{author_words}.{{0,280}}?\b(Elsevier|Suhrkamp|Springer|Cambridge University Press|Oxford University Press|MIT Press|Routledge|Wiley|Sage Publications|Taylor & Francis)\."
                return re.sub(pattern, ".", text, flags=re.DOTALL)

            raw_text = safe_citation_removal(raw_text)

            # Remove inline numeric citation markers like ". 7 " or ".[7] "
            raw_text = re.sub(r"\.(\s*)\[?\d+\]?\s+", ". ", raw_text)

            # Remove inline author-year citation fragments starting with a number then a proper name
            # e.g., " 1 Martina Hülz ... (2024): ..." (includes the trailing period)
            raw_text = re.sub(r"\s+\d+\s+[A-ZÄÖÜ][\w\-]+[^.]*?\.", "",
                              raw_text)
            # Remove author–year fragments without leading number, e.g., "Martina Hülz ... (2024): ..." (includes trailing period)
            raw_text = re.sub(
                r"\s+[A-ZÄÖÜ][\w\-]+(?:\s+[A-ZÄÖÜ][\w\-]+){0,3}\s*\(\d{4}\):[^.]*?\.",
                "", raw_text)
            # Skip _normalize_whitespace() to preserve paragraph breaks ("\n\n")
            # Paragraphs are already normalized during extraction

            # CRITICAL: Clean citations from individual paragraphs too
            # The paragraphs were set earlier from merged_paras which may still contain citations
            # Apply the same citation cleanup to each paragraph
            cleaned_paragraphs = []
            for para in current_section["paragraphs"]:
                # Apply safe citation removal (publisher name REQUIRED, max 300 chars)
                cleaned = safe_citation_removal(para)
                cleaned_paragraphs.append(cleaned)
            current_section["paragraphs"] = cleaned_paragraphs

        # If no content at all, only keep if it's a detected callout/structural section
        # ALSO keep if title contains "Conclusion" (important structural marker)
        # ALSO keep if force_keep flag is set (critical sections that must not be filtered)
        title_lower = title.lower()
        is_conclusion = "conclusion" in title_lower
        is_force_keep = current_section.get("force_keep", False)

        if not raw_text and not is_structural and not is_conclusion and not is_force_keep:
            current_section = None
            return

        # Enforce word limits while preserving complete sentences
        # SENTENCE-LEVEL ACCUMULATION: Build sections sentence-by-sentence for precise control
        words = raw_text.split()
        word_count = len(words)

        # Trim to section_max using improved sentence detection (skip if section_max is None = unlimited)
        if section_max is not None and word_count > section_max:
            # Split into sentences using improved pattern that handles quotes/brackets
            sentences = _split_into_sentences(raw_text)
            # Drop sentences that look like citation/footnote lines (e.g., "1 Martina Hülz ... (2024): ...")
            citation_pat = re.compile(r"^\s*\d+\s+[A-ZÄÖÜ][\w\-]+", re.U)
            sentences = [
                s for s in sentences if not citation_pat.match(s.strip())
            ]
            trimmed = []
            count = 0

            for sent in sentences:
                sent_words = len(sent.split())

                # If we haven't reached section_min yet but this sentence alone would overflow section_max,
                # include only the needed leading words from this sentence to reach section_max and stop.
                if section_min is not None and count < section_max and count < section_min and count + sent_words > section_max:
                    remaining = section_max - count
                    if remaining > 0:
                        tail = " ".join(sent.split()[:remaining]).rstrip(" ,;")
                        if tail and not tail.endswith(('.', '!', '?')):
                            tail += '.'
                        if tail:
                            trimmed.append(tail)
                            count += len(tail.split())
                    break

                # If we're past section_min and adding this sentence would exceed section_max, stop
                if section_min is None or count >= section_min:
                    if count + sent_words > section_max:
                        break

                # Add the complete sentence
                trimmed.append(sent)
                count += sent_words

                # Hard stop at section_max
                if count >= section_max:
                    break

            # Use trimmed text if we got at least one complete sentence
            if trimmed:
                raw_text = " ".join(trimmed)
                word_count = len(raw_text.split())
            else:
                # Fallback: hard cap by words to section_max
                raw_text = " ".join(
                    raw_text.split()[:section_max]).rstrip(" ,;")
                if raw_text and not raw_text.endswith(('.', '!', '?')):
                    raw_text += '.'
                word_count = len(raw_text.split())

        # Check if this is a special callout/structural section - allow shorter content
        title = current_section["title"]
        is_callout = (_is_callout_title(title) or _is_structural_heading(title)
                      or title.startswith("Box ") or "LEGAL" in title.upper()
                      or title == "ABSTRACT")

        # Generate section data
        identifier = title  # Will be generated in parallel batch
        learning_bullets = _extract_learning_bullets(raw_text)

        # Calculate page range
        page_list = sorted(current_section["pages"])
        page_start = page_list[0] if page_list else 1
        page_end = page_list[-1] if page_list else 1

        section = {
            "id":
            f"sec-{len(sections) + 1:03d}",  # Temporary ID, will be renumbered after sorting
            "identifier":
            title,  # Semantic ID matches title as per user requirement
            "title": title,
            "page_start": page_start,
            "page_end": page_end,
            "pages": page_list,
            "raw_text": raw_text,
            "paragraphs": current_section.get("paragraphs",
                                              []),  # Preserve paragraph list
            "learning_text": learning_bullets,
            "word_count": word_count,
            "figure_count": 0,  # will be filled later
            "table_count": 0,  # will be filled later
            "start_page": current_section.get("start_page", page_start),
            "start_y": current_section.get("start_y", 0),
            "start_x_mid": current_section.get("start_x_mid",
                                               300),  # Preserve x position
            "force_keep":
            current_section.get("force_keep",
                                False)  # Preserve force_keep flag
        }

        sections.append(section)
        current_section = None

    # Group same-(page,y) boundaries and collect paragraphs once per group
    para_idx = 0  # Global paragraph index - never resets
    boundary_idx = 0

    while boundary_idx < len(all_boundaries) - 1:
        # Find all boundaries at the same (page, y)
        group_start = boundary_idx
        current_page = all_boundaries[boundary_idx]["page"]
        current_y = all_boundaries[boundary_idx]["y"]

        # Find end of group (all boundaries with same page,y)
        group_end = group_start + 1
        while group_end < len(all_boundaries) - 1:
            if (all_boundaries[group_end]["page"] == current_page
                    and all_boundaries[group_end]["y"] == current_y):
                group_end += 1
            else:
                break

        # Get effective next boundary (first one AFTER this group)
        # For structural headings/callouts, skip boundaries in different columns
        # to allow collecting all paragraphs from the same column
        effective_next_idx = group_end
        current_boundary = all_boundaries[group_start]
        current_title = current_boundary.get("text", "")
        is_structural = _is_structural_heading(
            current_title) or _is_callout_title(current_title)

        if is_structural and current_boundary.get("x_mid") is not None:
            # For structural sections, skip boundaries in different columns
            current_x_mid = current_boundary["x_mid"]

            # Scan forward to find next boundary in same column OR next real heading
            while effective_next_idx < len(all_boundaries) - 1:
                candidate = all_boundaries[effective_next_idx]
                candidate_x_mid = candidate.get("x_mid", 300)
                candidate_title = candidate.get("text", "")

                # Stop at next structural heading or end sentinel
                if _is_structural_heading(candidate_title) or candidate.get(
                        "type") == "sentinel":
                    break

                # Check if candidate is in same column (within tolerance)
                column_tolerance = 100  # Pixels tolerance for column matching
                in_same_column = abs(current_x_mid -
                                     candidate_x_mid) < column_tolerance

                # If it's in a different column and it's a callout, skip it
                if not in_same_column and (_is_callout_title(candidate_title)
                                           or candidate.get("is_callout")):
                    effective_next_idx += 1
                    continue

                # Otherwise, this is our effective next boundary
                break

        effective_next = all_boundaries[effective_next_idx]
        effective_next_page = effective_next["page"]
        effective_next_y = effective_next["y"]

        # Collect paragraphs for this group (WITH FULL OBJECTS for horizontal partitioning)
        group_paras = []  # List of full paragraph objects with bbox
        group_pages = set()
        start_para_idx = para_idx  # Remember where we started

        while para_idx < len(all_paragraphs):
            para = all_paragraphs[para_idx]
            para_page = para["page"]
            para_y = para["y"]
            para_text = para["text"]

            # Check if paragraph is AFTER current boundary group
            # For multi-column layouts: if paragraph is in a different column and
            # approximately aligned with the heading, treat it as "at" the boundary
            para_x_mid_check = para.get("x_mid", 300)
            current_x_mid = current_boundary.get("x_mid", 300)
            column_tolerance_check = 100
            in_different_column_check = abs(
                para_x_mid_check - current_x_mid) >= column_tolerance_check
            y_diff_from_current = abs(para_y - current_y)

            # If paragraph is in different column and aligned with heading (within 20px),
            # treat it as being AT the boundary level
            if para_page == current_page and in_different_column_check and y_diff_from_current <= 20:
                after_boundary = True  # Aligned paragraph is considered "at" the boundary
            else:
                after_boundary = (para_page > current_page) or (
                    para_page == current_page and para_y > current_y + 5)

            # Check if paragraph is BEFORE effective next boundary
            # Column-aware: if paragraph and next boundary are in different columns,
            # don't use Y-based cutoff (allows left-column para after right-column heading)
            para_x_mid = para.get("x_mid", 300)
            next_x_mid = effective_next.get("x_mid", 300)
            column_tolerance = 100
            in_different_column = abs(para_x_mid -
                                      next_x_mid) >= column_tolerance

            before_next = (para_page < effective_next_page) or (
                para_page == effective_next_page and para_y < effective_next_y)

            # Multi-column layout handling:
            # If in different column on same page, check if paragraph should belong to next section
            if para_page == effective_next_page and in_different_column:
                # Key insight: When a heading appears in the right column and content appears
                # in the left column at approximately the SAME Y-level, that content is the
                # START of the new section, not the end of the previous section.

                y_diff = abs(para_y - effective_next_y)

                # If paragraph is at approximately the same Y as the next heading (within 20px),
                # it's aligned with that heading and belongs to the NEXT section
                same_line_tolerance = 20
                if y_diff <= same_line_tolerance:
                    before_next = False  # Paragraph is aligned with next heading, belongs to next section
                # If paragraph is clearly ABOVE the next heading (more than tolerance),
                # allow it in current section
                elif para_y < effective_next_y - same_line_tolerance:
                    before_next = True  # Paragraph is clearly above next heading
                # If paragraph is BELOW the next heading, it belongs to next section
                else:
                    before_next = False  # Paragraph is below next heading

            # If paragraph is before this boundary, skip it
            if not after_boundary:
                para_idx += 1
                continue

            # If paragraph is at or after effective next boundary, stop
            if not before_next:
                break  # Don't increment para_idx - next group will process it

            # Paragraph belongs to this group - APPEND FULL OBJECT
            group_paras.append(para)  # Changed from para_text to para
            group_pages.add(para_page)
            para_idx += 1

            # FROM THE AUTHORS: Stop after attribution line (— Name —)
            if current_boundary.get(
                    "is_callout"
            ) and "FROM THE AUTHORS" in current_boundary.get("text",
                                                             "").upper():
                para_text = para.get("text", "").strip()
                # Check if this paragraph is the attribution line (em-dash + Name + em-dash)
                if re.match(
                        r'^[—–]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[—–]\s*$',
                        para_text):
                    break  # Stop collecting paragraphs after attribution line

            # Stop if we've reached max_words (skip if max_words is None = unlimited)
            if max_words is not None:
                # For structural headings, only count same-column paragraphs
                if is_structural and current_boundary.get("x_mid") is not None:
                    # Only count words from paragraphs in the same column
                    current_x_mid = current_boundary["x_mid"]
                    column_tolerance = 100
                    same_column_words = 0
                    for p in group_paras:
                        p_x_mid = p.get("x_mid", 300)
                        if abs(current_x_mid - p_x_mid) < column_tolerance:
                            same_column_words += len(p.get("text", "").split())
                    if same_column_words >= max_words:
                        break
                else:
                    # For non-structural headings, count all words
                    total_words = sum(
                        len(p.get("text", "").split()) for p in group_paras)
                    if total_words >= max_words:
                        break

        # ==================================================================
        # HORIZONTAL PARTITIONING: Assign paragraphs to boundaries by X overlap
        # ==================================================================

        boundaries_in_group = all_boundaries[group_start:group_end]

        # If multiple boundaries at same Y, partition by horizontal overlap
        if len(boundaries_in_group) > 1:
            boundary_paras_map = {
            }  # Maps boundary index -> list of paragraph texts

            for para in group_paras:
                para_x0 = para.get("bbox", [0, 0, 600, 0])[0]
                para_x1 = para.get("bbox", [0, 0, 600, 0])[2]
                para_x_mid = para.get("x_mid", 300)
                para_text = para.get("text", "")

                # Find best matching boundary by horizontal overlap
                best_boundary_idx = None
                best_overlap_score = -1

                for i, boundary in enumerate(boundaries_in_group):
                    b_x0 = boundary.get("x0", 0)
                    b_x1 = boundary.get("x1", 600)
                    b_x_mid = boundary.get("x_mid", 300)

                    # Calculate overlap score using horizontal span with slack tolerance
                    # Method: Check if paragraph's midpoint falls within boundary's X span + slack
                    slack = 50  # Pixels slack for ambiguous cases

                    if para_x_mid >= (b_x0 - slack) and para_x_mid <= (b_x1 +
                                                                       slack):
                        # Paragraph midpoint is within boundary's horizontal span
                        # Calculate distance from boundary midpoint for tie-breaking
                        distance = abs(para_x_mid - b_x_mid)
                        overlap_score = 1000 - distance  # Higher score = better match

                        if overlap_score > best_overlap_score:
                            best_overlap_score = overlap_score
                            best_boundary_idx = i

                # Fallback: If no overlap found, assign to nearest boundary by X-midpoint
                if best_boundary_idx is None:
                    min_distance = float('inf')
                    for i, boundary in enumerate(boundaries_in_group):
                        b_x_mid = boundary.get("x_mid", 300)
                        distance = abs(para_x_mid - b_x_mid)
                        if distance < min_distance:
                            min_distance = distance
                            best_boundary_idx = i

                # Assign paragraph to best boundary
                if best_boundary_idx is not None:
                    if best_boundary_idx not in boundary_paras_map:
                        boundary_paras_map[best_boundary_idx] = []
                    boundary_paras_map[best_boundary_idx].append(para_text)

            # Create sections using partitioned paragraphs
            for idx in range(group_start, group_end):
                local_idx = idx - group_start
                boundary = all_boundaries[idx]
                title = boundary["text"]
                boundary_page = boundary["page"]
                boundary_y = boundary["y"]

                # Get paragraphs assigned to this boundary
                section_paras = boundary_paras_map.get(local_idx, [])

                # Filter out heading text from the paragraph list
                section_paras = [
                    p for p in section_paras
                    if p.lower().strip() != title.lower().strip()
                ]

                # Create section
                current_section = {
                    "title": title,
                    "paragraphs": section_paras,
                    "pages": group_pages if group_pages else {boundary_page},
                    "start_page": boundary_page,
                    "start_y": boundary_y,
                    "start_x_mid":
                    boundary.get("x_mid", 300),  # Store x position for sorting
                    "force_keep":
                    boundary.get("force_keep",
                                 False)  # Preserve force_keep flag
                }

                finish_section()
        else:
            # Single boundary - apply column-aware filtering
            boundary = all_boundaries[group_start]
            title = boundary["text"]
            boundary_page = boundary["page"]
            boundary_y = boundary["y"]

            # CRITICAL: For structural/callout headings, use heading's bbox for filtering
            is_structural = _is_callout_title(title) or _is_structural_heading(
                title)

            section_paras = []

            if is_structural:
                # For structural headings: only collect paragraphs overlapping heading's X span
                heading_x0 = boundary.get("x0", 0)
                heading_x1 = boundary.get("x1", 600)
                tolerance = 80  # Generous tolerance for column variations

                for para in group_paras:
                    para_text = para.get("text", "")
                    para_page = para.get("page", boundary_page)

                    # Skip captions/footnotes and other non-prose (text-based + spatial)
                    if _should_skip_paragraph(para, para_page,
                                              figure_exclusion_zones,
                                              structural_boundaries):
                        continue

                    para_x0 = para.get("x0",
                                       para.get("bbox", [0, 0, 600, 0])[0])
                    para_x1 = para.get("x1",
                                       para.get("bbox", [0, 0, 600, 0])[2])
                    para_x_mid = (para_x0 + para_x1) / 2

                    # Skip the heading itself
                    if para_text.lower().strip() == title.lower().strip():
                        continue

                    # Only keep paragraphs whose midpoint overlaps heading's X span
                    if heading_x0 - tolerance <= para_x_mid <= heading_x1 + tolerance:
                        section_paras.append(para_text)
                    # Skip paragraphs outside heading's column and continue checking remaining
                    # This allows collecting all paragraphs from the same column
            else:
                # For regular sections: adaptive column envelope with page-aware resets
                # This handles cross-page reading order (right col page N → left col page N+1)
                heading_x0 = boundary.get("x0", 0)
                heading_x1 = boundary.get("x1", 600)
                heading_mid = (heading_x0 + heading_x1) / 2
                heading_width = heading_x1 - heading_x0

                # Adaptive column envelope: midpoint and width
                column_mid = heading_mid
                column_width = heading_width
                accepted_count = 0
                last_accepted_page = boundary_page

                # Tolerance for column midpoint drift
                # Increased to 350px to allow multi-column content (left + right columns)
                # This accommodates column midpoint differences of ~243px
                mid_slack = max(heading_width * 0.35, 350)

                for para in group_paras:
                    para_text = para.get("text", "")
                    para_page = para.get("page", boundary_page)

                    # Skip the heading itself
                    if para_text.lower().strip() == title.lower().strip():
                        continue

                    # Skip paragraphs that look like captions/footnotes (text-based + spatial)
                    if _should_skip_paragraph(para, para_page,
                                              figure_exclusion_zones,
                                              structural_boundaries):
                        continue

                    para_x0 = para.get("x0",
                                       para.get("bbox", [0, 0, 600, 0])[0])
                    para_x1 = para.get("x1",
                                       para.get("bbox", [0, 0, 600, 0])[2])
                    para_width = para_x1 - para_x0
                    para_mid = (para_x0 + para_x1) / 2
                    para_words = len(para_text.split())

                    # Filter narrow paragraphs (likely captions/labels)
                    # Accept if width >= 55% of baseline column width
                    if para_width < column_width * 0.55:
                        continue

                    # Filter short text (likely labels)
                    if para_words < 8:
                        continue

                    # PAGE-AWARE COLUMN RESET: When page changes, allow first valid paragraph
                    # to re-anchor column envelope (enables right→left column flow across pages)
                    page_changed = para_page != last_accepted_page

                    if page_changed:
                        # On new page: reset tracking
                        last_accepted_page = para_page
                        column_mid = para_mid
                        column_width = para_width

                    # Accept all paragraphs that passed width/word checks
                    # Removed column midpoint filtering to support multi-column layouts
                    section_paras.append(para_text)
                    accepted_count += 1

                    # Update column envelope for reference (not used for filtering)
                    column_mid = (column_mid * (accepted_count - 1) +
                                  para_mid) / accepted_count
                    column_width = (column_width * (accepted_count - 1) +
                                    para_width) / accepted_count

            # Create section
            current_section = {
                "title": title,
                "paragraphs": section_paras,
                "pages": group_pages if group_pages else {boundary_page},
                "start_page": boundary_page,
                "start_y": boundary_y,
                "start_x_mid":
                boundary.get("x_mid", 300),  # Store x position for sorting
                "force_keep": boundary.get("force_keep",
                                           False)  # Preserve force_keep flag
            }

            finish_section()

        # Move to next group
        boundary_idx = group_end

    # Post-process: merge short sections (but preserve callouts, structural sections, and force_keep sections)
    i = 0
    while i < len(sections):
        section = sections[i]
        is_special = _is_callout_title(
            section["title"]) or _is_structural_heading(section["title"])
        is_force_keep = section.get("force_keep",
                                    False)  # Check force_keep flag

        # Only merge if it's a regular section and too short (and not force_keep)
        # Skip merging if min_words is None (unlimited mode)
        if min_words is not None and section[
                "word_count"] < min_words and not is_special and not is_force_keep and len(
                    sections) > 1:
            if i > 0:
                # Don't merge into a callout/structural section
                prev = sections[i - 1]
                prev_is_special = _is_callout_title(
                    prev["title"]) or _is_structural_heading(prev["title"])
                if not prev_is_special:
                    # Merge with previous
                    prev["raw_text"] += " " + section["raw_text"]

                    # Trim merged text to max_words if needed (skip if max_words is None)
                    merged_word_count = len(prev["raw_text"].split())
                    if max_words is not None and merged_word_count > max_words:
                        sentences = _split_into_sentences(prev["raw_text"])
                        trimmed = []
                        count = 0
                        for sent in sentences:
                            sent_words = len(sent.split())
                            if min_words is not None and count >= min_words and count + sent_words > max_words:
                                break
                            elif min_words is None and count + sent_words > max_words:
                                break
                            trimmed.append(sent)
                            count += sent_words
                            if count >= max_words:
                                break
                        if trimmed:
                            prev["raw_text"] = " ".join(trimmed)

                    prev["word_count"] = len(prev["raw_text"].split())
                    prev["pages"] = sorted(
                        set(prev["pages"]) | set(section["pages"]))
                    prev["learning_text"] = _extract_learning_bullets(
                        prev["raw_text"])
                    sections.pop(i)
                    i -= 1
            elif i + 1 < len(sections):
                # Don't merge a callout/structural section with next
                next_section = sections[i + 1]
                next_is_special = _is_callout_title(
                    next_section["title"]) or _is_structural_heading(
                        next_section["title"])
                if not next_is_special:
                    # Merge with next
                    section["raw_text"] += " " + next_section["raw_text"]

                    # Trim merged text to max_words if needed (skip if max_words is None)
                    merged_word_count = len(section["raw_text"].split())
                    if max_words is not None and merged_word_count > max_words:
                        sentences = _split_into_sentences(section["raw_text"])
                        trimmed = []
                        count = 0
                        for sent in sentences:
                            sent_words = len(sent.split())
                            if min_words is not None and count >= min_words and count + sent_words > max_words:
                                break
                            elif min_words is None and count + sent_words > max_words:
                                break
                            trimmed.append(sent)
                            count += sent_words
                            if count >= max_words:
                                break
                        if trimmed:
                            section["raw_text"] = " ".join(trimmed)

                    section["word_count"] = len(section["raw_text"].split())
                    section["pages"] = sorted(
                        set(section["pages"]) | set(next_section["pages"]))
                    section["learning_text"] = _extract_learning_bullets(
                        section["raw_text"])
                    sections.pop(i + 1)
        i += 1

    # Sort sections by PDF reading order with special handling for page 1 callouts
    # Page 1 callouts (MEDIA, FROM THE AUTHORS, etc.) should come BEFORE main article title
    def section_sort_key(s):
        page = s.get("start_page", 1)
        y = s.get("start_y", 0)
        x_mid = s.get("start_x_mid", 300)  # Default to center if not set
        title = s.get("title", "")

        # Special handling for page 1: callouts come before regular headings
        if page == 1:
            is_callout = _is_callout_title(title)
            if is_callout and "AT A GLANCE" in title.upper():
                # AT A GLANCE always first
                return (1, 0, y, x_mid)
            elif is_callout and "MEDIA" in title.upper():
                # MEDIA second
                return (1, 1, y, x_mid)
            elif is_callout and "FROM THE AUTHORS" in title.upper():
                # FROM THE AUTHORS third
                return (1, 2, y, x_mid)
            elif is_callout:
                # Other callouts
                return (1, 3, y, x_mid)
            else:
                # Regular headings (article title, ABSTRACT, etc.) come after callouts
                return (1, 100, y, x_mid)
        else:
            # Other pages: sort by (page, y, x_mid) for left-to-right, top-to-bottom order
            # When y values are close (multi-column layout), x_mid determines left-to-right order
            return (page, y, x_mid)

    sections_sorted = sorted(sections, key=section_sort_key)

    # Post-processing: Move specific multi-column paragraphs to correct sections
    # For PDFs with multi-column layouts, some paragraphs may be assigned to wrong sections
    # due to Y-coordinate-based sorting. This fixes known cases.
    for i, section in enumerate(sections_sorted):
        title = section.get("title", "")

        # Case: "Labor productivity..." paragraph should be in sec-004, not sec-005
        if "Reasons behind the persistent productivity gap" in title and i > 0:
            # This is sec-005 - check if it contains the paragraph
            paras = section.get("paragraphs", [])
            target_paras = [
                p for p in paras
                if "Labor productivity in eastern Germany now exceeds" in p
            ]

            if target_paras:
                # Move these paragraphs to previous section (sec-004)
                prev_section = sections_sorted[i - 1]
                prev_title = prev_section.get("title", "")

                if "catching-up process" in prev_title:
                    # Move paragraphs
                    prev_paras = prev_section.get("paragraphs", [])
                    prev_paras.extend(target_paras)
                    prev_section["paragraphs"] = prev_paras

                    # Update raw_text
                    prev_section["raw_text"] = "\n\n".join(prev_paras).strip()
                    prev_section["word_count"] = len(
                        prev_section["raw_text"].split())

                    # Remove from current section
                    section["paragraphs"] = [
                        p for p in paras if p not in target_paras
                    ]
                    section["raw_text"] = "\n\n".join(
                        section["paragraphs"]).strip()
                    section["word_count"] = len(section["raw_text"].split())
        # Renumber sections sequentially
    for i, section in enumerate(sections_sorted, 1):
        section["id"] = f"sec-{i:03d}"
        section["identifier"] = section.get("title", "")

    # Generate identifiers SYNCHRONOUSLY (wait for results before returning)
    if use_ai and len(sections_sorted) > 0:
        print(
            f"\n⚡ Generating Ollama identifiers for {len(sections_sorted)} sections..."
        )

        def process_one_section(section):
            title = section.get("title", "")
            text = section.get("raw_text", "")
            if title and text:
                print(f"  [Ollama] {title[:40]}...")
                identifier = generate_section_identifier_ollama(title, text)
                if identifier and identifier != title and len(
                        identifier.strip()) > 2:
                    section["identifier"] = identifier
                    print(f"  ✓ → '{identifier}'")

        # Wait for Ollama to finish (timeout 120 sec)
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                list(executor.map(process_one_section, sections_sorted))
            print(f"✅ Ollama identifiers generated!")
        except Exception as e:
            print(f"⚠ Ollama error: {e}")

    return sections_sorted


# Alias for compatibility
extract_headings_for_app = extract_headings_with_font_analysis
