from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber


# ----------------- small helpers -----------------

def _grid_dims(grid: List[List[str]]) -> Tuple[int, int]:
    nrows = len(grid)
    ncols = max((len(r) for r in grid), default=0)
    return nrows, ncols


def _normalize_grid(raw: List[List[object]]) -> List[List[str]]:
    nrows = len(raw)
    ncols = max((len(r) for r in raw), default=0)
    out: List[List[str]] = []
    for r in range(nrows):
        row: List[str] = []
        for c in range(ncols):
            try:
                v = raw[r][c]
            except Exception:
                v = ""
            s = ("" if v is None else str(v)).strip()
            row.append(s)
        out.append(row)
    return out


def _fill_ratio(grid: List[List[str]]) -> float:
    nrows, ncols = _grid_dims(grid)
    total = nrows * ncols
    if total == 0:
        return 0.0
    nonempty = sum(1 for r in grid for v in r if v)
    return nonempty / total


def _looks_like_header(text: str) -> bool:
    alpha = sum(ch.isalpha() for ch in text)
    return alpha >= 2


def _has_numbers(grid: List[List[str]]) -> bool:
    cnt = 0
    for r in grid:
        for v in r:
            if any(ch.isdigit() for ch in v):
                cnt += 1
                if cnt >= 3:
                    return True
    return False


def _table_quality(grid: List[List[str]]) -> int:
    nrows, ncols = _grid_dims(grid)
    ratio = _fill_ratio(grid)

    score = 0
    if nrows >= 3:
        score += 1
    if ncols >= 2:
        score += 1
    if ratio >= 0.35:
        score += 1
    if _has_numbers(grid):
        score += 1
    if nrows > 0:
        header_hits = sum(1 for v in grid[0] if _looks_like_header(v))
        if header_hits >= 2:
            score += 1
    return score


def _clip_png(doc: fitz.Document, page_idx: int, bbox: Tuple[float, float, float, float],
              out_path: Path, zoom: float = 2.0) -> None:
    page = doc[page_idx - 1]
    rect = fitz.Rect(*bbox)
    pm = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.save(str(out_path))


# ----------------- line-grid support (reject charts) -----------------

def _line_support(pl_page: pdfplumber.page.Page, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, float, float]:
    x0, y0, x1, y1 = bbox

    def inside(seg):
        sx0 = float(seg.get("x0", seg.get("x1", 0)))
        sx1 = float(seg.get("x1", seg.get("x0", 0)))
        sy0 = float(seg.get("top", seg.get("y0", 0)))
        sy1 = float(seg.get("bottom", seg.get("y1", 0)))
        return not (sx1 < x0 or sx0 > x1 or sy1 < y0 or sy0 > y1)

    h_count = v_count = 0
    h_len = v_len = 0.0

    edges = list(getattr(pl_page, "edges", []))
    curves = list(getattr(pl_page, "curves", []))

    for e in edges:
        if not inside(e):
            continue
        x0e, x1e = float(e.get("x0", 0)), float(e.get("x1", 0))
        top, bottom = float(e.get("top", 0)), float(e.get("bottom", 0))
        dx = abs(x1e - x0e)
        dy = abs(bottom - top)
        if dx >= dy:
            h_count += 1
            h_len += dx
        else:
            v_count += 1
            v_len += dy

    for c in curves:
        if not inside(c):
            continue
        pts = c.get("pts") or []
        if len(pts) >= 2:
            (cx0, cy0), (cx1, cy1) = pts[0], pts[-1]
            dx, dy = abs(cx1 - cx0), abs(cy1 - cy0)
            if dx >= dy:
                h_count += 1
                h_len += dx
            else:
                v_count += 1
                v_len += dy

    return h_count, v_count, h_len, v_len


# ----------------- main extractor -----------------

def extract_tables_with_coords(
    pdf_path: Path,
    render_dir: Optional[Path] = None,
    zoom: float = 2.0,
    min_quality_score: int = 4,
    min_h_edges: int = 2,
    min_v_edges: int = 2,
    max_page_frac: float = 0.55,
) -> List[Dict[str, Any]]:
    pdf_path = Path(pdf_path)
    results: List[Dict[str, Any]] = []

    lattice_settings = dict(
        vertical_strategy="lines",
        horizontal_strategy="lines",
        snap_tolerance=3,
        join_tolerance=3,
        edge_min_length=35,
        min_words_vertical=1,
        min_words_horizontal=1,
    )

    with pdfplumber.open(str(pdf_path)) as pl_doc, fitz.open(str(pdf_path)) as fz_doc:
        for pageno, pl_page in enumerate(pl_doc.pages, start=1):
            fz_page = fz_doc[pageno - 1]
            page_area = float(fz_page.rect.width * fz_page.rect.height)

            try:
                candidates = pl_page.find_tables(table_settings=lattice_settings)
            except TypeError:
                candidates = pl_page.find_tables(lattice_settings)
            except Exception:
                candidates = []

            k = 1
            for tbl in candidates:
                try:
                    raw = tbl.extract()
                except Exception:
                    continue

                grid = _normalize_grid(raw)
                quality = _table_quality(grid)
                if quality < min_quality_score:
                    continue

                try:
                    x0, top, x1, bottom = tbl.bbox
                except Exception:
                    continue

                bbox = (float(x0), float(top), float(x1), float(bottom))
                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if page_area > 0 and (box_area / page_area) > max_page_frac:
                    continue

                h_cnt, v_cnt, h_len, v_len = _line_support(pl_page, bbox)
                if h_cnt < min_h_edges or v_cnt < min_v_edges:
                    continue

                nrows, ncols = _grid_dims(grid)
                fill = _fill_ratio(grid)

                conf = round(
                    quality + min(h_cnt, 6) / 6 + min(v_cnt, 6) / 6 + min(fill, 1.0),
                    2,
                )

                cells = []
                if hasattr(tbl, "cells") and isinstance(tbl.cells, list):
                    for cell in tbl.cells[:200]:
                        try:
                            if isinstance(cell, dict):
                                r = int(cell.get("row", -1))
                                c = int(cell.get("col", -1))
                                bx0 = float(cell.get("x0", 0.0))
                                bt = float(cell.get("top", 0.0))
                                bx1 = float(cell.get("x1", 0.0))
                                bb = float(cell.get("bottom", 0.0))
                                text = (cell.get("text") or "").strip()
                            else:
                                continue
                            if r >= 0 and c >= 0:
                                cells.append(
                                    {"r": r, "c": c, "text": text, "bbox": [bx0, bt, bx1, bb]}
                                )
                        except Exception:
                            continue
                
                # FALLBACK: If no cells found, try to extract from pdfplumber table structure
                if not cells and grid and nrows > 0 and ncols > 0:
                    try:
                        # Try to get actual row/column boundaries from pdfplumber
                        row_y = []
                        col_x = []
                        
                        # Extract row boundaries
                        if hasattr(tbl, 'rows') and tbl.rows:
                            for row in tbl.rows:
                                if hasattr(row, 'bbox'):
                                    row_y.append(float(row.bbox[1]))  # top
                            row_y.append(float(tbl.rows[-1].bbox[3]))  # bottom of last row
                        
                        # Extract column boundaries  
                        if hasattr(tbl, 'columns') and tbl.columns:
                            for col in tbl.columns:
                                if hasattr(col, 'bbox'):
                                    col_x.append(float(col.bbox[0]))  # left
                            col_x.append(float(tbl.columns[-1].bbox[2]))  # right of last col
                        
                        # If we got actual boundaries, use them
                        if row_y and col_x and len(row_y) == nrows + 1 and len(col_x) == ncols + 1:
                            for r in range(min(nrows, 50)):
                                for c in range(min(ncols, 20)):
                                    try:
                                        text = grid[r][c] if r < len(grid) and c < len(grid[r]) else ""
                                        cells.append({
                                            "r": r,
                                            "c": c,
                                            "text": text.strip(),
                                            "bbox": [round(col_x[c], 2), round(row_y[r], 2), 
                                                    round(col_x[c + 1], 2), round(row_y[r + 1], 2)]
                                        })
                                    except Exception:
                                        continue
                        else:
                            # Fallback to uniform division if actual boundaries not available
                            cell_width = (bbox[2] - bbox[0]) / ncols
                            cell_height = (bbox[3] - bbox[1]) / nrows
                            
                            for r in range(min(nrows, 50)):
                                for c in range(min(ncols, 20)):
                                    try:
                                        text = grid[r][c] if r < len(grid) and c < len(grid[r]) else ""
                                        cx0 = bbox[0] + (c * cell_width)
                                        cy0 = bbox[1] + (r * cell_height)
                                        cx1 = cx0 + cell_width
                                        cy1 = cy0 + cell_height
                                        
                                        cells.append({
                                            "r": r,
                                            "c": c,
                                            "text": text.strip(),
                                            "bbox": [round(cx0, 2), round(cy0, 2), round(cx1, 2), round(cy1, 2)]
                                        })
                                    except Exception:
                                        continue
                    except Exception:
                        pass  # If extraction fails, cells will remain empty

                entry = {
                    "id": f"tbl-{pageno:03d}-{k:02d}",
                    "page": pageno,
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                    "nrows": nrows,
                    "ncols": ncols,
                    "data": grid,
                    "cells": cells,
                    "confidence": conf,
                }

                tmp_png = Path("data") / "runs" / "tmp" / "tables" / f"{entry['id']}.png"
                _clip_png(fz_doc, pageno, bbox, tmp_png, zoom=2.0)
                entry["image_path"] = str(tmp_png)

                results.append(entry)
                k += 1

    results.sort(key=lambda t: t.get("confidence", 0), reverse=True)

    if render_dir:
        render_dir = Path(render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        for t in results:
            old = Path(t["image_path"])
            if old.exists():
                new = render_dir / old.name
                try:
                    new.write_bytes(old.read_bytes())
                    t["image_path"] = str(new)
                except Exception:
                    pass

    return results
