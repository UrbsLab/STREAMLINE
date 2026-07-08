from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    hgap: float = 0.12 * inch
    vgap: float = 0.12 * inch
    cell_pad: float = 0.06 * inch


def _fit_into(w: float, h: float, max_w: float, max_h: float) -> tuple[float, float]:
    """Scale (w,h) to fit within (max_w,max_h) preserving aspect ratio."""
    if w <= 0 or h <= 0:
        return max_w, max_h
    s = min(max_w / w, max_h / h)
    return w * s, h * s


def _draw_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, lw: float = 1.0) -> None:
    c.setLineWidth(lw)
    c.rect(x, y, w, h, stroke=1, fill=0)


def _draw_header(
    c: canvas.Canvas,
    title: str,
    subtitle: str | None = None,
    left: float = 0.75 * inch,
    top: float = 10.75 * inch,
    width: float = 7.0 * inch,
) -> float:
    """
    Draw a boxed header similar to the attached report.
    Returns the y coordinate below the header block.
    """
    header_h = 0.55 * inch if subtitle else 0.40 * inch
    y = top - header_h

    _draw_box(c, left, y, width, header_h, lw=1.2)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left + 0.12 * inch, y + header_h - 0.26 * inch, title)

    if subtitle:
        c.setFont("Helvetica", 9)
        c.drawString(left + 0.12 * inch, y + 0.12 * inch, subtitle)

    return y - 0.15 * inch


def draw_image_grid_page(
    c: canvas.Canvas,
    title: str,
    images: Sequence[tuple[str, Path]],
    gridspec: GridSpec,
    subtitle: str | None = None,
    page_size=letter,
    footer: str | None = "Generated with STREAMLINE",
) -> None:
    """
    Draw a single page with:
      - boxed header
      - a grid of images with optional captions
      - optional footer
    """
    page_w, page_h = page_size

    margin_l = 0.75 * inch
    margin_r = 0.75 * inch
    margin_b = 0.65 * inch
    top = page_h - 0.65 * inch
    content_w = page_w - margin_l - margin_r

    y0 = _draw_header(c, title=title, subtitle=subtitle, left=margin_l, top=top, width=content_w)

    # Grid bounding box (leave room for footer)
    footer_h = 0.28 * inch if footer else 0.0
    grid_top = y0
    grid_bottom = margin_b + footer_h
    grid_h = grid_top - grid_bottom

    cell_w = (content_w - (gridspec.cols - 1) * gridspec.hgap) / gridspec.cols
    cell_h = (grid_h - (gridspec.rows - 1) * gridspec.vgap) / gridspec.rows

    # Place images in row-major order
    for idx, (caption, img_path) in enumerate(images[: gridspec.rows * gridspec.cols]):
        r = idx // gridspec.cols
        col = idx % gridspec.cols

        x = margin_l + col * (cell_w + gridspec.hgap)
        y = grid_top - (r + 1) * cell_h - r * gridspec.vgap

        # cell outline (matches “boxed sections” vibe)
        _draw_box(c, x, y, cell_w, cell_h, lw=1.0)

        # reserve a small caption strip at top of each cell
        cap_h = 0.22 * inch if caption else 0.0
        if caption:
            c.setFont("Helvetica-Bold", 9)
            c.drawString(x + gridspec.cell_pad, y + cell_h - cap_h + 0.06 * inch, caption)

        # image area
        img_x = x + gridspec.cell_pad
        img_y = y + gridspec.cell_pad
        img_max_w = cell_w - 2 * gridspec.cell_pad
        img_max_h = cell_h - 2 * gridspec.cell_pad - cap_h

        ir = ImageReader(str(img_path))
        iw, ih = ir.getSize()
        dw, dh = _fit_into(iw, ih, img_max_w, img_max_h)

        # center within image area
        cx = img_x + (img_max_w - dw) / 2.0
        cy = img_y + (img_max_h - dh) / 2.0
        c.drawImage(ir, cx, cy, width=dw, height=dh, preserveAspectRatio=True, mask="auto")

    if footer:
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(page_w / 2.0, margin_b - 0.10 * inch, footer)


def build_pdf(
    pdf_path: Path,
    pages: Iterable[dict],
    page_size=letter,
) -> None:
    """
    pages: each dict describes a page:
      {
        "title": str,
        "subtitle": str|None,
        "images": list[ (caption: str, img_path: Path) ],
        "grid": GridSpec(...)
      }
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)

    for p in pages:
        draw_image_grid_page(
            c,
            title=p["title"],
            subtitle=p.get("subtitle"),
            images=p["images"],
            gridspec=p["grid"],
            page_size=page_size,
            footer=p.get("footer", "Generated with STREAMLINE"),
        )
        c.showPage()

    c.save()
