"""Bead pattern rendering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .palette import PaletteEntry, load_palette_entries


@dataclass
class PatternLegendItem:
    """Legend item for a bead pattern."""

    entry: PaletteEntry
    count: int
    symbol: str


_FALLBACK_SYMBOLS = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%&*+=?"
)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _assign_symbols(entries: List[PaletteEntry]) -> Dict[Tuple[int, int, int], str]:
    used = set()
    fallback_idx = 0
    symbols: Dict[Tuple[int, int, int], str] = {}

    for entry in entries:
        raw = entry.symbol.strip() if entry.symbol else ""
        symbol = ""
        if len(raw) == 1 and raw not in used:
            symbol = raw
        else:
            while fallback_idx < len(_FALLBACK_SYMBOLS):
                candidate = _FALLBACK_SYMBOLS[fallback_idx]
                fallback_idx += 1
                if candidate not in used:
                    symbol = candidate
                    break
        if not symbol:
            symbol = "?"
        used.add(symbol)
        symbols[entry.rgb] = symbol

    return symbols


def _contrast_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)


def render_bead_pattern(
    grid_img: Image.Image,
    palette_name: str,
    title: Optional[str] = None,
    cell_size: int = 18,
    major_every: int = 5,
    include_symbols: bool = True,
) -> Image.Image:
    """Render a printable bead pattern image from a pixel grid.

    Args:
        grid_img: Pixel-grid image (one pixel per bead).
        palette_name: Palette name or path.
        title: Optional title for the pattern.
        cell_size: Pixel size of each bead cell in the output.
        major_every: Major gridline interval.
        include_symbols: Whether to overlay symbols on each cell.

    Returns:
        PIL Image with the rendered pattern.
    """
    grid = grid_img.convert("RGB")
    width, height = grid.size

    entries = load_palette_entries(palette_name)
    entries_by_rgb: Dict[Tuple[int, int, int], PaletteEntry] = {
        entry.rgb: entry for entry in entries
    }

    counts: Dict[Tuple[int, int, int], int] = {}
    pixels = grid.load()
    for y in range(height):
        for x in range(width):
            rgb = pixels[x, y]
            counts[rgb] = counts.get(rgb, 0) + 1

    legend_entries: List[PaletteEntry] = []
    for rgb in counts:
        entry = entries_by_rgb.get(rgb)
        if entry is None:
            r, g, b = rgb
            entry = PaletteEntry(
                reference_code="",
                name=f"#{r:02X}{g:02X}{b:02X}",
                symbol="",
                rgb=rgb,
                lab=(0.0, 0.0, 0.0),
                contributor="",
            )
        legend_entries.append(entry)

    symbol_map = _assign_symbols(legend_entries)

    legend_items: List[PatternLegendItem] = []
    for entry in legend_entries:
        legend_items.append(
            PatternLegendItem(
                entry=entry,
                count=counts[entry.rgb],
                symbol=symbol_map[entry.rgb],
            )
        )

    legend_items.sort(key=lambda item: (-item.count, item.entry.name.lower()))

    title_text = title or "Bead Pattern"
    total_beads = width * height

    dummy = Image.new("RGB", (1, 1), "white")
    dummy_draw = ImageDraw.Draw(dummy)
    font = ImageFont.load_default()
    line_w, line_h = _text_size(dummy_draw, "Ag", font)

    header_lines = [
        title_text,
        f"Palette: {palette_name}",
        f"Grid: {width}x{height} beads | Colors: {len(legend_items)} | Total: {total_beads}",
    ]
    header_height = line_h * len(header_lines) + 2 * (len(header_lines) - 1)

    margin = 20
    col_label_h = line_h + 4
    max_row_label = str(height)
    row_label_w, _ = _text_size(dummy_draw, max_row_label, font)
    row_label_w += 6

    grid_left = margin + row_label_w
    grid_top = margin + header_height + col_label_h
    grid_w_px = width * cell_size
    grid_h_px = height * cell_size

    legend_header = "Legend"
    legend_lines = []
    for item in legend_items:
        ref = f"{item.entry.reference_code} " if item.entry.reference_code else ""
        legend_lines.append(
            f"{item.symbol} {ref}{item.entry.name} x{item.count}"
        )
    legend_text_width = 0
    for text in legend_lines + [legend_header]:
        text_w, _ = _text_size(dummy_draw, text, font)
        legend_text_width = max(legend_text_width, text_w)

    legend_swatch = max(12, min(cell_size, 18))
    legend_row_h = max(line_h + 4, legend_swatch + 4)
    legend_width = legend_swatch + 8 + legend_text_width + margin

    legend_x = grid_left + grid_w_px + margin
    legend_y = margin + header_height
    legend_height = line_h + 4 + len(legend_items) * legend_row_h

    total_w = max(legend_x + legend_width + margin, grid_left + grid_w_px + margin)
    total_h = max(grid_top + grid_h_px + margin, legend_y + legend_height + margin)

    pattern = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(pattern)

    # Header
    header_y = margin
    for line in header_lines:
        draw.text((margin, header_y), line, fill=(0, 0, 0), font=font)
        header_y += line_h + 2

    # Column labels
    col_label_y = margin + header_height
    for col in range(major_every, width + 1, major_every):
        label = str(col)
        text_w, text_h = _text_size(draw, label, font)
        x = grid_left + col * cell_size - text_w / 2
        y = col_label_y
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    # Row labels
    for row in range(major_every, height + 1, major_every):
        label = str(row)
        text_w, text_h = _text_size(draw, label, font)
        x = grid_left - text_w - 4
        y = grid_top + row * cell_size - text_h / 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    # Grid cells
    for y in range(height):
        for x in range(width):
            rgb = pixels[x, y]
            x0 = grid_left + x * cell_size
            y0 = grid_top + y * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size, y0 + cell_size], fill=rgb
            )
            if include_symbols:
                symbol = symbol_map.get(rgb, "?")
                text_w, text_h = _text_size(draw, symbol, font)
                tx = x0 + (cell_size - text_w) / 2
                ty = y0 + (cell_size - text_h) / 2
                draw.text(
                    (tx, ty), symbol, fill=_contrast_color(rgb), font=font
                )

    # Grid lines
    for i in range(width + 1):
        x = grid_left + i * cell_size
        line_width = 2 if (i % major_every == 0) else 1
        draw.line(
            [(x, grid_top), (x, grid_top + grid_h_px)],
            fill=(0, 0, 0),
            width=line_width,
        )
    for i in range(height + 1):
        y = grid_top + i * cell_size
        line_width = 2 if (i % major_every == 0) else 1
        draw.line(
            [(grid_left, y), (grid_left + grid_w_px, y)],
            fill=(0, 0, 0),
            width=line_width,
        )

    # Legend
    draw.text((legend_x, legend_y), legend_header, fill=(0, 0, 0), font=font)
    legend_cursor = legend_y + line_h + 4
    for item, text in zip(legend_items, legend_lines):
        swatch_y = legend_cursor + (legend_row_h - legend_swatch) / 2
        draw.rectangle(
            [
                (legend_x, swatch_y),
                (legend_x + legend_swatch, swatch_y + legend_swatch),
            ],
            fill=item.entry.rgb,
            outline=(0, 0, 0),
        )
        draw.text(
            (legend_x + legend_swatch + 6, legend_cursor),
            text,
            fill=(0, 0, 0),
            font=font,
        )
        legend_cursor += legend_row_h

    return pattern
