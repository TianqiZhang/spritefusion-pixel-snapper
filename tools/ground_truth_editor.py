#!/usr/bin/env python
"""Ground truth grid editor — Tkinter GUI for creating/editing ground truth cuts.

Usage:
    python tools/ground_truth_editor.py testdata/ash.png
    python tools/ground_truth_editor.py testdata/ash.png --resolution-hint 64
    python tools/ground_truth_editor.py testdata/ --missing-only
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import List, Literal, Optional, Tuple

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageTk

from pixel_snapper import Config, process_image_bytes_with_grid
from pixel_snapper.ground_truth import (
    GroundTruth,
    get_test_images,
    ground_truth_dir,
    load_ground_truth,
    save_ground_truth,
)

# ---------------------------------------------------------------------------
# GridState — pure data model with undo/redo
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    col_cuts: List[int]
    row_cuts: List[int]


@dataclass
class GridState:
    """Mutable grid state with undo/redo support."""

    col_cuts: List[int]
    row_cuts: List[int]
    image_width: int
    image_height: int
    _undo_stack: List[_Snapshot] = field(default_factory=list, repr=False)
    _redo_stack: List[_Snapshot] = field(default_factory=list, repr=False)
    _dirty: bool = field(default=False, repr=False)

    # -- snapshot helpers ---------------------------------------------------

    def _push_undo(self) -> None:
        self._undo_stack.append(_Snapshot(list(self.col_cuts), list(self.row_cuts)))
        self._redo_stack.clear()
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    # -- mutations ----------------------------------------------------------

    def move_cut(self, axis: Literal["col", "row"], index: int, new_pos: int, *, record_undo: bool = True) -> None:
        cuts = self.col_cuts if axis == "col" else self.row_cuts
        if index <= 0 or index >= len(cuts) - 1:
            return  # endpoints immutable
        lo = cuts[index - 1] + 1
        hi = cuts[index + 1] - 1
        new_pos = max(lo, min(hi, new_pos))
        if new_pos == cuts[index]:
            return
        if record_undo:
            self._push_undo()
        else:
            self._dirty = True
        cuts[index] = new_pos

    def add_cut(self, axis: Literal["col", "row"], pos: int) -> None:
        cuts = self.col_cuts if axis == "col" else self.row_cuts
        dim = self.image_width if axis == "col" else self.image_height
        if pos <= 0 or pos >= dim:
            return
        if pos in cuts:
            return
        self._push_undo()
        cuts.append(pos)
        cuts.sort()

    def delete_cut(self, axis: Literal["col", "row"], index: int) -> None:
        cuts = self.col_cuts if axis == "col" else self.row_cuts
        if index <= 0 or index >= len(cuts) - 1:
            return  # endpoints undeletable
        self._push_undo()
        cuts.pop(index)

    def subdivide_cell(self, axis: Literal["col", "row"], cell_index: int) -> None:
        cuts = self.col_cuts if axis == "col" else self.row_cuts
        if cell_index < 0 or cell_index >= len(cuts) - 1:
            return
        mid = (cuts[cell_index] + cuts[cell_index + 1]) // 2
        if mid == cuts[cell_index] or mid == cuts[cell_index + 1]:
            return
        self._push_undo()
        cuts.insert(cell_index + 1, mid)

    def uniformize(self, axis: Literal["col", "row"], step: int) -> None:
        dim = self.image_width if axis == "col" else self.image_height
        if step < 1 or step >= dim:
            return
        self._push_undo()
        new_cuts = list(range(0, dim, step))
        if new_cuts[-1] != dim:
            new_cuts.append(dim)
        if axis == "col":
            self.col_cuts = new_cuts
        else:
            self.row_cuts = new_cuts

    def replace_all(self, col_cuts: List[int], row_cuts: List[int]) -> None:
        self._push_undo()
        self.col_cuts = sorted(col_cuts)
        self.row_cuts = sorted(row_cuts)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(_Snapshot(list(self.col_cuts), list(self.row_cuts)))
        snap = self._undo_stack.pop()
        self.col_cuts = snap.col_cuts
        self.row_cuts = snap.row_cuts
        self._dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(_Snapshot(list(self.col_cuts), list(self.row_cuts)))
        snap = self._redo_stack.pop()
        self.col_cuts = snap.col_cuts
        self.row_cuts = snap.row_cuts
        self._dirty = True
        return True


# ---------------------------------------------------------------------------
# CanvasPanel — renders image + grid lines
# ---------------------------------------------------------------------------

_ZOOM_LEVELS = [1, 2, 4, 8, 16]
_COL_COLOR = "#00e5ff"
_ROW_COLOR = "#ff40ff"
_HIGHLIGHT_COLOR = "#ffff00"
_SNAP_DIST = 5  # pixels (in scaled coords)


class CanvasPanel(tk.Canvas):
    """Canvas that renders the source image with grid overlay."""

    def __init__(self, master: tk.Widget, image: Image.Image, state: GridState, **kw):
        super().__init__(master, bg="#222222", highlightthickness=0, **kw)
        self._src_image = image.convert("RGBA")
        self._state = state
        self._zoom_index = 0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._pan_start: Optional[Tuple[int, int]] = None

        # Interaction state
        self._selected: Optional[Tuple[Literal["col", "row"], int]] = None
        self._dragging = False
        self._add_mode = False
        self._subdivide_mode = False

        # Bindings
        self.bind("<ButtonPress-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_up)
        self.bind("<Double-Button-1>", self._on_double_click)
        self.bind("<ButtonPress-3>", self._on_right_click)
        self.bind("<ButtonPress-2>", self._on_mid_down)
        self.bind("<B2-Motion>", self._on_mid_drag)
        self.bind("<ButtonRelease-2>", self._on_mid_up)
        self.bind("<Motion>", self._on_motion)

        # Mouse wheel zoom (macOS uses MouseWheel, Linux uses Button-4/5)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", lambda e: self._zoom_in())
        self.bind("<Button-5>", lambda e: self._zoom_out())

        self._render_image()

    # -- zoom ---------------------------------------------------------------

    @property
    def zoom(self) -> int:
        return _ZOOM_LEVELS[self._zoom_index]

    def _zoom_in(self) -> None:
        if self._zoom_index < len(_ZOOM_LEVELS) - 1:
            self._zoom_index += 1
            self._render_image()
            self._redraw_lines()

    def _zoom_out(self) -> None:
        if self._zoom_index > 0:
            self._zoom_index -= 1
            self._render_image()
            self._redraw_lines()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self._zoom_in()
        elif event.delta < 0:
            self._zoom_out()

    def fit_to_window(self) -> None:
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = self._src_image.size
        best = 0
        for i, z in enumerate(_ZOOM_LEVELS):
            if iw * z <= cw and ih * z <= ch:
                best = i
        self._zoom_index = best
        self._render_image()
        self._redraw_lines()

    # -- rendering ----------------------------------------------------------

    def _render_image(self) -> None:
        z = self.zoom
        w, h = self._src_image.size
        scaled = self._src_image.resize((w * z, h * z), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(scaled)
        self.delete("img")
        self.create_image(0, 0, anchor="nw", image=self._photo, tags="img")
        self.config(scrollregion=(0, 0, w * z, h * z))

    def _redraw_lines(self) -> None:
        self.delete("grid")
        z = self.zoom
        h = self._state.image_height * z
        w = self._state.image_width * z

        for i, cx in enumerate(self._state.col_cuts):
            x = cx * z
            color = _HIGHLIGHT_COLOR if self._selected == ("col", i) else _COL_COLOR
            self.create_line(x, 0, x, h, fill="black", width=3, tags="grid")
            self.create_line(x, 0, x, h, fill=color, width=1, tags="grid")

        for i, ry in enumerate(self._state.row_cuts):
            y = ry * z
            color = _HIGHLIGHT_COLOR if self._selected == ("row", i) else _ROW_COLOR
            self.create_line(0, y, w, y, fill="black", width=3, tags="grid")
            self.create_line(0, y, w, y, fill=color, width=1, tags="grid")

    def refresh(self) -> None:
        self._redraw_lines()

    # -- coordinate helpers -------------------------------------------------

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[int, int]:
        z = self.zoom
        return int(self.canvasx(cx)) // z, int(self.canvasy(cy)) // z

    def _find_nearest_cut(self, cx: int, cy: int) -> Optional[Tuple[Literal["col", "row"], int]]:
        z = self.zoom
        sx = int(self.canvasx(cx))
        sy = int(self.canvasy(cy))
        threshold = _SNAP_DIST * z

        best: Optional[Tuple[Literal["col", "row"], int]] = None
        best_dist = threshold + 1

        for i, c in enumerate(self._state.col_cuts):
            d = abs(sx - c * z)
            if d < best_dist:
                best_dist = d
                best = ("col", i)

        for i, r in enumerate(self._state.row_cuts):
            d = abs(sy - r * z)
            if d < best_dist:
                best_dist = d
                best = ("row", i)

        return best if best_dist <= threshold else None

    # -- mouse events -------------------------------------------------------

    def _on_motion(self, event: tk.Event) -> None:
        ix, iy = self._canvas_to_image(event.x, event.y)
        # Update status bar via the app
        self._notify_status(ix, iy)

        near = self._find_nearest_cut(event.x, event.y)
        if near and not self._dragging:
            axis, idx = near
            cuts = self._state.col_cuts if axis == "col" else self._state.row_cuts
            if 0 < idx < len(cuts) - 1:
                self.config(cursor="sb_h_double_arrow" if axis == "col" else "sb_v_double_arrow")
            else:
                self.config(cursor="")
        elif not self._dragging:
            self.config(cursor="crosshair" if self._add_mode or self._subdivide_mode else "")

    def _app(self) -> Optional["GridEditorApp"]:
        try:
            return self.winfo_toplevel()
        except Exception:
            return None

    def _notify_status(self, ix: int, iy: int) -> None:
        app = self._app()
        if app:
            app._update_status(ix, iy)

    def _on_left_down(self, event: tk.Event) -> None:
        if self._subdivide_mode:
            self._do_subdivide(event)
            return
        if self._add_mode:
            self._do_add(event)
            return

        near = self._find_nearest_cut(event.x, event.y)
        if near:
            axis, idx = near
            cuts = self._state.col_cuts if axis == "col" else self._state.row_cuts
            if 0 < idx < len(cuts) - 1:
                self._selected = near
                self._dragging = True
                self._state._push_undo()  # single undo entry for entire drag
                self._redraw_lines()
                return
        self._selected = None
        self._dragging = False
        self._redraw_lines()

    def _on_left_drag(self, event: tk.Event) -> None:
        if not self._dragging or self._selected is None:
            return
        axis, idx = self._selected
        ix, iy = self._canvas_to_image(event.x, event.y)
        new_pos = ix if axis == "col" else iy
        self._state.move_cut(axis, idx, new_pos, record_undo=False)
        self._redraw_lines()
        self._notify_status(ix, iy)

    def _on_left_up(self, event: tk.Event) -> None:
        self._dragging = False

    def _on_double_click(self, event: tk.Event) -> None:
        near = self._find_nearest_cut(event.x, event.y)
        if near is None:
            self._do_add(event)

    def _on_right_click(self, event: tk.Event) -> None:
        near = self._find_nearest_cut(event.x, event.y)
        if near:
            axis, idx = near
            self._state.delete_cut(axis, idx)
            self._selected = None
            self._redraw_lines()
            self._notify_status(*self._canvas_to_image(event.x, event.y))

    def _on_mid_down(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)
        self.config(cursor="fleur")

    def _on_mid_drag(self, event: tk.Event) -> None:
        if self._pan_start:
            dx = self._pan_start[0] - event.x
            dy = self._pan_start[1] - event.y
            self.xview_scroll(dx, "units")
            self.yview_scroll(dy, "units")
            self._pan_start = (event.x, event.y)

    def _on_mid_up(self, event: tk.Event) -> None:
        self._pan_start = None
        self.config(cursor="")

    # -- actions ------------------------------------------------------------

    def _do_add(self, event: tk.Event) -> None:
        ix, iy = self._canvas_to_image(event.x, event.y)
        # Decide axis: which is farther from a cut
        col_dists = [abs(ix - c) for c in self._state.col_cuts]
        row_dists = [abs(iy - r) for r in self._state.row_cuts]
        if min(col_dists) > min(row_dists):
            self._state.add_cut("col", ix)
        else:
            self._state.add_cut("row", iy)
        self._add_mode = False
        self.config(cursor="")
        self._redraw_lines()
        self._notify_status(ix, iy)

    def _do_subdivide(self, event: tk.Event) -> None:
        ix, iy = self._canvas_to_image(event.x, event.y)
        # Find which cell was clicked, subdivide both axes
        col_idx = 0
        for i, c in enumerate(self._state.col_cuts[:-1]):
            if ix >= c:
                col_idx = i
        row_idx = 0
        for i, r in enumerate(self._state.row_cuts[:-1]):
            if iy >= r:
                row_idx = i
        self._state.subdivide_cell("col", col_idx)
        self._state.subdivide_cell("row", row_idx)
        self._subdivide_mode = False
        self.config(cursor="")
        self._redraw_lines()
        self._notify_status(ix, iy)


# ---------------------------------------------------------------------------
# GridEditorApp — main window
# ---------------------------------------------------------------------------

class GridEditorApp(tk.Tk):
    """Root editor window."""

    def __init__(
        self,
        image_path: Path,
        initial_col_cuts: List[int],
        initial_row_cuts: List[int],
        algorithm_source: str = "",
        gt_dir: Optional[Path] = None,
    ):
        super().__init__()

        self._image_path = image_path
        self._image = Image.open(image_path).convert("RGBA")
        w, h = self._image.size

        self._state = GridState(
            col_cuts=sorted(initial_col_cuts),
            row_cuts=sorted(initial_row_cuts),
            image_width=w,
            image_height=h,
        )
        self._initial_col_cuts = list(initial_col_cuts)
        self._initial_row_cuts = list(initial_row_cuts)
        self._algorithm_source = algorithm_source

        stem = image_path.stem
        self._gt_dir = gt_dir or ground_truth_dir(image_path.parent)
        self._save_path = self._gt_dir / f"{stem}.json"

        self.title(f"Ground Truth Editor — {image_path.name}")
        self.geometry("1200x800")

        # Layout
        self._canvas = CanvasPanel(self, self._image, self._state)

        # Scrollbars
        xscroll = tk.Scrollbar(self, orient="horizontal", command=self._canvas.xview)
        yscroll = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.config(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        self._status = tk.Label(
            self, text="", anchor="w", bg="#333333", fg="#cccccc",
            font=("Courier", 11), padx=6, pady=2,
        )

        # Grid layout
        self._canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._status.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Key bindings
        self.bind("<Control-z>", self._on_undo)
        self.bind("<Control-Z>", self._on_redo)
        self.bind("<Control-Shift-Z>", self._on_redo)
        self.bind("<Command-z>", self._on_undo)
        self.bind("<Command-Z>", self._on_redo)
        self.bind("<Command-Shift-Z>", self._on_redo)
        self.bind("<Control-s>", self._on_save)
        self.bind("<Command-s>", self._on_save)
        self.bind("<Key-a>", self._on_add_mode)
        self.bind("<Key-d>", self._on_delete_selected)
        self.bind("<Delete>", self._on_delete_selected)
        self.bind("<BackSpace>", self._on_delete_selected)
        self.bind("<Key-s>", self._on_subdivide_mode)
        self.bind("<Key-u>", self._on_uniformize)
        self.bind("<Key-r>", self._on_reset)
        self.bind("<Key-f>", self._on_fit)
        self.bind("<Left>", lambda e: self._on_arrow("col", -1, e))
        self.bind("<Right>", lambda e: self._on_arrow("col", 1, e))
        self.bind("<Up>", lambda e: self._on_arrow("row", -1, e))
        self.bind("<Down>", lambda e: self._on_arrow("row", 1, e))

        self._canvas._redraw_lines()
        self._update_status(0, 0)

    # -- status bar ---------------------------------------------------------

    def _update_status(self, ix: int = 0, iy: int = 0) -> None:
        s = self._state
        cx = len(s.col_cuts) - 1
        cy = len(s.row_cuts) - 1
        avg_step_x = s.image_width / cx if cx > 0 else 0
        avg_step_y = s.image_height / cy if cy > 0 else 0
        avg = (avg_step_x + avg_step_y) / 2 if cx > 0 and cy > 0 else 0
        dirty = " | Unsaved changes" if s.dirty else ""
        zoom = self._canvas.zoom
        self._status.config(
            text=f"{cx}\u00d7{cy} cells | step ~{avg:.1f} px | "
                 f"({ix}, {iy}) | zoom {zoom}x{dirty}"
        )

    # -- keyboard handlers --------------------------------------------------

    def _on_undo(self, event: tk.Event = None) -> None:
        self._state.undo()
        self._canvas.refresh()
        self._update_status()

    def _on_redo(self, event: tk.Event = None) -> None:
        self._state.redo()
        self._canvas.refresh()
        self._update_status()

    def _on_save(self, event: tk.Event = None) -> None:
        gt = GroundTruth(
            image_file=self._image_path.name,
            image_width=self._state.image_width,
            image_height=self._state.image_height,
            col_cuts=list(self._state.col_cuts),
            row_cuts=list(self._state.row_cuts),
            metadata={
                "algorithm_source": self._algorithm_source,
                "notes": "",
            },
        )
        # Preserve created timestamp from existing file
        if self._save_path.exists():
            try:
                existing = load_ground_truth(self._save_path)
                if "created" in existing.metadata:
                    gt.metadata["created"] = existing.metadata["created"]
            except Exception:
                pass

        save_ground_truth(gt, self._save_path)
        self._state.mark_clean()
        self._update_status()
        self._status.config(text=self._status.cget("text") + f" | Saved to {self._save_path.name}")

    def _on_add_mode(self, event: tk.Event = None) -> None:
        self._canvas._add_mode = True
        self._canvas._subdivide_mode = False
        self._canvas.config(cursor="crosshair")
        self._status.config(text="ADD MODE: Click to add a cut (col or row auto-detected)")

    def _on_delete_selected(self, event: tk.Event = None) -> None:
        if self._canvas._selected:
            axis, idx = self._canvas._selected
            self._state.delete_cut(axis, idx)
            self._canvas._selected = None
            self._canvas.refresh()
            self._update_status()

    def _on_subdivide_mode(self, event: tk.Event = None) -> None:
        self._canvas._subdivide_mode = True
        self._canvas._add_mode = False
        self._canvas.config(cursor="crosshair")
        self._status.config(text="SUBDIVIDE MODE: Click a cell to split at midpoint")

    def _on_uniformize(self, event: tk.Event = None) -> None:
        step = simpledialog.askinteger(
            "Uniformize", "Enter step size (pixels):",
            minvalue=2, maxvalue=max(self._state.image_width, self._state.image_height),
        )
        if step:
            self._state.uniformize("col", step)
            self._state.uniformize("row", step)
            self._canvas.refresh()
            self._update_status()

    def _on_reset(self, event: tk.Event = None) -> None:
        if messagebox.askyesno("Reset", "Revert to algorithm's original prediction?"):
            self._state.replace_all(
                list(self._initial_col_cuts),
                list(self._initial_row_cuts),
            )
            self._canvas.refresh()
            self._update_status()

    def _on_fit(self, event: tk.Event = None) -> None:
        self._canvas.fit_to_window()
        self._update_status()

    def _on_arrow(self, axis: str, delta: int, event: tk.Event = None) -> None:
        if self._canvas._selected is None:
            return
        sel_axis, idx = self._canvas._selected
        if sel_axis != axis:
            return
        cuts = self._state.col_cuts if axis == "col" else self._state.row_cuts
        if 0 < idx < len(cuts) - 1:
            step = 5 if (event and event.state & 0x1) else 1  # Shift = ±5
            self._state.move_cut(axis, idx, cuts[idx] + delta * step)
            self._canvas.refresh()
            self._update_status()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _get_prediction(image_path: Path, resolution_hint: Optional[int] = None) -> Tuple[List[int], List[int], str]:
    """Run algorithm and return (col_cuts, row_cuts, source_name)."""
    with open(image_path, "rb") as f:
        data = f.read()
    config = Config(
        use_uniformity_scoring=True,
        use_autocorrelation=True,
        resolution_hint=resolution_hint,
    )
    result = process_image_bytes_with_grid(data, config)
    source = ""
    if result.scored_candidates:
        source = result.scored_candidates[0].source
    return list(result.col_cuts), list(result.row_cuts), source


def _open_editor(image_path: Path, resolution_hint: Optional[int] = None) -> None:
    """Open editor for a single image."""
    gt_dir = ground_truth_dir(image_path.parent)
    gt_path = gt_dir / f"{image_path.stem}.json"

    if gt_path.exists():
        print(f"Loading existing ground truth: {gt_path}")
        gt = load_ground_truth(gt_path)
        col_cuts = gt.col_cuts
        row_cuts = gt.row_cuts
        source = gt.metadata.get("algorithm_source", "ground_truth")
    else:
        print(f"Running algorithm on {image_path.name}...")
        col_cuts, row_cuts, source = _get_prediction(image_path, resolution_hint)
        print(f"  Prediction: {len(col_cuts)-1}x{len(row_cuts)-1} cells (source: {source})")

    app = GridEditorApp(
        image_path=image_path,
        initial_col_cuts=col_cuts,
        initial_row_cuts=row_cuts,
        algorithm_source=source,
        gt_dir=gt_dir,
    )
    app.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground truth grid editor")
    parser.add_argument("path", type=Path, help="Image file or testdata directory")
    parser.add_argument("--resolution-hint", type=int, default=None, help="Resolution hint for algorithm")
    parser.add_argument("--missing-only", action="store_true", help="Only open images without ground truth")
    args = parser.parse_args()

    path: Path = args.path

    if path.is_dir():
        images = get_test_images(path)
        if args.missing_only:
            gt_d = ground_truth_dir(path)
            images = [
                img for img in images
                if not (gt_d / f"{img.stem}.json").exists()
            ]
        if not images:
            print("No images to process.")
            return
        print(f"Found {len(images)} image(s)")
        for img in images:
            _open_editor(img, args.resolution_hint)
    else:
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        _open_editor(path, args.resolution_hint)


if __name__ == "__main__":
    main()
