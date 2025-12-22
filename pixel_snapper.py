from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image


class PixelSnapperError(Exception):
    pass


@dataclass
class Config:
    k_colors: int = 16
    k_seed: int = 42
    input_path: str = "samples/2/skeleton.png"
    output_path: str = "samples/2/skeleton_fixed_clean2.png"
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8
    preview: bool = False


def validate_image_dimensions(width: int, height: int) -> None:
    if width == 0 or height == 0:
        raise PixelSnapperError("Image dimensions cannot be zero")
    if width > 10000 or height > 10000:
        raise PixelSnapperError("Image dimensions too large (max 10000x10000)")


def quantize_image(img: Image.Image, config: Config) -> Image.Image:
    if config.k_colors <= 0:
        raise PixelSnapperError("Number of colors must be greater than 0")

    pixels = img.load()
    width, height = img.size
    opaque_pixels: List[Tuple[float, float, float]] = []

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            opaque_pixels.append((float(r), float(g), float(b)))

    n_pixels = len(opaque_pixels)
    if n_pixels == 0:
        return img.copy()

    import random

    rng = random.Random(config.k_seed)
    k = min(config.k_colors, n_pixels)

    def dist_sq(p: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
        dr = p[0] - c[0]
        dg = p[1] - c[1]
        db = p[2] - c[2]
        return dr * dr + dg * dg + db * db

    centroids: List[Tuple[float, float, float]] = []
    first_idx = rng.randrange(n_pixels)
    centroids.append(opaque_pixels[first_idx])
    distances = [float("inf")] * n_pixels

    for _ in range(1, k):
        last_c = centroids[-1]
        sum_sq_dist = 0.0
        for i, p in enumerate(opaque_pixels):
            d_sq = dist_sq(p, last_c)
            if d_sq < distances[i]:
                distances[i] = d_sq
            sum_sq_dist += distances[i]

        if sum_sq_dist <= 0.0:
            idx = rng.randrange(n_pixels)
            centroids.append(opaque_pixels[idx])
        else:
            r = rng.random() * sum_sq_dist
            cumulative = 0.0
            chosen_idx = 0
            for i, d in enumerate(distances):
                cumulative += d
                if cumulative >= r:
                    chosen_idx = i
                    break
            centroids.append(opaque_pixels[chosen_idx])

    prev_centroids = centroids[:]
    for iteration in range(config.max_kmeans_iterations):
        sums = [(0.0, 0.0, 0.0) for _ in range(k)]
        counts = [0 for _ in range(k)]

        for p in opaque_pixels:
            min_dist = float("inf")
            best_k = 0
            for i, c in enumerate(centroids):
                d = dist_sq(p, c)
                if d < min_dist:
                    min_dist = d
                    best_k = i
            sr, sg, sb = sums[best_k]
            sums[best_k] = (sr + p[0], sg + p[1], sb + p[2])
            counts[best_k] += 1

        new_centroids: List[Tuple[float, float, float]] = list(centroids)
        for i in range(k):
            if counts[i] > 0:
                fcount = float(counts[i])
                sr, sg, sb = sums[i]
                new_centroids[i] = (sr / fcount, sg / fcount, sb / fcount)
        centroids = new_centroids

        if iteration > 0:
            max_movement = 0.0
            for new_c, old_c in zip(centroids, prev_centroids):
                movement = dist_sq(new_c, old_c)
                if movement > max_movement:
                    max_movement = movement
            if max_movement < 0.01:
                break

        prev_centroids = centroids[:]

    new_img = Image.new("RGBA", (width, height))
    out_pixels = new_img.load()
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                out_pixels[x, y] = (r, g, b, a)
                continue
            p = (float(r), float(g), float(b))
            min_dist = float("inf")
            best_c = (r, g, b)
            for c in centroids:
                d = dist_sq(p, c)
                if d < min_dist:
                    min_dist = d
                    best_c = (
                        int(round(c[0])),
                        int(round(c[1])),
                        int(round(c[2])),
                    )
            out_pixels[x, y] = (best_c[0], best_c[1], best_c[2], a)

    return new_img


def compute_profiles(img: Image.Image) -> Tuple[List[float], List[float]]:
    width, height = img.size
    if width < 3 or height < 3:
        raise PixelSnapperError("Image too small (minimum 3x3)")

    col_proj = [0.0 for _ in range(width)]
    row_proj = [0.0 for _ in range(height)]
    pixels = img.load()

    def gray(x: int, y: int) -> float:
        r, g, b, a = pixels[x, y]
        if a == 0:
            return 0.0
        return 0.299 * r + 0.587 * g + 0.114 * b

    for y in range(height):
        for x in range(1, width - 1):
            left = gray(x - 1, y)
            right = gray(x + 1, y)
            grad = abs(right - left)
            col_proj[x] += grad

    for x in range(width):
        for y in range(1, height - 1):
            top = gray(x, y - 1)
            bottom = gray(x, y + 1)
            grad = abs(bottom - top)
            row_proj[y] += grad

    return col_proj, row_proj


def estimate_step_size(profile: Sequence[float], config: Config) -> Optional[float]:
    if not profile:
        return None

    max_val = max(profile)
    if max_val == 0.0:
        return None

    threshold = max_val * config.peak_threshold_multiplier
    peaks: List[int] = []
    for i in range(1, len(profile) - 1):
        if profile[i] > threshold and profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        return None

    clean_peaks = [peaks[0]]
    for p in peaks[1:]:
        if p - clean_peaks[-1] > (config.peak_distance_filter - 1):
            clean_peaks.append(p)

    if len(clean_peaks) < 2:
        return None

    diffs = [float(clean_peaks[i + 1] - clean_peaks[i]) for i in range(len(clean_peaks) - 1)]
    diffs.sort()
    return diffs[len(diffs) // 2]


def resolve_step_sizes(
    step_x_opt: Optional[float],
    step_y_opt: Optional[float],
    width: int,
    height: int,
    config: Config,
) -> Tuple[float, float]:
    if step_x_opt is not None and step_y_opt is not None:
        ratio = step_x_opt / step_y_opt if step_x_opt > step_y_opt else step_y_opt / step_x_opt
        if ratio > config.max_step_ratio:
            smaller = min(step_x_opt, step_y_opt)
            return smaller, smaller
        avg = (step_x_opt + step_y_opt) / 2.0
        return avg, avg

    if step_x_opt is not None:
        return step_x_opt, step_x_opt
    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback_step = max(min(width, height) / float(config.fallback_target_segments), 1.0)
    return fallback_step, fallback_step


def walk(profile: Sequence[float], step_size: float, limit: int, config: Config) -> List[int]:
    if not profile:
        raise PixelSnapperError("Cannot walk on empty profile")

    cuts = [0]
    current_pos = 0.0
    search_window = max(step_size * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = sum(profile) / float(len(profile))

    while current_pos < float(limit):
        target = current_pos + step_size
        if target >= float(limit):
            cuts.append(limit)
            break

        start_search = max(int(target - search_window), int(current_pos + 1.0))
        end_search = min(int(target + search_window), limit)
        if end_search <= start_search:
            current_pos = target
            continue

        max_val = -1.0
        max_idx = start_search
        for i in range(start_search, end_search):
            if profile[i] > max_val:
                max_val = profile[i]
                max_idx = i

        if max_val > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target

    return cuts


def sanitize_cuts(cuts: List[int], limit: int) -> List[int]:
    if limit == 0:
        return [0]

    has_zero = False
    has_limit = False
    sanitized = []
    for value in cuts:
        v = value
        if v == 0:
            has_zero = True
        if v >= limit:
            v = limit
        if v == limit:
            has_limit = True
        sanitized.append(v)

    if not has_zero:
        sanitized.append(0)
    if not has_limit:
        sanitized.append(limit)

    sanitized.sort()
    deduped: List[int] = []
    for v in sanitized:
        if not deduped or deduped[-1] != v:
            deduped.append(v)
    return deduped


def snap_uniform_cuts(
    profile: Sequence[float],
    limit: int,
    target_step: float,
    config: Config,
    min_required: int,
) -> List[int]:
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    if math.isfinite(target_step) and target_step > 0.0:
        desired_cells = int(round(limit / target_step))
    else:
        desired_cells = 0

    desired_cells = max(desired_cells, min_required - 1, 1)
    desired_cells = min(desired_cells, limit)

    cell_width = limit / float(desired_cells)
    search_window = max(cell_width * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = sum(profile) / float(len(profile)) if profile else 0.0

    cuts = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break

        start = int(math.floor(target - search_window))
        start = max(start, prev + 1, 0)
        end = int(math.ceil(target + search_window))
        end = min(end, limit - 1)
        if end < start:
            start = prev + 1
            end = start

        best_idx = min(start, len(profile) - 1) if profile else start
        best_val = -1.0
        last_index = min(end, len(profile) - 1)
        for i in range(start, last_index + 1):
            v = profile[i] if i < len(profile) else 0.0
            if v > best_val:
                best_val = v
                best_idx = i

        strength_threshold = mean_val * config.walker_strength_threshold
        if best_val < strength_threshold:
            fallback_idx = int(round(target))
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx

        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)

    return sanitize_cuts(cuts, limit)


def stabilize_cuts(
    profile: Sequence[float],
    cuts: List[int],
    limit: int,
    sibling_cuts: Sequence[int],
    sibling_limit: int,
    config: Config,
) -> List[int]:
    if limit == 0:
        return [0]

    cuts = sanitize_cuts(cuts, limit)
    min_required = min(max(config.min_cuts_per_axis, 2), limit + 1)
    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = (
        sibling_limit > 0
        and sibling_cells >= max(min_required - 1, 0)
        and sibling_cells > 0
    )
    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / float(axis_cells)
        sibling_step = sibling_limit / float(sibling_cells)
        step_ratio = axis_step / sibling_step
        steps_skewed = step_ratio > config.max_step_ratio or step_ratio < 1.0 / config.max_step_ratio

    has_enough = len(cuts) >= min_required
    if has_enough and not steps_skewed:
        return cuts

    if sibling_has_grid:
        target_step = sibling_limit / float(sibling_cells)
    elif config.fallback_target_segments > 1:
        target_step = limit / float(config.fallback_target_segments)
    elif axis_cells > 0:
        target_step = limit / float(axis_cells)
    else:
        target_step = float(limit)

    if not math.isfinite(target_step) or target_step <= 0.0:
        target_step = 1.0

    return snap_uniform_cuts(profile, limit, target_step, config, min_required)


def stabilize_both_axes(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    raw_col_cuts: List[int],
    raw_row_cuts: List[int],
    width: int,
    height: int,
    config: Config,
) -> Tuple[List[int], List[int]]:
    col_cuts_pass1 = stabilize_cuts(profile_x, raw_col_cuts[:], width, raw_row_cuts, height, config)
    row_cuts_pass1 = stabilize_cuts(profile_y, raw_row_cuts[:], height, raw_col_cuts, width, config)

    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / float(col_cells)
    row_step = height / float(row_cells)

    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step
    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)
        if col_step > target_step * 1.2:
            final_col_cuts = snap_uniform_cuts(
                profile_x, width, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_col_cuts = col_cuts_pass1
        if row_step > target_step * 1.2:
            final_row_cuts = snap_uniform_cuts(
                profile_y, height, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_row_cuts = row_cuts_pass1
        return final_col_cuts, final_row_cuts

    return col_cuts_pass1, row_cuts_pass1


def resample(img: Image.Image, cols: Sequence[int], rows: Sequence[int]) -> Image.Image:
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = Image.new("RGBA", (out_w, out_h))

    in_pixels = img.load()
    out_pixels = final_img.load()
    width, height = img.size

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue

            counts: Dict[Tuple[int, int, int, int], int] = {}
            for y in range(ys, ye):
                if y >= height:
                    break
                for x in range(xs, xe):
                    if x >= width:
                        break
                    p = in_pixels[x, y]
                    counts[p] = counts.get(p, 0) + 1

            best_pixel = (0, 0, 0, 0)
            best_count = -1
            for p, count in counts.items():
                if count > best_count or (count == best_count and p < best_pixel):
                    best_pixel = p
                    best_count = count

            out_pixels[x_i, y_i] = best_pixel

    return final_img


def process_image_bytes_common(input_bytes: bytes, config: Optional[Config] = None) -> bytes:
    config = config or Config()

    img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    width, height = img.size
    validate_image_dimensions(width, height)

    quantized = quantize_image(img, config)
    profile_x, profile_y = compute_profiles(quantized)

    step_x_opt = estimate_step_size(profile_x, config)
    step_y_opt = estimate_step_size(profile_y, config)
    step_x, step_y = resolve_step_sizes(step_x_opt, step_y_opt, width, height, config)

    raw_col_cuts = walk(profile_x, step_x, width, config)
    raw_row_cuts = walk(profile_y, step_y, height, config)

    col_cuts, row_cuts = stabilize_both_axes(
        profile_x,
        profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width,
        height,
        config,
    )

    output_img = resample(quantized, col_cuts, row_cuts)

    out_buf = io.BytesIO()
    output_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


def process_image(config: Config) -> None:
    print(f"Processing: {config.input_path}")
    with open(config.input_path, "rb") as f:
        img_bytes = f.read()

    output_bytes = process_image_bytes_common(img_bytes, config)
    with open(config.output_path, "wb") as f:
        f.write(output_bytes)

    print(f"Saved to: {config.output_path}")
    if config.preview:
        preview_side_by_side(img_bytes, output_bytes)


def preview_side_by_side(input_bytes: bytes, output_bytes: bytes) -> None:
    input_img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    in_w, in_h = input_img.size
    scaled_output = output_img.resize((in_w, in_h), resample=Image.NEAREST)

    preview_img = Image.new("RGBA", (in_w * 2, in_h))
    preview_img.paste(input_img, (0, 0))
    preview_img.paste(scaled_output, (in_w, 0))
    preview_img.show(title="Pixel Snapper Preview")


def parse_args(argv: Sequence[str]) -> Config:
    args = list(argv[1:])
    preview = False
    if "--preview" in args:
        preview = True
        args.remove("--preview")

    if len(args) < 2:
        raise PixelSnapperError(
            "Usage: python pixel_snapper.py input.png output.png [k_colors] [--preview]"
        )

    config = Config(input_path=args[0], output_path=args[1], preview=preview)
    if len(args) >= 3:
        try:
            k = int(args[2])
            if k > 0:
                config.k_colors = k
            else:
                print(
                    f"Warning: invalid k_colors '{args[2]}', falling back to default ({config.k_colors})"
                )
        except ValueError:
            print(
                f"Warning: invalid k_colors '{args[2]}', falling back to default ({config.k_colors})"
            )
    if len(args) > 3:
        raise PixelSnapperError(
            "Usage: python pixel_snapper.py input.png output.png [k_colors] [--preview]"
        )

    return config


def main(argv: Sequence[str]) -> int:
    try:
        config = parse_args(argv)
        process_image(config)
        return 0
    except PixelSnapperError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
