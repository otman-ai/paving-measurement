"""Image composition, grid, and mask helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib
import numpy as np
from PIL import Image


def create_grid_tiles(image: Image.Image, grid_size: int) -> list[dict[str, object]]:
    """Split an image into an evenly spaced grid, retaining origin coordinates."""
    if grid_size < 1:
        raise ValueError("Grid size must be at least 1.")
    image_width, image_height = image.size
    tile_width, tile_height = image_width // grid_size, image_height // grid_size
    tiles: list[dict[str, object]] = []
    for row in range(grid_size):
        for column in range(grid_size):
            x_start, y_start = column * tile_width, row * tile_height
            x_end = image_width if column == grid_size - 1 else x_start + tile_width
            y_end = image_height if row == grid_size - 1 else y_start + tile_height
            if x_end > x_start and y_end > y_start:
                tiles.append({"image": image.crop((x_start, y_start, x_end, y_end)), "x": x_start, "y": y_start})
    return tiles


def overlay_masks(image: Image.Image, masks: Any | np.ndarray | None) -> Image.Image:
    """Return an RGBA image with semi-transparent, distinct colors for each mask."""
    output = image.convert("RGBA")
    if masks is None:
        return output
    if hasattr(masks, "detach"):
        array = masks.detach().cpu().numpy()
    else:
        array = np.asarray(masks)
    if array.size == 0:
        return output
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("Masks must have shape (count, height, width).")
    colors = matplotlib.colormaps.get_cmap("rainbow").resampled(len(array))
    for index, mask in enumerate(array.astype(np.uint8)):
        color = tuple(int(channel * 255) for channel in colors(index)[:3])
        alpha = Image.fromarray(mask * 128).convert("L")
        layer = Image.new("RGBA", output.size, color + (0,))
        layer.putalpha(alpha)
        output = Image.alpha_composite(output, layer)
    return output


def create_comparison(images: Sequence[Image.Image]) -> Image.Image:
    """Place result images horizontally for an easy grid-size comparison."""
    if not images:
        raise ValueError("At least one image is required for a comparison.")
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    comparison = Image.new("RGB", (width, height))
    offset = 0
    for image in images:
        comparison.paste(image.convert("RGB"), (offset, 0))
        offset += image.width
    return comparison
