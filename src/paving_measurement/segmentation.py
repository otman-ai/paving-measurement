"""SAM 3 inference and the multi-grid segmentation workflow."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from paving_measurement.config import Settings
from paving_measurement.geospatial import calculate_area
from paving_measurement.image_ops import create_comparison, create_grid_tiles, overlay_masks

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    final_overlay: Image.Image
    comparison: Image.Image
    summary: str
    grid_rows: list[list[str | int]]
    grid_overlays: list[Image.Image]


class Sam3Segmenter:
    """Loads SAM 3 once and executes batched text-prompted segmentation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading %s on %s", settings.model_id, self.device)
        self.model = Sam3Model.from_pretrained(settings.model_id, token=settings.hf_token).to(self.device)
        self.processor = Sam3Processor.from_pretrained(settings.model_id, token=settings.hf_token)
        self.model.eval()

    def _run_batch(self, tiles: list[dict[str, object]]) -> list[dict[str, Any]]:
        images = [tile["image"] for tile in tiles]
        inputs = self.processor(images=images, text=[self.settings.prompt] * len(images), return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs.to(self.device))
        target_sizes = [(image.height, image.width) for image in images]
        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.settings.segmentation_threshold,
            mask_threshold=self.settings.mask_threshold,
            target_sizes=target_sizes,
        )

    def process_grid(self, original_image: Image.Image, grid_size: int) -> torch.Tensor:
        """Segment a grid and project every local mask into original-image coordinates."""
        image_width, image_height = original_image.size
        masks: list[torch.Tensor] = []
        tiles = create_grid_tiles(original_image, grid_size)
        for start in range(0, len(tiles), self.settings.batch_size):
            batch = tiles[start : start + self.settings.batch_size]
            for tile, result in zip(batch, self._run_batch(batch)):
                result_masks = result.get("masks")
                if result_masks is None:
                    continue
                for local_mask in result_masks:
                    local_mask = local_mask.bool().cpu()
                    full_mask = torch.zeros((image_height, image_width), dtype=torch.bool)
                    x, y = int(tile["x"]), int(tile["y"])
                    height, width = local_mask.shape
                    full_mask[y : y + height, x : x + width] = local_mask
                    masks.append(full_mask)
        return torch.stack(masks) if masks else torch.empty((0, image_height, image_width), dtype=torch.bool)

    def run(self, satellite_image: Image.Image, latitude: float, zoom: int) -> PipelineResult:
        """Run all configured grid sizes and calculate approximate asphalt area."""
        start = time.perf_counter()
        original = satellite_image.convert("RGB")
        all_masks: list[torch.Tensor] = []
        rows: list[list[str | int]] = []
        overlays: list[Image.Image] = []
        for grid_size in self.settings.grid_sizes:
            masks = self.process_grid(original, grid_size)
            if len(masks):
                all_masks.extend(masks)
                union = torch.any(masks, dim=0)
                pixels = int(union.sum().item())
                _, area_m2, area_ft2 = calculate_area(pixels, latitude, zoom)
            else:
                pixels, area_m2, area_ft2 = 0, 0.0, 0.0
            overlays.append(overlay_masks(original, masks).convert("RGB"))
            rows.append([f"{grid_size}x{grid_size}", len(masks), f"{pixels:,}", f"{area_m2:,.2f}", f"{area_ft2:,.2f}"])
        final_mask = torch.any(torch.stack(all_masks), dim=0) if all_masks else torch.zeros(original.size[::-1], dtype=torch.bool)
        pixels = int(final_mask.sum().item())
        resolution, area_m2, area_ft2 = calculate_area(pixels, latitude, zoom)
        total_tiles = sum(size**2 for size in self.settings.grid_sizes)
        total_batches = sum(math.ceil(size**2 / self.settings.batch_size) for size in self.settings.grid_sizes)
        summary = f"""# Asphalt Segmentation Results

## Location

Latitude: `{latitude:.6f}`  
Zoom: `{zoom}`

## Processing

Prompt: `{self.settings.prompt}`  
Grid strategy: `1x1` through `7x7`  
Image tiles analyzed: `{total_tiles}`  
Model batches: `{total_batches}`  
Device: `{self.device}`  
Elapsed time: `{time.perf_counter() - start:.2f} seconds`

## Estimated Asphalt Area

Mask pixels: `{pixels:,}`  
Meters per pixel: `{resolution:.6f}`  
Area: `{area_m2:,.2f} m²` (`{area_ft2:,.2f} ft²`)

Area is an approximate Web Mercator estimate, not a survey-grade measurement.
"""
        return PipelineResult(overlay_masks(original, final_mask).convert("RGB"), create_comparison(overlays), summary, rows, overlays)
