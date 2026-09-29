"""YOLO instance-segmentation inference for satellite mosaics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from paving_measurement.geospatial import calculate_area
from paving_measurement.image_ops import create_comparison, overlay_masks

LOGGER = logging.getLogger(__name__)


@dataclass
class YoloSegmentationResult:
    final_overlay: Image.Image
    comparison: Image.Image
    summary: str
    grid_rows: list[list[str | int]]
    grid_overlays: list[Image.Image]
    polygons: list[list[list[int]]]
    meters_per_pixel: float
    area_m2: float
    area_ft2: float


class YoloSegmentationDetector:
    """Load a Hugging Face YOLO segmentation checkpoint once per worker."""

    def __init__(self, model: Any, model_id: str, confidence: float = 0.25) -> None:
        self.model = model
        self.model_id = model_id
        self.confidence = confidence

    @staticmethod
    def load_huggingface_model(model_id: str, token: str | None) -> "YoloSegmentationDetector":
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        # The trained repository contains ``best.pt``. Download that exact
        # file instead of scanning the shared cache, which can also contain
        # the previous SAM3 ``sam3.pt`` checkpoint.
        checkpoint = hf_hub_download(repo_id=model_id, filename="best.pt", token=token)
        LOGGER.info("Loading YOLO segmentation checkpoint %s", checkpoint)
        return YoloSegmentationDetector(YOLO(checkpoint), model_id)

    def predict_polygons(
        self, image: Image.Image, imgsz: int = 1280, confidence: float | None = None
    ) -> list[list[list[int]]]:
        """Return mask contours in the pixel coordinates of ``image``.

        Ultralytics' ``masks.xyn`` is normalized to the original image passed
        to ``predict``. Converting it with that image's dimensions avoids using
        the model's internal 640x640 mask tensor as if it were the mosaic size.
        """
        original = image.convert("RGB")
        result = self.model.predict(
            original,
            imgsz=imgsz,
            conf=confidence if confidence is not None else self.confidence,
            verbose=False,
            max_det=2000,
        )[0]
        masks = getattr(result, "masks", None)
        if masks is None:
            return []
        normalized_polygons = getattr(masks, "xyn", None)
        if normalized_polygons is None:
            return []
        width, height = original.size
        contours: list[list[list[int]]] = []
        for normalized in normalized_polygons:
            points = np.asarray(normalized, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3:
                continue
            pixel_points = np.round(points * np.asarray([width, height], dtype=np.float32)).astype(np.int32)
            pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, width - 1)
            pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, height - 1)
            contour = [[int(point[0]), int(point[1])] for point in pixel_points]
            if len(contour) >= 3:
                contours.append(contour)
        return contours

    def run(
        self,
        satellite_image: Image.Image,
        latitude: float,
        zoom: int,
        prompts: list[str] | None = None,
        grid_sizes: tuple[int, ...] | None = None,
    ) -> YoloSegmentationResult:
        """Run one whole-image YOLO segmentation pass for the legacy route."""
        import cv2

        start = time.perf_counter()
        original = satellite_image.convert("RGB")
        polygons = self.predict_polygons(original)
        masks = np.zeros((len(polygons), original.height, original.width), dtype=np.uint8)
        for index, polygon in enumerate(polygons):
            cv2.fillPoly(masks[index], [np.asarray(polygon, dtype=np.int32)], 1)
        union = np.any(masks, axis=0) if len(masks) else np.zeros((original.height, original.width), dtype=bool)
        resolution, area_m2, area_ft2 = calculate_area(int(union.sum()), latitude, zoom)
        overlay = overlay_masks(original, masks).convert("RGB")
        summary = (
            f"# YOLO Segmentation Results\n\n"
            f"Model: `{self.model_id}`\n\n"
            f"Objects: `{len(polygons)}`\n\n"
            f"Meters per pixel: `{resolution:.6f}`\n\n"
            f"Area: `{area_m2:,.2f} m²` (`{area_ft2:,.2f} ft²`)\n\n"
            f"Elapsed time: `{time.perf_counter() - start:.2f} seconds`\n"
        )
        return YoloSegmentationResult(
            overlay,
            create_comparison([overlay]),
            summary,
            [["whole", len(polygons), f"{int(union.sum()):,}", f"{area_m2:,.2f}", f"{area_ft2:,.2f}"]],
            [overlay],
            polygons,
            resolution,
            area_m2,
            area_ft2,
        )
