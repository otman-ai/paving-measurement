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
        geometries = self.predict_geometries(image, imgsz, confidence)
        return [geometry[0] for geometry in geometries if geometry]

    def predict_geometries(
        self, image: Image.Image, imgsz: int = 640, confidence: float | None = None
    ) -> list[list[list[list[int]]]]:
        """Return mask geometries as outer rings followed by interior holes."""
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
        mask_data = getattr(masks, "data", None)
        if mask_data is not None:
            import cv2

            mask_array = mask_data.detach().cpu().numpy() if hasattr(mask_data, "detach") else np.asarray(mask_data)
            geometries: list[list[list[list[int]]]] = []
            width, height = original.size
            for mask_index, mask in enumerate(mask_array):
                resized = cv2.resize(mask.astype(np.uint8), (original.width, original.height), interpolation=cv2.INTER_NEAREST)
                binary = (resized > 0).astype(np.uint8)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                external_indices = [] if hierarchy is None else [
                    index for index in range(len(contours))
                    if hierarchy[0][index][3] == -1 and cv2.contourArea(contours[index]) >= 1
                ]
                # A YOLO mask can occasionally contain disconnected blobs.
                # Do not use one normalized ring for that case: it would draw
                # straight connector segments between the separate blobs.
                if len(external_indices) > 1:
                    for external_index in external_indices:
                        component_rings = [[[int(point[0][0]), int(point[0][1])] for point in contours[external_index]]]
                        child = hierarchy[0][external_index][2]
                        while child != -1:
                            if cv2.contourArea(contours[child]) >= 1:
                                component_rings.append([[int(point[0][0]), int(point[0][1])] for point in contours[child]])
                            child = hierarchy[0][child][0]
                        if len(component_rings[0]) >= 3:
                            geometries.append(component_rings)
                    continue
                outer: list[list[int]] | None = None
                if normalized_polygons is not None and mask_index < len(normalized_polygons):
                    points = np.asarray(normalized_polygons[mask_index], dtype=np.float32)
                    if points.ndim == 2 and points.shape[0] >= 3:
                        pixel_points = np.round(points * np.asarray([width, height], dtype=np.float32)).astype(np.int32)
                        pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, width - 1)
                        pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, height - 1)
                        outer = [[int(point[0]), int(point[1])] for point in pixel_points]
                if outer is None and hierarchy is not None:
                    for index, contour in enumerate(contours):
                        if hierarchy[0][index][3] == -1 and cv2.contourArea(contour) >= 1:
                            outer = [[int(point[0][0]), int(point[0][1])] for point in contour]
                            break
                if outer is None or len(outer) < 3:
                    continue
                rings = [outer]
                if hierarchy is not None:
                    for index, contour in enumerate(contours):
                        if hierarchy[0][index][3] != -1 and cv2.contourArea(contour) >= 1:
                            rings.append([[int(point[0][0]), int(point[0][1])] for point in contour])
                geometries.append(rings)
            if geometries:
                return geometries

        if normalized_polygons is None:
            return []
        width, height = original.size
        geometries = []
        for normalized in normalized_polygons:
            points = np.asarray(normalized, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3:
                continue
            pixel_points = np.round(points * np.asarray([width, height], dtype=np.float32)).astype(np.int32)
            pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, width - 1)
            pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, height - 1)
            contour = [[int(point[0]), int(point[1])] for point in pixel_points]
            if len(contour) >= 3:
                geometries.append([contour])
        return geometries

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
