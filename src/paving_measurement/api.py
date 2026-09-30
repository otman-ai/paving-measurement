"""FastAPI service for programmatic pavement analysis."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import math
from io import BytesIO
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from paving_measurement.config import Settings, load_settings
from paving_measurement.geospatial import latlon_to_tile, meters_per_pixel
from paving_measurement.mapbox import MapboxClient
from paving_measurement.parking_detection import pixel_to_longitude_latitude, tile_range_for_polygons, validate_polygon
from paving_measurement.satellite import (
    get_satellite_image,
    get_satellite_mosaic_for_tile_bounds,
)
from paving_measurement.yolo_segmentation import YoloSegmentationDetector

SEGMENTATION_INFERENCE_ZOOM = 20
SEGMENTATION_MODEL_ID = "otmanheddouch/yolo26n-seg"
LOGGER = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """Location input. An address takes precedence over coordinates."""

    address: str | None = Field(default=None, examples=["1600 Pennsylvania Avenue NW, Washington, DC"])
    latitude: float | None = Field(default=None, ge=-85.05112878, le=85.05112878)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    zoom: int | None = None
    prompt: str | None = Field(default=None, examples=["asphalt pavement, parking lot"])

    @model_validator(mode="after")
    def has_location(self) -> "AnalyzeRequest":
        if self.address and self.address.strip():
            return self
        if self.latitude is None or self.longitude is None:
            raise ValueError("Provide an address or both latitude and longitude.")
        return self


class AnalyzePolygonRequest(BaseModel):
    """User-drawn map regions for YOLO segmentation, in GeoJSON coordinate order."""

    polygons: list[list[list[float]]] = Field(
        min_length=1,
        description="One or more polygon rings; each position is [longitude, latitude].",
    )
    zoom: int | None = Field(default=None, description="Deprecated; segmentation always uses fixed inference zoom 20.")
    prompt: str | None = Field(default=None, examples=["asphalt pavement, parking lot"])
    processing_mode: Literal["whole", "chunked"] = Field(
        default="whole", description="Run one full-mosaic inference or overlapping whole-image chunks."
    )
    return_debug_image: bool = Field(
        default=False, description="Return the exact stitched YOLO input image as a base64 data URL."
    )
    max_tiles: int = Field(default=400, ge=1, le=900, description="Maximum source tiles across all chunks.")
    tiles_per_chunk: int = Field(default=9, ge=1, le=9, description="Maximum source tiles in each whole-image SAM pass.")

    @model_validator(mode="after")
    def has_valid_polygons(self) -> "AnalyzePolygonRequest":
        for polygon in self.polygons:
            validate_polygon(polygon)
        return self


@dataclass
class Services:
    settings: Settings
    client: MapboxClient
    segmenter: YoloSegmentationDetector


def _geographic_polygon_area_m2(polygon: list[list[float]]) -> float:
    """Approximate a longitude/latitude polygon's area in square metres."""
    if len(polygon) < 3:
        return 0.0
    latitude = sum(point[1] for point in polygon) / len(polygon)
    longitude_scale = 111_320.0 * math.cos(math.radians(latitude))
    latitude_scale = 111_320.0
    return abs(
        sum(
            polygon[index][0] * longitude_scale * polygon[(index + 1) % len(polygon)][1] * latitude_scale
            - polygon[(index + 1) % len(polygon)][0] * longitude_scale * polygon[index][1] * latitude_scale
            for index in range(len(polygon))
        )
        / 2.0
    )


def _mask_geometries(mask: np.ndarray) -> list[list[list[list[int]]]]:
    """Convert a binary mask to GeoJSON-style rings, retaining holes."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    geometries: list[list[list[list[int]]]] = []
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] != -1 or cv2.contourArea(contour) < 4:
            continue
        rings = [[[int(point[0][0]), int(point[0][1])] for point in contour]]
        child = hierarchy[0][index][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= 4:
                rings.append([[int(point[0][0]), int(point[0][1])] for point in contours[child]])
            child = hierarchy[0][child][0]
        geometries.append(rings)
    return geometries


def _clip_geometries_to_input(
    geometries: list[list[list[list[int]]]],
    input_polygons: list[list[tuple[float, float]]],
    mosaic_origin: tuple[int, int],
    zoom: int,
    image_size: tuple[int, int],
) -> list[list[list[list[int]]]]:
    """Clip mask geometries to drawn geographic regions without dropping holes."""
    width, height = image_size
    input_mask = np.zeros((height, width), dtype=np.uint8)
    origin_x, origin_y = mosaic_origin
    for polygon in input_polygons:
        points = [
            [
                round(tile_x * 256 - origin_x * 256),
                round(tile_y * 256 - origin_y * 256),
            ]
            for longitude, latitude in polygon
            for tile_x, tile_y in [latlon_to_tile(latitude, longitude, zoom)]
        ]
        cv2.fillPoly(input_mask, [np.asarray(points, dtype=np.int32)], 1)

    clipped: list[list[list[list[int]]]] = []
    for geometry in geometries:
        if not geometry:
            continue
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [np.asarray(geometry[0], dtype=np.int32)], 1)
        for hole in geometry[1:]:
            cv2.fillPoly(contour_mask, [np.asarray(hole, dtype=np.int32)], 0)
        intersection = cv2.bitwise_and(contour_mask, input_mask)
        for clipped_geometry in _mask_geometries(intersection):
            simplified_geometry: list[list[list[int]]] = []
            for ring in clipped_geometry:
                simplified = cv2.approxPolyDP(np.asarray(ring, dtype=np.int32), 1.5, True)
                points = [[int(point[0][0]), int(point[0][1])] for point in simplified]
                if len(points) >= 3:
                    simplified_geometry.append(points)
            if simplified_geometry:
                clipped.append(simplified_geometry)
    return clipped


def _geographic_geometry_area_m2(geometry: list[list[list[float]]]) -> float:
    if not geometry:
        return 0.0
    return max(0.0, _geographic_polygon_area_m2(geometry[0]) - sum(
        _geographic_polygon_area_m2(ring) for ring in geometry[1:]
    ))


def _as_data_url(image: Any, image_format: str = "PNG") -> str:
    """Encode a PIL image for a JSON API response."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{payload}"


def _chunk_tile_bounds(
    min_x: int, max_x: int, min_y: int, max_y: int, tiles_per_chunk: int
) -> list[tuple[int, int, int, int]]:
    """Split a tile rectangle into overlapping whole-image chunks."""
    side = max(1, min(3, math.isqrt(tiles_per_chunk)))
    step = max(1, side - 1)
    chunks: list[tuple[int, int, int, int]] = []
    for chunk_y in range(min_y, max_y + 1, step):
        for chunk_x in range(min_x, max_x + 1, step):
            chunks.append(
                (
                    chunk_x,
                    min(max_x, chunk_x + side - 1),
                    chunk_y,
                    min(max_y, chunk_y + side - 1),
                )
            )
    return chunks


def _resolve_location(request: AnalyzeRequest, client: MapboxClient) -> tuple[float, float]:
    if request.address and request.address.strip():
        return client.geocode_address(request.address)
    return float(request.latitude), float(request.longitude)


def create_api() -> FastAPI:
    """Create an API whose lifespan loads the model exactly once per worker."""

    configured_settings = load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.services = Services(
                settings=configured_settings,
                client=MapboxClient(configured_settings.mapbox_token, configured_settings.request_timeout_seconds),
                segmenter=YoloSegmentationDetector.load_huggingface_model(
                    SEGMENTATION_MODEL_ID, configured_settings.hf_token
                ),
            )
        except Exception:
            LOGGER.exception("Failed to load YOLO segmentation model %s", configured_settings.model_id)
            raise
        yield

    api = FastAPI(
        title="Paving Measurement API",
        version="0.1.0",
        description="Satellite-image pavement segmentation and approximate area measurement.",
        lifespan=lifespan,
    )
    # cors_origins = [origin.strip() for origin in configured_settings.cors_origins.split(",") if origin.strip()]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["https://paving-measurement-frontend.vercel.app"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/v1/satellite")
    def satellite(request: AnalyzeRequest) -> dict[str, Any]:
        services: Services = api.state.services
        try:
            latitude, longitude = _resolve_location(request, services.client)
            zoom = request.zoom or services.settings.default_zoom
            if not services.settings.min_zoom <= zoom <= services.settings.max_zoom:
                raise ValueError(f"Zoom must be between {services.settings.min_zoom} and {services.settings.max_zoom}.")
            image = get_satellite_image(services.client, latitude, longitude, zoom)
            return {"latitude": latitude, "longitude": longitude, "zoom": zoom, "image": _as_data_url(image, "JPEG")}
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.post("/v1/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        services: Services = api.state.services
        try:
            latitude, longitude = _resolve_location(request, services.client)
            zoom = SEGMENTATION_INFERENCE_ZOOM
            satellite_image = get_satellite_image(services.client, latitude, longitude, zoom)
            prompts = [item.strip() for item in (request.prompt or services.settings.prompt).split(",") if item.strip()]
            result = services.segmenter.run(satellite_image, latitude, zoom, prompts)
            return {
                "latitude": latitude,
                "longitude": longitude,
                "zoom": zoom,
                "summary": result.summary,
                "grid_results": result.grid_rows,
                "satellite_image": _as_data_url(satellite_image, "JPEG"),
                "final_overlay": _as_data_url(result.final_overlay),
                "comparison": _as_data_url(result.comparison),
                "polygons": result.polygons,
                "meters_per_pixel": result.meters_per_pixel,
                "area_m2": result.area_m2,
                "area_ft2": result.area_ft2,
            }
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.post("/v1/analyze-polygon")
    def analyze_polygon(request: AnalyzePolygonRequest) -> dict[str, Any]:
        """Segment only the map region drawn by the user and return geographic contours."""
        services: Services = api.state.services
        try:
            selected_polygons = [validate_polygon(polygon) for polygon in request.polygons]
            min_x, max_x, min_y, max_y = tile_range_for_polygons(selected_polygons, SEGMENTATION_INFERENCE_ZOOM)
            tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
            if tile_count > request.max_tiles:
                raise ValueError(
                    f"Polygon covers {tile_count} tiles at zoom {SEGMENTATION_INFERENCE_ZOOM}, exceeding the request limit "
                    f"of {request.max_tiles}. Increase max_tiles or draw a smaller area."
                )
            prompts = [item.strip() for item in (request.prompt or services.settings.prompt).split(",") if item.strip()]
            latitude = sum(latitude for polygon in selected_polygons for _, latitude in polygon) / sum(
                len(polygon) for polygon in selected_polygons
            )
            # All chunk contours are rasterized into one shared mask. This is
            # a true geometric union, so an object detected in two overlapping
            # chunks cannot produce two translucent fills on the map.
            full_width = (max_x - min_x + 1) * 256
            full_height = (max_y - min_y + 1) * 256
            union_mask = np.zeros((full_height, full_width), dtype=np.uint8)
            grid_rows: list[list[str | int]] = []
            summaries: list[str] = []
            debug_images: list[str] = []
            debug_overlays: list[str] = []
            mask_counts: list[int] = []
            input_dimensions: list[list[int]] = []
            return_debug_image = getattr(request, "return_debug_image", False)
            processing_mode = getattr(request, "processing_mode", "whole")
            chunks = (
                [(min_x, max_x, min_y, max_y)]
                if processing_mode == "whole"
                else _chunk_tile_bounds(min_x, max_x, min_y, max_y, request.tiles_per_chunk)
            )
            for chunk_min_x, chunk_max_x, chunk_min_y, chunk_max_y in chunks:
                mosaic = get_satellite_mosaic_for_tile_bounds(
                    services.client,
                    chunk_min_x,
                    chunk_max_x,
                    chunk_min_y,
                    chunk_max_y,
                    SEGMENTATION_INFERENCE_ZOOM,
                )
                if return_debug_image:
                    debug_images.append(_as_data_url(mosaic.image, "JPEG"))
                geometries = services.segmenter.predict_geometries(mosaic.image)
                mask_counts.append(len(geometries))
                input_dimensions.append([mosaic.image.width, mosaic.image.height])
                if return_debug_image:
                    overlay = np.asarray(mosaic.image.convert("RGB")).copy()
                    for geometry in geometries:
                        for ring in geometry:
                            cv2.polylines(
                                overlay,
                                [np.asarray(ring, dtype=np.int32)],
                                isClosed=True,
                                color=(255, 0, 0) if ring is geometry[0] else (0, 0, 0),
                                thickness=2,
                            )
                    debug_overlays.append(_as_data_url(Image.fromarray(overlay), "JPEG"))
                summaries.append(f"YOLO segmentation detected {len(geometries)} masks in this whole-image chunk.")
                clipped_geometries = _clip_geometries_to_input(
                    geometries,
                    selected_polygons,
                    (mosaic.min_tile_x, mosaic.min_tile_y),
                    SEGMENTATION_INFERENCE_ZOOM,
                    mosaic.image.size,
                )
                for pixel_geometry in clipped_geometries:
                    global_geometry = []
                    for pixel_ring in pixel_geometry:
                        global_geometry.append([
                            [
                                pixel_x + (mosaic.min_tile_x - min_x) * 256,
                                pixel_y + (mosaic.min_tile_y - min_y) * 256,
                            ]
                            for pixel_x, pixel_y in pixel_ring
                        ])
                    geometry_mask = np.zeros_like(union_mask)
                    cv2.fillPoly(geometry_mask, [np.asarray(global_geometry[0], dtype=np.int32)], 1)
                    for hole in global_geometry[1:]:
                        cv2.fillPoly(geometry_mask, [np.asarray(hole, dtype=np.int32)], 0)
                    union_mask = np.maximum(union_mask, geometry_mask)
            merged_geometries = _mask_geometries(union_mask)
            geographic_geometries: list[list[list[list[float]]]] = []
            for geometry in merged_geometries:
                geographic_geometry = []
                for ring in geometry:
                    simplified = cv2.approxPolyDP(np.asarray(ring, dtype=np.int32), 1.5, True)
                    if len(simplified) < 3:
                        continue
                    geographic_geometry.append([
                        list(pixel_to_longitude_latitude(
                            min_x,
                            min_y,
                            int(point[0][0]),
                            int(point[0][1]),
                            SEGMENTATION_INFERENCE_ZOOM,
                        ))
                        for point in simplified
                    ])
                if geographic_geometry:
                    geographic_geometries.append(geographic_geometry)
            geographic_polygons = [geometry[0] for geometry in geographic_geometries]
            area_m2 = sum(_geographic_geometry_area_m2(geometry) for geometry in geographic_geometries)
            return {
                "zoom": SEGMENTATION_INFERENCE_ZOOM,
                "tile_count": tile_count,
                "chunk_count": len(chunks),
                "processing_mode": processing_mode,
                "model_id": SEGMENTATION_MODEL_ID,
                "summary": f"Processed {len(chunks)} whole-image YOLO segmentation pass(es) at fixed zoom {SEGMENTATION_INFERENCE_ZOOM}.\n\n"
                + "\n\n".join(summaries),
                "grid_results": grid_rows,
                "polygons": geographic_polygons,
                "polygon_geometries": geographic_geometries,
                "debug_images": debug_images,
                "debug_overlays": debug_overlays,
                "mask_counts": mask_counts,
                "input_dimensions": input_dimensions,
                "meters_per_pixel": meters_per_pixel(latitude, SEGMENTATION_INFERENCE_ZOOM),
                "area_m2": area_m2,
                "area_ft2": area_m2 * 10.7639,
            }
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return api
