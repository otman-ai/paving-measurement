"""FastAPI service for programmatic pavement analysis."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from paving_measurement.config import Settings, load_settings
from paving_measurement.geospatial import latlon_to_tile
from paving_measurement.mapbox import MapboxClient
from paving_measurement.parking_detection import pixel_to_longitude_latitude, tile_range_for_polygons, validate_polygon
from paving_measurement.satellite import (
    get_satellite_image,
    get_satellite_mosaic_for_tile_bounds,
)
from paving_measurement.segmentation import Sam3Segmenter

SAM_INFERENCE_ZOOM = 20


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
    """User-drawn map regions for SAM3 analysis, in GeoJSON coordinate order."""

    polygons: list[list[list[float]]] = Field(
        min_length=1,
        description="One or more polygon rings; each position is [longitude, latitude].",
    )
    zoom: int | None = Field(default=None, description="Deprecated; SAM3 always uses fixed inference zoom 20.")
    prompt: str | None = Field(default=None, examples=["asphalt pavement, parking lot"])
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
    segmenter: Sam3Segmenter


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


def _clip_contours_to_input(
    contours: list[list[list[int]]],
    input_polygons: list[list[tuple[float, float]]],
    mosaic_origin: tuple[int, int],
    zoom: int,
    image_size: tuple[int, int],
) -> list[list[list[int]]]:
    """Clip SAM contours to the drawn geographic regions before returning them."""
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

    clipped: list[list[list[int]]] = []
    for contour in contours:
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        points = np.asarray(contour, dtype=np.int32)
        cv2.fillPoly(contour_mask, [points], 1)
        intersection = cv2.bitwise_and(contour_mask, input_mask)
        clipped_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for clipped_contour in clipped_contours:
            if cv2.contourArea(clipped_contour) < 4:
                continue
            simplified = cv2.approxPolyDP(clipped_contour, 1.5, True)
            clipped_points = [[int(point[0][0]), int(point[0][1])] for point in simplified]
            if len(clipped_points) >= 3:
                clipped.append(clipped_points)
    return clipped


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
        application.state.services = Services(
            settings=configured_settings,
            client=MapboxClient(configured_settings.mapbox_token, configured_settings.request_timeout_seconds),
            segmenter=Sam3Segmenter(configured_settings),
        )
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
            zoom = SAM_INFERENCE_ZOOM
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
            min_x, max_x, min_y, max_y = tile_range_for_polygons(selected_polygons, SAM_INFERENCE_ZOOM)
            tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
            if tile_count > request.max_tiles:
                raise ValueError(
                    f"Polygon covers {tile_count} tiles at zoom {SAM_INFERENCE_ZOOM}, exceeding the request limit "
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
            meters_per_pixel = 0.0
            chunks = _chunk_tile_bounds(min_x, max_x, min_y, max_y, request.tiles_per_chunk)
            for chunk_min_x, chunk_max_x, chunk_min_y, chunk_max_y in chunks:
                mosaic = get_satellite_mosaic_for_tile_bounds(
                    services.client,
                    chunk_min_x,
                    chunk_max_x,
                    chunk_min_y,
                    chunk_max_y,
                    SAM_INFERENCE_ZOOM,
                )
                result = services.segmenter.run(
                    mosaic.image, latitude, SAM_INFERENCE_ZOOM, prompts, grid_sizes=(1,)
                )
                meters_per_pixel = result.meters_per_pixel
                summaries.append(result.summary)
                grid_rows.extend(result.grid_rows)
                clipped_pixel_polygons = _clip_contours_to_input(
                    result.polygons,
                    selected_polygons,
                    (mosaic.min_tile_x, mosaic.min_tile_y),
                    SAM_INFERENCE_ZOOM,
                    mosaic.image.size,
                )
                for pixel_polygon in clipped_pixel_polygons:
                    global_points = np.asarray(
                        [
                            [
                                pixel_x + (mosaic.min_tile_x - min_x) * 256,
                                pixel_y + (mosaic.min_tile_y - min_y) * 256,
                            ]
                            for pixel_x, pixel_y in pixel_polygon
                        ],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(union_mask, [global_points], 1)
            merged_contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            geographic_polygons: list[list[list[float]]] = []
            for contour in merged_contours:
                if cv2.contourArea(contour) < 4:
                    continue
                simplified = cv2.approxPolyDP(contour, 1.5, True)
                if len(simplified) < 3:
                    continue
                geographic_polygons.append(
                    [
                        list(
                            pixel_to_longitude_latitude(
                                min_x,
                                min_y,
                                int(point[0][0]),
                                int(point[0][1]),
                                SAM_INFERENCE_ZOOM,
                            )
                        )
                        for point in simplified
                    ]
                )
            area_m2 = sum(_geographic_polygon_area_m2(polygon) for polygon in geographic_polygons)
            return {
                "zoom": SAM_INFERENCE_ZOOM,
                "tile_count": tile_count,
                "chunk_count": len(chunks),
                "processing_mode": "whole_per_chunk",
                "summary": f"Processed {len(chunks)} overlapping whole-image chunks at fixed zoom {SAM_INFERENCE_ZOOM}.\n\n"
                + "\n\n".join(summaries),
                "grid_results": grid_rows,
                "polygons": geographic_polygons,
                "meters_per_pixel": meters_per_pixel,
                "area_m2": area_m2,
                "area_ft2": area_m2 * 10.7639,
            }
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return api
