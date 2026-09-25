"""Tiled satellite-image inference and coordinate helpers for parking stalls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from paving_measurement.geospatial import WEB_MERCATOR_MAX_LATITUDE, latlon_to_tile
from paving_measurement.mapbox import MapboxClient

TILE_SIZE = 256


@dataclass(frozen=True)
class ParkingSpot:
    """A detected oriented stall box and its centre in GeoJSON order."""

    longitude: float
    latitude: float
    confidence: float
    polygon_index: int
    corners: list[tuple[float, float]] = field(default_factory=list)
    angle_degrees: float = 0.0
    class_id: int = 0

    def as_dict(self) -> dict[str, float | int | list[float] | list[list[float]]]:
        return {
            "coordinates": [self.longitude, self.latitude],
            "longitude": self.longitude,
            "latitude": self.latitude,
            "confidence": self.confidence,
            "polygon_index": self.polygon_index,
            "corners": [[longitude, latitude] for longitude, latitude in self.corners],
            "angle_degrees": self.angle_degrees,
            "class_id": self.class_id,
        }


def validate_polygon(ring: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Validate a GeoJSON-style ring and return ``(longitude, latitude)`` points."""
    if len(ring) < 3:
        raise ValueError("Each polygon needs at least three [longitude, latitude] positions.")
    points: list[tuple[float, float]] = []
    for position in ring:
        if len(position) != 2:
            raise ValueError("Polygon positions must be [longitude, latitude].")
        longitude, latitude = float(position[0]), float(position[1])
        if not -180 <= longitude <= 180 or not -WEB_MERCATOR_MAX_LATITUDE <= latitude <= WEB_MERCATOR_MAX_LATITUDE:
            raise ValueError("Polygon coordinate is outside Web Mercator coverage.")
        points.append((longitude, latitude))
    if max(longitude for longitude, _ in points) - min(longitude for longitude, _ in points) > 180:
        raise ValueError("Polygons crossing the antimeridian are not supported.")
    return points


def point_in_polygon(longitude: float, latitude: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Return whether a longitude/latitude point is inside a polygon ring."""
    inside = False
    previous_longitude, previous_latitude = polygon[-1]
    for current_longitude, current_latitude in polygon:
        crosses = (current_latitude > latitude) != (previous_latitude > latitude)
        if crosses:
            crossing_longitude = (
                (previous_longitude - current_longitude)
                * (latitude - current_latitude)
                / (previous_latitude - current_latitude)
                + current_longitude
            )
            if longitude < crossing_longitude:
                inside = not inside
        previous_longitude, previous_latitude = current_longitude, current_latitude
    return inside


def tile_range_for_polygons(polygons: Sequence[Sequence[tuple[float, float]]], zoom: int) -> tuple[int, int, int, int]:
    """Return inclusive Mapbox tile bounds containing the supplied rings."""
    tile_positions = [latlon_to_tile(latitude, longitude, zoom) for polygon in polygons for longitude, latitude in polygon]
    x_values, y_values = zip(*tile_positions)
    tile_count = 2**zoom
    return (
        max(0, math.floor(min(x_values))),
        min(tile_count - 1, math.floor(max(x_values))),
        max(0, math.floor(min(y_values))),
        min(tile_count - 1, math.floor(max(y_values))),
    )


def pixel_to_longitude_latitude(tile_x: int, tile_y: int, pixel_x: float, pixel_y: float, zoom: int) -> tuple[float, float]:
    """Project a pixel in a Mapbox tile back to GeoJSON longitude/latitude."""
    world_pixels = TILE_SIZE * (2**zoom)
    normalized_x = (tile_x * TILE_SIZE + pixel_x) / world_pixels
    normalized_y = (tile_y * TILE_SIZE + pixel_y) / world_pixels
    longitude = normalized_x * 360.0 - 180.0
    latitude = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * normalized_y))))
    return longitude, latitude


def _distance_meters(first: ParkingSpot, second: ParkingSpot) -> float:
    latitude_scale = 111_320.0
    latitude_delta = (first.latitude - second.latitude) * latitude_scale
    longitude_delta = (first.longitude - second.longitude) * latitude_scale * math.cos(math.radians(first.latitude))
    return math.hypot(latitude_delta, longitude_delta)


def deduplicate_spots(spots: Iterable[ParkingSpot], distance_meters: float) -> list[ParkingSpot]:
    """Keep the highest-confidence member of nearby, overlapping-tile detections."""
    selected: list[ParkingSpot] = []
    for spot in sorted(spots, key=lambda item: item.confidence, reverse=True):
        if all(_distance_meters(spot, existing) > distance_meters for existing in selected):
            selected.append(spot)
    return selected


class ParkingStallDetector:
    """Run YOLO on overlapping high-resolution Mapbox tile mosaics."""

    def __init__(self, model: Any, client: MapboxClient, chunk_tiles: int = 2, overlap_tiles: int = 1) -> None:
        self.model = model
        self.client = client
        self.chunk_tiles = chunk_tiles
        self.overlap_tiles = overlap_tiles

    @staticmethod
    def load_huggingface_model(model_id: str, token: str | None) -> Any:
        """Download a YOLO checkpoint from Hugging Face and initialize it once."""
        from huggingface_hub import snapshot_download
        from ultralytics import YOLO

        model_dir = Path(snapshot_download(repo_id=model_id, token=token))
        checkpoints = sorted(model_dir.rglob("*.pt"))
        if not checkpoints:
            raise RuntimeError(f"No .pt YOLO checkpoint was found in Hugging Face repository {model_id!r}.")
        return YOLO(str(checkpoints[0]))

    def detect(
        self,
        polygons: Sequence[Sequence[tuple[float, float]]],
        zoom: int,
        confidence: float,
        max_tiles: int,
        imgsz: int,
        duplicate_distance_meters: float,
        processing_mode: str = "tiled",
    ) -> tuple[list[ParkingSpot], int]:
        min_x, max_x, min_y, max_y = tile_range_for_polygons(polygons, zoom)
        tile_total = (max_x - min_x + 1) * (max_y - min_y + 1)
        if tile_total > max_tiles:
            raise ValueError(
                f"Polygon covers {tile_total} tiles at zoom {zoom}; the limit is {max_tiles}. "
                "Split the polygon into smaller areas or use a lower zoom."
            )

        tiles = {
            (x, y): self.client.download_tile(x, y, zoom)
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
        }
        if processing_mode not in {"tiled", "whole"}:
            raise ValueError("processing_mode must be 'tiled' or 'whole'.")
        step = max(1, self.chunk_tiles - self.overlap_tiles)
        windows = (
            [(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)]
            if processing_mode == "whole"
            else [
                (start_x, start_y, min(self.chunk_tiles, max_x - start_x + 1), min(self.chunk_tiles, max_y - start_y + 1))
                for start_y in range(min_y, max_y + 1, step)
                for start_x in range(min_x, max_x + 1, step)
            ]
        )
        detections: list[ParkingSpot] = []
        for start_x, start_y, width, height in windows:
                mosaic = Image.new("RGB", (width * TILE_SIZE, height * TILE_SIZE))
                for offset_y in range(height):
                    for offset_x in range(width):
                        mosaic.paste(tiles[(start_x + offset_x, start_y + offset_y)], (offset_x * TILE_SIZE, offset_y * TILE_SIZE))

                # Keep each source window small and upscale it before inference. A 2x2
                # window at 1280px preserves stall detail better than shrinking a 3x3
                # window to 640px. The inverse scale maps YOLO boxes back to tile pixels.
                source_width, source_height = mosaic.size
                scale = max(1.0, imgsz / max(source_width, source_height))
                if scale != 1.0:
                    inference_image = mosaic.resize(
                        (round(source_width * scale), round(source_height * scale)), Image.Resampling.LANCZOS
                    )
                else:
                    inference_image = mosaic
                result = self.model.predict(inference_image, imgsz=imgsz, conf=confidence, verbose=False, max_det=2000)[0]
                obb = getattr(result, "obb", None)
                if obb is None or len(obb) == 0:
                    continue
                corners_array = obb.xyxyxyxy.cpu().numpy()
                angles = obb.xywhr.cpu().numpy()[:, 4]
                scores = obb.conf.cpu().numpy()
                classes = obb.cls.cpu().numpy()
                for corner_pixels, angle, score, class_id in zip(corners_array, angles, scores, classes):
                    source_corners = [
                        (float(point[0]) / scale, float(point[1]) / scale)
                        for point in corner_pixels
                    ]
                    center_x = sum(point[0] for point in source_corners) / len(source_corners)
                    center_y = sum(point[1] for point in source_corners) / len(source_corners)
                    longitude, latitude = pixel_to_longitude_latitude(
                        start_x,
                        start_y,
                        center_x,
                        center_y,
                        zoom,
                    )
                    geographic_corners = [
                        pixel_to_longitude_latitude(start_x, start_y, pixel_x, pixel_y, zoom)
                        for pixel_x, pixel_y in source_corners
                    ]
                    for polygon_index, polygon in enumerate(polygons):
                        if point_in_polygon(longitude, latitude, polygon):
                            detections.append(
                                ParkingSpot(
                                    longitude,
                                    latitude,
                                    float(score),
                                    polygon_index,
                                    geographic_corners,
                                    math.degrees(float(angle)),
                                    int(class_id),
                                )
                            )
                            break
        return deduplicate_spots(detections, duplicate_distance_meters), tile_total
