"""Satellite-mosaic retrieval service."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from paving_measurement.geospatial import latlon_to_tile
from paving_measurement.mapbox import MapboxClient
from paving_measurement.parking_detection import tile_range_for_polygons


@dataclass(frozen=True)
class TileMosaic:
    """A Mapbox tile mosaic together with the origin needed to georeference pixels."""

    image: Image.Image
    min_tile_x: int
    min_tile_y: int
    tile_count: int


def get_satellite_image(
    client: MapboxClient, latitude: float, longitude: float, zoom: int
) -> Image.Image:
    """Download the 3 by 3 satellite mosaic centered on a coordinate."""
    fractional_x, fractional_y = latlon_to_tile(latitude, longitude, zoom)
    center_x, center_y = int(fractional_x), int(fractional_y)
    tile_size = 256
    mosaic = Image.new("RGB", (tile_size * 3, tile_size * 3))
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            tile = client.download_tile(center_x + x_offset, center_y + y_offset, zoom)
            mosaic.paste(tile, ((x_offset + 1) * tile_size, (y_offset + 1) * tile_size))
    return mosaic


def get_satellite_mosaic_for_polygons(
    client: MapboxClient,
    polygons: list[list[tuple[float, float]]],
    zoom: int,
    max_tiles: int = 9,
) -> TileMosaic:
    """Download the exact tile rectangle containing drawn GeoJSON polygon rings."""
    min_x, max_x, min_y, max_y = tile_range_for_polygons(polygons, zoom)
    tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
    if tile_count > max_tiles:
        raise ValueError(
            f"Polygon covers {tile_count} tiles at zoom {zoom}; SAM3 analysis is limited to {max_tiles}. "
            "Draw a smaller area or use a lower zoom."
        )
    return get_satellite_mosaic_for_tile_bounds(client, min_x, max_x, min_y, max_y, zoom)


def get_satellite_mosaic_for_tile_bounds(
    client: MapboxClient,
    min_tile_x: int,
    max_tile_x: int,
    min_tile_y: int,
    max_tile_y: int,
    zoom: int,
) -> TileMosaic:
    """Download an inclusive tile rectangle without applying a size limit."""
    if min_tile_x > max_tile_x or min_tile_y > max_tile_y:
        raise ValueError("Tile bounds must describe a non-empty rectangle.")
    width, height = max_tile_x - min_tile_x + 1, max_tile_y - min_tile_y + 1
    mosaic = Image.new("RGB", (width * 256, height * 256))
    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            mosaic.paste(
                client.download_tile(tile_x, tile_y, zoom),
                ((tile_x - min_tile_x) * 256, (tile_y - min_tile_y) * 256),
            )
    return TileMosaic(
        mosaic,
        min_tile_x,
        min_tile_y,
        width * height,
    )
