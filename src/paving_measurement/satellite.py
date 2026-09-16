"""Satellite-mosaic retrieval service."""

from __future__ import annotations

from PIL import Image

from paving_measurement.geospatial import latlon_to_tile
from paving_measurement.mapbox import MapboxClient


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
