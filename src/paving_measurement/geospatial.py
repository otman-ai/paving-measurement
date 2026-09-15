"""Web Mercator and ground-resolution helpers."""

from __future__ import annotations

import math

WEB_MERCATOR_MAX_LATITUDE = 85.05112878
METERS_PER_PIXEL_AT_ZOOM_0 = 156543.03392
SQUARE_METERS_TO_SQUARE_FEET = 10.7639


def latlon_to_tile(latitude: float, longitude: float, zoom: int) -> tuple[float, float]:
    """Convert latitude/longitude to fractional Web Mercator tile coordinates."""
    if not -WEB_MERCATOR_MAX_LATITUDE <= latitude <= WEB_MERCATOR_MAX_LATITUDE:
        raise ValueError("Latitude is outside the Web Mercator coverage area.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    scale = 2**zoom
    x = (longitude + 180.0) / 360.0 * scale
    latitude_radians = math.radians(latitude)
    y = (1 - math.asinh(math.tan(latitude_radians)) / math.pi) / 2 * scale
    return x, y


def meters_per_pixel(latitude: float, zoom: int) -> float:
    """Return approximate ground resolution for a 256-pixel Web Mercator tile."""
    return METERS_PER_PIXEL_AT_ZOOM_0 * math.cos(math.radians(latitude)) / (2**zoom)


def calculate_area(pixel_count: int, latitude: float, zoom: int) -> tuple[float, float, float]:
    """Return meters per pixel, square meters, and square feet for mask pixels."""
    resolution = meters_per_pixel(latitude, zoom)
    square_meters = pixel_count * resolution**2
    return resolution, square_meters, square_meters * SQUARE_METERS_TO_SQUARE_FEET
