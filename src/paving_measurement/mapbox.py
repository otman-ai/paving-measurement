"""Small Mapbox HTTP client for address lookup and satellite tiles."""

from __future__ import annotations

import io

import requests
from PIL import Image


class MapboxClient:
    def __init__(self, token: str, timeout_seconds: int = 30) -> None:
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def geocode_address(self, address: str) -> tuple[float, float]:
        """Resolve an address and return latitude, longitude."""
        if not address or not address.strip():
            raise ValueError("Address is empty.")
        response = self.session.get(
            "https://api.mapbox.com/search/geocode/v6/forward",
            params={"q": address.strip(), "access_token": self.token, "limit": 1},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        features = response.json().get("features", [])
        if not features:
            raise ValueError(f"Address not found: {address}")
        longitude, latitude = features[0]["geometry"]["coordinates"][:2]
        return float(latitude), float(longitude)

    def download_tile(self, x: int, y: int, zoom: int) -> Image.Image:
        """Download one 256 by 256 Mapbox satellite tile."""
        tile_count = 2**zoom
        if not 0 <= y < tile_count:
            raise ValueError(f"Invalid tile Y coordinate: {y}")
        response = self.session.get(
            f"https://api.mapbox.com/v4/mapbox.satellite/{zoom}/{x % tile_count}/{y}.jpg90",
            params={"access_token": self.token},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
