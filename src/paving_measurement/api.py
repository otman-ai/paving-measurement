"""FastAPI service for programmatic pavement analysis."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from paving_measurement.config import Settings, load_settings
from paving_measurement.mapbox import MapboxClient
from paving_measurement.satellite import get_satellite_image
from paving_measurement.segmentation import Sam3Segmenter


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


@dataclass
class Services:
    settings: Settings
    client: MapboxClient
    segmenter: Sam3Segmenter


def _as_data_url(image: Any, image_format: str = "PNG") -> str:
    """Encode a PIL image for a JSON API response."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{payload}"


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
    cors_origins = [origin.strip() for origin in configured_settings.cors_origins.split(",") if origin.strip()]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
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
            zoom = request.zoom or services.settings.default_zoom
            if not services.settings.min_zoom <= zoom <= services.settings.max_zoom:
                raise ValueError(f"Zoom must be between {services.settings.min_zoom} and {services.settings.max_zoom}.")
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

    return api
