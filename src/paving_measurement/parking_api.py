"""FastAPI application for tiled parking-stall centre detection."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from paving_measurement.mapbox import MapboxClient
from paving_measurement.parking_detection import ParkingStallDetector, validate_polygon

PARKING_MODEL_ID = "otmanheddouch/yolov8n-09-19-2026"
PARKING_INFERENCE_ZOOM = 20


class ParkingDetectionRequest(BaseModel):
    """GeoJSON rings marking the areas where parking stalls should be found."""

    polygons: list[list[list[float]]] = Field(
        min_length=1,
        description="One or more polygon rings; every position is [longitude, latitude].",
        examples=[[[[-77.0366, 38.8975], [-77.0359, 38.8975], [-77.0359, 38.8971]]]],
    )
    zoom: int | None = Field(default=None, description="Deprecated; parking inference always uses fixed zoom 20.")
    confidence: float = Field(default=0.25, gt=0, lt=1)
    max_tiles: int = Field(default=400, ge=1, le=900)
    imgsz: int = Field(default=1280, ge=256, le=1536)
    duplicate_distance_meters: float = Field(default=2.0, gt=0, le=20)

    @field_validator("polygons")
    @classmethod
    def valid_rings(cls, polygons: list[list[list[float]]]) -> list[list[list[float]]]:
        for polygon in polygons:
            validate_polygon(polygon)
        return polygons


@dataclass
class ParkingServices:
    detector: ParkingStallDetector


def create_parking_api() -> FastAPI:
    """Create the independent, model-once-per-worker parking detector API."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        mapbox_token = os.getenv("MAPBOX_TOKEN", "").strip()
        if not mapbox_token:
            raise RuntimeError("MAPBOX_TOKEN is required.")
        model = ParkingStallDetector.load_huggingface_model(PARKING_MODEL_ID, os.getenv("HF_TOKEN"))
        application.state.services = ParkingServices(ParkingStallDetector(model, MapboxClient(mapbox_token)))
        yield

    api = FastAPI(
        title="Parking Stall Detection API",
        version="0.1.0",
        description="Runs a Hugging Face YOLO model on high-resolution satellite tiles and returns stall centres.",
        lifespan=lifespan,
    )
    origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_id": PARKING_MODEL_ID}

    @api.post("/v1/detect-parking-stalls")
    def detect_parking_stalls(request: ParkingDetectionRequest) -> dict[str, Any]:
        services: ParkingServices = api.state.services
        try:
            polygons = [validate_polygon(polygon) for polygon in request.polygons]
            spots, tile_count = services.detector.detect(
                polygons=polygons,
                zoom=PARKING_INFERENCE_ZOOM,
                confidence=request.confidence,
                max_tiles=request.max_tiles,
                imgsz=request.imgsz,
                duplicate_distance_meters=request.duplicate_distance_meters,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "model_id": PARKING_MODEL_ID,
            "zoom": PARKING_INFERENCE_ZOOM,
            "tile_count": tile_count,
            "spots": [spot.as_dict() for spot in spots],
        }

    return api
