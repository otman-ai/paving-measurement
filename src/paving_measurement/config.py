"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


class ConfigurationError(ValueError):
    """Raised when required configuration is absent or invalid."""


@dataclass(frozen=True)
class Settings:
    hf_token: str
    mapbox_token: str
    model_id: str = "facebook/sam3"
    prompt: str = "asphalt pavement"
    batch_size: int = 8
    min_zoom: int = 14
    max_zoom: int = 20
    default_zoom: int = 18
    host: str = "0.0.0.0"
    port: int = 7860
    cors_origins: str = "*"
    request_timeout_seconds: int = 30
    grid_sizes: tuple[int, ...] = tuple(range(1, 8))
    segmentation_threshold: float = 0.5
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ConfigurationError("BATCH_SIZE must be at least 1.")
        if not 0 < self.min_zoom <= self.default_zoom <= self.max_zoom:
            raise ConfigurationError("Zoom settings must satisfy 0 < MIN_ZOOM <= DEFAULT_ZOOM <= MAX_ZOOM.")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required. Set it in the environment or .env file.")
    return value


def _integer(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error


def load_settings() -> Settings:
    """Load and validate runtime settings without exposing secret values."""
    return Settings(
        hf_token=_required("HF_TOKEN"),
        mapbox_token=_required("MAPBOX_TOKEN"),
        model_id=os.getenv("MODEL_ID", "facebook/sam3"),
        prompt=os.getenv("SEGMENTATION_PROMPT", "asphalt pavement"),
        batch_size=_integer("BATCH_SIZE", 8),
        min_zoom=_integer("MIN_ZOOM", 14),
        max_zoom=_integer("MAX_ZOOM", 20),
        default_zoom=_integer("DEFAULT_ZOOM", 18),
        host=os.getenv("HOST", "0.0.0.0"),
        port=_integer("PORT", 7860),
        cors_origins=os.getenv("CORS_ORIGINS", "*"),
    )
