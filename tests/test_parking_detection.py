import pytest
import numpy as np
from PIL import Image

from paving_measurement.parking_detection import (
    ParkingSpot,
    deduplicate_spots,
    pixel_to_longitude_latitude,
    point_in_polygon,
    tile_range_for_polygons,
    validate_polygon,
)


def test_tile_origin_round_trips_to_longitude_latitude():
    longitude, latitude = pixel_to_longitude_latitude(0, 0, 128, 128, 0)
    assert longitude == pytest.approx(0)
    assert latitude == pytest.approx(0)


def test_polygon_filter_and_tile_bounds():
    polygon = validate_polygon([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    assert point_in_polygon(0, 0, polygon)
    assert not point_in_polygon(2, 0, polygon)
    assert tile_range_for_polygons([polygon], 2) == (1, 2, 1, 2)


def test_overlapping_tile_detections_keep_best_confidence():
    spots = deduplicate_spots(
        [ParkingSpot(0, 0, 0.7, 0), ParkingSpot(0.000001, 0, 0.9, 0)], distance_meters=2
    )
    assert spots == [ParkingSpot(0.000001, 0, 0.9, 0)]


def test_detector_upscales_small_windows_and_maps_boxes_back():
    class TensorLike:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __getitem__(self, item):
            return TensorLike(self.value[item])

    class FakeClient:
        def download_tile(self, x: int, y: int, zoom: int) -> Image.Image:
            return Image.new("RGB", (256, 256))

    class FakeModel:
        def __init__(self):
            self.sizes = []

        def predict(self, image, **kwargs):
            self.sizes.append(image.size)
            result = type("Result", (), {})()
            result.obb = type(
                "OBB",
                (),
                {
                    "xyxyxyxy": TensorLike(np.array([[[300, 300], [340, 300], [340, 340], [300, 340]]])),
                    "xywhr": TensorLike(np.array([[320, 320, 40, 40, 0.0]])),
                    "conf": TensorLike(np.array([0.9])),
                    "cls": TensorLike(np.array([0])),
                    "__len__": lambda _self: 1,
                },
            )()
            return [result]

    from paving_measurement.parking_detection import ParkingStallDetector

    model = FakeModel()
    polygon = validate_polygon([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    spots, tile_count = ParkingStallDetector(model, FakeClient()).detect(
        [polygon], zoom=2, confidence=0.25, max_tiles=10, imgsz=1280, duplicate_distance_meters=2
    )
    assert tile_count == 4
    assert model.sizes[0] == (1280, 1280)
    assert len(model.sizes) == 4
