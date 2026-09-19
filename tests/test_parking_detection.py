import pytest

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
