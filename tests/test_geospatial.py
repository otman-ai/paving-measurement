import pytest

from paving_measurement.geospatial import calculate_area, latlon_to_tile, meters_per_pixel


def test_origin_maps_to_center_tile_at_zoom_zero():
    assert latlon_to_tile(0, 0, 0) == (0.5, 0.5)


def test_resolution_halves_for_each_zoom_level():
    assert meters_per_pixel(0, 1) == pytest.approx(meters_per_pixel(0, 0) / 2)


def test_area_conversion():
    resolution, square_meters, square_feet = calculate_area(1, 0, 0)
    assert square_meters == pytest.approx(resolution**2)
    assert square_feet == pytest.approx(square_meters * 10.7639)
