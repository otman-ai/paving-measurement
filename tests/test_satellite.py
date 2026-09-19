from PIL import Image

from paving_measurement.satellite import get_satellite_mosaic_for_polygons


class FakeMapboxClient:
    def __init__(self):
        self.calls: list[tuple[int, int, int]] = []

    def download_tile(self, x: int, y: int, zoom: int) -> Image.Image:
        self.calls.append((x, y, zoom))
        return Image.new("RGB", (256, 256), (x % 255, y % 255, 0))


def test_polygon_mosaic_records_georeferencing_origin():
    client = FakeMapboxClient()
    mosaic = get_satellite_mosaic_for_polygons(client, [[(-1, -1), (1, -1), (1, 1)]], zoom=2)
    assert mosaic.image.size == (512, 512)
    assert (mosaic.min_tile_x, mosaic.min_tile_y, mosaic.tile_count) == (1, 1, 4)
    assert len(client.calls) == 4
