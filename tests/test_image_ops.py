from PIL import Image

from paving_measurement.image_ops import create_grid_tiles


def test_grid_tiles_cover_image_and_keep_last_pixel_remainder():
    tiles = create_grid_tiles(Image.new("RGB", (10, 7)), 3)
    assert len(tiles) == 9
    assert tiles[-1]["image"].size == (4, 3)
