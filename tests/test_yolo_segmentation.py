import numpy as np
from PIL import Image

from paving_measurement.yolo_segmentation import YoloSegmentationDetector


def test_normalized_yolo_mask_coordinates_map_to_image_pixels():
    class FakeModel:
        def predict(self, image, **kwargs):
            result = type("Result", (), {})()
            result.masks = type(
                "Masks",
                (),
                {"xyn": [np.array([[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]], dtype=np.float32)]},
            )()
            return [result]

    detector = YoloSegmentationDetector(FakeModel(), "test-model")
    polygons = detector.predict_polygons(Image.new("RGB", (100, 200)))
    assert polygons == [[[10, 40], [80, 40], [80, 180], [10, 180]]]
