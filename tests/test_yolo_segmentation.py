import numpy as np
import pytest
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


def test_semantic_mask_data_is_used_when_instance_masks_are_missing():
    pytest.importorskip("cv2")

    class FakeModel:
        def predict(self, image, **kwargs):
            result = type("Result", (), {})()
            result.masks = None
            result.semantic_mask = type(
                "SemanticMask",
                (),
                {"data": np.array(
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    dtype=np.uint8,
                )},
            )()
            return [result]

    detector = YoloSegmentationDetector(FakeModel(), "semantic-test")
    polygons = detector.predict_polygons(Image.new("RGB", (40, 40)))
    assert polygons
    assert min(point[0] for point in polygons[0]) >= 0
    assert max(point[0] for point in polygons[0]) <= 39
