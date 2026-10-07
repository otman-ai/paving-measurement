"""RF-DETR segmentation inference adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class RfDetrSegmentationDetector:
    """Adapt RF-DETR's boolean instance masks to the API geometry format."""

    def __init__(self, model: Any, model_id: str, confidence: float = 0.5) -> None:
        self.model = model
        self.model_id = model_id
        self.confidence = confidence

    @staticmethod
    def load_huggingface_model(
        model_id: str, token: str | None, filename: str = "rfdetr_best_total.pth"
    ) -> "RfDetrSegmentationDetector":
        from huggingface_hub import hf_hub_download
        from rfdetr import RFDETRSegNano

        checkpoint = hf_hub_download(repo_id=model_id, filename=filename, token=token)
        model = RFDETRSegNano(pretrain_weights=checkpoint)
        optimize = getattr(model, "optimize_for_inference", None)
        if callable(optimize):
            optimize()
        return RfDetrSegmentationDetector(model, model_id)

    def predict_geometries(
        self, image: Image.Image, imgsz: int = 640, confidence: float | None = None
    ) -> list[list[list[list[int]]]]:
        import cv2

        source = image.convert("RGB")
        detections = self.model.predict(source, threshold=confidence if confidence is not None else self.confidence)
        mask_data = getattr(detections, "mask", None)
        if mask_data is None:
            return []
        if hasattr(mask_data, "detach"):
            masks = mask_data.detach().cpu().numpy()
        else:
            masks = np.asarray(mask_data)
        if masks.ndim == 2:
            masks = masks[None, ...]

        geometries: list[list[list[list[int]]]] = []
        for mask in masks:
            resized = cv2.resize(mask.astype(np.uint8), source.size, interpolation=cv2.INTER_NEAREST)
            contours, hierarchy = cv2.findContours((resized > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            for index, contour in enumerate(contours):
                if hierarchy[0][index][3] != -1 or cv2.contourArea(contour) < 16:
                    continue
                rings = [[[int(point[0][0]), int(point[0][1])] for point in contour]]
                child = hierarchy[0][index][2]
                while child != -1:
                    if cv2.contourArea(contours[child]) >= 16:
                        rings.append([[int(point[0][0]), int(point[0][1])] for point in contours[child]])
                    child = hierarchy[0][child][0]
                if len(rings[0]) >= 3:
                    geometries.append(rings)
        return geometries
