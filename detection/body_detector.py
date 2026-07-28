"""Person detection with YOLOv8s.

YOLOv8s is the size that fits this problem: roughly 11M parameters, strong COCO
person AP, and small enough that it, SCRFD and AdaFace coexist in a T4's 16 GB
alongside decoded frames.  YOLOv8n is meaningfully worse on small and occluded
people, which is most of what a ceiling-mounted camera sees; YOLOv8m buys little
person AP for the extra memory once the face branch is added.

Only class 0 (``person``) is retained.  The remaining 79 COCO classes are
computed regardless -- the head is shared -- but filtering here keeps every
downstream stage from re-checking.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from configs.config import DetectionConfig, PathConfig, RuntimeConfig
from detection.base import Detector, DetectorError
from utils.log import get_logger
from utils.types import BoundingBox, Detection

__all__ = ["BodyDetector"]

LOGGER = get_logger(__name__)


class BodyDetector(Detector):
    """YOLOv8s person detector.

    Ultralytics performs letterboxing and coordinate inversion internally and
    returns boxes already in original frame coordinates, so this class does not
    reimplement either.  Reimplementing them is a common source of boxes that
    are plausibly placed but consistently wrong.

    Args:
        config: Detector thresholds and geometry.
        runtime: Device and precision policy.
        paths: Filesystem layout used to resolve the checkpoint.
    """

    def __init__(
        self,
        config: DetectionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
    ) -> None:
        super().__init__(config, runtime, paths, name="yolov8s-body")
        self._model = None
        self._class_ids = set(config.body_classes)

    def load(self) -> "BodyDetector":
        """Load YOLOv8s onto the target device.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            DetectorError: If ultralytics is unavailable or the checkpoint
                cannot be loaded.
        """
        if self._loaded:
            return self

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "ultralytics is required: pip install 'ultralytics>=8.2.0,<8.4.0'"
            ) from exc

        weights = self._resolve_weights(self._config.body_weights)
        try:
            self._model = YOLO(weights)
            self._model.to(self._device)
            # Folding batch-norm into the preceding convolution is a free
            # inference speedup and changes no output values.
            self._model.fuse()
        except Exception as exc:  # noqa: BLE001 - ultralytics raises widely
            raise DetectorError(f"Cannot load {weights}: {exc}") from exc

        self._loaded = True
        LOGGER.info(
            "Loaded %s on %s (half=%s, imgsz=%d, classes=%s)",
            weights,
            self._device,
            self._half,
            self._config.body_imgsz,
            sorted(self._class_ids),
        )
        return self

    def close(self) -> None:
        """Release the model and free device memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False

        try:
            import torch

            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        LOGGER.debug("%s closed", self._name)

    def _infer(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Run YOLOv8s over a batch of frames.

        Args:
            frames: BGR ``uint8`` arrays shaped ``(H, W, 3)``.

        Returns:
            One detection list per frame, in input order.
        """
        import torch

        with torch.inference_mode():
            outputs = self._model.predict(
                source=list(frames),
                imgsz=self._config.body_imgsz,
                conf=self._config.body_conf,
                iou=self._config.body_iou,
                classes=sorted(self._class_ids),
                max_det=self._config.body_max_det,
                half=self._half,
                device=self._device,
                verbose=False,
            )

        return [self._convert(result) for result in outputs]

    def _convert(self, result: object) -> List[Detection]:
        """Convert one ultralytics result into :class:`Detection` objects.

        Boxes are pulled out as arrays and filtered with vectorised operations
        rather than a Python loop over detections, which matters on crowded
        frames where ``max_det`` allows hundreds of boxes.

        Args:
            result: A single ultralytics ``Results`` object.

        Returns:
            Detections passing the class and minimum-area filters.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)

        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        keep = (widths * heights) >= self._config.min_body_area
        keep &= (widths > 0) & (heights > 0)

        if not keep.any():
            return []

        xyxy = xyxy[keep]
        scores = scores[keep]
        classes = classes[keep]

        return [
            Detection(
                box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                # Clamp: FP16 accumulation occasionally yields 1.0000001, which
                # would trip Detection's range validation.
                score=float(min(max(score, 0.0), 1.0)),
                class_id=int(class_id),
            )
            for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, classes)
        ]