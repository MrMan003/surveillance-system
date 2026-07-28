"""Face detection with SCRFD, via the InsightFace model zoo.

SCRFD is used rather than a general object detector for two reasons.  It is
trained specifically on faces at the scales CCTV produces, and it emits five
landmarks alongside each box.  Those landmarks are not a bonus: the Phase 5
Umeyama alignment needs them, and a face box without them cannot be aligned,
which means it cannot be recognised.

Only ``det_10g.onnx`` is loaded.  The ``buffalo_l`` pack ships five models
totalling 341 MB; the detector is 17 MB of that.  Loading the pack through
``FaceAnalysis`` would also resident-load 3D landmarks, 2D landmarks, gender
and age estimation, and a recognition model this project replaces with AdaFace
-- roughly 324 MB of VRAM spent on nothing, on a device where YOLOv8s, SCRFD
and AdaFace must coexist.

A note on batching
------------------
``det_10g.onnx`` declares its input as ``[1, 3, ?, ?]``.  The batch dimension
is fixed at one: spatial dimensions are dynamic, batch is not.  This class
therefore loops over the batch rather than pretending to fuse it.  The loop is
honest; the alternative would be an API that accepts a batch and silently
serialises it while reporting batched throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from configs.config import DetectionConfig, PathConfig, RuntimeConfig
from detection.base import Detector, DetectorError
from utils.log import get_logger
from utils.types import BoundingBox, FaceDetection

__all__ = ["FaceDetector"]

LOGGER = get_logger(__name__)

#: Filename of the SCRFD detector inside an InsightFace model pack.
SCRFD_FILENAME = "det_10g.onnx"


class FaceDetector(Detector[FaceDetection]):
    """SCRFD face detector emitting boxes and five-point landmarks.

    Args:
        config: Detector thresholds and geometry.
        runtime: Device and precision policy.
        paths: Filesystem layout; the model pack is resolved under
            ``weights_dir/models/<pack name>``.
    """

    def __init__(
        self,
        config: DetectionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
    ) -> None:
        super().__init__(config, runtime, paths, name="scrfd-face")
        self._model = None

    @property
    def _warmup_size(self) -> Tuple[int, int]:
        """Warm up at SCRFD's configured input resolution, not YOLO's."""
        width, height = self._config.face_det_size
        return height, width

    # -- weights ----------------------------------------------------------- #
    def _pack_directory(self) -> Path:
        """Return the directory the model pack should live in."""
        return self._paths.weights_dir / "models" / self._config.face_model_name

    def _ensure_pack(self) -> Path:
        """Locate the SCRFD checkpoint, downloading the pack if necessary.

        Returns:
            Path to ``det_10g.onnx``.

        Raises:
            DetectorError: If the pack cannot be found or downloaded.
        """
        model_path = self._pack_directory() / SCRFD_FILENAME
        if model_path.is_file():
            return model_path

        LOGGER.info(
            "SCRFD not found at %s; downloading %s pack",
            model_path,
            self._config.face_model_name,
        )
        try:
            from insightface.utils import storage

            storage.ensure_available(
                "models",
                self._config.face_model_name,
                root=str(self._paths.weights_dir),
            )
        except Exception as exc:  # noqa: BLE001 - insightface raises widely
            raise DetectorError(
                f"Cannot obtain the {self._config.face_model_name} pack: {exc}. "
                f"Download it manually into {self._pack_directory()}."
            ) from exc

        if not model_path.is_file():
            raise DetectorError(f"{SCRFD_FILENAME} missing after download at {model_path}")
        return model_path

    def _providers(self) -> List[str]:
        """Choose ONNXRuntime execution providers for the resolved device.

        Returns:
            Provider names in priority order.

        Raises:
            DetectorError: If CUDA was requested but the CUDA provider is
                absent.  Falling back silently would move SCRFD onto the CPU,
                where it dominates wall-clock and quietly destroys throughput
                while everything still appears to work.
        """
        import onnxruntime as ort

        available = ort.get_available_providers()
        if not self._device.startswith("cuda"):
            return ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" not in available:
            raise DetectorError(
                "CUDA was requested but CUDAExecutionProvider is unavailable "
                f"(providers: {available}). SCRFD would silently run on CPU. "
                "Install onnxruntime-gpu, or set runtime.device='cpu' to accept "
                "CPU inference explicitly."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # -- lifecycle --------------------------------------------------------- #
    def load(self) -> "FaceDetector":
        """Load SCRFD onto the target device.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            DetectorError: If InsightFace is unavailable or the model cannot
                be prepared.
        """
        if self._loaded:
            return self

        try:
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise DetectorError(
                "insightface is required: pip install insightface==0.7.3"
            ) from exc

        model_path = self._ensure_pack()
        providers = self._providers()
        # InsightFace maps ctx_id < 0 to CPU and otherwise to that CUDA index.
        ctx_id = -1 if not self._device.startswith("cuda") else (
            int(self._device.split(":")[1]) if ":" in self._device else 0
        )

        try:
            self._model = get_model(str(model_path), providers=providers)
            self._model.prepare(
                ctx_id=ctx_id,
                input_size=tuple(self._config.face_det_size),
                det_thresh=self._config.face_conf,
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"Cannot load SCRFD from {model_path}: {exc}") from exc

        self._loaded = True
        LOGGER.info(
            "Loaded SCRFD %s on %s (det_size=%s, conf=%.2f, providers=%s)",
            model_path.name,
            self._device,
            self._config.face_det_size,
            self._config.face_conf,
            providers[0],
        )
        return self

    def close(self) -> None:
        """Release the ONNX session and free device memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        LOGGER.debug("%s closed", self._name)

    # -- inference --------------------------------------------------------- #
    def _infer(self, frames: Sequence[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces in each frame.

        Serialised by necessity: see the module docstring on batching.

        Args:
            frames: BGR ``uint8`` arrays shaped ``(H, W, 3)``.

        Returns:
            One face list per frame, in input order.
        """
        return [self._detect_single(frame) for frame in frames]

    def _detect_single(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in one frame and apply the size filter.

        Args:
            frame: BGR ``uint8`` array shaped ``(H, W, 3)``.

        Returns:
            Faces whose shorter side meets ``min_face_size``.
        """
        boxes, landmarks = self._model.detect(
            frame,
            input_size=tuple(self._config.face_det_size),
            metric="default",
        )

        if boxes is None or len(boxes) == 0:
            return []
        if landmarks is None:
            # Without landmarks a face cannot be aligned, so it cannot be
            # recognised. Emitting it would push the failure into Phase 5.
            LOGGER.warning("SCRFD returned boxes without landmarks; discarding frame")
            return []

        xyxy = boxes[:, :4].astype(np.float32)
        scores = boxes[:, 4].astype(np.float32)

        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        keep = np.minimum(widths, heights) >= self._config.min_face_size
        keep &= (widths > 0) & (heights > 0)

        if not keep.any():
            return []

        xyxy = xyxy[keep]
        scores = scores[keep]
        landmarks = landmarks[keep].astype(np.float32)

        return [
            FaceDetection(
                box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                score=float(min(max(score, 0.0), 1.0)),
                landmarks=np.ascontiguousarray(points),
            )
            for (x1, y1, x2, y2), score, points in zip(xyxy, scores, landmarks)
        ]