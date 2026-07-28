"""Runs the body and face detectors together over a shared frame batch.

This is composition, not inheritance.  ``CombinedDetector`` is not a
:class:`~detection.base.Detector`: it returns a *pair* of detection lists per
frame rather than one list, so making it a subclass would force the base
class's single-type contract to widen to ``Any`` and cost both detectors their
static typing.  It owns two detectors and coordinates them.

The two models are independent -- neither consumes the other's output -- so
running both on the same batch of frames is the whole job.  Pairing faces to
bodies is Phase 4's problem, and deliberately not done here: geometry-based
assignment needs the tracker's identities to be useful, and doing it early
would fix a policy the association module exists to own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from configs.config import DetectionConfig, PathConfig, RuntimeConfig
from detection.base import DetectorError
from detection.body_detector import BodyDetector
from detection.face_detector import FaceDetector
from utils.log import get_logger
from utils.types import Detection, FaceDetection

__all__ = ["FrameDetections", "CombinedDetector"]

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FrameDetections:
    """Everything both detectors found in a single frame.

    Attributes:
        frame_number: Emission index of the source frame.
        bodies: Person detections from YOLOv8s.
        faces: Face detections with landmarks from SCRFD.
    """

    frame_number: int
    bodies: List[Detection] = field(default_factory=list)
    faces: List[FaceDetection] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether neither detector found anything."""
        return not self.bodies and not self.faces

    def __len__(self) -> int:
        """Total detections across both detectors."""
        return len(self.bodies) + len(self.faces)

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"FrameDetections(n={self.frame_number}, "
            f"bodies={len(self.bodies)}, faces={len(self.faces)})"
        )


class CombinedDetector:
    """Coordinates the body and face detectors over shared frame batches.

    Args:
        config: Detector thresholds and geometry, shared by both detectors.
        runtime: Device and precision policy.
        paths: Filesystem layout used to resolve checkpoints.
        detect_bodies: Enable the person detector.
        detect_faces: Enable the face detector.

    Raises:
        ValueError: If both detectors are disabled.
    """

    def __init__(
        self,
        config: DetectionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
        detect_bodies: bool = True,
        detect_faces: bool = True,
    ) -> None:
        if not detect_bodies and not detect_faces:
            raise ValueError("At least one of detect_bodies or detect_faces must be enabled")

        self._config = config
        self._runtime = runtime
        self._paths = paths or PathConfig()

        self.body: Optional[BodyDetector] = (
            BodyDetector(config, runtime, self._paths) if detect_bodies else None
        )
        self.face: Optional[FaceDetector] = (
            FaceDetector(config, runtime, self._paths) if detect_faces else None
        )
        self._loaded = False

    # -- properties -------------------------------------------------------- #
    @property
    def is_loaded(self) -> bool:
        """Whether every enabled detector has loaded its weights."""
        return self._loaded

    @property
    def device(self) -> str:
        """Resolved compute device, shared by both detectors."""
        return self._runtime.resolve_device()

    # -- lifecycle --------------------------------------------------------- #
    def load(self) -> "CombinedDetector":
        """Load every enabled detector.

        If the second detector fails, the first is released before the error
        propagates, so a partial load never leaves weights stranded in VRAM.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            DetectorError: If any enabled detector cannot be loaded.
        """
        if self._loaded:
            return self

        try:
            if self.body is not None:
                self.body.load()
            if self.face is not None:
                self.face.load()
        except DetectorError:
            self.close()
            raise

        self._loaded = True
        LOGGER.info(
            "CombinedDetector ready on %s (bodies=%s, faces=%s)",
            self.device,
            self.body is not None,
            self.face is not None,
        )
        return self

    def close(self) -> None:
        """Release every detector. Safe to call more than once."""
        for detector in (self.body, self.face):
            if detector is not None:
                detector.close()
        self._loaded = False

    def __enter__(self) -> "CombinedDetector":
        """Load on entering a ``with`` block."""
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Release on leaving a ``with`` block."""
        self.close()

    def warmup(self, iterations: int = 2) -> None:
        """Warm up every detector at its own native input resolution.

        Args:
            iterations: Number of warmup batches per detector.
        """
        for detector in (self.body, self.face):
            if detector is not None:
                detector.warmup(iterations=iterations)

    # -- inference --------------------------------------------------------- #
    def detect(self, frame: np.ndarray, frame_number: int = -1) -> FrameDetections:
        """Detect bodies and faces in a single frame.

        Args:
            frame: BGR ``uint8`` array shaped ``(H, W, 3)``.
            frame_number: Emission index recorded on the result.

        Returns:
            Both detection lists for that frame.
        """
        return self.detect_batch([frame], [frame_number])[0]

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> List[FrameDetections]:
        """Detect bodies and faces across a batch of frames.

        Args:
            frames: BGR ``uint8`` arrays.
            frame_numbers: Emission indices, one per frame.  Defaults to
                positional indices when omitted.

        Returns:
            One :class:`FrameDetections` per input frame, in input order.

        Raises:
            DetectorError: If the detectors are not loaded.
            ValueError: If ``frame_numbers`` length does not match ``frames``.
        """
        if not self._loaded:
            raise DetectorError("CombinedDetector not loaded; call load() or use a with block")
        if not frames:
            return []

        if frame_numbers is None:
            frame_numbers = list(range(len(frames)))
        elif len(frame_numbers) != len(frames):
            raise ValueError(
                f"frame_numbers has {len(frame_numbers)} entries "
                f"for {len(frames)} frame(s)"
            )

        empty: List[List[Any]] = [[] for _ in frames]
        bodies = self.body.detect_batch(frames) if self.body is not None else empty
        faces = self.face.detect_batch(frames) if self.face is not None else empty

        return [
            FrameDetections(frame_number=number, bodies=list(body_list), faces=list(face_list))
            for number, body_list, face_list in zip(frame_numbers, bodies, faces)
        ]

    # -- diagnostics ------------------------------------------------------- #
    def stats_summary(self) -> Dict[str, Any]:
        """Collect per-detector statistics.

        Returns:
            A mapping with a ``body`` and/or ``face`` entry, plus the combined
            end-to-end frame rate.  The combined figure is the reciprocal of
            the summed per-frame latencies, not the minimum of the two rates:
            the detectors run in sequence, so their costs add.
        """
        summary: Dict[str, Any] = {}
        total_ms = 0.0

        for key, detector in (("body", self.body), ("face", self.face)):
            if detector is None:
                continue
            report = detector.stats.summary()
            summary[key] = report
            if report.get("frames"):
                total_ms += report["total_seconds"] * 1000.0 / report["frames"]

        summary["combined_fps"] = 1000.0 / total_ms if total_ms > 0 else 0.0
        return summary

    def reset_stats(self) -> None:
        """Clear accumulated statistics on every detector."""
        for detector in (self.body, self.face):
            if detector is not None:
                detector.stats.reset()