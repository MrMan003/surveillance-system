"""Shared value types passed between pipeline stages.

Every stage from detection onward consumes and produces these.  Defining them
once, here, is what lets the tracker accept detections from any detector and
the renderer draw results from any tracker: stages depend on these types, not
on each other.

Design constraints
------------------
* Immutable.  A detection handed to the tracker must not be mutable by it.
  ``frozen=True`` makes accidental coupling a runtime error.
* ``slots=True``.  A busy frame carries dozens of these objects and a long
  recording creates millions; slots remove the per-instance ``__dict__`` and
  cut memory materially.
* Boxes are ``(x1, y1, x2, y2)`` in absolute pixel coordinates, top-left
  origin.  This matches YOLO's and SCRFD's native output, so no conversion
  happens at the boundary where conversion bugs hide.
* Geometry is exposed as vectorised helpers operating on ``(N, 4)`` arrays.
  Per-object Python loops over boxes are the default performance mistake in
  this kind of pipeline; the array functions exist so no stage needs them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BoundingBox",
    "Detection",
    "FaceDetection",
    "TrackState",
    "boxes_to_array",
    "box_areas",
    "box_centres",
    "pairwise_iou",
    "containment_matrix",
    "clip_boxes",
    "scale_boxes",
]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in absolute pixel coordinates.

    Attributes:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        """Reject inverted boxes at construction.

        Raises:
            ValueError: If either extent is non-positive.
        """
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Degenerate box ({self.x1}, {self.y1}, {self.x2}, {self.y2}): "
                "x2 must exceed x1 and y2 must exceed y1"
            )

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def centre(self) -> Tuple[float, float]:
        """Box centre as ``(x, y)``."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.width / self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)``."""
        return self.x1, self.y1, self.x2, self.y2

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        """Return integer corners, for slicing and drawing."""
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def as_array(self) -> np.ndarray:
        """Return the corners as a ``float32`` array of shape ``(4,)``."""
        return np.array(self.as_tuple(), dtype=np.float32)

    def as_xywh(self) -> Tuple[float, float, float, float]:
        """Return ``(x, y, width, height)`` with ``(x, y)`` at the top left."""
        return self.x1, self.y1, self.width, self.height

    def as_cxcywh(self) -> Tuple[float, float, float, float]:
        """Return ``(centre_x, centre_y, width, height)``.

        This is the parameterisation the Kalman filter tracks in Phase 3.
        """
        cx, cy = self.centre
        return cx, cy, self.width, self.height

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> "BoundingBox":
        """Build a box from top-left corner and extents.

        Args:
            x: Left edge.
            y: Top edge.
            width: Box width.
            height: Box height.

        Returns:
            The equivalent corner-form box.
        """
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, width: float, height: float) -> "BoundingBox":
        """Build a box from centre and extents.

        Args:
            cx: Centre x coordinate.
            cy: Centre y coordinate.
            width: Box width.
            height: Box height.

        Returns:
            The equivalent corner-form box.
        """
        return cls(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)

    def clip(self, width: int, height: int) -> "BoundingBox":
        """Clamp the box to a frame, preserving at least one pixel of extent.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            A box wholly inside the frame.
        """
        x1 = min(max(self.x1, 0.0), width - 1.0)
        y1 = min(max(self.y1, 0.0), height - 1.0)
        x2 = min(max(self.x2, x1 + 1.0), float(width))
        y2 = min(max(self.y2, y1 + 1.0), float(height))
        return BoundingBox(x1, y1, x2, y2)

    def scale(self, factor: float) -> "BoundingBox":
        """Grow or shrink the box about its own centre.

        Args:
            factor: Multiplier applied to both extents.

        Returns:
            The rescaled box.

        Raises:
            ValueError: If ``factor`` is not positive.
        """
        if factor <= 0:
            raise ValueError(f"scale factor must be > 0, got {factor}")
        cx, cy = self.centre
        return BoundingBox.from_cxcywh(cx, cy, self.width * factor, self.height * factor)

    def truncation(self, width: int, height: int) -> float:
        """Fraction of the box lying outside the frame.

        Used by the Phase 7 quality gate to discard partial faces, which
        produce embeddings that are numerically valid and semantically noise.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            A value in ``[0, 1]``; ``0`` means fully inside.
        """
        inside_w = max(0.0, min(self.x2, width) - max(self.x1, 0.0))
        inside_h = max(0.0, min(self.y2, height) - max(self.y1, 0.0))
        return 1.0 - (inside_w * inside_h) / self.area

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union with another box.

        Args:
            other: The box to compare against.

        Returns:
            A value in ``[0, 1]``.
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_fraction(self, other: "BoundingBox") -> float:
        """Fraction of ``other`` that lies inside this box.

        Asymmetric by design: this is the face-in-body test used by the Phase 4
        association, where IoU is the wrong measure because a face box is
        tiny relative to a body box even when perfectly contained.

        Args:
            other: The box being tested for containment.

        Returns:
            A value in ``[0, 1]``.
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter / other.area if other.area > 0 else 0.0

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Extract this region from an image, clipped to its bounds.

        Args:
            image: Source array shaped ``(H, W, C)``.

        Returns:
            A view of the cropped region.
        """
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self.clip(width, height).as_int_tuple()
        return image[y1:y2, x1:x2]


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Detection:
    """A single body detection from the Phase 2 detector.

    Attributes:
        box: Location in absolute pixel coordinates.
        score: Detector confidence in ``[0, 1]``.
        class_id: COCO class identifier; ``0`` is ``person``.
        frame_number: Emission index of the source frame.
    """

    box: BoundingBox
    score: float
    class_id: int = 0
    frame_number: int = -1

    def __post_init__(self) -> None:
        """Validate the confidence range.

        Raises:
            ValueError: If ``score`` falls outside ``[0, 1]``.
        """
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """A face detection with SCRFD's five-point landmarks.

    The landmark order is fixed by the ArcFace convention and must not be
    permuted: the Phase 5 Umeyama alignment matches these against a canonical
    template positionally, so a reordering silently produces warped crops
    rather than an error.

    Attributes:
        box: Face location in absolute pixel coordinates.
        score: Detector confidence in ``[0, 1]``.
        landmarks: Array of shape ``(5, 2)`` holding, in order, the left eye,
            right eye, nose tip, left mouth corner and right mouth corner.
        frame_number: Emission index of the source frame.
    """

    box: BoundingBox
    score: float
    landmarks: np.ndarray
    frame_number: int = -1

    #: Index of each landmark within the ``(5, 2)`` array. These are ClassVar,
    #: not fields: as plain annotations @dataclass would turn them into
    #: constructor arguments, and under slots=True the class attribute would
    #: resolve to a slot descriptor rather than the integer, so indexing with
    #: FaceDetection.LEFT_EYE would raise instead of returning 0.
    LEFT_EYE: ClassVar[int] = 0
    RIGHT_EYE: ClassVar[int] = 1
    NOSE: ClassVar[int] = 2
    LEFT_MOUTH: ClassVar[int] = 3
    RIGHT_MOUTH: ClassVar[int] = 4

    def __post_init__(self) -> None:
        """Validate confidence and landmark geometry.

        Raises:
            ValueError: If the score is out of range or the landmark array is
                not shaped ``(5, 2)``.
        """
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.landmarks.shape != (5, 2):
            raise ValueError(
                f"landmarks must have shape (5, 2), got {self.landmarks.shape}"
            )

    @property
    def eye_distance(self) -> float:
        """Pixel distance between the eye landmarks.

        A cheap sharpness and scale proxy: below roughly 20 pixels, recognition
        accuracy degrades sharply regardless of what the detector reports.
        """
        return float(np.linalg.norm(self.landmarks[self.RIGHT_EYE] - self.landmarks[self.LEFT_EYE]))

    @property
    def roll_degrees(self) -> float:
        """In-plane rotation implied by the eye line, in degrees.

        Returns:
            Signed angle; ``0`` means the eyes are level.
        """
        delta = self.landmarks[self.RIGHT_EYE] - self.landmarks[self.LEFT_EYE]
        return float(np.degrees(np.arctan2(delta[1], delta[0])))


class TrackState(str, Enum):
    """Lifecycle state of a track.

    Attributes:
        TENTATIVE: Seen, but not yet confirmed by ``min_hits`` observations.
        CONFIRMED: Actively matched and published downstream.
        LOST: Unmatched recently, still within ``max_age``, position predicted.
        REMOVED: Exceeded ``max_age`` and deleted.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


# --------------------------------------------------------------------------- #
# Vectorised geometry
# --------------------------------------------------------------------------- #
def boxes_to_array(boxes: Sequence[BoundingBox]) -> np.ndarray:
    """Stack boxes into a single array for vectorised work.

    Args:
        boxes: Any sequence of boxes.

    Returns:
        A ``float32`` array of shape ``(N, 4)``; ``(0, 4)`` when empty.
    """
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray([box.as_tuple() for box in boxes], dtype=np.float32)


def box_areas(boxes: np.ndarray) -> np.ndarray:
    """Compute areas for an array of boxes.

    Args:
        boxes: Array of shape ``(N, 4)`` in corner form.

    Returns:
        A ``float32`` array of shape ``(N,)``.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_centres(boxes: np.ndarray) -> np.ndarray:
    """Compute centres for an array of boxes.

    Args:
        boxes: Array of shape ``(N, 4)`` in corner form.

    Returns:
        A ``float32`` array of shape ``(N, 2)``.
    """
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1
    )


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the full IoU matrix between two sets of boxes.

    Broadcasting replaces the nested Python loop that would otherwise dominate
    the tracker's per-frame cost.  This is the primary association cost in
    Phase 3.

    Args:
        a: Array of shape ``(N, 4)``.
        b: Array of shape ``(M, 4)``.

    Returns:
        A ``float32`` array of shape ``(N, M)`` with values in ``[0, 1]``.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_areas(a)[:, None] + box_areas(b)[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0).astype(np.float32)


def containment_matrix(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compute, for every pair, the fraction of an inner box inside an outer one.

    Asymmetric counterpart to :func:`pairwise_iou`, and the right measure for
    face-to-body assignment: a face perfectly inside a body still has a tiny
    IoU with it, so IoU would reject every correct pairing.

    Args:
        outer: Array of shape ``(N, 4)``, typically body boxes.
        inner: Array of shape ``(M, 4)``, typically face boxes.

    Returns:
        A ``float32`` array of shape ``(N, M)`` with values in ``[0, 1]``.
    """
    if outer.size == 0 or inner.size == 0:
        return np.zeros((outer.shape[0], inner.shape[0]), dtype=np.float32)

    x1 = np.maximum(outer[:, None, 0], inner[None, :, 0])
    y1 = np.maximum(outer[:, None, 1], inner[None, :, 1])
    x2 = np.minimum(outer[:, None, 2], inner[None, :, 2])
    y2 = np.minimum(outer[:, None, 3], inner[None, :, 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    inner_areas = box_areas(inner)[None, :]
    return np.where(inner_areas > 0, intersection / inner_areas, 0.0).astype(np.float32)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp an array of boxes to frame bounds.

    Args:
        boxes: Array of shape ``(N, 4)``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A new ``float32`` array of the same shape.
    """
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    clipped[:, 2] = np.clip(clipped[:, 2], 1, width)
    clipped[:, 3] = np.clip(clipped[:, 3], 1, height)
    return clipped.astype(np.float32)


def scale_boxes(
    boxes: np.ndarray, scale_x: float, scale_y: float, pad_x: float = 0.0, pad_y: float = 0.0
) -> np.ndarray:
    """Map boxes from a letterboxed inference resolution back to frame pixels.

    Detectors run on a padded, resized copy of the frame.  Their output is in
    that coordinate system, and forgetting to invert the transform produces
    boxes that are plausibly placed but consistently wrong -- a bug that
    survives visual inspection at a glance.

    Args:
        boxes: Array of shape ``(N, 4)`` in inference coordinates.
        scale_x: Horizontal scale applied during preprocessing.
        scale_y: Vertical scale applied during preprocessing.
        pad_x: Horizontal padding added during preprocessing.
        pad_y: Vertical padding added during preprocessing.

    Returns:
        A new ``float32`` array in original frame coordinates.
    """
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale_x
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale_y
    return out