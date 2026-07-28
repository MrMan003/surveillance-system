"""Face alignment by Umeyama similarity transform.

The problem
-----------
A recognition model is trained on faces in a canonical pose: eyes on a fixed
horizontal line, nose at a fixed height, face filling a fixed fraction of the
frame.  A raw detector crop satisfies none of that.  The same person at two
head angles produces two crops that differ more, pixel-wise, than crops of two
different people at the same angle.  Alignment removes that variation so the
embedding encodes identity rather than pose.

Why a similarity transform
--------------------------
A *similarity* transform is rotation, uniform scale and translation -- four
degrees of freedom.  That is deliberately restrictive.  An affine transform
(six degrees of freedom) would additionally shear and scale each axis
independently, and with only five noisy landmarks it will happily do so: it
fits the landmark noise by distorting the face, and a face stretched to make
its landmarks match the template is no longer the face the model was trained
on.  A homography is worse still.

Four degrees of freedom is exactly what is recoverable from five landmarks
without overfitting them.

Umeyama's method
----------------
Umeyama (1991) gives the closed-form least-squares similarity transform between
two point sets, including the reflection correction that a naive SVD solution
gets wrong.  Without that correction, a face whose landmarks are noisy enough
to make the covariance determinant negative aligns to a *mirrored* template --
producing a crop that looks almost right and embeds as a different person.

This module implements the algorithm rather than calling
``cv2.estimateAffinePartial2D``, which is RANSAC-based, non-deterministic
across runs, and overkill for five inlier points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from configs.config import AlignmentConfig
from utils.log import get_logger
from utils.types import FaceDetection

__all__ = ["AlignmentError", "AlignedFace", "umeyama", "FaceAligner"]

LOGGER = get_logger(__name__)

#: Below this the point set is effectively collinear or coincident and no
#: meaningful scale can be recovered.
_MIN_VARIANCE = 1e-8


class AlignmentError(RuntimeError):
    """Raised when a face cannot be aligned."""


@dataclass(frozen=True, slots=True)
class AlignedFace:
    """A face warped to the canonical template.

    Attributes:
        image: Aligned crop, ``uint8``, shaped ``(H, W, 3)``, in the same
            channel order as the source frame.
        matrix: The ``2 x 3`` affine matrix mapping source pixels to crop
            pixels.  Retained so detections can be mapped back onto the frame
            for rendering without re-estimating.
        scale: Uniform scale factor applied.  Below 1 means the source face was
            larger than the template and was downsampled; above 1 means it was
            upsampled, which caps the real information available regardless of
            the crop's nominal resolution.
        rotation_degrees: In-plane rotation removed by the warp.
        residual: Root-mean-square landmark error after alignment, in crop
            pixels.  A large residual means the five points were not
            consistent with any similarity transform, which usually indicates
            a bad detection rather than an unusual pose.
        source_eye_distance: Inter-eye distance in the original frame, in
            pixels.  The most direct measure of how much real detail the face
            had before warping.
    """

    image: np.ndarray
    matrix: np.ndarray
    scale: float
    rotation_degrees: float
    residual: float
    source_eye_distance: float

    @property
    def was_upsampled(self) -> bool:
        """Whether the source face was smaller than the template.

        An upsampled crop has no more information than its source; the extra
        pixels are interpolation.  Phase 7 uses this as a quality signal.
        """
        return self.scale > 1.0

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"AlignedFace({self.image.shape[1]}x{self.image.shape[0]}, "
            f"scale={self.scale:.3f}, roll={self.rotation_degrees:+.1f}deg, "
            f"residual={self.residual:.2f}px)"
        )


def umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Estimate the least-squares similarity transform mapping source to target.

    Implements Umeyama (1991), "Least-squares estimation of transformation
    parameters between two point patterns", including the reflection
    correction of equation 39.

    Args:
        source: Points to be transformed, shape ``(N, 2)``.
        target: Points to align onto, shape ``(N, 2)``.

    Returns:
        A ``3 x 3`` homogeneous matrix ``T`` such that ``T @ [x, y, 1]``
        approximates the corresponding target point.

    Raises:
        AlignmentError: If the shapes disagree, fewer than two points are
            supplied, or the source points are degenerate.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise AlignmentError(
            f"Point sets must match: got {source.shape} and {target.shape}"
        )
    if source.ndim != 2 or source.shape[1] != 2:
        raise AlignmentError(f"Points must have shape (N, 2), got {source.shape}")
    if source.shape[0] < 2:
        raise AlignmentError("At least two point correspondences are required")

    count, dimensions = source.shape

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centred = source - source_mean
    target_centred = target - target_mean

    source_variance = (source_centred**2).sum() / count
    if source_variance < _MIN_VARIANCE:
        raise AlignmentError("Source landmarks are coincident; no transform exists")

    covariance = (target_centred.T @ source_centred) / count
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)

    # Reflection correction. A plain SVD solution can produce a rotation with
    # determinant -1, which mirrors the face. The result looks nearly correct
    # and embeds as a different person, so this is not a cosmetic guard.
    correction = np.ones(dimensions, dtype=np.float64)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        correction[-1] = -1.0

    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        raise AlignmentError("Degenerate landmark configuration; covariance has rank 0")
    if rank == dimensions - 1:
        # Rank-deficient but recoverable: the sign is determined by the
        # determinants of U and V rather than by the covariance itself.
        if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) > 0:
            rotation = u_matrix @ vt_matrix
        else:
            saved = correction[-1]
            correction[-1] = -1.0
            rotation = u_matrix @ np.diag(correction) @ vt_matrix
            correction[-1] = saved
    else:
        rotation = u_matrix @ np.diag(correction) @ vt_matrix

    scale = float((singular_values * correction).sum() / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)

    transform = np.eye(dimensions + 1, dtype=np.float64)
    transform[:dimensions, :dimensions] = scale * rotation
    transform[:dimensions, dimensions] = translation
    return transform


class FaceAligner:
    """Warps detected faces onto the canonical recognition template.

    The aligner is colour-agnostic: it applies a geometric transform and
    returns pixels in whatever channel order it was given.  Colour conversion
    belongs at the model boundary, where the model's own expectation is known,
    not here.

    Args:
        config: Crop geometry, template and resampling settings.
    """

    def __init__(self, config: AlignmentConfig) -> None:
        self._config = config
        self._template = np.asarray(config.scaled_template(), dtype=np.float64)

    @property
    def template(self) -> np.ndarray:
        """The canonical landmark template in crop coordinates, shape ``(5, 2)``."""
        return self._template.copy()

    @property
    def output_size(self) -> Tuple[int, int]:
        """Crop resolution as ``(width, height)``."""
        return self._config.output_size

    # -- single face ------------------------------------------------------- #
    def align(
        self, frame: np.ndarray, face: FaceDetection, strict: bool = False
    ) -> Optional[AlignedFace]:
        """Warp one detected face onto the template.

        Args:
            frame: Source frame, ``uint8``, shaped ``(H, W, 3)``.
            face: Detection carrying the five landmarks.
            strict: Raise instead of returning ``None`` when the face is
                rejected for excessive roll.

        Returns:
            The aligned crop, or ``None`` when the face was rejected and
            ``strict`` is ``False``.

        Raises:
            AlignmentError: If the frame is malformed, the transform cannot be
                estimated, or the face is rejected while ``strict`` is set.
        """
        import cv2

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise AlignmentError(f"Frame must be (H, W, 3), got {frame.shape}")

        roll = face.roll_degrees
        limit = self._config.max_roll_degrees
        if limit is not None and abs(roll) > limit:
            message = f"Face roll {roll:+.1f}deg exceeds the {limit:.0f}deg limit"
            if strict:
                raise AlignmentError(message)
            LOGGER.debug("%s; skipping", message)
            return None

        landmarks = np.asarray(face.landmarks, dtype=np.float64)
        transform = umeyama(landmarks, self._template)
        matrix = transform[:2, :].astype(np.float64)

        width, height = self._config.output_size
        crop = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=self._config.interpolation.to_cv2(),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self._config.border_value,
        )

        # Recover scale and rotation from the estimated matrix rather than
        # tracking them separately, so the reported values always describe the
        # warp that was actually applied.
        linear = matrix[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))

        projected = (landmarks @ linear.T) + matrix[:, 2]
        residual = float(np.sqrt(((projected - self._template) ** 2).sum(axis=1).mean()))

        return AlignedFace(
            image=np.ascontiguousarray(crop),
            matrix=matrix,
            scale=scale,
            rotation_degrees=rotation,
            residual=residual,
            source_eye_distance=face.eye_distance,
        )

    # -- many faces -------------------------------------------------------- #
    def align_batch(
        self, frame: np.ndarray, faces: Sequence[FaceDetection]
    ) -> List[Optional[AlignedFace]]:
        """Align several faces from one frame.

        The warp itself is inherently per-face -- each has its own matrix and
        OpenCV resamples one image at a time -- so this is a loop by necessity
        rather than an unvectorised oversight.  It preserves positional
        correspondence with the input so callers can zip results back to
        detections.

        Args:
            frame: Source frame, ``uint8``, shaped ``(H, W, 3)``.
            faces: Detections to align.

        Returns:
            One entry per input face, ``None`` where the face was rejected.
        """
        return [self.align(frame, face) for face in faces]

    # -- inverse ----------------------------------------------------------- #
    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Invert a ``2 x 3`` affine matrix.

        Lets crop-space coordinates be mapped back onto the source frame, for
        drawing aligned landmarks over the original video.

        Args:
            matrix: The forward ``2 x 3`` affine matrix.

        Returns:
            The inverse ``2 x 3`` affine matrix.

        Raises:
            AlignmentError: If the matrix is singular.
        """
        homogeneous = np.eye(3, dtype=np.float64)
        homogeneous[:2, :] = matrix
        try:
            return np.linalg.inv(homogeneous)[:2, :]
        except np.linalg.LinAlgError as exc:
            raise AlignmentError(f"Affine matrix is singular: {exc}") from exc