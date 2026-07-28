"""Tests for the Umeyama similarity transform and face alignment."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.umeyama import AlignmentError, FaceAligner, umeyama
from configs import SurveillanceConfig
from utils.types import BoundingBox, FaceDetection


def rotation_matrix(degrees: float) -> np.ndarray:
    """Build a 2x2 rotation matrix."""
    angle = np.deg2rad(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )


@pytest.fixture()
def aligner() -> FaceAligner:
    """An aligner on the stock 112x112 ArcFace template."""
    return FaceAligner(SurveillanceConfig.default().alignment)


def synthetic_face(
    aligner: FaceAligner,
    scale: float = 1.6,
    degrees: float = 15.0,
    offset: tuple = (230.0, 170.0),
) -> tuple:
    """Place the template into a frame under a known similarity transform.

    Returns:
        ``(frame, detection, landmarks)`` where the landmarks are the template
        mapped by the known transform, so alignment has an exact ground truth.
    """
    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (360, 340), (90, 120, 160), -1)

    landmarks = (scale * (aligner.template @ rotation_matrix(degrees).T)) + np.asarray(offset)
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)

    box = BoundingBox(
        float(landmarks[:, 0].min() - 20),
        float(landmarks[:, 1].min() - 20),
        float(landmarks[:, 0].max() + 20),
        float(landmarks[:, 1].max() + 20),
    )
    detection = FaceDetection(
        box=box, score=0.95, landmarks=landmarks.astype(np.float32)
    )
    return frame, detection, landmarks


# --------------------------------------------------------------------------- #
# Umeyama
# --------------------------------------------------------------------------- #
def test_recovers_a_known_similarity_transform() -> None:
    """The estimate must reproduce a synthesised transform to machine precision."""
    rng = np.random.default_rng(0)
    source = rng.uniform(0, 100, size=(5, 2))
    scale, degrees, translation = 1.7, 25.0, np.array([13.0, -7.0])
    target = (scale * (source @ rotation_matrix(degrees).T)) + translation

    transform = umeyama(source, target)
    linear = transform[:2, :2]

    assert np.sqrt(abs(np.linalg.det(linear))) == pytest.approx(scale, abs=1e-9)
    assert np.degrees(np.arctan2(linear[1, 0], linear[0, 0])) == pytest.approx(
        degrees, abs=1e-9
    )
    assert transform[:2, 2] == pytest.approx(translation, abs=1e-9)


def test_transform_maps_points_onto_target() -> None:
    """Applying the estimate must land the source points on the target."""
    rng = np.random.default_rng(1)
    source = rng.uniform(0, 100, size=(5, 2))
    target = (2.0 * (source @ rotation_matrix(-40.0).T)) + np.array([5.0, 5.0])

    transform = umeyama(source, target)
    projected = (source @ transform[:2, :2].T) + transform[:2, 2]
    assert projected == pytest.approx(target, abs=1e-9)


def test_identity_when_sets_match() -> None:
    """Aligning a point set to itself must yield the identity."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    assert umeyama(points, points) == pytest.approx(np.eye(3), abs=1e-9)


def test_rotation_has_positive_determinant() -> None:
    """The rotation must never mirror the face.

    A naive SVD solution can return a determinant of -1. The resulting crop
    looks nearly correct and embeds as a different person, so this correction
    is not cosmetic.
    """
    rng = np.random.default_rng(2)
    source = rng.uniform(0, 100, size=(5, 2))
    mirrored = source.copy()
    mirrored[:, 0] *= -1

    transform = umeyama(source, mirrored)
    linear = transform[:2, :2]
    normalised = linear / np.sqrt(abs(np.linalg.det(linear)))
    assert np.linalg.det(normalised) > 0


def test_least_squares_under_noise() -> None:
    """With noisy correspondences the estimate must stay close to the truth."""
    rng = np.random.default_rng(3)
    source = rng.uniform(0, 100, size=(5, 2))
    target = (1.5 * (source @ rotation_matrix(10.0).T)) + np.array([2.0, 3.0])
    noisy = target + rng.normal(0, 0.5, size=target.shape)

    transform = umeyama(source, noisy)
    assert np.sqrt(abs(np.linalg.det(transform[:2, :2]))) == pytest.approx(1.5, abs=0.05)


@pytest.mark.parametrize(
    ("source", "target", "reason"),
    [
        (np.zeros((5, 2)), np.ones((5, 2)), "coincident points"),
        (np.zeros((1, 2)), np.zeros((1, 2)), "too few points"),
        (np.zeros((5, 2)), np.zeros((4, 2)), "shape mismatch"),
        (np.zeros((5, 3)), np.zeros((5, 3)), "wrong dimensionality"),
    ],
)
def test_degenerate_inputs_rejected(source, target, reason) -> None:
    """Degenerate correspondences must raise rather than return nonsense."""
    with pytest.raises(AlignmentError):
        umeyama(source, target)


# --------------------------------------------------------------------------- #
# Aligner
# --------------------------------------------------------------------------- #
def test_template_matches_configuration(aligner: FaceAligner) -> None:
    """The aligner must use the configured ArcFace template."""
    assert aligner.template.shape == (5, 2)
    assert aligner.template[0] == pytest.approx([38.2946, 51.6963], abs=1e-3)
    assert aligner.output_size == (112, 112)


def test_aligned_crop_has_configured_shape(aligner: FaceAligner) -> None:
    """Output geometry must match the recognition model's expected input."""
    frame, detection, _ = synthetic_face(aligner)
    aligned = aligner.align(frame, detection)
    assert aligned is not None
    assert aligned.image.shape == (112, 112, 3)
    assert aligned.image.dtype == np.uint8
    assert aligned.image.flags["C_CONTIGUOUS"]


def test_landmarks_land_on_the_template(aligner: FaceAligner) -> None:
    """The whole point of alignment: warped landmarks must hit the template."""
    frame, detection, _ = synthetic_face(aligner)
    aligned = aligner.align(frame, detection)
    assert aligned is not None
    assert aligned.residual < 1e-3


def test_scale_and_rotation_are_reported(aligner: FaceAligner) -> None:
    """Reported warp parameters must describe the transform actually applied."""
    frame, detection, _ = synthetic_face(aligner, scale=1.6, degrees=15.0)
    aligned = aligner.align(frame, detection)
    assert aligned is not None
    assert aligned.scale == pytest.approx(1.0 / 1.6, abs=1e-3)
    assert aligned.rotation_degrees == pytest.approx(-15.0, abs=1e-3)


def test_upsampling_is_flagged(aligner: FaceAligner) -> None:
    """A face smaller than the template gains no information from warping."""
    frame, small, _ = synthetic_face(aligner, scale=0.4, degrees=0.0, offset=(300, 200))
    aligned = aligner.align(frame, small)
    assert aligned is not None
    assert aligned.was_upsampled is True

    frame, large, _ = synthetic_face(aligner, scale=2.0, degrees=0.0)
    assert aligner.align(frame, large).was_upsampled is False


def test_rotation_is_normalised_away(aligner: FaceAligner) -> None:
    """Two rotations of the same face must produce near-identical crops.

    This is what alignment is for: removing pose variation so the embedding
    encodes identity instead.
    """
    upright_frame, upright, _ = synthetic_face(aligner, degrees=0.0)
    tilted_frame, tilted, _ = synthetic_face(aligner, degrees=25.0)

    first = aligner.align(upright_frame, upright)
    second = aligner.align(tilted_frame, tilted)
    assert first is not None and second is not None
    assert first.residual < 1e-3
    assert second.residual < 1e-3
    assert second.rotation_degrees == pytest.approx(-25.0, abs=1e-3)


def test_excessive_roll_is_rejected(aligner: FaceAligner) -> None:
    """A face rotated past the limit must be skipped, not warped."""
    frame, detection, landmarks = synthetic_face(aligner)
    centre = landmarks.mean(axis=0)
    rolled = ((landmarks - centre) @ rotation_matrix(60.0).T) + centre
    rolled_detection = FaceDetection(
        box=detection.box, score=0.9, landmarks=rolled.astype(np.float32)
    )

    assert abs(rolled_detection.roll_degrees) > 45
    assert aligner.align(frame, rolled_detection) is None
    with pytest.raises(AlignmentError, match="roll"):
        aligner.align(frame, rolled_detection, strict=True)


def test_roll_check_can_be_disabled(aligner: FaceAligner) -> None:
    """With no roll limit, extreme rotations must still align."""
    permissive = FaceAligner(
        SurveillanceConfig.from_dict(
            {"alignment": {"max_roll_degrees": None}}
        ).alignment
    )
    frame, detection, landmarks = synthetic_face(aligner)
    centre = landmarks.mean(axis=0)
    rolled = ((landmarks - centre) @ rotation_matrix(75.0).T) + centre
    rolled_detection = FaceDetection(
        box=detection.box, score=0.9, landmarks=rolled.astype(np.float32)
    )
    assert permissive.align(frame, rolled_detection) is not None


def test_malformed_frame_rejected(aligner: FaceAligner) -> None:
    """A frame without three channels must raise."""
    _, detection, _ = synthetic_face(aligner)
    with pytest.raises(AlignmentError, match="H, W, 3"):
        aligner.align(np.zeros((100, 100), dtype=np.uint8), detection)


# --------------------------------------------------------------------------- #
# Batch and inverse
# --------------------------------------------------------------------------- #
def test_batch_preserves_positional_correspondence(aligner: FaceAligner) -> None:
    """Results must zip back to their input detections, rejections included."""
    frame, good, landmarks = synthetic_face(aligner)
    centre = landmarks.mean(axis=0)
    rolled = ((landmarks - centre) @ rotation_matrix(70.0).T) + centre
    bad = FaceDetection(box=good.box, score=0.9, landmarks=rolled.astype(np.float32))

    results = aligner.align_batch(frame, [good, bad, good])
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


def test_empty_batch(aligner: FaceAligner) -> None:
    """An empty batch must return an empty list."""
    frame, _, _ = synthetic_face(aligner)
    assert aligner.align_batch(frame, []) == []


def test_inverse_maps_crop_coordinates_back(aligner: FaceAligner) -> None:
    """The inverse must recover the original landmark positions."""
    frame, detection, landmarks = synthetic_face(aligner)
    aligned = aligner.align(frame, detection)
    assert aligned is not None

    inverse = FaceAligner.invert(aligned.matrix)
    recovered = (aligner.template @ inverse[:, :2].T) + inverse[:, 2]
    assert recovered == pytest.approx(landmarks, abs=1e-3)


def test_inverse_rejects_singular_matrix() -> None:
    """A singular affine matrix must raise rather than emit garbage."""
    with pytest.raises(AlignmentError, match="singular"):
        FaceAligner.invert(np.zeros((2, 3), dtype=np.float64))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_output_size_is_configurable() -> None:
    """A different crop size must rescale the template consistently."""
    aligner = FaceAligner(
        SurveillanceConfig.from_dict({"alignment": {"output_size": [224, 224]}}).alignment
    )
    frame, detection, _ = synthetic_face(aligner)
    aligned = aligner.align(frame, detection)
    assert aligned is not None
    assert aligned.image.shape == (224, 224, 3)
    assert aligned.residual < 1e-3


def test_padding_shrinks_the_face_within_the_crop() -> None:
    """Padding must pull the template inward, leaving margin around the face."""
    default = FaceAligner(SurveillanceConfig.default().alignment)
    padded = FaceAligner(
        SurveillanceConfig.from_dict({"alignment": {"padding_ratio": 0.3}}).alignment
    )
    default_span = np.ptp(default.template[:, 0])
    padded_span = np.ptp(padded.template[:, 0])
    assert padded_span < default_span