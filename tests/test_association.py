"""Tests for geometric face-to-body association."""

from __future__ import annotations

import numpy as np
import pytest

from association.face_body import AssociationResult, FaceBodyAssociator
from configs import SurveillanceConfig
from tracking.track import Track
from utils.types import BoundingBox, FaceDetection


def make_body(x: float, y: float, width: float = 60, height: float = 340) -> BoundingBox:
    """Build a full-body box from its top-left corner."""
    return BoundingBox(x, y, x + width, y + height)


def make_face(centre_x: float, centre_y: float, size: float = 26) -> FaceDetection:
    """Build a face detection centred on a point, with plausible landmarks."""
    landmarks = np.array(
        [
            [centre_x - 6, centre_y - 4],
            [centre_x + 6, centre_y - 4],
            [centre_x, centre_y + 1],
            [centre_x - 5, centre_y + 7],
            [centre_x + 5, centre_y + 7],
        ],
        dtype=np.float32,
    )
    half = size / 2.0
    return FaceDetection(
        box=BoundingBox(centre_x - half, centre_y - half, centre_x + half, centre_y + half),
        score=0.9,
        landmarks=landmarks,
    )


def make_track(box: BoundingBox) -> Track:
    """Build a confirmed-eligible track seeded on a box."""
    return Track(box, frame_number=0, min_hits=1)


@pytest.fixture()
def associator() -> FaceBodyAssociator:
    """An associator on stock weights and gates."""
    return FaceBodyAssociator(SurveillanceConfig.default().association)


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #
def test_no_faces(associator: FaceBodyAssociator) -> None:
    """A frame with no faces must report the track as unmatched."""
    result = associator.associate([make_track(make_body(100, 100))], [])
    assert len(result) == 0
    assert result.unmatched_tracks == [0]


def test_no_tracks(associator: FaceBodyAssociator) -> None:
    """A frame with no tracks must report the face as unmatched."""
    result = associator.associate([], [make_face(130, 130)])
    assert len(result) == 0
    assert result.unmatched_faces == [0]


def test_both_empty(associator: FaceBodyAssociator) -> None:
    """Empty input must produce an empty result, not an error."""
    assert len(associator.associate([], [])) == 0


# --------------------------------------------------------------------------- #
# Correct pairings
# --------------------------------------------------------------------------- #
def test_single_face_on_single_body(associator: FaceBodyAssociator) -> None:
    """A face on its own body must associate with near-maximal score."""
    track = make_track(make_body(100, 100))
    result = associator.associate([track], [make_face(130, 130)])
    assert len(result) == 1
    association = result.associations[0]
    assert association.track_id == track.track_id
    assert association.containment == pytest.approx(1.0)
    assert association.score > 0.9


def test_faces_are_matched_regardless_of_input_order(
    associator: FaceBodyAssociator,
) -> None:
    """Assignment must be by geometry, not by list position."""
    left = make_track(make_body(100, 100))
    right = make_track(make_body(400, 120))
    faces = [make_face(430, 150), make_face(130, 130)]

    result = associator.associate([left, right], faces)
    pairs = {a.track_id: a.face_index for a in result.associations}
    assert pairs[left.track_id] == 1
    assert pairs[right.track_id] == 0


def test_each_face_used_at_most_once(associator: FaceBodyAssociator) -> None:
    """Overlapping bodies must not both claim the same face."""
    tracks = [make_track(make_body(100, 100)), make_track(make_body(150, 100))]
    faces = [make_face(133, 130), make_face(178, 130)]
    result = associator.associate(tracks, faces)
    assert len({a.face_index for a in result.associations}) == len(result)
    assert len({a.track_id for a in result.associations}) == len(result)


def test_result_is_indexable_by_track(associator: FaceBodyAssociator) -> None:
    """The result must be addressable by track id for downstream stages."""
    track = make_track(make_body(100, 100))
    result = associator.associate([track], [make_face(130, 130)], frame_number=42)
    assert track.track_id in result.by_track_id()
    assert result.frame_number == 42


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_face_outside_every_body_is_unmatched(associator: FaceBodyAssociator) -> None:
    """A face with no containing body must not be forced onto one.

    Hungarian assignment produces a complete matching, so without a gate it
    would pair this face with whatever track was left over.
    """
    result = associator.associate([make_track(make_body(100, 100))], [make_face(600, 130)])
    assert len(result) == 0
    assert result.unmatched_faces == [0]


def test_face_below_the_head_region_is_rejected(associator: FaceBodyAssociator) -> None:
    """A contained face at knee height belongs to someone else.

    Containment alone accepts this pairing; the head-region gate is what
    rejects it.
    """
    track = make_track(make_body(100, 100))
    knee_height = make_face(130, 380)
    assert track.box.contains_fraction(knee_height.box) == pytest.approx(1.0)
    assert len(associator.associate([track], [knee_height])) == 0


def test_implausibly_large_face_is_rejected(associator: FaceBodyAssociator) -> None:
    """A face far too large for its body is a depth-ordering artefact."""
    small_body = make_track(make_body(500, 200, width=20, height=110))
    huge_face = make_face(510, 215, size=90)
    assert len(associator.associate([small_body], [huge_face])) == 0


def test_partially_contained_face_below_threshold_is_rejected(
    associator: FaceBodyAssociator,
) -> None:
    """A face mostly outside the body must not associate."""
    track = make_track(make_body(100, 100))
    edge_face = make_face(95, 130, size=40)
    containment = track.box.contains_fraction(edge_face.box)
    assert containment < associator._config.containment_threshold  # noqa: SLF001
    assert len(associator.associate([track], [edge_face])) == 0


# --------------------------------------------------------------------------- #
# Cost matrix
# --------------------------------------------------------------------------- #
def test_cost_matrix_shape_and_range(associator: FaceBodyAssociator) -> None:
    """Feasible costs must lie in [0, 1]; infeasible ones above max_cost."""
    bodies = np.array([[100, 100, 160, 440], [400, 120, 460, 460]], dtype=np.float32)
    faces = np.array([[117, 117, 143, 143], [600, 600, 626, 626]], dtype=np.float32)
    cost, containment, feasible = associator.cost_matrix(bodies, faces)

    assert cost.shape == (2, 2)
    assert containment.shape == (2, 2)
    assert feasible.shape == (2, 2)
    assert cost[feasible].max() <= 1.0
    assert cost[~feasible].min() > associator._config.max_cost  # noqa: SLF001


def test_containment_beats_iou_for_nested_boxes() -> None:
    """The measure this module rests on: IoU fails where containment works."""
    body = BoundingBox(100, 100, 160, 440)
    face = BoundingBox(117, 117, 143, 143)
    assert body.contains_fraction(face) == pytest.approx(1.0)
    assert body.iou(face) < 0.05


def test_better_aligned_face_costs_less(associator: FaceBodyAssociator) -> None:
    """A centred face must beat one at the body's edge."""
    bodies = np.array([[100, 100, 200, 440]], dtype=np.float32)
    centred = np.array([[137, 117, 163, 143]], dtype=np.float32)
    offset = np.array([[102, 117, 128, 143]], dtype=np.float32)
    centred_cost, _, _ = associator.cost_matrix(bodies, centred)
    offset_cost, _, _ = associator.cost_matrix(bodies, offset)
    assert centred_cost[0, 0] < offset_cost[0, 0]


def test_cost_matrix_is_finite(associator: FaceBodyAssociator) -> None:
    """Infeasible pairs must be large but finite; the solver rejects inf."""
    bodies = np.array([[0, 0, 10, 20]], dtype=np.float32)
    faces = np.array([[900, 900, 910, 910]], dtype=np.float32)
    cost, _, _ = associator.cost_matrix(bodies, faces)
    assert np.isfinite(cost).all()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_stricter_containment_rejects_more() -> None:
    """Raising the containment gate must not increase the match count."""
    track = make_track(make_body(100, 100))
    face = make_face(112, 130, size=34)

    lenient = FaceBodyAssociator(
        SurveillanceConfig.from_dict(
            {"association": {"containment_threshold": 0.3}}
        ).association
    )
    strict = FaceBodyAssociator(
        SurveillanceConfig.from_dict(
            {"association": {"containment_threshold": 0.99}}
        ).association
    )
    assert len(strict.associate([track], [face])) <= len(lenient.associate([track], [face]))


def test_head_region_ratio_controls_acceptance() -> None:
    """Widening the head region must admit faces lower on the body."""
    track = make_track(make_body(100, 100))
    torso_face = make_face(130, 300)

    default = FaceBodyAssociator(SurveillanceConfig.default().association)
    permissive = FaceBodyAssociator(
        SurveillanceConfig.from_dict(
            {"association": {"head_region_ratio": 0.95}}
        ).association
    )
    assert len(default.associate([track], [torso_face])) == 0
    assert len(permissive.associate([track], [torso_face])) == 1


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
def test_crowded_frame_is_resolved(associator: FaceBodyAssociator) -> None:
    """Ten people, ten faces, one pairing each."""
    tracks = [make_track(make_body(80 * index, 100)) for index in range(10)]
    faces = [make_face(80 * index + 30, 130) for index in range(10)]
    result = associator.associate(tracks, faces)
    assert len(result) == 10
    assert result.unmatched_faces == []


def test_association_is_vectorised(associator: FaceBodyAssociator) -> None:
    """A large frame must resolve in well under a frame budget.

    The guard is loose deliberately -- it catches a regression to a nested
    Python loop over pairs, not small performance drift.
    """
    import time

    rng = np.random.default_rng(0)
    tracks = [
        make_track(make_body(float(rng.uniform(0, 1800)), float(rng.uniform(0, 600))))
        for _ in range(60)
    ]
    faces = [
        make_face(float(rng.uniform(0, 1800)), float(rng.uniform(0, 600)))
        for _ in range(40)
    ]

    start = time.perf_counter()
    for _ in range(50):
        associator.associate(tracks, faces)
    elapsed_ms = (time.perf_counter() - start) * 1000 / 50
    assert elapsed_ms < 20.0