"""Unit tests for shared types, vectorised geometry and logging."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from utils.log import AuditLogger, get_logger, setup_logging
from utils.types import (
    BoundingBox,
    Detection,
    FaceDetection,
    TrackState,
    box_areas,
    box_centres,
    boxes_to_array,
    clip_boxes,
    containment_matrix,
    pairwise_iou,
    scale_boxes,
)


# --------------------------------------------------------------------------- #
# BoundingBox
# --------------------------------------------------------------------------- #
def test_box_geometry() -> None:
    """Derived properties must follow from the corners."""
    box = BoundingBox(10, 20, 110, 220)
    assert box.width == 100
    assert box.height == 200
    assert box.area == 20000
    assert box.centre == (60.0, 120.0)
    assert box.aspect_ratio == pytest.approx(0.5)


def test_degenerate_box_rejected() -> None:
    """Inverted or zero-extent boxes must fail at construction."""
    with pytest.raises(ValueError):
        BoundingBox(10, 10, 5, 20)
    with pytest.raises(ValueError):
        BoundingBox(10, 10, 20, 10)


def test_box_is_immutable() -> None:
    """A box handed to another stage must not be mutable by it."""
    box = BoundingBox(0, 0, 10, 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.x1 = 5  # type: ignore[misc]


@pytest.mark.parametrize("form", ["xywh", "cxcywh"])
def test_box_conversions_round_trip(form: str) -> None:
    """Alternate parameterisations must survive a round trip exactly."""
    box = BoundingBox(10, 20, 110, 220)
    if form == "xywh":
        assert BoundingBox.from_xywh(*box.as_xywh()).as_tuple() == box.as_tuple()
    else:
        assert BoundingBox.from_cxcywh(*box.as_cxcywh()).as_tuple() == box.as_tuple()


def test_scale_preserves_centre() -> None:
    """Scaling must grow the box about its own centre."""
    box = BoundingBox(10, 20, 110, 220)
    scaled = box.scale(0.5)
    assert scaled.centre == box.centre
    assert scaled.width == pytest.approx(box.width * 0.5)


def test_scale_rejects_non_positive() -> None:
    """A non-positive scale factor is meaningless and must raise."""
    with pytest.raises(ValueError):
        BoundingBox(0, 0, 10, 10).scale(0)


def test_clip_keeps_box_inside_frame() -> None:
    """Clipping must leave the box wholly within bounds and non-degenerate."""
    clipped = BoundingBox(-20, -30, 700, 500).clip(640, 480)
    assert clipped.x1 >= 0 and clipped.y1 >= 0
    assert clipped.x2 <= 640 and clipped.y2 <= 480
    assert clipped.area > 0


def test_truncation_measures_overhang() -> None:
    """Truncation must report the fraction of the box outside the frame."""
    assert BoundingBox(0, 0, 100, 100).truncation(640, 480) == pytest.approx(0.0)
    assert BoundingBox(-50, 0, 50, 100).truncation(640, 480) == pytest.approx(0.5)


def test_iou_bounds() -> None:
    """Identical boxes score 1; disjoint boxes score 0."""
    a = BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BoundingBox(100, 100, 110, 110)) == pytest.approx(0.0)


def test_containment_is_asymmetric_and_beats_iou() -> None:
    """A contained face has containment 1 but negligible IoU.

    This asymmetry is the reason Phase 4 uses containment rather than IoU: a
    correctly nested face/body pair would be rejected by any IoU threshold.
    """
    body = BoundingBox(20, 20, 120, 300)
    face = BoundingBox(40, 30, 70, 70)
    assert body.contains_fraction(face) == pytest.approx(1.0)
    assert body.iou(face) < 0.1


def test_crop_matches_clipped_bounds() -> None:
    """Cropping must return the region the box describes, clipped to frame."""
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    crop = BoundingBox(10, 20, 60, 70).crop(image)
    assert crop.shape == (50, 50, 3)
    assert BoundingBox(-10, -10, 20, 20).crop(image).shape[0] > 0


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #
def test_detection_rejects_out_of_range_score() -> None:
    """Confidence outside [0, 1] indicates a preprocessing bug."""
    box = BoundingBox(0, 0, 10, 10)
    with pytest.raises(ValueError):
        Detection(box=box, score=1.5)
    with pytest.raises(ValueError):
        Detection(box=box, score=-0.1)


def test_face_requires_five_landmarks() -> None:
    """SCRFD emits exactly five points; anything else is a shape bug."""
    with pytest.raises(ValueError):
        FaceDetection(
            box=BoundingBox(0, 0, 10, 10), score=0.9, landmarks=np.zeros((3, 2))
        )


def test_eye_distance_and_roll() -> None:
    """Landmark-derived pose measures must match the geometry."""
    level = np.array([[30, 40], [50, 40], [40, 50], [32, 60], [48, 60]], dtype=np.float32)
    face = FaceDetection(box=BoundingBox(20, 25, 60, 70), score=0.9, landmarks=level)
    assert face.eye_distance == pytest.approx(20.0)
    assert face.roll_degrees == pytest.approx(0.0)

    tilted = level.copy()
    tilted[FaceDetection.RIGHT_EYE] = [50, 50]
    rolled = FaceDetection(box=face.box, score=0.9, landmarks=tilted)
    assert rolled.roll_degrees == pytest.approx(26.565, abs=0.01)


def test_landmark_indices_follow_arcface_order() -> None:
    """The template match in Phase 5 is positional; this order is load-bearing."""
    assert (
        FaceDetection.LEFT_EYE,
        FaceDetection.RIGHT_EYE,
        FaceDetection.NOSE,
        FaceDetection.LEFT_MOUTH,
        FaceDetection.RIGHT_MOUTH,
    ) == (0, 1, 2, 3, 4)


def test_track_states() -> None:
    """The lifecycle must expose exactly the four expected states."""
    assert {s.value for s in TrackState} == {"tentative", "confirmed", "lost", "removed"}


# --------------------------------------------------------------------------- #
# Vectorised geometry
# --------------------------------------------------------------------------- #
def test_boxes_to_array_shape() -> None:
    """Stacking must produce (N, 4), and (0, 4) when empty."""
    assert boxes_to_array([]).shape == (0, 4)
    assert boxes_to_array([BoundingBox(0, 0, 1, 1)] * 3).shape == (3, 4)


def test_vectorised_matches_scalar_iou() -> None:
    """The matrix form must agree with the per-pair method."""
    a = [BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)]
    b = [BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)]
    matrix = pairwise_iou(boxes_to_array(a), boxes_to_array(b))
    for i, box_a in enumerate(a):
        for j, box_b in enumerate(b):
            assert matrix[i, j] == pytest.approx(box_a.iou(box_b), abs=1e-6)


def test_vectorised_matches_scalar_containment() -> None:
    """The containment matrix must agree with the per-pair method."""
    bodies = [BoundingBox(20, 20, 120, 300), BoundingBox(200, 20, 300, 300)]
    faces = [BoundingBox(40, 30, 70, 70), BoundingBox(210, 30, 240, 70)]
    matrix = containment_matrix(boxes_to_array(bodies), boxes_to_array(faces))
    for i, body in enumerate(bodies):
        for j, face in enumerate(faces):
            assert matrix[i, j] == pytest.approx(body.contains_fraction(face), abs=1e-6)


@pytest.mark.parametrize("fn", [pairwise_iou, containment_matrix])
def test_empty_inputs_give_empty_matrices(fn) -> None:
    """Frames with no detections must not raise on the association path."""
    boxes = boxes_to_array([BoundingBox(0, 0, 10, 10)])
    empty = np.zeros((0, 4), dtype=np.float32)
    assert fn(empty, boxes).shape == (0, 1)
    assert fn(boxes, empty).shape == (1, 0)


def test_areas_and_centres() -> None:
    """Array helpers must match the scalar properties."""
    boxes = [BoundingBox(0, 0, 10, 20), BoundingBox(5, 5, 15, 15)]
    arr = boxes_to_array(boxes)
    assert box_areas(arr).tolist() == [b.area for b in boxes]
    assert box_centres(arr).tolist() == [list(b.centre) for b in boxes]


def test_clip_boxes_array() -> None:
    """Array clipping must keep every box inside the frame."""
    clipped = clip_boxes(np.array([[-10, -10, 700, 500]], dtype=np.float32), 640, 480)
    assert clipped[0, 0] >= 0 and clipped[0, 2] <= 640
    assert clipped[0, 1] >= 0 and clipped[0, 3] <= 480


def test_scale_boxes_inverts_letterboxing() -> None:
    """Undoing the preprocessing transform must recover frame coordinates."""
    original = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
    scale_x, scale_y, pad_x, pad_y = 0.5, 0.5, 20.0, 10.0
    letterboxed = original.copy()
    letterboxed[:, [0, 2]] = letterboxed[:, [0, 2]] * scale_x + pad_x
    letterboxed[:, [1, 3]] = letterboxed[:, [1, 3]] * scale_y + pad_y
    recovered = scale_boxes(letterboxed, scale_x, scale_y, pad_x, pad_y)
    assert recovered == pytest.approx(original, abs=1e-4)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_setup_logging_writes_a_file(tmp_path: Path) -> None:
    """File logging must produce a readable log."""
    setup_logging(level="INFO", log_dir=tmp_path)
    get_logger("test.logging").info("hello")
    logging.shutdown()
    assert "hello" in (tmp_path / "pipeline.log").read_text(encoding="utf-8")


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls must not duplicate handlers or log lines."""
    setup_logging(level="INFO", log_dir=tmp_path)
    first = len(logging.getLogger().handlers)
    setup_logging(level="INFO", log_dir=tmp_path)
    assert len(logging.getLogger().handlers) == first


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
def test_audit_records_are_valid_jsonl(tmp_path: Path) -> None:
    """Every line must be an independently parseable JSON object."""
    audit = AuditLogger(tmp_path / "audit.jsonl", source="cam-07.mkv")
    audit.run_started({"threshold": 0.35})
    audit.identification("t-001", "alice", 0.71, 0.19, 0.35, 120, 4.033, 12)
    audit.run_finished(frames=300, tracks=1, identifications=1)

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert "timestamp_utc" in record
        assert record["source"] == "cam-07.mkv"


def test_audit_records_rejections(tmp_path: Path) -> None:
    """Non-matches must be logged too.

    A log holding only positive matches cannot answer whether the system
    considered and dismissed someone, which is what an audit asks.
    """
    audit = AuditLogger(tmp_path / "audit.jsonl")
    audit.identification("t-002", "UNKNOWN", 0.31, 0.02, 0.35, 145)
    record = audit.read_all()[0]
    assert record["accepted"] is False
    assert record["identity"] == "UNKNOWN"
    assert record["similarity"] == pytest.approx(0.31)


def test_audit_is_append_only(tmp_path: Path) -> None:
    """A second instance must extend the file, never truncate it."""
    path = tmp_path / "audit.jsonl"
    AuditLogger(path).identification("t-001", "alice", 0.7, 0.2, 0.35, 1)
    AuditLogger(path).identification("t-002", "bob", 0.8, 0.3, 0.35, 2)
    assert len(AuditLogger(path).read_all()) == 2


def test_disabled_audit_writes_nothing(tmp_path: Path) -> None:
    """Disabling the audit must be a no-op, not an error."""
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(path, enabled=False)
    audit.identification("t-001", "alice", 0.7, 0.2, 0.35, 1)
    assert not path.exists()
    assert audit.count == 0


def test_audit_coerces_numpy_and_fractions(tmp_path: Path) -> None:
    """Numpy scalars and Fractions must serialise rather than raise."""
    from fractions import Fraction

    audit = AuditLogger(tmp_path / "audit.jsonl")
    audit.identification(
        "t-001",
        "alice",
        np.float32(0.71),
        np.float64(0.19),
        0.35,
        int(np.int64(120)),
        media_seconds=Fraction(121, 30),
    )
    record = audit.read_all()[0]
    assert isinstance(record["media_seconds"], float)


def test_read_all_on_missing_file(tmp_path: Path) -> None:
    """Reading before anything is written must return an empty list."""
    assert AuditLogger(tmp_path / "nothing.jsonl", enabled=False).read_all() == []