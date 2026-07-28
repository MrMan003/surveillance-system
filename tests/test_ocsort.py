"""Tests for the OC-SORT tracker.

Scenarios are synthetic and exact, so a failure points at the tracker rather
than at detector noise.  The hard cases -- crossing paths and long occlusions --
are the ones SORT gets wrong, and they are the reason this module exists.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from configs import SurveillanceConfig
from tracking.ocsort import OCSort, direction_consistency
from utils.types import BoundingBox, Detection, boxes_to_array


def make_detection(
    x: float, y: float, score: float = 0.9, width: float = 50, height: float = 100
) -> Detection:
    """Build a detection from a top-left corner."""
    return Detection(box=BoundingBox(x, y, x + width, y + height), score=score)


@pytest.fixture()
def tracker() -> OCSort:
    """A tracker on stock hyper-parameters."""
    return OCSort(SurveillanceConfig.default().tracking)


# --------------------------------------------------------------------------- #
# Direction consistency
# --------------------------------------------------------------------------- #
def test_direction_consistency_scores_heading() -> None:
    """A detection ahead must score 1, behind must score 0."""
    reference = boxes_to_array([BoundingBox(100, 100, 150, 200)])
    detections = boxes_to_array(
        [BoundingBox(200, 100, 250, 200), BoundingBox(0, 100, 50, 200)]
    )
    scores = direction_consistency(reference, detections, np.array([[1.0, 0.0]]))
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert scores[0, 1] == pytest.approx(0.0, abs=1e-3)


def test_unknown_heading_is_neutral() -> None:
    """A track with no established heading must not bias the assignment."""
    reference = boxes_to_array([BoundingBox(100, 100, 150, 200)])
    detections = boxes_to_array(
        [BoundingBox(200, 100, 250, 200), BoundingBox(0, 100, 50, 200)]
    )
    scores = direction_consistency(reference, detections, np.array([[0.0, 0.0]]))
    assert scores == pytest.approx(0.5)


def test_direction_consistency_handles_empty() -> None:
    """Frames with no tracks or no detections must not raise."""
    empty = np.zeros((0, 4), dtype=np.float32)
    boxes = boxes_to_array([BoundingBox(0, 0, 10, 10)])
    assert direction_consistency(empty, boxes, np.zeros((0, 2))).shape == (0, 1)
    assert direction_consistency(boxes, empty, np.zeros((1, 2))).shape == (1, 0)


# --------------------------------------------------------------------------- #
# Basic tracking
# --------------------------------------------------------------------------- #
def test_empty_input_creates_nothing(tracker: OCSort) -> None:
    """A frame with no detections must be a no-op."""
    assert tracker.update([], 0) == []
    assert tracker.total_created == 0


def test_track_is_created_and_confirmed(tracker: OCSort) -> None:
    """A consistent object must produce exactly one confirmed track."""
    for frame in range(12):
        tracker.update([make_detection(100 + 10 * frame, 100)], frame)
    assert tracker.total_created == 1
    assert len(tracker.confirmed_tracks) == 1


def test_confirmation_requires_min_hits(tracker: OCSort) -> None:
    """A track must not be published before min_hits observations."""
    tracker.update([make_detection(100, 100)], 0)
    assert tracker.confirmed_tracks == []


def test_velocity_is_recovered(tracker: OCSort) -> None:
    """The tracker's filter must learn the true motion."""
    for frame in range(15):
        tracker.update([make_detection(100 + 10 * frame, 100)], frame)
    assert tracker.confirmed_tracks[0].velocity == pytest.approx([10.0, 0.0], abs=0.5)


def test_track_id_is_stable(tracker: OCSort) -> None:
    """Identity must persist across frames for an unambiguous object."""
    ids = set()
    for frame in range(20):
        for track in tracker.update([make_detection(100 + 5 * frame, 100)], frame):
            ids.add(track.track_id)
    assert len(ids) == 1


def test_reset_clears_state(tracker: OCSort) -> None:
    """Reset must return the tracker to its initial condition."""
    for frame in range(10):
        tracker.update([make_detection(100 + 10 * frame, 100)], frame)
    tracker.reset()
    assert tracker.tracks == []
    assert tracker.total_created == 0


# --------------------------------------------------------------------------- #
# The hard cases
# --------------------------------------------------------------------------- #
def test_crossing_paths_do_not_switch_identity(tracker: OCSort) -> None:
    """Two objects passing through each other must keep their identities.

    This is the canonical SORT failure. At the crossing point both predictions
    overlap both detections, so IoU alone is ambiguous and the assignment can
    swap. The momentum term resolves it: each track prefers the detection that
    continues its established heading.
    """
    trajectories: Dict[str, List[float]] = {}
    for frame in range(30):
        detections = [
            make_detection(100 + 8 * frame, 100),
            make_detection(340 - 8 * frame, 100),
        ]
        for track in tracker.update(detections, frame):
            trajectories.setdefault(track.track_id, []).append(track.box.centre[0])

    assert len(trajectories) == 2, "identity switch or spurious track"
    movements = [path[-1] - path[0] for path in trajectories.values()]
    assert max(movements) > 100, "expected one track moving right"
    assert min(movements) < -100, "expected one track moving left"


def test_identity_survives_occlusion(tracker: OCSort) -> None:
    """A track must be recovered after a gap, not replaced."""
    for frame in range(8):
        tracker.update([make_detection(100 + 10 * frame, 100)], frame)
    original = tracker.confirmed_tracks[0].track_id

    for frame in range(8, 18):
        tracker.update([], frame)
    assert tracker.confirmed_tracks == [], "a fully occluded track must not stay confirmed"

    for frame in range(18, 24):
        tracker.update([make_detection(100 + 10 * frame, 100)], frame)

    assert original in {t.track_id for t in tracker.confirmed_tracks}
    assert tracker.total_created == 1, "recovery must not create a second track"


def test_track_is_deleted_after_max_age(tracker: OCSort) -> None:
    """A track absent beyond max_age must be removed, not held forever."""
    for frame in range(10):
        tracker.update([make_detection(100 + 10 * frame, 100)], frame)
    max_age = SurveillanceConfig.default().tracking.max_age
    for frame in range(10, 10 + max_age + 5):
        tracker.update([], frame)
    assert tracker.tracks == []


def test_distant_detection_starts_a_new_track(tracker: OCSort) -> None:
    """A detection with no plausible overlap must not be forced onto a track.

    Hungarian assignment produces a complete matching, so without an IoU gate
    it will pair a track with any leftover detection regardless of distance.
    """
    for frame in range(6):
        tracker.update([make_detection(100, 100)], frame)
    tracker.update([make_detection(600, 400)], 6)
    assert tracker.total_created == 2


# --------------------------------------------------------------------------- #
# BYTE second pass
# --------------------------------------------------------------------------- #
def test_low_confidence_sustains_an_existing_track(tracker: OCSort) -> None:
    """Detections that dip below threshold must not end a track.

    Confidence usually drops because of partial occlusion, which is exactly
    when continuity matters most.
    """
    for frame in range(6):
        tracker.update([make_detection(100 + 10 * frame, 100, score=0.9)], frame)
    original = tracker.confirmed_tracks[0].track_id

    for frame in range(6, 14):
        tracker.update([make_detection(100 + 10 * frame, 100, score=0.20)], frame)

    assert original in {t.track_id for t in tracker.confirmed_tracks}
    assert tracker.total_created == 1


def test_low_confidence_cannot_create_a_track(tracker: OCSort) -> None:
    """The BYTE band may sustain tracks but never spawn them."""
    for frame in range(10):
        tracker.update([make_detection(100 + 10 * frame, 100, score=0.20)], frame)
    assert tracker.total_created == 0


def test_byte_can_be_disabled() -> None:
    """With BYTE off, a track must not survive on low-confidence detections."""
    config = SurveillanceConfig.from_dict({"tracking": {"use_byte": False}}).tracking
    tracker = OCSort(config)
    for frame in range(6):
        tracker.update([make_detection(100 + 10 * frame, 100, score=0.9)], frame)
    for frame in range(6, 6 + config.max_age + 5):
        tracker.update([make_detection(100 + 10 * frame, 100, score=0.20)], frame)
    assert tracker.confirmed_tracks == []


def test_sub_threshold_noise_is_discarded(tracker: OCSort) -> None:
    """Detections below the BYTE floor are noise and must be ignored."""
    rng = np.random.default_rng(0)
    for frame in range(20):
        detection = make_detection(
            float(rng.integers(0, 600)), float(rng.integers(0, 400)), score=0.05
        )
        tracker.update([detection], frame)
    assert tracker.total_created == 0


# --------------------------------------------------------------------------- #
# Crowds
# --------------------------------------------------------------------------- #
def test_many_objects_are_tracked_independently(tracker: OCSort) -> None:
    """A crowded scene must produce one track per object."""
    count = 12
    for frame in range(15):
        detections = [
            make_detection(60 * index, 100 + 4 * frame, width=40, height=80)
            for index in range(count)
        ]
        tracker.update(detections, frame)
    assert len(tracker.confirmed_tracks) == count
    assert tracker.total_created == count


def test_summary_reports_population(tracker: OCSort) -> None:
    """The diagnostic summary must reflect live and confirmed counts."""
    for frame in range(10):
        tracker.update(
            [make_detection(100 + 10 * frame, 100), make_detection(400, 200)], frame
        )
    summary = tracker.summary()
    assert summary["confirmed"] == 2
    assert summary["created_total"] == 2
    assert summary["frame"] == 9


def test_frame_numbers_are_recorded(tracker: OCSort) -> None:
    """Bookkeeping must use the supplied frame numbers, not a loop counter.

    Frame numbers arrive from the decoder and may be strided or offset.
    """
    for frame in range(100, 115):
        tracker.update([make_detection(100 + 10 * (frame - 100), 100)], frame)
    track = tracker.confirmed_tracks[0]
    assert track.birth_frame == 100
    assert track.last_seen_frame == 114