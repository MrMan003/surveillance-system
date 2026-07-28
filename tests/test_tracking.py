"""Tests for the Kalman filter and track lifecycle."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tracking.kalman import KalmanBoxFilter, box_to_measurement, measurement_to_box
from tracking.track import Track
from utils.types import BoundingBox, TrackState


def make_box(x: float, y: float, width: float = 50, height: float = 100) -> BoundingBox:
    """Build a box from its top-left corner and extents."""
    return BoundingBox(x, y, x + width, y + height)


# --------------------------------------------------------------------------- #
# Measurement conversion
# --------------------------------------------------------------------------- #
def test_measurement_round_trip_is_exact() -> None:
    """box -> measurement -> box must recover the original corners."""
    box = BoundingBox(100, 100, 150, 200)
    recovered = measurement_to_box(box_to_measurement(box))
    assert recovered.as_tuple() == pytest.approx(box.as_tuple(), abs=1e-6)


def test_measurement_uses_area_and_ratio() -> None:
    """The parameterisation must be centre, area, aspect ratio."""
    measurement = box_to_measurement(BoundingBox(0, 0, 40, 80)).ravel()
    assert measurement[0] == pytest.approx(20.0)
    assert measurement[1] == pytest.approx(40.0)
    assert measurement[2] == pytest.approx(3200.0)
    assert measurement[3] == pytest.approx(0.5)


def test_non_positive_area_is_clamped() -> None:
    """A degenerate state must still yield a constructible box."""
    box = measurement_to_box(np.array([100.0, 100.0, -5.0, 0.5]))
    assert box.area > 0


# --------------------------------------------------------------------------- #
# Kalman filter
# --------------------------------------------------------------------------- #
def test_filter_learns_constant_velocity() -> None:
    """Velocity is unobservable in one frame and must be inferred over several."""
    kf = KalmanBoxFilter(make_box(100, 100))
    for step in range(1, 15):
        kf.predict()
        kf.update(make_box(100 + 5 * step, 100 + 2 * step))
    assert kf.velocity == pytest.approx([5.0, 2.0], abs=0.05)
    assert kf.speed == pytest.approx(np.hypot(5.0, 2.0), abs=0.05)


def test_prediction_extrapolates_motion() -> None:
    """Blind prediction must follow the learned trajectory."""
    kf = KalmanBoxFilter(make_box(100, 100))
    for step in range(1, 15):
        kf.predict()
        kf.update(make_box(100 + 5 * step, 100 + 2 * step))
    for _ in range(8):
        predicted = kf.predict()
    assert predicted.centre == pytest.approx((235.0, 194.0), abs=1.0)


def test_uncertainty_grows_without_updates() -> None:
    """Positional covariance must grow while a track coasts."""
    kf = KalmanBoxFilter(make_box(100, 100))
    for step in range(1, 10):
        kf.predict()
        kf.update(make_box(100 + 5 * step, 100))
    before = kf.position_uncertainty
    for _ in range(8):
        kf.predict()
    assert kf.position_uncertainty > before


def test_update_none_is_a_no_op() -> None:
    """An unmatched frame must leave the prediction standing."""
    kf = KalmanBoxFilter(make_box(0, 0))
    kf.predict()
    state = kf.x.copy()
    kf.update(None)
    assert np.array_equal(state, kf.x)


def test_shrinking_box_never_goes_degenerate() -> None:
    """A negative area trend must not drive predicted area to zero."""
    kf = KalmanBoxFilter(BoundingBox(0, 0, 200, 400))
    for step in range(6):
        kf.predict()
        kf.update(BoundingBox(0, 0, max(20, 200 - 30 * step), max(40, 400 - 60 * step)))
    areas = [kf.predict().area for _ in range(40)]
    assert all(area > 0 for area in areas)


def test_covariance_stays_symmetric_and_positive_definite() -> None:
    """Joseph-form update must remain numerically stable over a long run.

    The textbook form P = (I - KH)P is algebraically equivalent but loses
    symmetry to rounding over thousands of frames.
    """
    kf = KalmanBoxFilter(make_box(10, 10))
    for step in range(500):
        kf.predict()
        kf.update(make_box(10 + step, 10))
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.linalg.eigvalsh(kf.P).min() > 0


def test_time_since_update_tracks_misses() -> None:
    """The miss counter must increment on predict and reset on update."""
    kf = KalmanBoxFilter(make_box(0, 0))
    for _ in range(3):
        kf.predict()
    assert kf.time_since_update == 3
    kf.update(make_box(5, 0))
    assert kf.time_since_update == 0


# --------------------------------------------------------------------------- #
# Track identity and lifecycle
# --------------------------------------------------------------------------- #
def test_track_ids_are_unique() -> None:
    """Every track must carry a distinct UUID."""
    ids = {Track(make_box(0, 0), 0).track_id for _ in range(2000)}
    assert len(ids) == 2000


def test_track_starts_tentative() -> None:
    """A new track must not be published before confirmation."""
    track = Track(make_box(0, 0), 0, min_hits=3)
    assert track.state is TrackState.TENTATIVE
    assert track.is_confirmed is False


def test_track_confirms_after_min_hits() -> None:
    """Reaching min_hits must promote the track."""
    track = Track(make_box(0, 0), 0, min_hits=3)
    for frame in range(1, 3):
        track.predict()
        track.update(make_box(10 * frame, 0), frame)
    assert track.is_confirmed is True


def test_tentative_track_dies_on_first_miss() -> None:
    """An unconfirmed track is more likely a false positive than an object."""
    track = Track(make_box(0, 0), 0, min_hits=3, max_age=30)
    track.predict()
    track.mark_missed()
    assert track.state is TrackState.REMOVED


def test_confirmed_track_goes_lost_then_removed() -> None:
    """A confirmed track must survive max_age misses before deletion."""
    track = Track(make_box(0, 0), 0, min_hits=2, max_age=3)
    track.predict()
    track.update(make_box(10, 0), 1)
    assert track.is_confirmed

    states = []
    for _ in range(5):
        track.predict()
        track.mark_missed()
        states.append(track.state)
    assert states[0] is TrackState.LOST
    assert states[-1] is TrackState.REMOVED


def test_confidence_decays_while_unmatched() -> None:
    """Confidence must fall as a track coasts on predictions."""
    track = Track(make_box(0, 0), 0, min_hits=2, max_age=10, score=0.9)
    track.predict()
    track.update(make_box(10, 0), 1, score=0.9)
    initial = track.confidence
    for _ in range(4):
        track.predict()
        track.mark_missed()
    assert track.confidence < initial


def test_birth_and_last_seen_frames() -> None:
    """Birth and last-seen bookkeeping must reflect real observations."""
    track = Track(make_box(0, 0), frame_number=17)
    track.predict()
    track.update(make_box(10, 0), frame_number=23)
    assert track.birth_frame == 17
    assert track.last_seen_frame == 23


# --------------------------------------------------------------------------- #
# Observation-centric behaviour
# --------------------------------------------------------------------------- #
def test_velocity_direction_is_a_unit_vector() -> None:
    """The momentum cue must be normalised."""
    track = Track(make_box(100, 100), 0, delta_t=3)
    for frame in range(1, 6):
        track.predict()
        track.update(make_box(100 + 10 * frame, 100), frame)
    direction = track.velocity_direction
    assert direction == pytest.approx([1.0, 0.0], abs=1e-3)
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)


def test_velocity_direction_is_zero_without_history() -> None:
    """With one observation there is no direction; the cue must contribute nothing."""
    track = Track(make_box(0, 0), 0)
    assert np.linalg.norm(track.velocity_direction) == pytest.approx(0.0)


def test_velocity_direction_comes_from_observations_not_predictions() -> None:
    """The cue must survive an occlusion that corrupts the filter's velocity.

    This is the point of storing raw observations separately: after a long gap
    the filter's velocity is biased by its own extrapolation, but the direction
    between two real detections is not.
    """
    track = Track(make_box(100, 100), 0, min_hits=2, max_age=50, delta_t=3)
    for frame in range(1, 6):
        track.predict()
        track.update(make_box(100 + 10 * frame, 100), frame)
    for _ in range(20):
        track.predict()
        track.mark_missed()
    assert track.velocity_direction == pytest.approx([1.0, 0.0], abs=1e-3)


def test_re_update_recovers_velocity_across_a_gap() -> None:
    """Re-anchoring must reset the filter from the two bracketing observations."""
    track = Track(make_box(100, 100), 0, min_hits=2, max_age=50)
    for frame in range(1, 6):
        track.predict()
        track.update(make_box(100 + 8 * frame, 100), frame)
    for _ in range(10):
        track.predict()
        track.mark_missed()

    track.re_update(make_box(100 + 8 * 15, 100), 15)
    assert track.box.centre == pytest.approx((245.0, 150.0), abs=1.0)
    assert track.velocity == pytest.approx([8.0, 0.0], abs=0.5)
    assert track.state is TrackState.CONFIRMED


def test_re_update_on_non_linear_motion() -> None:
    """Recovery must anchor on the observation, not on extrapolated motion.

    A track moving right, then occluded, then reappearing *above* its start
    would be extrapolated far to the right by prediction alone.
    """
    track = Track(make_box(100, 100), 0, min_hits=2, max_age=50)
    for frame in range(1, 6):
        track.predict()
        track.update(make_box(100 + 20 * frame, 100), frame)
    for _ in range(10):
        track.predict()
        track.mark_missed()

    recovery = make_box(200, 20)
    track.re_update(recovery, 15)
    assert track.box.centre == pytest.approx(recovery.centre, abs=1.0)


# --------------------------------------------------------------------------- #
# Memory bounds
# --------------------------------------------------------------------------- #
def test_observations_are_pruned() -> None:
    """Observation history must stay bounded over a long track."""
    track = Track(make_box(0, 0), 0, delta_t=3)
    for frame in range(1, 500):
        track.predict()
        track.update(make_box(frame, 0), frame)
    assert track.observation_count < 50


def test_trajectory_is_capped() -> None:
    """Centroid history must respect max_trajectory."""
    track = Track(make_box(0, 0), 0, max_trajectory=32)
    for frame in range(1, 200):
        track.predict()
        track.update(make_box(frame, 0), frame)
    assert len(track.trajectory) == 32
    assert track.trajectory_array().shape == (32, 2)


def test_empty_trajectory_array_shape() -> None:
    """An empty history must still return a well-shaped array."""
    track = Track(make_box(0, 0), 0, max_trajectory=1)
    track.trajectory.clear()
    assert track.trajectory_array().shape == (0, 2)


def test_to_dict_is_json_serialisable() -> None:
    """Track state must serialise for manifests and the audit log."""
    track = Track(make_box(0, 0), 0)
    track.predict()
    track.update(make_box(10, 0), 1)
    payload = track.to_dict()
    json.dumps(payload)
    assert payload["track_id"] == track.track_id
    assert len(payload["box"]) == 4