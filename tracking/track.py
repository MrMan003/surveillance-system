"""Track lifecycle and observation history.

A :class:`Track` owns one Kalman filter plus the bookkeeping OC-SORT needs on
top of plain SORT: a sparse history of *actual observations*, kept separately
from the filter's smoothed estimates.

Why observations are stored separately
--------------------------------------
SORT's failure mode is that during an occlusion the filter keeps predicting.
Those predictions accumulate error, and because each prediction feeds the next,
the error compounds in whatever direction the velocity happened to point when
contact was lost.  When the object reappears, the track has drifted somewhere
plausible-looking and wrong, and it either fails to re-associate or steals a
neighbouring detection.

OC-SORT's answer is to treat detections as the anchor and predictions as
disposable.  That requires keeping raw observations rather than only the
filtered state, which is what ``_observations`` here is for.  Two things read
it: the velocity direction used by the momentum term, and the re-update
performed when a lost track is recovered.
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from tracking.kalman import KalmanBoxFilter
from utils.types import BoundingBox, TrackState

__all__ = ["Track"]

#: Returned by :attr:`Track.velocity_direction` when no direction can be
#: estimated yet. Zero rather than an arbitrary unit vector so that the
#: momentum cost contributes nothing instead of contributing noise.
NO_DIRECTION = np.zeros(2, dtype=np.float64)


class Track:
    """A single tracked object across frames.

    Args:
        box: Detection that created the track.
        frame_number: Frame the track was born on.
        score: Detector confidence of the creating detection.
        max_trajectory: Number of centroids retained for rendering.
        delta_t: Temporal window, in frames, over which the observation-centric
            velocity direction is measured.
        min_hits: Consecutive matches required before the track is confirmed.
        max_age: Frames a track may go unmatched before removal.
    """

    def __init__(
        self,
        box: BoundingBox,
        frame_number: int,
        score: float = 1.0,
        max_trajectory: int = 64,
        delta_t: int = 3,
        min_hits: int = 3,
        max_age: int = 30,
    ) -> None:
        self.track_id: str = uuid.uuid4().hex
        self.birth_frame: int = frame_number
        self.last_seen_frame: int = frame_number

        self.kf = KalmanBoxFilter(box)
        self.state: TrackState = TrackState.TENTATIVE

        self._delta_t = delta_t
        self._min_hits = min_hits
        self._max_age = max_age

        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        self.time_since_update: int = 0

        self.score: float = score
        self._score_history: Deque[float] = deque([score], maxlen=32)

        # Sparse map of frame number to observed box. Only real detections
        # land here; predictions never do.
        self._observations: Dict[int, BoundingBox] = {frame_number: box}
        self.last_observation: BoundingBox = box
        self.last_observation_frame: int = frame_number

        self.trajectory: Deque[Tuple[float, float]] = deque(
            [box.centre], maxlen=max_trajectory
        )

    # -- identity ---------------------------------------------------------- #
    @property
    def short_id(self) -> str:
        """First eight hex characters of the UUID, for display."""
        return self.track_id[:8]

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"Track({self.short_id}, {self.state.value}, hits={self.hits}, "
            f"age={self.age}, tsu={self.time_since_update})"
        )

    # -- geometry ---------------------------------------------------------- #
    @property
    def box(self) -> BoundingBox:
        """Current filtered box estimate."""
        return self.kf.box

    @property
    def centre(self) -> Tuple[float, float]:
        """Current filtered centre."""
        return self.box.centre

    @property
    def velocity(self) -> np.ndarray:
        """Filtered centre velocity in pixels per frame."""
        return self.kf.velocity

    @property
    def velocity_direction(self) -> np.ndarray:
        """Unit direction between two real observations ``delta_t`` apart.

        Deliberately computed from observations rather than from the filter's
        velocity state.  The filter's velocity is contaminated by its own
        predictions during an occlusion; this is not, which is what makes it
        usable as an association cue after one.

        Returns:
            A unit vector, or :data:`NO_DIRECTION` when there is no earlier
            observation to measure against.
        """
        if len(self._observations) < 2:
            return NO_DIRECTION.copy()

        target = self.last_observation_frame - self._delta_t
        previous: Optional[BoundingBox] = None
        for offset in range(self._delta_t):
            candidate = self._observations.get(target + offset)
            if candidate is not None:
                previous = candidate
                break
        if previous is None:
            earliest = min(self._observations)
            if earliest == self.last_observation_frame:
                return NO_DIRECTION.copy()
            previous = self._observations[earliest]

        current_centre = np.asarray(self.last_observation.centre, dtype=np.float64)
        previous_centre = np.asarray(previous.centre, dtype=np.float64)
        delta = current_centre - previous_centre
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1e-6 else NO_DIRECTION.copy()

    # -- lifecycle --------------------------------------------------------- #
    @property
    def is_confirmed(self) -> bool:
        """Whether the track has been confirmed and may be published."""
        return self.state is TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        """Whether the track is still alive, matched or not."""
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)

    @property
    def confidence(self) -> float:
        """Composite tracking confidence in ``[0, 1]``.

        Blends three independent signals so that no single one dominates:
        recent detector scores, how well established the track is, and how
        recently it was actually observed.  A track coasting on predictions
        loses confidence even while its detector scores stay high, because the
        detector has not confirmed it for several frames.
        """
        detector = float(np.mean(self._score_history)) if self._score_history else 0.0
        maturity = min(self.hits / max(self._min_hits, 1), 1.0)
        recency = max(0.0, 1.0 - self.time_since_update / max(self._max_age, 1))
        return float(np.clip(0.5 * detector + 0.2 * maturity + 0.3 * recency, 0.0, 1.0))

    # -- stepping ---------------------------------------------------------- #
    def predict(self) -> BoundingBox:
        """Advance the filter one frame.

        Returns:
            The predicted box for the current frame.
        """
        predicted = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        return predicted

    def update(self, box: BoundingBox, frame_number: int, score: float = 1.0) -> None:
        """Fold a matched detection into the track.

        Args:
            box: The matched detection.
            frame_number: Frame the detection came from.
            score: Detector confidence.
        """
        self.kf.update(box)

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_seen_frame = frame_number

        self.score = score
        self._score_history.append(score)

        self._observations[frame_number] = box
        self.last_observation = box
        self.last_observation_frame = frame_number
        self._prune_observations()

        self.trajectory.append(box.centre)

        if self.state in (TrackState.TENTATIVE, TrackState.LOST) and self.hits >= self._min_hits:
            self.state = TrackState.CONFIRMED
        elif self.state is TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Record that no detection matched this track on the current frame."""
        if self.state is TrackState.TENTATIVE:
            # An unconfirmed track that misses even once is far more likely to
            # be a detector false positive than a real object, so it is dropped
            # immediately rather than allowed to coast for max_age frames.
            self.state = TrackState.REMOVED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.REMOVED
        elif self.state is TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def re_update(self, box: BoundingBox, frame_number: int, score: float = 1.0) -> None:
        """Recover a lost track and re-anchor the filter on the new observation.

        This is OC-SORT's observation-centric re-update.  A track that was lost
        for ``k`` frames has a filter whose velocity was extrapolated across
        that whole gap, so its state is not merely uncertain but biased.  Rather
        than feed a normal update into that biased state, the filter is
        re-seeded from the two real observations bracketing the gap: the one
        before it was lost and the one recovering it.  The velocity implied by
        that pair is the correct average over the occlusion, which a chain of
        predictions cannot recover.

        Args:
            box: The detection recovering the track.
            frame_number: Frame the detection came from.
            score: Detector confidence.
        """
        gap = max(frame_number - self.last_observation_frame, 1)
        previous_centre = np.asarray(self.last_observation.centre, dtype=np.float64)
        current_centre = np.asarray(box.centre, dtype=np.float64)
        implied_velocity = (current_centre - previous_centre) / gap

        self.kf = KalmanBoxFilter(box)
        self.kf.x[4, 0] = implied_velocity[0]
        self.kf.x[5, 0] = implied_velocity[1]

        self.update(box, frame_number, score)

    # -- housekeeping ------------------------------------------------------ #
    def _prune_observations(self) -> None:
        """Discard observations older than the momentum window needs.

        Without this the dictionary grows for the lifetime of the track, which
        on an hour of footage is a slow memory leak per track.
        """
        cutoff = self.last_observation_frame - max(self._delta_t * 4, 16)
        stale = [frame for frame in self._observations if frame < cutoff]
        for frame in stale:
            del self._observations[frame]

    @property
    def observation_count(self) -> int:
        """Number of retained observations."""
        return len(self._observations)

    def trajectory_array(self) -> np.ndarray:
        """Return the retained centroid history.

        Returns:
            A ``float32`` array of shape ``(N, 2)``.
        """
        if not self.trajectory:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(self.trajectory, dtype=np.float32)

    def to_dict(self) -> Dict[str, object]:
        """Serialise the track's public state.

        Returns:
            A JSON-friendly mapping for manifests and the audit log.
        """
        box = self.box
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "birth_frame": self.birth_frame,
            "last_seen_frame": self.last_seen_frame,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 2) for v in box.as_tuple()],
            "velocity": [round(float(v), 4) for v in self.velocity],
        }