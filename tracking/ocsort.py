"""OC-SORT: observation-centric multi-object tracking, implemented from scratch.

What this adds over SORT
------------------------
SORT associates detections to Kalman predictions by IoU and nothing else.  That
works while objects are visible and fails in the two situations CCTV is made of:
occlusion, and people passing close to one another.

Three mechanisms address that, and all three share a premise -- a *detection* is
evidence, a *prediction* is a guess, so when they disagree the detection wins.

Observation-centric momentum (OCM)
    Adds a direction-consistency term to the association cost.  A track moving
    left should prefer a detection to its left even when a stationary
    neighbour's box overlaps more.  Direction is measured between two real
    observations ``delta_t`` apart, never from the filter's velocity state,
    which is contaminated by its own extrapolation during a gap.

Observation-centric recovery (OCR)
    A third association pass that matches leftover tracks against their **last
    real observation** rather than their current prediction.  A track occluded
    for twenty frames has a prediction that has drifted; the place it was last
    actually seen is often the better anchor.

BYTE two-stage association
    Detections below the tracking threshold are usually not noise -- they are
    real objects the detector became unsure about, which is exactly what
    happens under partial occlusion.  Discarding them ends tracks precisely
    when continuity matters most.  They are used in a second pass, allowed to
    sustain existing tracks, and never allowed to create new ones.

Every cost matrix is built with broadcasting.  A crowded frame can hold 60
tracks and 80 detections; a nested Python loop over that pairing is what makes
naive trackers slower than the detector feeding them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from configs.config import TrackingConfig
from tracking.track import Track
from utils.log import get_logger
from utils.types import BoundingBox, Detection, TrackState, boxes_to_array, pairwise_iou

__all__ = ["OCSort", "direction_consistency"]

LOGGER = get_logger(__name__)

#: Guards the division when two boxes share a centre exactly.
_EPS = 1e-6


def direction_consistency(
    reference_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Score how well each detection agrees with each track's heading.

    For every (track, detection) pair this measures the angle between the
    track's established direction of travel and the direction from the track's
    last observation to that detection, then maps it to ``[0, 1]``.

    Args:
        reference_boxes: Track anchor boxes, shape ``(N, 4)``.  These are last
            *observations*, not predictions.
        detection_boxes: Candidate detections, shape ``(M, 4)``.
        directions: Unit heading per track, shape ``(N, 2)`` as ``(dx, dy)``.
            A zero row means the heading is unknown.

    Returns:
        A ``float32`` array of shape ``(N, M)``.  ``1`` means the detection
        lies exactly along the track's heading, ``0`` means opposite.  Tracks
        with an unknown heading score a flat ``0.5``, so the term neither
        rewards nor penalises them.
    """
    n, m = reference_boxes.shape[0], detection_boxes.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    ref_cx = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2.0
    ref_cy = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2.0
    det_cx = (detection_boxes[:, 0] + detection_boxes[:, 2]) / 2.0
    det_cy = (detection_boxes[:, 1] + detection_boxes[:, 3]) / 2.0

    delta_x = det_cx[None, :] - ref_cx[:, None]
    delta_y = det_cy[None, :] - ref_cy[:, None]
    norm = np.sqrt(delta_x**2 + delta_y**2) + _EPS
    delta_x /= norm
    delta_y /= norm

    # Cosine between the track heading and the track-to-detection direction.
    cosine = directions[:, 0:1] * delta_x + directions[:, 1:2] * delta_y
    consistency = (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0

    # A track with no established heading must not bias the assignment either
    # way; a flat 0.5 leaves IoU to decide.
    unknown = np.linalg.norm(directions, axis=1) < _EPS
    consistency[unknown, :] = 0.5
    return consistency.astype(np.float32)


def _solve(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Run Hungarian assignment on a cost matrix.

    Args:
        cost: Cost matrix of shape ``(N, M)``; lower is better.

    Returns:
        Matched ``(row, column)`` index pairs.
    """
    if cost.size == 0:
        return []
    rows, columns = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), columns.tolist()))


class OCSort:
    """Observation-centric SORT tracker.

    Args:
        config: Tracker hyper-parameters.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._tracks: List[Track] = []
        self._frame_number = -1
        self._total_created = 0

    # -- properties -------------------------------------------------------- #
    @property
    def tracks(self) -> List[Track]:
        """All live tracks, confirmed or not."""
        return list(self._tracks)

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Tracks that have passed ``min_hits`` and may be published."""
        return [track for track in self._tracks if track.is_confirmed]

    @property
    def total_created(self) -> int:
        """Number of tracks created over the tracker's lifetime."""
        return self._total_created

    def reset(self) -> None:
        """Drop all tracks and counters."""
        self._tracks.clear()
        self._frame_number = -1
        self._total_created = 0

    # -- main loop --------------------------------------------------------- #
    def update(self, detections: Sequence[Detection], frame_number: int) -> List[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: Body detections for this frame, at any confidence.
            frame_number: Emission index of the source frame.  Used for track
                bookkeeping and for the observation history, so it must be the
                real frame number rather than a loop counter.

        Returns:
            Confirmed tracks after the update, in creation order.
        """
        self._frame_number = frame_number

        for track in self._tracks:
            track.predict()

        high, low = self._split_by_confidence(detections)
        unmatched_tracks = list(range(len(self._tracks)))

        # Stage 1 -- IoU plus momentum, high-confidence detections.
        matches, unmatched_tracks, unmatched_high = self._associate_primary(
            unmatched_tracks, high
        )
        for track_index, detection_index in matches:
            detection = high[detection_index]
            self._tracks[track_index].update(detection.box, frame_number, detection.score)

        # Stage 2 -- BYTE. Low-confidence detections may sustain a track but
        # never create one.
        if self._config.use_byte and low:
            byte_matches, unmatched_tracks, _ = self._associate_iou(
                unmatched_tracks, low, self._config.iou_threshold
            )
            for track_index, detection_index in byte_matches:
                detection = low[detection_index]
                self._tracks[track_index].update(
                    detection.box, frame_number, detection.score
                )

        # Stage 3 -- OCR. Match against last observations rather than
        # predictions, which is what recovers a track after a long occlusion.
        if unmatched_tracks and unmatched_high:
            recovered, unmatched_tracks, unmatched_high = self._associate_recovery(
                unmatched_tracks, high, unmatched_high
            )
            for track_index, detection_index in recovered:
                detection = high[detection_index]
                self._tracks[track_index].re_update(
                    detection.box, frame_number, detection.score
                )

        for track_index in unmatched_tracks:
            self._tracks[track_index].mark_missed()

        for detection_index in unmatched_high:
            self._create_track(high[detection_index], frame_number)

        removed = [t for t in self._tracks if t.state is TrackState.REMOVED]
        if removed:
            LOGGER.debug("Frame %d: removing %d track(s)", frame_number, len(removed))
        self._tracks = [t for t in self._tracks if t.state is not TrackState.REMOVED]

        return self.confirmed_tracks

    # -- association stages ------------------------------------------------ #
    def _split_by_confidence(
        self, detections: Sequence[Detection]
    ) -> Tuple[List[Detection], List[Detection]]:
        """Partition detections into the high and low confidence bands.

        Args:
            detections: All detections for the frame.

        Returns:
            ``(high, low)``.  Detections below ``low_threshold`` are dropped
            entirely; they are noise rather than uncertain objects.
        """
        high = [d for d in detections if d.score >= self._config.det_threshold]
        low = [
            d
            for d in detections
            if self._config.low_threshold <= d.score < self._config.det_threshold
        ]
        return high, low

    def _associate_primary(
        self, track_indices: List[int], detections: Sequence[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate on IoU blended with direction consistency.

        Args:
            track_indices: Indices into ``self._tracks`` still unmatched.
            detections: High-confidence detections.

        Returns:
            ``(matches, unmatched_tracks, unmatched_detections)``.
        """
        if not track_indices or not detections:
            return [], list(track_indices), list(range(len(detections)))

        predicted = boxes_to_array([self._tracks[i].box for i in track_indices])
        anchors = boxes_to_array(
            [self._tracks[i].last_observation for i in track_indices]
        )
        detection_boxes = boxes_to_array([d.box for d in detections])
        directions = np.asarray(
            [self._tracks[i].velocity_direction for i in track_indices], dtype=np.float64
        )

        iou = pairwise_iou(predicted, detection_boxes)
        consistency = direction_consistency(anchors, detection_boxes, directions)

        # Momentum modulates IoU rather than replacing it: geometry decides,
        # heading breaks ties. Weighting it too heavily would let a fast track
        # steal a detection it does not overlap at all.
        affinity = iou + self._config.inertia * consistency

        # Hard-gate on raw IoU before solving. Without this the Hungarian
        # solver, which must produce a full assignment, will pair a track with
        # a detection it does not touch simply because nothing better remains.
        gate = iou < self._config.iou_threshold
        affinity[gate] = -1.0

        matches: List[Tuple[int, int]] = []
        matched_tracks: set = set()
        matched_detections: set = set()

        for row, column in _solve(-affinity):
            if affinity[row, column] <= 0.0:
                continue
            matches.append((track_indices[row], column))
            matched_tracks.add(row)
            matched_detections.add(column)

        unmatched_tracks = [
            track_indices[i] for i in range(len(track_indices)) if i not in matched_tracks
        ]
        unmatched_detections = [
            j for j in range(len(detections)) if j not in matched_detections
        ]
        return matches, unmatched_tracks, unmatched_detections

    def _associate_iou(
        self,
        track_indices: List[int],
        detections: Sequence[Detection],
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate on IoU alone.

        Args:
            track_indices: Indices into ``self._tracks`` still unmatched.
            detections: Candidate detections.
            threshold: Minimum IoU for a pair to be assignable.

        Returns:
            ``(matches, unmatched_tracks, unmatched_detections)``.
        """
        if not track_indices or not detections:
            return [], list(track_indices), list(range(len(detections)))

        predicted = boxes_to_array([self._tracks[i].box for i in track_indices])
        detection_boxes = boxes_to_array([d.box for d in detections])
        iou = pairwise_iou(predicted, detection_boxes)

        matches: List[Tuple[int, int]] = []
        matched_tracks: set = set()
        matched_detections: set = set()

        for row, column in _solve(-iou):
            if iou[row, column] < threshold:
                continue
            matches.append((track_indices[row], column))
            matched_tracks.add(row)
            matched_detections.add(column)

        unmatched_tracks = [
            track_indices[i] for i in range(len(track_indices)) if i not in matched_tracks
        ]
        unmatched_detections = [
            j for j in range(len(detections)) if j not in matched_detections
        ]
        return matches, unmatched_tracks, unmatched_detections

    def _associate_recovery(
        self,
        track_indices: List[int],
        detections: Sequence[Detection],
        candidate_indices: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match leftover tracks against their last observation.

        The prediction of a long-occluded track has drifted; where it was last
        actually seen has not.  This pass uses that as the anchor, with a
        relaxed threshold because the object has genuinely moved in the interim.

        Args:
            track_indices: Track indices still unmatched.
            detections: The full high-confidence detection list.
            candidate_indices: Indices within ``detections`` still unmatched.

        Returns:
            ``(matches, unmatched_tracks, unmatched_detections)``.
        """
        if not track_indices or not candidate_indices:
            return [], list(track_indices), list(candidate_indices)

        anchors = boxes_to_array(
            [self._tracks[i].last_observation for i in track_indices]
        )
        detection_boxes = boxes_to_array([detections[j].box for j in candidate_indices])
        iou = pairwise_iou(anchors, detection_boxes)

        threshold = self._config.iou_threshold * 0.5
        matches: List[Tuple[int, int]] = []
        matched_tracks: set = set()
        matched_detections: set = set()

        for row, column in _solve(-iou):
            if iou[row, column] < threshold:
                continue
            matches.append((track_indices[row], candidate_indices[column]))
            matched_tracks.add(row)
            matched_detections.add(column)

        unmatched_tracks = [
            track_indices[i] for i in range(len(track_indices)) if i not in matched_tracks
        ]
        unmatched_detections = [
            candidate_indices[j]
            for j in range(len(candidate_indices))
            if j not in matched_detections
        ]
        return matches, unmatched_tracks, unmatched_detections

    # -- track creation ---------------------------------------------------- #
    def _create_track(self, detection: Detection, frame_number: int) -> Track:
        """Start a new track from an unmatched high-confidence detection.

        Args:
            detection: The originating detection.
            frame_number: Frame the detection came from.

        Returns:
            The newly created track.
        """
        track = Track(
            box=detection.box,
            frame_number=frame_number,
            score=detection.score,
            max_trajectory=self._config.max_trajectory,
            delta_t=self._config.delta_t,
            min_hits=self._config.min_hits,
            max_age=self._config.max_age,
        )
        self._tracks.append(track)
        self._total_created += 1
        return track

    # -- diagnostics ------------------------------------------------------- #
    def summary(self) -> Dict[str, object]:
        """Summarise the tracker's current population.

        Returns:
            Counts by lifecycle state plus totals.
        """
        by_state: Dict[str, int] = {}
        for track in self._tracks:
            by_state[track.state.value] = by_state.get(track.state.value, 0) + 1
        return {
            "frame": self._frame_number,
            "live": len(self._tracks),
            "confirmed": len(self.confirmed_tracks),
            "created_total": self._total_created,
            "by_state": by_state,
        }