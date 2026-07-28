"""Assign detected faces to tracked bodies by geometry.

The problem
-----------
The two detectors are independent.  YOLO reports people, SCRFD reports faces,
and nothing connects them.  Recognition needs that link: an embedding is only
useful attached to a *track*, because a track is what persists across frames
and accumulates enough observations to identify.

Why not IoU
-----------
The obvious cost is IoU, and it does not work.  A face perfectly nested inside
a body has an IoU around 0.04 -- the face is a few percent of the body's area,
so the union is dominated by body pixels the face does not cover.  Any IoU
threshold that admits correct pairings also admits nonsense.

Containment is the right measure: *what fraction of the face lies inside this
body*.  It is asymmetric, it is 1.0 for a correct pairing regardless of the
size difference, and :func:`utils.types.containment_matrix` computes it for all
pairs at once.

The cost
--------
Containment alone is not sufficient either.  In a crowd, one person standing
behind another can contain a face entirely, and two bodies can both contain the
same face.  Two more terms disambiguate:

Horizontal alignment
    A face belongs near its body's vertical axis.  A face at the far left edge
    of a wide box more likely belongs to a neighbour.

Scale plausibility
    A face occupies a fairly stable fraction of a full-body box.  A face far too
    large or small for the body it sits in is a depth-ordering artefact -- a
    near face over a far body.

All three are weighted, summed, gated, and resolved by a single Hungarian solve
so the assignment is globally optimal rather than greedy.  A greedy pass that
takes the best pair first can strand two other faces on the wrong bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from configs.config import AssociationConfig
from tracking.track import Track
from utils.log import get_logger
from utils.types import BoundingBox, FaceDetection, boxes_to_array, containment_matrix

__all__ = ["FaceBodyAssociation", "AssociationResult", "FaceBodyAssociator"]

LOGGER = get_logger(__name__)

#: Guards divisions by a zero-extent box.
_EPS = 1e-6

#: Scale cost saturates at this many octaves from the expected face/body ratio.
#: Three octaves is a factor of eight, which is well outside anything a correct
#: pairing produces and well inside what a depth-ordering error produces.
_SCALE_OCTAVES = 3.0


@dataclass(frozen=True, slots=True)
class FaceBodyAssociation:
    """One accepted face-to-track pairing.

    Attributes:
        track_id: UUID of the track the face was assigned to.
        track_index: Index of that track in the input sequence.
        face_index: Index of the face in the input sequence.
        face: The face detection, carrying its landmarks.
        body: The track's current box at assignment time.
        score: Association confidence in ``[0, 1]``, defined as ``1 - cost``.
        containment: Fraction of the face box inside the body box.
    """

    track_id: str
    track_index: int
    face_index: int
    face: FaceDetection
    body: BoundingBox
    score: float
    containment: float

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"FaceBodyAssociation({self.track_id[:8]}, face={self.face_index}, "
            f"score={self.score:.3f}, containment={self.containment:.3f})"
        )


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Outcome of associating one frame's faces with its tracks.

    Attributes:
        associations: Accepted pairings.
        unmatched_faces: Indices of faces left unassigned.
        unmatched_tracks: Indices of tracks with no face this frame.
        frame_number: Frame the association was computed for.
    """

    associations: List[FaceBodyAssociation]
    unmatched_faces: List[int]
    unmatched_tracks: List[int]
    frame_number: int = -1

    def __len__(self) -> int:
        """Number of accepted pairings."""
        return len(self.associations)

    def by_track_id(self) -> dict:
        """Index the associations by track UUID.

        Returns:
            A mapping from track id to its association.
        """
        return {a.track_id: a for a in self.associations}


class FaceBodyAssociator:
    """Assigns faces to tracked bodies by geometry alone.

    Deliberately stateless and appearance-free.  Appearance matching would
    need an embedding, and embeddings come from aligned crops, which come from
    faces that have already been associated -- so using appearance here would
    invert the pipeline's dependency order.

    Args:
        config: Cost weights, gates and geometric priors.
    """

    def __init__(self, config: AssociationConfig) -> None:
        self._config = config

    # -- public API -------------------------------------------------------- #
    def associate(
        self,
        tracks: Sequence[Track],
        faces: Sequence[FaceDetection],
        frame_number: int = -1,
    ) -> AssociationResult:
        """Assign each face to at most one track.

        Args:
            tracks: Live tracks for this frame.
            faces: Face detections for this frame.
            frame_number: Frame number recorded on the result.

        Returns:
            Accepted pairings plus whatever was left unmatched.
        """
        if not tracks or not faces:
            return AssociationResult(
                associations=[],
                unmatched_faces=list(range(len(faces))),
                unmatched_tracks=list(range(len(tracks))),
                frame_number=frame_number,
            )

        body_boxes = boxes_to_array([track.box for track in tracks])
        face_boxes = boxes_to_array([face.box for face in faces])

        cost, containment, feasible = self.cost_matrix(body_boxes, face_boxes)

        matched_tracks: set = set()
        matched_faces: set = set()
        associations: List[FaceBodyAssociation] = []

        # Infeasible pairs are pushed above max_cost rather than removed, so
        # the matrix stays rectangular for the solver; they are filtered out
        # after the solve.
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if not feasible[row, column]:
                continue
            if cost[row, column] > self._config.max_cost:
                continue

            associations.append(
                FaceBodyAssociation(
                    track_id=tracks[row].track_id,
                    track_index=row,
                    face_index=column,
                    face=faces[column],
                    body=tracks[row].box,
                    score=float(1.0 - cost[row, column]),
                    containment=float(containment[row, column]),
                )
            )
            matched_tracks.add(row)
            matched_faces.add(column)

        result = AssociationResult(
            associations=associations,
            unmatched_faces=[i for i in range(len(faces)) if i not in matched_faces],
            unmatched_tracks=[i for i in range(len(tracks)) if i not in matched_tracks],
            frame_number=frame_number,
        )
        LOGGER.debug(
            "Frame %d: %d/%d faces associated to %d track(s)",
            frame_number,
            len(associations),
            len(faces),
            len(tracks),
        )
        return result

    # -- cost construction ------------------------------------------------- #
    def cost_matrix(
        self, body_boxes: np.ndarray, face_boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the full assignment cost matrix.

        Entirely vectorised: every term is computed for all ``N x M`` pairs by
        broadcasting, with no Python loop over pairs.

        Args:
            body_boxes: Track boxes, shape ``(N, 4)``.
            face_boxes: Face boxes, shape ``(M, 4)``.

        Returns:
            A tuple ``(cost, containment, feasible)``, each of shape
            ``(N, M)``.  ``cost`` lies in ``[0, 1]`` for feasible pairs and is
            set above ``max_cost`` for infeasible ones.  ``feasible`` is the
            boolean gate.
        """
        containment = containment_matrix(body_boxes, face_boxes)

        body_x1 = body_boxes[:, 0:1]
        body_y1 = body_boxes[:, 1:2]
        body_width = (body_boxes[:, 2] - body_boxes[:, 0])[:, None]
        body_height = (body_boxes[:, 3] - body_boxes[:, 1])[:, None]
        body_area = (body_width * body_height) + _EPS
        body_centre_x = body_x1 + body_width / 2.0

        face_centre_x = ((face_boxes[:, 0] + face_boxes[:, 2]) / 2.0)[None, :]
        face_centre_y = ((face_boxes[:, 1] + face_boxes[:, 3]) / 2.0)[None, :]
        face_area = (
            (face_boxes[:, 2] - face_boxes[:, 0]) * (face_boxes[:, 3] - face_boxes[:, 1])
        )[None, :] + _EPS

        # Term 1 -- containment. The dominant signal.
        containment_cost = 1.0 - containment

        # Term 2 -- horizontal alignment, normalised by body width so it is
        # scale-free: half a body width off is a cost of 1 regardless of how
        # near or far the person is.
        centre_cost = np.clip(
            np.abs(face_centre_x - body_centre_x) / (body_width / 2.0 + _EPS), 0.0, 1.0
        )

        # Term 3 -- scale plausibility, measured in octaves from the expected
        # face/body area ratio. Log scale because the error is multiplicative:
        # a face twice too big and one half too big are equally implausible.
        ratio = face_area / body_area
        octaves = np.abs(np.log2(ratio / self._config.expected_face_ratio))
        scale_cost = np.clip(octaves / _SCALE_OCTAVES, 0.0, 1.0)

        cost = (
            self._config.containment_weight * containment_cost
            + self._config.centre_weight * centre_cost
            + self._config.scale_weight * scale_cost
        ).astype(np.float32)

        # Gate 1 -- the face must be substantially inside the body.
        feasible = containment >= self._config.containment_threshold

        # Gate 2 -- the face must sit in the body's head region. A face level
        # with someone's knees belongs to a different person, however well it
        # is contained.
        head_limit = body_y1 + body_height * self._config.head_region_ratio
        feasible &= face_centre_y <= head_limit

        # Push infeasible pairs above the acceptance threshold rather than to
        # infinity: linear_sum_assignment cannot handle inf, and a merely large
        # finite value keeps the solve well conditioned.
        cost = np.where(feasible, cost, 1.0 + self._config.max_cost).astype(np.float32)
        return cost, containment, feasible