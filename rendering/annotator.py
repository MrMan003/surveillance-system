"""Overlay drawing for annotated output video.

Scope
-----
This module draws on frames.  It does not decode, encode, or decide anything.
Keeping it free of I/O means overlays can be unit tested on a synthetic frame
without a codec, and the encoder can be swapped without touching the drawing.

What gets drawn, and why
------------------------
Body box and track identity
    The track UUID rather than a sequence number.  A sequence number is only
    meaningful within one run; a UUID is stable across a manifest, an audit log
    and a rendered video, which is what makes a claim about a specific person
    checkable afterwards.

Identity and similarity
    Both, always.  A label without its score invites treating a 0.36 match and
    a 0.91 match as the same statement.

Timestamp and frame number
    The presentation time decoded from the container in Phase 1, not
    ``frame_number / fps``.  This is the number a claim about *when* something
    happened rests on, so it is rendered from the real PTS.

Trajectory
    Centroid history, which makes tracking failures visible at a glance --
    an identity switch shows up as a trail that teleports.

Redaction
    Optional blurring of faces that resolved to ``UNKNOWN``.  Counterintuitive
    until you consider that people *not* in the gallery never consented to
    appear, so they are precisely the ones to obscure in any output leaving the
    machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from association.face_body import FaceBodyAssociation
from configs.config import RenderingConfig
from datasets.frame_stream import TimedFrame
from search.open_set import Identification
from tracking.track import Track
from utils.log import get_logger
from utils.types import BoundingBox, FaceDetection

__all__ = ["FrameAnnotation", "Annotator"]

LOGGER = get_logger(__name__)

#: Padding inside a text background box, in pixels.
_LABEL_PADDING = 4

#: Alpha applied to filled label backgrounds, so overlays remain readable
#: without hiding the pixels underneath them.
_LABEL_ALPHA = 0.65


@dataclass
class FrameAnnotation:
    """Everything the annotator needs to draw one frame.

    Grouped into a single object rather than passed as eight parameters so the
    renderer's signature does not change every time a stage adds an output.

    Attributes:
        timed: The decoded frame with its container timing.
        tracks: Confirmed tracks for this frame.
        faces: Face detections for this frame.
        associations: Face-to-track pairings from Phase 4.
        identifications: Resolved identities keyed by track id.
    """

    timed: TimedFrame
    tracks: Sequence[Track] = field(default_factory=list)
    faces: Sequence[FaceDetection] = field(default_factory=list)
    associations: Sequence[FaceBodyAssociation] = field(default_factory=list)
    identifications: Dict[str, Identification] = field(default_factory=dict)


class Annotator:
    """Draws pipeline output onto video frames.

    Args:
        config: Colours, thicknesses and which overlays to draw.
        unknown_label: Label treated as "not identified", for colouring and
            redaction decisions.
    """

    def __init__(self, config: RenderingConfig, unknown_label: str = "UNKNOWN") -> None:
        self._config = config
        self._unknown_label = unknown_label

    # -- public API -------------------------------------------------------- #
    def draw(self, annotation: FrameAnnotation) -> np.ndarray:
        """Render every enabled overlay onto a copy of the frame.

        Args:
            annotation: The frame and everything to draw on it.

        Returns:
            A new annotated ``uint8`` array; the input frame is not modified.
        """
        canvas = annotation.timed.frame.copy()
        height, width = canvas.shape[:2]

        faces_by_track = {a.track_id: a.face for a in annotation.associations}

        # Redaction runs first so later overlays are drawn on top of the blur
        # rather than being blurred themselves.
        if self._config.redact_unknown:
            canvas = self._redact_unidentified(canvas, annotation, faces_by_track)

        for track in annotation.tracks:
            identification = annotation.identifications.get(track.track_id)
            colour = self._track_colour(identification)

            if self._config.draw_trajectory:
                self._draw_trajectory(canvas, track, colour)

            if self._config.draw_body:
                self._draw_box(canvas, track.box, colour, width, height)

            self._draw_track_label(canvas, track, identification, colour, width, height)

            face = faces_by_track.get(track.track_id)
            if face is not None:
                if self._config.draw_face:
                    self._draw_box(canvas, face.box, self._config.face_color, width, height)
                if self._config.draw_landmarks:
                    self._draw_landmarks(canvas, face)

        # Unassociated faces are drawn too. A face the association stage could
        # not attach to a body is exactly the kind of failure worth seeing.
        if self._config.draw_face:
            associated = {id(a.face) for a in annotation.associations}
            for face in annotation.faces:
                if id(face) not in associated:
                    self._draw_box(canvas, face.box, self._config.face_color, width, height)

        if self._config.draw_timestamp:
            self._draw_header(canvas, annotation)

        return canvas

    # -- overlays ---------------------------------------------------------- #
    def _track_colour(self, identification: Optional[Identification]) -> Tuple[int, int, int]:
        """Choose a colour reflecting identification outcome.

        Args:
            identification: The decision for this track, if any.

        Returns:
            A BGR triplet: the known colour only when an identity was accepted.
        """
        if identification is not None and identification.accepted:
            return self._config.known_color
        return self._config.unknown_color

    def _draw_box(
        self,
        canvas: np.ndarray,
        box: BoundingBox,
        colour: Tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        """Draw a rectangle clipped to the frame.

        Args:
            canvas: Image to draw on, modified in place.
            box: The box to draw.
            colour: BGR triplet.
            width: Frame width.
            height: Frame height.
        """
        import cv2

        x1, y1, x2, y2 = box.clip(width, height).as_int_tuple()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, self._config.box_thickness)

    def _draw_landmarks(self, canvas: np.ndarray, face: FaceDetection) -> None:
        """Draw the five facial landmarks.

        Eyes are drawn larger than the other points: their separation is the
        clearest visual indicator of whether a face carried enough resolution
        to recognise.

        Args:
            canvas: Image to draw on, modified in place.
            face: The detection carrying the landmarks.
        """
        import cv2

        for index, (x, y) in enumerate(face.landmarks):
            radius = 3 if index in (FaceDetection.LEFT_EYE, FaceDetection.RIGHT_EYE) else 2
            cv2.circle(canvas, (int(x), int(y)), radius, self._config.face_color, -1)

    def _draw_trajectory(
        self, canvas: np.ndarray, track: Track, colour: Tuple[int, int, int]
    ) -> None:
        """Draw the track's centroid history as a fading polyline.

        Older points are drawn thinner, so direction of travel is readable from
        a still frame.

        Args:
            canvas: Image to draw on, modified in place.
            track: The track whose history to draw.
            colour: BGR triplet.
        """
        import cv2

        points = track.trajectory_array()
        if points.shape[0] < 2:
            return

        points = points[-self._config.trajectory_length :]
        count = points.shape[0]
        for index in range(1, count):
            thickness = 1 + int(2 * index / count)
            cv2.line(
                canvas,
                (int(points[index - 1, 0]), int(points[index - 1, 1])),
                (int(points[index, 0]), int(points[index, 1])),
                colour,
                thickness,
            )

    def _draw_track_label(
        self,
        canvas: np.ndarray,
        track: Track,
        identification: Optional[Identification],
        colour: Tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        """Draw the identity block above a track's box.

        Args:
            canvas: Image to draw on, modified in place.
            track: The track being labelled.
            identification: Its resolved identity, if any.
            colour: BGR triplet.
            width: Frame width.
            height: Frame height.
        """
        lines = [f"#{track.short_id}"]

        if self._config.draw_identity:
            if identification is None:
                lines.append("unresolved")
            elif identification.accepted:
                lines.append(f"{identification.identity} {identification.similarity:.2f}")
            else:
                lines.append(self._unknown_label)

        lines.append(f"conf {track.confidence:.2f}")

        x1, y1, _, _ = track.box.clip(width, height).as_int_tuple()
        self._draw_text_block(canvas, lines, (x1, y1), colour, above=True)

    def _draw_header(self, canvas: np.ndarray, annotation: FrameAnnotation) -> None:
        """Draw the timing and population header.

        Args:
            canvas: Image to draw on, modified in place.
            annotation: The frame's data.
        """
        timed = annotation.timed
        identified = sum(1 for i in annotation.identifications.values() if i.accepted)
        lines = [
            f"t={timed.timestamp_string()}  pts={timed.pts}  frame={timed.frame_number}",
            f"tracks={len(annotation.tracks)}  faces={len(annotation.faces)}  "
            f"identified={identified}",
        ]
        self._draw_text_block(canvas, lines, (8, 8), (240, 240, 240), above=False)

    # -- text -------------------------------------------------------------- #
    def _draw_text_block(
        self,
        canvas: np.ndarray,
        lines: Sequence[str],
        anchor: Tuple[int, int],
        colour: Tuple[int, int, int],
        above: bool,
    ) -> None:
        """Draw multiple text lines over a translucent background.

        The background is what makes overlays legible against arbitrary video;
        plain text disappears over a busy or bright scene.

        Args:
            canvas: Image to draw on, modified in place.
            lines: Text lines, top to bottom.
            anchor: Reference corner in pixels.
            colour: BGR triplet for the text and background border.
            above: Place the block above the anchor rather than below.
        """
        import cv2

        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._config.font_scale
        thickness = self._config.font_thickness

        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        line_height = max(size[1] for size in sizes) + 5
        block_width = max(size[0] for size in sizes) + 2 * _LABEL_PADDING
        block_height = line_height * len(lines) + _LABEL_PADDING

        x, y = anchor
        top = y - block_height if above else y
        # Clamp so a track near the top edge keeps its label on screen.
        top = max(0, min(top, canvas.shape[0] - block_height))
        left = max(0, min(x, canvas.shape[1] - block_width))

        region = canvas[top : top + block_height, left : left + block_width]
        if region.size:
            overlay = np.full_like(region, 20)
            cv2.addWeighted(overlay, _LABEL_ALPHA, region, 1 - _LABEL_ALPHA, 0, region)

        for index, line in enumerate(lines):
            baseline = top + line_height * (index + 1) - 4
            cv2.putText(
                canvas,
                line,
                (left + _LABEL_PADDING, baseline),
                font,
                scale,
                colour,
                thickness,
                cv2.LINE_AA,
            )

    # -- redaction --------------------------------------------------------- #
    def _redact_unidentified(
        self,
        canvas: np.ndarray,
        annotation: FrameAnnotation,
        faces_by_track: Dict[str, FaceDetection],
    ) -> np.ndarray:
        """Blur faces belonging to tracks that were not identified.

        Args:
            canvas: Image to draw on.
            annotation: The frame's data.
            faces_by_track: Associated face per track id.

        Returns:
            The canvas, modified in place and returned for chaining.
        """
        import cv2

        height, width = canvas.shape[:2]
        kernel = self._config.redaction_kernel

        identified_tracks = {
            track_id
            for track_id, identification in annotation.identifications.items()
            if identification.accepted
        }

        targets: List[BoundingBox] = [
            face.box
            for track_id, face in faces_by_track.items()
            if track_id not in identified_tracks
        ]
        associated = {id(face) for face in faces_by_track.values()}
        targets.extend(face.box for face in annotation.faces if id(face) not in associated)

        for box in targets:
            x1, y1, x2, y2 = box.clip(width, height).as_int_tuple()
            patch = canvas[y1:y2, x1:x2]
            if patch.size:
                canvas[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (kernel, kernel), 0)

        return canvas