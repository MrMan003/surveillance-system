"""Tests for overlay drawing and the timestamp-preserving video writer."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import List

import numpy as np
import pytest

from association.face_body import FaceBodyAssociation
from configs import SurveillanceConfig
from datasets.frame_stream import TimedFrame
from rendering.annotator import Annotator, FrameAnnotation
from rendering.writer import AnnotatedVideoWriter, VideoWriterError
from search.open_set import Identification
from tracking.track import Track
from utils.types import BoundingBox, FaceDetection

pytest.importorskip("cv2", reason="OpenCV is required for drawing")
pytest.importorskip("av", reason="PyAV is required for encoding")

TIME_BASE = Fraction(1, 90000)


def make_timed(frame_number: int = 0, pts: int = 0, size=(480, 640)) -> TimedFrame:
    """Build a TimedFrame over a mid-grey image."""
    return TimedFrame(
        frame=np.full((size[0], size[1], 3), 128, dtype=np.uint8),
        pts=pts,
        dts=pts,
        time_base=TIME_BASE,
        frame_number=frame_number,
    )


def make_track(x: float = 100, y: float = 100, frames: int = 5) -> Track:
    """Build a confirmed track with some trajectory history."""
    track = Track(BoundingBox(x, y, x + 60, y + 200), 0, min_hits=2, max_trajectory=32)
    for step in range(1, frames):
        track.predict()
        track.update(BoundingBox(x + 5 * step, y, x + 60 + 5 * step, y + 200), step)
    return track


def make_face(centre_x: float = 130, centre_y: float = 130) -> FaceDetection:
    """Build a face detection with plausible landmarks."""
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
    return FaceDetection(
        box=BoundingBox(centre_x - 13, centre_y - 13, centre_x + 13, centre_y + 13),
        score=0.92,
        landmarks=landmarks,
    )


def make_identification(track_id: str, accepted: bool) -> Identification:
    """Build an accepted or rejected identification."""
    return Identification(
        track_id=track_id,
        identity="alice" if accepted else "UNKNOWN",
        similarity=0.78 if accepted else 0.21,
        margin=0.31 if accepted else 0.01,
        accepted=accepted,
        threshold=0.35,
        rejection_reason="" if accepted else "similarity below threshold",
    )


@pytest.fixture()
def config() -> SurveillanceConfig:
    """A stock configuration."""
    return SurveillanceConfig.default()


@pytest.fixture()
def annotator(config) -> Annotator:
    """An annotator on stock rendering settings."""
    return Annotator(config.rendering)


# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #
def test_draw_does_not_modify_the_source(annotator: Annotator) -> None:
    """The pipeline reuses frames; drawing must not mutate them."""
    timed = make_timed()
    original = timed.frame.copy()
    annotator.draw(FrameAnnotation(timed=timed, tracks=[make_track()]))
    assert np.array_equal(timed.frame, original)


def test_draw_returns_same_geometry(annotator: Annotator) -> None:
    """Output geometry must match the input, or the encoder rejects it."""
    timed = make_timed()
    canvas = annotator.draw(FrameAnnotation(timed=timed, tracks=[make_track()]))
    assert canvas.shape == timed.frame.shape
    assert canvas.dtype == np.uint8


def test_empty_frame_is_still_annotated(annotator: Annotator) -> None:
    """A frame with nothing detected must still carry its timing header."""
    timed = make_timed()
    canvas = annotator.draw(FrameAnnotation(timed=timed))
    assert not np.array_equal(canvas, timed.frame)


def test_tracks_are_drawn(annotator: Annotator) -> None:
    """Drawing a track must change the frame."""
    timed = make_timed()
    blank = annotator.draw(FrameAnnotation(timed=timed))
    with_track = annotator.draw(FrameAnnotation(timed=timed, tracks=[make_track()]))
    assert not np.array_equal(blank, with_track)


def test_identified_and_unidentified_differ(annotator: Annotator) -> None:
    """Colour must reflect whether an identity was actually accepted."""
    timed = make_timed()
    track = make_track()

    identified = annotator.draw(
        FrameAnnotation(
            timed=timed,
            tracks=[track],
            identifications={track.track_id: make_identification(track.track_id, True)},
        )
    )
    rejected = annotator.draw(
        FrameAnnotation(
            timed=timed,
            tracks=[track],
            identifications={track.track_id: make_identification(track.track_id, False)},
        )
    )
    assert not np.array_equal(identified, rejected)


def test_unresolved_track_is_not_coloured_as_known(annotator: Annotator) -> None:
    """A track with no decision yet must not look identified."""
    timed = make_timed()
    track = make_track()

    unresolved = annotator.draw(FrameAnnotation(timed=timed, tracks=[track]))
    identified = annotator.draw(
        FrameAnnotation(
            timed=timed,
            tracks=[track],
            identifications={track.track_id: make_identification(track.track_id, True)},
        )
    )
    assert not np.array_equal(unresolved, identified)


def test_unassociated_faces_are_still_drawn(annotator: Annotator) -> None:
    """A face the association stage could not place is worth seeing."""
    timed = make_timed()
    without = annotator.draw(FrameAnnotation(timed=timed))
    with_face = annotator.draw(FrameAnnotation(timed=timed, faces=[make_face(400, 300)]))
    assert not np.array_equal(without, with_face)


def test_associated_face_is_drawn_with_its_track(annotator: Annotator) -> None:
    """An associated face must render without raising."""
    timed = make_timed()
    track = make_track()
    face = make_face()
    association = FaceBodyAssociation(
        track_id=track.track_id,
        track_index=0,
        face_index=0,
        face=face,
        body=track.box,
        score=0.9,
        containment=1.0,
    )
    canvas = annotator.draw(
        FrameAnnotation(
            timed=timed, tracks=[track], faces=[face], associations=[association]
        )
    )
    assert canvas.shape == timed.frame.shape


def test_boxes_outside_the_frame_do_not_raise(annotator: Annotator) -> None:
    """Predicted boxes can leave the frame; drawing must clip, not crash."""
    timed = make_timed()
    track = Track(BoundingBox(-200, -300, 50, 40), 0, min_hits=1)
    annotator.draw(FrameAnnotation(timed=timed, tracks=[track]))


def test_label_stays_on_screen_for_a_track_at_the_top(annotator: Annotator) -> None:
    """Labels are drawn above boxes and must be clamped into the frame."""
    timed = make_timed()
    track = Track(BoundingBox(10, 0, 70, 200), 0, min_hits=1)
    canvas = annotator.draw(FrameAnnotation(timed=timed, tracks=[track]))
    assert not np.array_equal(canvas[:40], timed.frame[:40])


def test_overlays_can_be_disabled() -> None:
    """Every overlay must be individually switchable."""
    minimal = SurveillanceConfig.from_dict(
        {
            "rendering": {
                "draw_body": False,
                "draw_face": False,
                "draw_landmarks": False,
                "draw_trajectory": False,
                "draw_identity": False,
                "draw_timestamp": False,
            }
        }
    )
    timed = make_timed()
    canvas = Annotator(minimal.rendering).draw(
        FrameAnnotation(timed=timed, tracks=[make_track()], faces=[make_face()])
    )
    assert canvas.shape == timed.frame.shape


def test_redaction_blurs_unidentified_faces() -> None:
    """People not in the gallery never consented to appear in the output."""
    redacting = SurveillanceConfig.from_dict({"rendering": {"redact_unknown": True}})
    timed = TimedFrame(
        frame=np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),
        pts=0,
        dts=0,
        time_base=TIME_BASE,
        frame_number=0,
    )
    face = make_face(400, 300)

    plain = Annotator(SurveillanceConfig.default().rendering).draw(
        FrameAnnotation(timed=timed, faces=[face])
    )
    redacted = Annotator(redacting.rendering).draw(
        FrameAnnotation(timed=timed, faces=[face])
    )

    x1, y1, x2, y2 = face.box.as_int_tuple()
    assert redacted[y1:y2, x1:x2].std() < plain[y1:y2, x1:x2].std()


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def test_writer_requires_open(config, tmp_path) -> None:
    """Writing before open() must raise."""
    writer = AnnotatedVideoWriter(
        tmp_path / "out.mp4", config.rendering, 640, 480, TIME_BASE
    )
    with pytest.raises(VideoWriterError, match="not open"):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8), make_timed())


def test_writer_rejects_odd_dimensions(config, tmp_path) -> None:
    """yuv420p subsamples chroma and cannot represent odd dimensions."""
    writer = AnnotatedVideoWriter(
        tmp_path / "out.mp4", config.rendering, 641, 480, TIME_BASE
    )
    with pytest.raises(VideoWriterError, match="even dimensions"):
        writer.open()


def test_writer_rejects_wrong_frame_size(config, tmp_path) -> None:
    """A frame that does not match the stream geometry must raise."""
    with AnnotatedVideoWriter(
        tmp_path / "out.mp4", config.rendering, 640, 480, TIME_BASE
    ) as writer:
        with pytest.raises(VideoWriterError, match="expected"):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8), make_timed())


def test_writer_rejects_non_monotonic_pts(config, tmp_path) -> None:
    """Out-of-order timestamps indicate an upstream bug, not a container quirk."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with AnnotatedVideoWriter(
        tmp_path / "out.mp4", config.rendering, 640, 480, TIME_BASE
    ) as writer:
        writer.write(image, make_timed(0, 3000))
        with pytest.raises(VideoWriterError, match="Non-monotonic"):
            writer.write(image, make_timed(1, 1500))


def test_written_video_is_decodable(config, tmp_path) -> None:
    """The output must be a valid, readable H.264 file."""
    from configs import VideoConfig
    from datasets.frame_stream import FrameStream, probe

    output = tmp_path / "out.mp4"
    rng = np.random.default_rng(0)
    with AnnotatedVideoWriter(output, config.rendering, 320, 240, TIME_BASE) as writer:
        for index in range(10):
            image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            writer.write(image, make_timed(index, index * 3600, size=(240, 320)))
        assert writer.frames_written == 10

    info = probe(output)
    assert info.codec == "h264"
    assert (info.width, info.height) == (320, 240)

    with FrameStream(VideoConfig(input_path=output)) as stream:
        assert len(list(stream)) == 10


def test_variable_timestamps_survive_the_round_trip(config, tmp_path) -> None:
    """The central requirement: output timing must match the source exactly.

    Encoding at a fixed frame rate would discard the exact timestamps Phase 1
    exists to recover, leaving a rendered video whose timeline disagrees with
    the source it was made from.
    """
    from configs import VideoConfig
    from datasets.frame_stream import FrameStream

    output = tmp_path / "out.mp4"
    # Deliberately irregular intervals, including a large gap.
    schedule = [0, 3000, 6000, 9000, 20250, 31500, 37500, 40500, 130500, 133500]
    rng = np.random.default_rng(1)

    with AnnotatedVideoWriter(output, config.rendering, 320, 240, TIME_BASE) as writer:
        for index, pts in enumerate(schedule):
            image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            writer.write(image, make_timed(index, pts, size=(240, 320)))

    with FrameStream(VideoConfig(input_path=output)) as stream:
        recovered = [frame.pts for frame in stream]

    assert recovered == schedule


def test_writer_reports_duration(config, tmp_path) -> None:
    """Duration must come from the timestamps, not the frame count."""
    with AnnotatedVideoWriter(
        tmp_path / "out.mp4", config.rendering, 320, 240, TIME_BASE
    ) as writer:
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8), make_timed(0, 0, (240, 320)))
        writer.write(
            np.zeros((240, 320, 3), dtype=np.uint8), make_timed(1, 90000, (240, 320))
        )
    assert writer.duration_seconds == pytest.approx(1.0)


def test_close_is_idempotent(config, tmp_path) -> None:
    """Closing twice must not raise."""
    writer = AnnotatedVideoWriter(
        tmp_path / "out.mp4", config.rendering, 320, 240, TIME_BASE
    ).open()
    writer.write(np.zeros((240, 320, 3), dtype=np.uint8), make_timed(0, 0, (240, 320)))
    writer.close()
    writer.close()


def test_from_stream_info_matches_the_source(config, tmp_path) -> None:
    """A writer built from stream metadata must adopt its geometry and time base."""
    from datasets.frame_stream import StreamInfo

    info = StreamInfo(
        path=Path("source.mp4"),
        codec="h264",
        width=1280,
        height=720,
        time_base=Fraction(1, 90000),
        average_rate=Fraction(30, 1),
        guessed_rate=Fraction(30, 1),
        duration_seconds=10.0,
        frame_count=300,
        pixel_format="yuv420p",
    )
    writer = AnnotatedVideoWriter.from_stream_info(
        tmp_path / "out.mp4", config.rendering, info
    )
    assert writer._width == 1280  # noqa: SLF001
    assert writer._time_base == Fraction(1, 90000)  # noqa: SLF001