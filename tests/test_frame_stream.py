"""Unit tests for forensic VFR decoding."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest

from configs import VideoConfig
from datasets.frame_stream import (
    FrameStream,
    FrameStreamError,
    MissingTimestampError,
    TimedFrame,
    probe,
)


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def test_probe_reads_geometry(vfr_clip: Path) -> None:
    """Probing must report codec and resolution without decoding."""
    info = probe(vfr_clip)
    assert info.codec == "h264"
    assert (info.width, info.height) == (320, 240)
    assert info.time_base == Fraction(1, 90000)


def test_probe_missing_file(tmp_path: Path) -> None:
    """A nonexistent path must fail loudly rather than return empty metadata."""
    with pytest.raises(FrameStreamError):
        probe(tmp_path / "nope.mp4")


def test_probe_rejects_out_of_range_stream(vfr_clip: Path) -> None:
    """Requesting a stream index the container lacks must raise."""
    with pytest.raises(FrameStreamError):
        probe(vfr_clip, stream_index=7)


# --------------------------------------------------------------------------- #
# The central claim: real timestamps, not reconstructed ones
# --------------------------------------------------------------------------- #
def test_cfr_clip_has_a_single_interval(cfr_clip: Path) -> None:
    """The control clip must show exactly one PTS delta."""
    with FrameStream(VideoConfig(input_path=cfr_clip)) as stream:
        list(stream)
        report = stream.timing_report()
    assert report["delta_unique_count"] == 1
    assert report["is_vfr"] is False


def test_vfr_clip_is_detected_as_variable(vfr_clip: Path) -> None:
    """The variable clip must show several distinct intervals."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        list(stream)
        report = stream.timing_report()
    assert report["is_vfr"] is True
    assert report["delta_unique_count"] > 1
    assert report["delta_max_seconds"] > report["delta_min_seconds"]


def test_gap_clip_preserves_the_discontinuity(gap_clip: Path) -> None:
    """A recorder stop/resume must survive decoding as a real gap."""
    with FrameStream(VideoConfig(input_path=gap_clip)) as stream:
        list(stream)
        report = stream.timing_report()
    assert report["delta_max_seconds"] > 2.0


def test_reconstructed_time_diverges_from_real_time(vfr_clip: Path) -> None:
    """frame_index / fps must measurably disagree with the true timeline.

    This is the whole justification for the module. If the naive
    reconstruction agreed with the container timestamps, PyAV would be
    unnecessary.
    """
    info = probe(vfr_clip)
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        frames = list(stream)

    nominal_fps = float(info.guessed_rate or 30)
    worst = max(
        abs(float(timed.seconds) - timed.frame_number / nominal_fps) for timed in frames
    )
    assert worst > 0.05, "expected the naive timeline to drift on a VFR clip"


def test_pts_is_strictly_increasing(vfr_clip: Path) -> None:
    """Presentation timestamps must be monotonic in emission order."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        pts = [timed.pts for timed in stream]
    assert pts == sorted(pts)
    assert len(set(pts)) == len(pts)


def test_seconds_is_an_exact_rational(vfr_clip: Path) -> None:
    """Time must be a Fraction so long recordings accumulate no float error."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        first = next(iter(stream))
    assert isinstance(first.seconds, Fraction)
    assert first.seconds == first.pts * first.time_base


def test_frame_numbers_are_contiguous(vfr_clip: Path) -> None:
    """Emission indices must be a gapless zero-based sequence."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        numbers = [timed.frame_number for timed in stream]
    assert numbers == list(range(len(numbers)))


# --------------------------------------------------------------------------- #
# strict_vfr
# --------------------------------------------------------------------------- #
class _FakeFrame:
    """A decoded frame with no presentation timestamp."""

    pts = None
    key_frame = False

    def to_ndarray(self, format: str) -> np.ndarray:  # noqa: A002, ARG002
        """Return a dummy image."""
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakePacket:
    """A packet that decodes to a single timestamp-less frame."""

    dts = 0

    def decode(self) -> List[_FakeFrame]:
        """Return one frame lacking a PTS."""
        return [_FakeFrame()]


class _FakeContainer:
    """A container yielding exactly one degenerate packet."""

    def demux(self, _stream: object) -> Iterator[_FakePacket]:
        """Yield the single fake packet."""
        yield _FakePacket()

    def close(self) -> None:
        """No-op."""


def _stream_without_timestamps(clip: Path, strict: bool) -> FrameStream:
    """Build a stream whose decoder emits frames with no PTS.

    Args:
        clip: A real clip, used only to satisfy path validation.
        strict: Value for ``strict_vfr``.

    Returns:
        An opened stream with its container swapped for a fake.
    """
    stream = FrameStream(VideoConfig(input_path=clip, strict_vfr=strict)).open()
    stream._container = _FakeContainer()  # noqa: SLF001 - deliberate injection
    return stream


def test_strict_vfr_refuses_to_invent_a_timestamp(vfr_clip: Path) -> None:
    """A frame without PTS must be fatal when strict_vfr is enabled."""
    stream = _stream_without_timestamps(vfr_clip, strict=True)
    with pytest.raises(MissingTimestampError, match="strict_vfr"):
        list(stream)


def test_non_strict_mode_drops_untimed_frames(vfr_clip: Path) -> None:
    """With strict_vfr disabled, untimed frames are dropped, never guessed."""
    stream = _stream_without_timestamps(vfr_clip, strict=False)
    assert list(stream) == []


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_max_frames_caps_output(vfr_clip: Path) -> None:
    """max_frames must bound the number of emitted frames."""
    with FrameStream(VideoConfig(input_path=vfr_clip, max_frames=5)) as stream:
        assert len(list(stream)) == 5


def test_stride_subsamples(vfr_clip: Path) -> None:
    """Stride must keep every Nth frame and preserve their real timestamps."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        every = [timed.pts for timed in stream]
    with FrameStream(VideoConfig(input_path=vfr_clip, stride=3)) as stream:
        strided = [timed.pts for timed in stream]
    assert strided == every[::3]


def test_end_seconds_truncates(vfr_clip: Path) -> None:
    """No emitted frame may fall beyond end_seconds."""
    with FrameStream(VideoConfig(input_path=vfr_clip, end_seconds=1.0)) as stream:
        frames = list(stream)
    assert frames
    assert all(float(timed.seconds) <= 1.0 for timed in frames)


def test_start_seconds_skips_the_head(vfr_clip: Path) -> None:
    """No emitted frame may fall before start_seconds."""
    with FrameStream(VideoConfig(input_path=vfr_clip, start_seconds=1.0)) as stream:
        frames = list(stream)
    assert frames
    assert all(float(timed.seconds) >= 1.0 for timed in frames)


# --------------------------------------------------------------------------- #
# Frame payload
# --------------------------------------------------------------------------- #
def test_frames_are_contiguous_bgr_uint8(vfr_clip: Path) -> None:
    """Downstream detectors require contiguous uint8 HxWx3 arrays."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        first = next(iter(stream))
    assert first.frame.dtype == np.uint8
    assert first.frame.shape == (240, 320, 3)
    assert first.frame.flags["C_CONTIGUOUS"]
    assert (first.width, first.height) == (320, 240)


def test_timestamp_string_format(vfr_clip: Path) -> None:
    """Timestamps must render as zero-padded HH:MM:SS.mmm."""
    timed = TimedFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        pts=90000 * 3661 + 45000,
        dts=None,
        time_base=Fraction(1, 90000),
        frame_number=0,
    )
    assert timed.timestamp_string() == "01:01:01.500"


def test_decode_seconds_is_none_without_dts() -> None:
    """A frame with no DTS must report no decode time rather than guessing."""
    timed = TimedFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        pts=0,
        dts=None,
        time_base=Fraction(1, 90000),
        frame_number=0,
    )
    assert timed.decode_seconds is None


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_iteration_requires_open(vfr_clip: Path) -> None:
    """Iterating before open() must raise rather than yield nothing."""
    with pytest.raises(FrameStreamError):
        list(FrameStream(VideoConfig(input_path=vfr_clip)))


def test_stream_is_single_pass(vfr_clip: Path) -> None:
    """A second iteration must raise rather than silently yield nothing."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        list(stream)
        with pytest.raises(FrameStreamError):
            list(stream)


def test_close_is_idempotent(vfr_clip: Path) -> None:
    """Closing twice must not raise."""
    stream = FrameStream(VideoConfig(input_path=vfr_clip)).open()
    stream.close()
    stream.close()


def test_missing_file_rejected(tmp_path: Path) -> None:
    """Construction must fail immediately for a nonexistent file."""
    with pytest.raises(FrameStreamError):
        FrameStream(VideoConfig(), path=tmp_path / "nope.mp4")


def test_no_input_path_rejected() -> None:
    """Construction must fail when neither config nor argument supplies a path."""
    with pytest.raises(FrameStreamError):
        FrameStream(VideoConfig())


def test_info_requires_open(vfr_clip: Path) -> None:
    """Reading metadata before open() must raise."""
    with pytest.raises(FrameStreamError):
        _ = FrameStream(VideoConfig(input_path=vfr_clip)).info


def test_timing_report_is_empty_before_iteration(vfr_clip: Path) -> None:
    """A report with no observations must say so rather than fabricate stats."""
    with FrameStream(VideoConfig(input_path=vfr_clip)) as stream:
        assert stream.timing_report() == {"frames": 0}