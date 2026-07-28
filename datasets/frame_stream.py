"""Forensic, variable-frame-rate video decoding built on PyAV.

Why this module exists
----------------------
``cv2.VideoCapture`` discards container timestamps.  It exposes a single
averaged ``CAP_PROP_FPS`` and a frame counter, which forces every downstream
consumer to reconstruct time as ``frame_index / fps``.  For constant-frame-rate
studio footage that is harmless.  For CCTV it is wrong: recorders drop frames
under load, stretch intervals during motion inactivity, and splice segments
with discontinuous clocks.  The reconstructed timeline drifts without bound,
and any claim about *when* an event occurred inherits that drift.

:class:`FrameStream` reads the presentation timestamp the muxer actually wrote,
keeps it as an exact rational in stream time base units, and refuses to
synthesise a timestamp it was not given.

Terminology
-----------
PTS
    Presentation timestamp -- when a frame should be shown.  This is the one
    that matters for forensic timing.
DTS
    Decode timestamp -- when a frame must enter the decoder.  With B-frames
    DTS and PTS differ, and DTS may be absent in remuxed containers.
time_base
    The rational unit both are counted in, e.g. ``1/90000``.  Seconds are
    ``pts * time_base`` computed as a :class:`fractions.Fraction`, never as a
    float multiply, so no rounding accumulates across a long recording.

Example
-------
>>> from configs import SurveillanceConfig
>>> cfg = SurveillanceConfig.default()
>>> with FrameStream(cfg.video) as stream:  # doctest: +SKIP
...     for timed in stream:
...         print(timed.frame_number, timed.pts, float(timed.seconds))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np

from configs.config import VideoConfig

__all__ = [
    "FrameStreamError",
    "MissingTimestampError",
    "TimedFrame",
    "StreamInfo",
    "FrameStream",
    "probe",
]

LOGGER = logging.getLogger(__name__)


class FrameStreamError(RuntimeError):
    """Base class for every decoding failure raised by this module."""


class MissingTimestampError(FrameStreamError):
    """Raised when a frame carries no PTS and ``strict_vfr`` is enabled.

    This is deliberately fatal.  The alternative -- inventing a timestamp from
    the frame index and a nominal frame rate -- silently produces a plausible
    but incorrect timeline, which is the exact failure this module exists to
    prevent.
    """


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TimedFrame:
    """A decoded frame together with its exact container timing.

    Attributes:
        frame: Decoded image as a contiguous ``uint8`` array in the configured
            pixel format (BGR by default), shaped ``(height, width, 3)``.
        pts: Presentation timestamp in ``time_base`` units, as written by the
            muxer.  Never synthesised.
        dts: Decode timestamp of the packet that produced this frame, or
            ``None`` when the container omits it.
        time_base: Rational unit for ``pts`` and ``dts``.
        frame_number: Zero-based index of this frame in emission order.  This
            is a counter for bookkeeping only and must never be used to derive
            time.
        key_frame: Whether this frame is an I-frame.
        packet_index: Zero-based index of the source packet in the container.
    """

    frame: np.ndarray
    pts: int
    dts: Optional[int]
    time_base: Fraction
    frame_number: int
    key_frame: bool = False
    packet_index: int = -1

    @property
    def seconds(self) -> Fraction:
        """Presentation time in seconds as an exact rational.

        Returns:
            ``pts * time_base``.  Kept rational so that summing or differencing
            timestamps across a multi-hour recording introduces no float drift.
        """
        return self.pts * self.time_base

    @property
    def decode_seconds(self) -> Optional[Fraction]:
        """Decode time in seconds, or ``None`` when DTS is unavailable.

        Returns:
            ``dts * time_base`` when DTS is present.
        """
        return None if self.dts is None else self.dts * self.time_base

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.frame.shape[0])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.frame.shape[1])

    def timestamp_string(self, decimals: int = 3) -> str:
        """Format the presentation time as ``HH:MM:SS.mmm``.

        Args:
            decimals: Fractional-second digits to render.

        Returns:
            A zero-padded wall-clock offset from the start of the stream.
        """
        total = self.seconds
        hours, remainder = divmod(int(total), 3600)
        minutes, whole_seconds = divmod(remainder, 60)
        frac = total - int(total)
        scaled = int(frac * (10**decimals))
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{scaled:0{decimals}d}"

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"TimedFrame(n={self.frame_number}, pts={self.pts}, dts={self.dts}, "
            f"t={self.timestamp_string()}, {self.width}x{self.height}"
            f"{', key' if self.key_frame else ''})"
        )


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """Static metadata describing the source video stream.

    Attributes:
        path: Container path.
        codec: Codec short name, e.g. ``h264``.
        width: Coded frame width in pixels.
        height: Coded frame height in pixels.
        time_base: Rational unit of the stream's timestamps.
        average_rate: Muxer-declared average frame rate, or ``None``.
        guessed_rate: Muxer-declared nominal frame rate, or ``None``.
        duration_seconds: Stream duration in seconds, or ``None`` if unknown.
        frame_count: Muxer-declared frame count; often ``0`` and untrustworthy.
        pixel_format: Source pixel format name.
    """

    path: Path
    codec: str
    width: int
    height: int
    time_base: Fraction
    average_rate: Optional[Fraction]
    guessed_rate: Optional[Fraction]
    duration_seconds: Optional[float]
    frame_count: int
    pixel_format: Optional[str]

    @property
    def declared_vfr(self) -> bool:
        """Whether the container's own metadata suggests variable frame rate.

        Compares the average rate against the nominal rate.  A mismatch is
        strong evidence of VFR, but agreement does **not** prove CFR -- many
        recorders declare a nominal rate they do not honour.  Only the decoded
        PTS deltas settle it; see :meth:`FrameStream.timing_report`.

        Returns:
            ``True`` when the two declared rates disagree.
        """
        if self.average_rate is None or self.guessed_rate is None:
            return False
        return self.average_rate != self.guessed_rate

    def summary(self) -> str:
        """Render a one-block human-readable digest.

        Returns:
            A multi-line string suitable for logging at start-up.
        """
        avg = f"{float(self.average_rate):.4f}" if self.average_rate else "unknown"
        gss = f"{float(self.guessed_rate):.4f}" if self.guessed_rate else "unknown"
        dur = f"{self.duration_seconds:.3f}s" if self.duration_seconds else "unknown"
        return "\n".join(
            [
                f"StreamInfo({self.path.name})",
                f"  codec        : {self.codec} [{self.pixel_format}]",
                f"  resolution   : {self.width}x{self.height}",
                f"  time_base    : {self.time_base}",
                f"  average_rate : {avg}",
                f"  guessed_rate : {gss}",
                f"  duration     : {dur}",
                f"  frame_count  : {self.frame_count} (declared; often 0)",
                f"  declared_vfr : {self.declared_vfr}",
            ]
        )


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def probe(path: Union[str, Path], stream_index: int = 0) -> StreamInfo:
    """Read stream metadata without decoding any frames.

    Args:
        path: Path to an ``.mp4`` or ``.mkv`` container.
        stream_index: Index of the video stream within the container.

    Returns:
        The stream's static metadata.

    Raises:
        FrameStreamError: If the file is missing, unreadable, or contains no
            video stream at ``stream_index``.
    """
    import av

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FrameStreamError(f"Video not found: {path}")

    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if stream_index >= len(streams):
                raise FrameStreamError(
                    f"{path.name} has {len(streams)} video stream(s); "
                    f"index {stream_index} requested"
                )
            stream = streams[stream_index]
            duration: Optional[float] = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / 1_000_000.0

            return StreamInfo(
                path=path,
                codec=stream.codec_context.name,
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                time_base=Fraction(stream.time_base),
                average_rate=Fraction(stream.average_rate) if stream.average_rate else None,
                guessed_rate=Fraction(stream.guessed_rate) if stream.guessed_rate else None,
                duration_seconds=duration,
                frame_count=int(stream.frames or 0),
                pixel_format=stream.codec_context.pix_fmt,
            )
    except FrameStreamError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyAV raises a wide error family
        raise FrameStreamError(f"Cannot probe {path.name}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Frame stream
# --------------------------------------------------------------------------- #
class FrameStream:
    """Iterate a video container, yielding frames with exact container timing.

    The stream is a context manager and a single-pass iterator.  Iterating a
    closed stream, or iterating twice, raises rather than silently producing
    nothing.

    Args:
        config: Decoding parameters.  ``config.input_path`` must be set, or
            ``path`` must be supplied.
        path: Optional override for ``config.input_path``.

    Raises:
        FrameStreamError: If no input path is configured, or the file is
            missing or unreadable.
    """

    def __init__(self, config: VideoConfig, path: Optional[Union[str, Path]] = None) -> None:
        source = path if path is not None else config.input_path
        if source is None:
            raise FrameStreamError(
                "No input path: set VideoConfig.input_path or pass path= explicitly"
            )

        self._config = config
        self._path = Path(source).expanduser().resolve()
        if not self._path.is_file():
            raise FrameStreamError(f"Video not found: {self._path}")
        if self._path.suffix.lower() not in config.allowed_suffixes:
            raise FrameStreamError(
                f"Unsupported container {self._path.suffix!r}; "
                f"allowed: {config.allowed_suffixes}"
            )

        self._container: Any = None
        self._stream: Any = None
        self._info: Optional[StreamInfo] = None
        self._closed = False
        self._consumed = False

        # Running counters, reset on open().
        self._emitted = 0
        self._decoded = 0
        self._skipped_no_pts = 0
        self._pts_history: List[int] = []
        self._non_monotonic = 0

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> "FrameStream":
        """Open the container and configure the decoder.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            FrameStreamError: If the container cannot be opened or has no video
                stream at the configured index.
        """
        import av

        if self._container is not None:
            return self

        try:
            self._container = av.open(str(self._path))
        except Exception as exc:  # noqa: BLE001
            raise FrameStreamError(f"Cannot open {self._path.name}: {exc}") from exc

        streams = self._container.streams.video
        if self._config.stream_index >= len(streams):
            self.close()
            raise FrameStreamError(
                f"{self._path.name} has {len(streams)} video stream(s); "
                f"index {self._config.stream_index} requested"
            )

        self._stream = streams[self._config.stream_index]
        # Frame-level threading is the single biggest decode speedup available
        # and costs nothing in correctness: timestamps come from the container,
        # not from decode order.
        self._stream.thread_type = self._config.thread_type.value
        if self._config.thread_count > 0:
            self._stream.codec_context.thread_count = self._config.thread_count

        self._info = probe(self._path, self._config.stream_index)
        self._closed = False
        self._emitted = 0
        self._decoded = 0
        self._skipped_no_pts = 0
        self._pts_history = []
        self._non_monotonic = 0

        LOGGER.info(
            "Opened %s [%s %dx%d, time_base=%s]",
            self._path.name,
            self._info.codec,
            self._info.width,
            self._info.height,
            self._info.time_base,
        )
        return self

    def close(self) -> None:
        """Close the container and release decoder resources.

        Safe to call more than once.
        """
        if self._container is not None:
            try:
                self._container.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Error closing %s: %s", self._path.name, exc)
            finally:
                self._container = None
                self._stream = None
        self._closed = True

    def __enter__(self) -> "FrameStream":
        """Open the stream on entering a ``with`` block."""
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close the stream on leaving a ``with`` block."""
        self.close()

    # -- properties -------------------------------------------------------- #
    @property
    def info(self) -> StreamInfo:
        """Static metadata for the opened stream.

        Returns:
            The :class:`StreamInfo` gathered at open time.

        Raises:
            FrameStreamError: If the stream has not been opened.
        """
        if self._info is None:
            raise FrameStreamError("Stream not opened; call open() or use a with block")
        return self._info

    @property
    def time_base(self) -> Fraction:
        """Rational unit of this stream's timestamps."""
        return self.info.time_base

    @property
    def path(self) -> Path:
        """Resolved path to the source container."""
        return self._path

    # -- iteration --------------------------------------------------------- #
    def __iter__(self) -> Iterator[TimedFrame]:
        """Yield decoded frames in presentation order.

        Frames are produced by demuxing packets and decoding each one, which is
        what makes DTS available: a decoded frame carries its own PTS, but DTS
        belongs to the packet that delivered it.

        Yields:
            One :class:`TimedFrame` per surviving frame, after ``stride``,
            ``start_seconds``, ``end_seconds`` and ``max_frames`` filtering.

        Raises:
            FrameStreamError: If the stream is closed or already consumed.
            MissingTimestampError: If a frame has no PTS and
                ``config.strict_vfr`` is enabled.
        """
        if self._container is None:
            raise FrameStreamError("Stream not opened; call open() or use a with block")
        if self._consumed:
            raise FrameStreamError(
                "FrameStream is single-pass; construct a new one to iterate again"
            )
        self._consumed = True

        config = self._config
        time_base = self.time_base
        start_pts = self._seconds_to_pts(config.start_seconds)
        end_pts = self._seconds_to_pts(config.end_seconds)

        if start_pts is not None:
            self._seek(start_pts)

        for packet_index, packet in enumerate(self._container.demux(self._stream)):
            # A packet with dts is None is the flush packet PyAV appends after
            # the last real packet; decoding it drains buffered B-frames.
            for frame in packet.decode():
                self._decoded += 1

                pts = frame.pts
                if pts is None:
                    if config.strict_vfr:
                        raise MissingTimestampError(
                            f"{self._path.name}: frame {self._decoded} has no PTS. "
                            "Refusing to synthesise one from frame_index/fps because "
                            "strict_vfr is enabled. Set video.strict_vfr=false to "
                            "accept an approximate timeline."
                        )
                    self._skipped_no_pts += 1
                    LOGGER.warning(
                        "Dropping frame %d: no PTS (strict_vfr disabled)", self._decoded
                    )
                    continue

                if start_pts is not None and pts < start_pts:
                    continue
                if end_pts is not None and pts > end_pts:
                    LOGGER.debug("Reached end_seconds at pts=%d", pts)
                    return

                if self._pts_history and pts <= self._pts_history[-1]:
                    self._non_monotonic += 1
                    LOGGER.warning(
                        "Non-monotonic PTS: %d follows %d (packet %d)",
                        pts,
                        self._pts_history[-1],
                        packet_index,
                    )
                self._pts_history.append(pts)

                if config.stride > 1 and (len(self._pts_history) - 1) % config.stride:
                    continue

                array = frame.to_ndarray(format=config.pixel_format)
                if not array.flags["C_CONTIGUOUS"]:
                    array = np.ascontiguousarray(array)

                dts = packet.dts
                if dts is None and not config.allow_missing_dts:
                    raise FrameStreamError(
                        f"{self._path.name}: packet {packet_index} has no DTS and "
                        "allow_missing_dts is disabled"
                    )

                yield TimedFrame(
                    frame=array,
                    pts=int(pts),
                    dts=None if dts is None else int(dts),
                    time_base=time_base,
                    frame_number=self._emitted,
                    key_frame=bool(frame.key_frame),
                    packet_index=packet_index,
                )

                self._emitted += 1
                if config.max_frames is not None and self._emitted >= config.max_frames:
                    LOGGER.info("Reached max_frames=%d", config.max_frames)
                    return

        LOGGER.info(
            "Decoded %d frame(s), emitted %d, dropped %d without PTS",
            self._decoded,
            self._emitted,
            self._skipped_no_pts,
        )

    # -- helpers ----------------------------------------------------------- #
    def _seconds_to_pts(self, seconds: Optional[float]) -> Optional[int]:
        """Convert a wall-clock offset to stream time base units.

        Args:
            seconds: Offset in seconds, or ``None``.

        Returns:
            The equivalent PTS, or ``None`` when ``seconds`` is ``None``.
        """
        if seconds is None:
            return None
        return int(Fraction(seconds).limit_denominator(1_000_000) / self.time_base)

    def _seek(self, target_pts: int) -> None:
        """Seek to the last keyframe at or before ``target_pts``.

        Decoding must begin at a keyframe, so the container seeks backwards and
        the iterator discards frames before the requested point.

        Args:
            target_pts: Desired start position in stream time base units.
        """
        try:
            self._container.seek(target_pts, stream=self._stream, backward=True, any_frame=False)
            LOGGER.debug("Sought to pts <= %d", target_pts)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Seek to %d failed (%s); decoding from start", target_pts, exc)

    # -- diagnostics ------------------------------------------------------- #
    def timing_report(self) -> Dict[str, Any]:
        """Summarise the timing actually observed during iteration.

        This is the empirical VFR test.  ``StreamInfo.declared_vfr`` reports
        what the container claims; this reports what the timestamps do.  A
        recording is variable-frame-rate when the inter-frame PTS deltas are
        not all equal.

        Returns:
            A mapping with frame counts, PTS delta statistics in both time base
            units and seconds, an ``is_vfr`` verdict, and any anomaly counts.
            Returns ``{"frames": 0}`` when fewer than two frames were seen.
        """
        if len(self._pts_history) < 2:
            return {"frames": len(self._pts_history)}

        pts = np.asarray(self._pts_history, dtype=np.int64)
        deltas = np.diff(pts)
        unique = np.unique(deltas)
        tb = float(self.time_base)
        mean_delta = float(deltas.mean())

        return {
            "frames": int(pts.size),
            "decoded": self._decoded,
            "emitted": self._emitted,
            "dropped_no_pts": self._skipped_no_pts,
            "non_monotonic": self._non_monotonic,
            "first_pts": int(pts[0]),
            "last_pts": int(pts[-1]),
            "span_seconds": float((pts[-1] - pts[0]) * tb),
            "delta_unique_count": int(unique.size),
            "delta_min": int(deltas.min()),
            "delta_max": int(deltas.max()),
            "delta_mean": mean_delta,
            "delta_std": float(deltas.std()),
            "delta_min_seconds": float(deltas.min() * tb),
            "delta_max_seconds": float(deltas.max() * tb),
            "is_vfr": bool(unique.size > 1),
            "effective_fps": (1.0 / (mean_delta * tb)) if mean_delta > 0 else 0.0,
        }