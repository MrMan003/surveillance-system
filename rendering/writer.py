"""H.264 video writer that preserves original container timestamps.

The point
---------
Phase 1 goes to some trouble to read exact presentation timestamps rather than
reconstructing them from a nominal frame rate.  Writing the annotated output at
a fixed frame rate would discard that work at the last step: the rendered video
would no longer align with the source, and any claim about when an event
occurred would have to be checked against the original rather than the
artefact people actually watch.

This writer therefore carries each frame's original PTS through to the output.
The result is a variable-frame-rate file whose timeline matches the source
exactly, which is the only version of the output that is safe to reason about.

``cv2.VideoWriter`` cannot do this.  It accepts a single ``fps`` and assigns
timestamps by counting, which is the failure this pipeline exists to avoid.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Optional, Type, Union

import numpy as np

from configs.config import RenderingConfig
from datasets.frame_stream import TimedFrame
from utils.log import get_logger

__all__ = ["VideoWriterError", "AnnotatedVideoWriter"]

LOGGER = get_logger(__name__)


class VideoWriterError(RuntimeError):
    """Raised when the output video cannot be opened, written, or closed."""


class AnnotatedVideoWriter:
    """Encodes annotated frames to H.264, preserving source timestamps.

    Args:
        path: Destination file.
        config: Encoder settings.
        width: Frame width in pixels.
        height: Frame height in pixels.
        time_base: Time base of the source stream.  Reusing it means output
            PTS values are the source's own integers, with no rescaling and so
            no rounding.
        nominal_rate: Frame rate written into the container header.  Advisory
            only for a variable-rate file, but players use it to size their
            seek bars, so a wildly wrong value makes the output awkward to
            scrub even though its timestamps are exact.
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: RenderingConfig,
        width: int,
        height: int,
        time_base: Fraction,
        nominal_rate: Fraction = Fraction(25, 1),
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._config = config
        self._width = int(width)
        self._height = int(height)
        self._time_base = time_base
        self._nominal_rate = nominal_rate

        self._container = None
        self._stream = None
        self._frames_written = 0
        self._first_pts: Optional[int] = None
        self._last_pts: Optional[int] = None
        self._closed = False

    # -- properties -------------------------------------------------------- #
    @property
    def path(self) -> Path:
        """Destination file."""
        return self._path

    @property
    def frames_written(self) -> int:
        """Frames encoded so far."""
        return self._frames_written

    @property
    def duration_seconds(self) -> float:
        """Span between the first and last written timestamps, in seconds."""
        if self._first_pts is None or self._last_pts is None:
            return 0.0
        return float((self._last_pts - self._first_pts) * self._time_base)

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> "AnnotatedVideoWriter":
        """Create the output container and configure the encoder.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            VideoWriterError: If the container or encoder cannot be created.
        """
        if self._container is not None:
            return self

        try:
            import av
        except ImportError as exc:
            raise VideoWriterError("PyAV is required to write video") from exc

        if self._width % 2 or self._height % 2:
            raise VideoWriterError(
                f"H.264 with {self._config.pixel_format} requires even dimensions, "
                f"got {self._width}x{self._height}"
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._container = av.open(str(self._path), mode="w")
            stream = self._container.add_stream(
                self._config.codec, rate=self._nominal_rate
            )
            stream.width = self._width
            stream.height = self._height
            stream.pix_fmt = self._config.pixel_format
            stream.time_base = self._time_base
            # Setting the stream time base alone is not enough: FFmpeg quantises
            # presentation timestamps to the encoder's frame-rate grid unless
            # the codec context shares the same base. Without this the output
            # silently becomes constant frame rate.
            stream.codec_context.time_base = self._time_base
            stream.options = {
                "crf": str(self._config.crf),
                "preset": self._config.preset,
            }
            self._stream = stream
        except Exception as exc:  # noqa: BLE001 - PyAV raises widely
            self._container = None
            raise VideoWriterError(f"Cannot open {self._path}: {exc}") from exc

        LOGGER.info(
            "Writing %s (%dx%d, %s crf=%d, time_base=%s)",
            self._path.name,
            self._width,
            self._height,
            self._config.codec,
            self._config.crf,
            self._time_base,
        )
        return self

    def write(self, image: np.ndarray, timed: TimedFrame) -> None:
        """Encode one annotated frame at its original timestamp.

        Args:
            image: Annotated BGR ``uint8`` frame.
            timed: The source frame, supplying the presentation timestamp.

        Raises:
            VideoWriterError: If the writer is closed, the frame geometry does
                not match, or encoding fails.
        """
        import av

        if self._container is None:
            raise VideoWriterError("Writer not open; call open() or use a with block")
        if self._closed:
            raise VideoWriterError("Writer already closed")

        if image.shape[0] != self._height or image.shape[1] != self._width:
            raise VideoWriterError(
                f"Frame is {image.shape[1]}x{image.shape[0]}, "
                f"expected {self._width}x{self._height}"
            )

        # Strictly increasing PTS is a container requirement. Frames arrive in
        # presentation order, so a violation means an upstream reordering bug
        # rather than something to paper over by renumbering.
        if self._last_pts is not None and timed.pts <= self._last_pts:
            raise VideoWriterError(
                f"Non-monotonic PTS: {timed.pts} follows {self._last_pts}"
            )

        try:
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format="bgr24"
            )
            frame.pts = timed.pts
            frame.time_base = self._time_base
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        except Exception as exc:  # noqa: BLE001
            raise VideoWriterError(f"Encoding frame {timed.frame_number} failed: {exc}") from exc

        if self._first_pts is None:
            self._first_pts = timed.pts
        self._last_pts = timed.pts
        self._frames_written += 1

    def close(self) -> None:
        """Flush the encoder and finalise the container.

        Safe to call more than once.  Flushing matters: H.264 buffers frames
        internally, so skipping it truncates the output by the buffer depth.
        """
        if self._closed or self._container is None:
            self._closed = True
            return

        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Error flushing encoder: %s", exc)
        finally:
            try:
                self._container.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Error closing %s: %s", self._path.name, exc)
            self._container = None
            self._stream = None
            self._closed = True

        LOGGER.info(
            "Wrote %d frame(s) spanning %.3fs to %s",
            self._frames_written,
            self.duration_seconds,
            self._path,
        )

    def __enter__(self) -> "AnnotatedVideoWriter":
        """Open the writer on entering a ``with`` block."""
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Finalise the output on leaving a ``with`` block."""
        self.close()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_stream_info(
        cls,
        path: Union[str, Path],
        config: RenderingConfig,
        info: object,
    ) -> "AnnotatedVideoWriter":
        """Build a writer matching a source stream's geometry and time base.

        Args:
            path: Destination file.
            config: Encoder settings.
            info: A :class:`datasets.frame_stream.StreamInfo` for the source.

        Returns:
            An unopened writer configured to match the source.
        """
        rate = getattr(info, "average_rate", None) or Fraction(25, 1)
        return cls(
            path=path,
            config=config,
            width=int(getattr(info, "width")),
            height=int(getattr(info, "height")),
            time_base=Fraction(getattr(info, "time_base")),
            nominal_rate=Fraction(rate),
        )