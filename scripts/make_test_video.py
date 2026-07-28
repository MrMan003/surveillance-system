#!/usr/bin/env python3
"""Generate synthetic test clips with controlled timestamps.

A clean constant-frame-rate MP4 never exercises the code paths this project
exists for.  This script writes clips whose PTS values are deliberately
irregular, so the decoder's timestamp handling can be tested against a known
ground truth rather than against whatever a particular camera happened to emit.

Modes:
    cfr     Constant frame rate. Every PTS delta identical. Control case.
    vfr     Variable frame rate. Intervals drawn from a repeating pattern that
            mimics a recorder idling and then bursting during motion.
    gap     Variable, with a large jump partway through, as produced by a
            recorder that stopped and resumed.

Usage:
    python scripts/make_test_video.py --mode vfr --seconds 6
    python scripts/make_test_video.py --mode cfr --output datasets/samples/cfr.mp4
    python scripts/make_test_video.py --mode gap --seconds 8 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import logging
import sys
from fractions import Fraction
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("make_test_video")

#: Stream time base. 1/90000 is the MPEG transport standard and is divisible by
#: common frame intervals, so exact timestamps stay representable as integers.
TIME_BASE = Fraction(1, 90000)

#: Inter-frame intervals in seconds, cycled to build the VFR pattern. Mimics a
#: recorder idling near 8 fps then bursting toward 30 fps during motion.
VFR_PATTERN = (1 / 30, 1 / 30, 1 / 30, 1 / 15, 1 / 8, 1 / 8, 1 / 15, 1 / 30)


def build_pts_schedule(mode: str, duration: float) -> List[int]:
    """Build the exact PTS sequence for the requested mode.

    Args:
        mode: One of ``cfr``, ``vfr`` or ``gap``.
        duration: Target clip duration in seconds.

    Returns:
        Strictly increasing PTS values in ``TIME_BASE`` units.

    Raises:
        ValueError: If ``mode`` is unrecognised.
    """
    ticks_per_second = int(1 / TIME_BASE)
    schedule: List[int] = [0]
    elapsed = 0.0

    if mode == "cfr":
        step = ticks_per_second // 25  # exactly 25 fps
        while elapsed < duration:
            schedule.append(schedule[-1] + step)
            elapsed += 1 / 25
        return schedule

    if mode in {"vfr", "gap"}:
        index = 0
        gap_inserted = False
        while elapsed < duration:
            interval = VFR_PATTERN[index % len(VFR_PATTERN)]
            index += 1
            if mode == "gap" and not gap_inserted and elapsed >= duration / 2:
                interval = 2.5  # recorder stopped and resumed
                gap_inserted = True
            schedule.append(schedule[-1] + round(interval * ticks_per_second))
            elapsed += interval
        return schedule

    raise ValueError(f"Unknown mode {mode!r}; expected cfr, vfr or gap")


def render_frame(index: int, total: int, width: int, height: int, pts: int) -> np.ndarray:
    """Draw a frame carrying its own index and timestamp as visible structure.

    Encoding the frame number into the image lets a test assert that decode
    order matches emission order, independently of the container metadata.

    Args:
        index: Zero-based frame index.
        total: Total frame count, used for the progress band.
        width: Frame width in pixels.
        height: Frame height in pixels.
        pts: This frame's presentation timestamp.

    Returns:
        An RGB ``uint8`` array shaped ``(height, width, 3)``.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient shifting with time, so consecutive frames differ
    # enough that the encoder cannot collapse them into near-empty P-frames.
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    frame[:, :, 0] = xs[np.newaxis, :]
    frame[:, :, 1] = np.uint8((index * 7) % 256)
    frame[:, :, 2] = np.uint8((pts // 512) % 256)

    # Progress band across the top.
    band = max(4, height // 20)
    filled = int(width * (index + 1) / max(total, 1))
    frame[:band, :filled] = 255

    # Binary frame-index marker: one block per bit, LSB at the left.
    block = max(6, width // 64)
    top = band + block
    for bit in range(16):
        if index & (1 << bit):
            x0 = bit * (block + 2) + block
            frame[top : top + block, x0 : x0 + block] = 255

    # Moving square, position derived from the index.
    size = max(8, min(width, height) // 12)
    cx = int((width - size) * (0.5 + 0.45 * np.sin(index * 0.20)))
    cy = int((height - size) * (0.5 + 0.45 * np.cos(index * 0.13)))
    frame[cy : cy + size, cx : cx + size] = (255, 64, 64)

    return frame


def write_clip(output: Path, mode: str, duration: float, width: int, height: int) -> Path:
    """Encode a clip whose frames carry the generated PTS schedule.

    Args:
        output: Destination file; the extension selects the container.
        mode: Timing mode passed to :func:`build_pts_schedule`.
        duration: Target duration in seconds.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        The resolved output path.

    Raises:
        SystemExit: If PyAV is not installed.
    """
    try:
        import av
    except ImportError:
        LOGGER.error("PyAV is required: pip install 'av>=16.0.0,<17.0.0'")
        raise SystemExit(2) from None

    schedule = build_pts_schedule(mode, duration)
    output.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(output), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = TIME_BASE
        # Setting the stream time base alone is not enough: FFmpeg quantises
        # presentation timestamps to the encoder's frame-rate grid unless the
        # codec context shares the same time base. Without this line a 11250
        # tick interval silently becomes 12000, and the "variable" clip is no
        # longer variable in the way the test expects.
        stream.codec_context.time_base = TIME_BASE
        stream.options = {"crf": "18", "preset": "veryfast", "x264-params": "keyint=30"}

        for index, pts in enumerate(schedule):
            array = render_frame(index, len(schedule), width, height, pts)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = pts
            frame.time_base = TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    LOGGER.info(
        "Wrote %s: mode=%s frames=%d span=%.3fs",
        output,
        mode,
        len(schedule),
        float(schedule[-1] * TIME_BASE),
    )
    return output


def main() -> int:
    """Parse arguments and generate the requested clip.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cfr", "vfr", "gap"), default="vfr")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    output = args.output or REPO_ROOT / "datasets" / "samples" / f"{args.mode}.mp4"
    write_clip(output, args.mode, args.seconds, args.width, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())