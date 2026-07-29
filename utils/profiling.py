"""Per-stage timing and memory profiling.

Why per stage
-------------
A pipeline reporting only end-to-end frames per second tells you it is slow and
nothing else.  Optimisation then proceeds by guesswork, and the guess is usually
wrong: the intuitive culprit is the largest model, while the actual cost is
often a colour conversion or a per-object Python loop that never appears in a
profile taken at the wrong granularity.

Timing each stage separately turns "the pipeline is slow" into "SCRFD is 60% of
wall clock", which is actionable.

Percentiles, not means
----------------------
A stage averaging 12 ms with a p99 of 400 ms will stall a real-time pipeline in
a way its mean never reveals.  Every report here carries p50, p95, p99 and max.

GPU memory
----------
``torch.cuda.max_memory_allocated`` is a high-water mark rather than an
instantaneous reading, which is what actually matters on a 16 GB card: the peak
decides whether a run survives, and it can occur inside a single forward pass
that no sampled measurement would catch.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

__all__ = ["StageStats", "PipelineProfiler"]

#: Conversion factor from bytes to gibibytes.
_GIB = 1024**3


@dataclass
class StageStats:
    """Accumulated timings for one pipeline stage.

    Attributes:
        name: Stage identifier.
        durations_ms: Per-call wall-clock durations in milliseconds.
        items: Units processed per call, for per-item cost.
    """

    name: str
    durations_ms: List[float] = field(default_factory=list)
    items: List[int] = field(default_factory=list)

    def record(self, duration_ms: float, count: int = 1) -> None:
        """Add one measurement.

        Args:
            duration_ms: Wall-clock duration of the call.
            count: Units processed.
        """
        self.durations_ms.append(duration_ms)
        self.items.append(count)

    @property
    def total_ms(self) -> float:
        """Total time spent in this stage."""
        return float(sum(self.durations_ms))

    @property
    def calls(self) -> int:
        """Number of recorded calls."""
        return len(self.durations_ms)

    def summary(self) -> Dict[str, Any]:
        """Summarise timings for this stage.

        Returns:
            A mapping of statistics; ``calls`` is ``0`` when nothing ran.
        """
        if not self.durations_ms:
            return {"name": self.name, "calls": 0}

        durations = np.asarray(self.durations_ms, dtype=np.float64)
        total_items = int(sum(self.items)) or 1
        return {
            "name": self.name,
            "calls": len(durations),
            "items": total_items,
            "total_ms": float(durations.sum()),
            "total_s": float(durations.sum() / 1000.0),
            "mean_ms": float(durations.mean()),
            "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95)),
            "p99_ms": float(np.percentile(durations, 99)),
            "max_ms": float(durations.max()),
            "ms_per_item": float(durations.sum() / total_items),
        }


class PipelineProfiler:
    """Collects per-stage timings and peak device memory for one run.

    Args:
        enabled: When ``False`` every method is a cheap no-op, so callers need
            no conditional branches around instrumentation.
        device: Resolved device string, used to decide whether to query CUDA.
    """

    def __init__(self, enabled: bool = True, device: str = "cpu") -> None:
        self._enabled = enabled
        self._device = device
        self._stages: Dict[str, StageStats] = {}
        self._started: Optional[float] = None
        self._finished: Optional[float] = None
        self._frames = 0

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        """Mark the beginning of the run and reset CUDA peak memory."""
        self._started = time.perf_counter()
        if self._enabled and self._device.startswith("cuda"):
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):  # pragma: no cover
                pass

    def finish(self) -> None:
        """Mark the end of the run."""
        self._finished = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock duration of the run so far."""
        if self._started is None:
            return 0.0
        end = self._finished if self._finished is not None else time.perf_counter()
        return end - self._started

    @property
    def frames(self) -> int:
        """Frames processed."""
        return self._frames

    def count_frame(self, count: int = 1) -> None:
        """Record that frames were processed.

        Args:
            count: Number of frames.
        """
        self._frames += count

    # -- measurement ------------------------------------------------------- #
    @contextmanager
    def stage(self, name: str, items: int = 1) -> Iterator[None]:
        """Time a block of work.

        Args:
            name: Stage identifier.
            items: Units processed in the block, for per-item cost.

        Yields:
            Control to the timed block.
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._stages.setdefault(name, StageStats(name=name)).record(elapsed_ms, items)

    def record(self, name: str, duration_ms: float, items: int = 1) -> None:
        """Record a timing measured elsewhere.

        Args:
            name: Stage identifier.
            duration_ms: Duration in milliseconds.
            items: Units processed.
        """
        if self._enabled:
            self._stages.setdefault(name, StageStats(name=name)).record(duration_ms, items)

    # -- memory ------------------------------------------------------------ #
    def memory(self) -> Dict[str, float]:
        """Report host and device memory usage.

        Returns:
            A mapping in gibibytes.  CUDA figures are peak values since
            :meth:`start`, because the peak is what decides whether a run fits.
        """
        report: Dict[str, float] = {}

        try:
            import psutil

            report["host_rss_gib"] = psutil.Process().memory_info().rss / _GIB
        except ImportError:  # pragma: no cover
            pass

        if self._device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    report["cuda_allocated_gib"] = torch.cuda.memory_allocated() / _GIB
                    report["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / _GIB
                    report["cuda_reserved_gib"] = torch.cuda.memory_reserved() / _GIB
                    free, total = torch.cuda.mem_get_info()
                    report["cuda_free_gib"] = free / _GIB
                    report["cuda_total_gib"] = total / _GIB
            except (ImportError, RuntimeError):  # pragma: no cover
                pass

        return report

    # -- reporting --------------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        """Assemble the full profiling report.

        Returns:
            Throughput, per-stage timings sorted by total cost, and memory.
        """
        elapsed = self.elapsed_seconds
        stages = sorted(
            (stats.summary() for stats in self._stages.values()),
            key=lambda entry: entry.get("total_ms", 0.0),
            reverse=True,
        )
        accounted = sum(entry.get("total_ms", 0.0) for entry in stages)

        return {
            "frames": self._frames,
            "elapsed_s": elapsed,
            "fps": self._frames / elapsed if elapsed > 0 else 0.0,
            "ms_per_frame": (elapsed * 1000.0 / self._frames) if self._frames else 0.0,
            "stages": stages,
            "accounted_ms": accounted,
            # The gap between measured stages and wall clock is unattributed
            # overhead -- iteration, allocation, garbage collection. A large
            # gap means the instrumentation is missing something real.
            "unaccounted_ms": max(0.0, elapsed * 1000.0 - accounted),
            "memory": self.memory(),
        }

    def report(self) -> str:
        """Render the summary as an aligned text table.

        Returns:
            A multi-line report suitable for logging or a benchmark file.
        """
        summary = self.summary()
        if not summary["frames"]:
            return "No frames profiled."

        lines = [
            "",
            "=" * 78,
            f"  {summary['frames']} frames in {summary['elapsed_s']:.2f}s  "
            f"({summary['fps']:.2f} fps, {summary['ms_per_frame']:.1f} ms/frame)",
            "=" * 78,
            f"  {'stage':<22}{'total_s':>9}{'share':>8}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}",
            "  " + "-" * 74,
        ]

        total_ms = max(summary["elapsed_s"] * 1000.0, 1e-9)
        for stage in summary["stages"]:
            if not stage.get("calls"):
                continue
            lines.append(
                f"  {stage['name']:<22}"
                f"{stage['total_s']:>9.2f}"
                f"{100 * stage['total_ms'] / total_ms:>7.1f}%"
                f"{stage['p50_ms']:>9.1f}"
                f"{stage['p95_ms']:>9.1f}"
                f"{stage['p99_ms']:>9.1f}"
                f"{stage['max_ms']:>9.1f}"
            )

        lines.append("  " + "-" * 74)
        lines.append(
            f"  {'unaccounted':<22}{summary['unaccounted_ms'] / 1000.0:>9.2f}"
            f"{100 * summary['unaccounted_ms'] / total_ms:>7.1f}%"
        )

        memory = summary["memory"]
        if memory:
            lines.append("")
            if "host_rss_gib" in memory:
                lines.append(f"  host rss      : {memory['host_rss_gib']:.2f} GiB")
            if "cuda_peak_gib" in memory:
                lines.append(
                    f"  cuda peak     : {memory['cuda_peak_gib']:.2f} GiB"
                    f"  (reserved {memory.get('cuda_reserved_gib', 0):.2f},"
                    f" total {memory.get('cuda_total_gib', 0):.2f})"
                )
        lines.append("=" * 78)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all measurements."""
        self._stages.clear()
        self._started = None
        self._finished = None
        self._frames = 0