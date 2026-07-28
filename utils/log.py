"""Logging configuration and the biometric audit trail.

Two separate concerns live here, deliberately kept apart:

Operational logging
    Human-readable diagnostics for the engineer running the pipeline.  Noisy,
    disposable, rotated.

Audit logging
    A machine-readable, append-only record of every identification decision.
    This is a governance artefact, not a debugging aid.  Under GDPR Art. 22 and
    India's DPDP Act a person may ask what an automated system concluded about
    them and on what basis; a system that cannot answer has a compliance
    problem, not a logging problem.  The audit log is therefore JSONL, one
    self-contained object per line, appended and never rewritten.

The module is named ``log`` rather than ``logging`` so that ``import logging``
inside this package unambiguously reaches the standard library.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = [
    "setup_logging",
    "get_logger",
    "AuditLogger",
    "ColourFormatter",
]

#: Default format for file output. Deliberately verbose: a log line that cannot
#: be traced back to a module and a timestamp is of little use after the fact.
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(funcName)-20s | %(message)s"

#: Console format. Shorter, because the console is read live.
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Third-party loggers that emit at INFO on import and drown everything else.
NOISY_LOGGERS = (
    "matplotlib",
    "PIL",
    "urllib3",
    "filelock",
    "asyncio",
    "libav",
    "numba",
)


class ColourFormatter(logging.Formatter):
    """Console formatter that tints the level name when attached to a TTY.

    Colour is suppressed when output is redirected, so log files never contain
    escape sequences.
    """

    COLOURS: Dict[int, str] = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str, use_colour: bool = True) -> None:
        """Initialise the formatter.

        Args:
            fmt: Format string.
            datefmt: Date format string.
            use_colour: Whether to emit ANSI colour codes.
        """
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colour = use_colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Render a record, optionally colouring the level name.

        Args:
            record: The record to render.

        Returns:
            The formatted line.
        """
        if not self._use_colour:
            return super().format(record)
        colour = self.COLOURS.get(record.levelno, "")
        original = record.levelname
        record.levelname = f"{colour}{original}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[Path] = None,
    log_name: str = "pipeline.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure root logging for the process.

    Idempotent: existing handlers on the root logger are removed first, so
    calling this from both a script and a notebook cell does not duplicate
    every line.

    Args:
        level: Threshold for both handlers, as a level name or number.
        log_dir: Directory for the rotating log file.  ``None`` disables file
            logging, which is the right choice inside a notebook.
        log_name: Filename within ``log_dir``.
        max_bytes: Size at which the log file rotates.
        backup_count: Number of rotated files retained.
        capture_warnings: Route :mod:`warnings` through logging as well.

    Returns:
        The configured root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(ColourFormatter(CONSOLE_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / log_name,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        root.addHandler(file_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    if capture_warnings:
        logging.captureWarnings(True)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Usually ``__name__`` from the calling module.

    Returns:
        The named logger.
    """
    return logging.getLogger(name)


def _to_jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts.

    Args:
        value: Any value destined for an audit record.

    Returns:
        A JSON-serialisable equivalent.
    """
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value


class AuditLogger:
    """Append-only JSONL record of identification decisions.

    Each line is a complete, self-contained event.  Nothing is rewritten, so a
    partially written file loses at most its final line, and the record cannot
    be silently amended after the fact.

    Args:
        path: Destination file.  Parent directories are created.
        enabled: When ``False`` every method is a no-op, so callers need no
            conditional branches around audit calls.
        source: Identifier for the video or camera being processed.
    """

    def __init__(self, path: Union[str, Path], enabled: bool = True, source: str = "") -> None:
        self._path = Path(path).expanduser()
        self._enabled = enabled
        self._source = source
        self._lock = threading.Lock()
        self._count = 0

        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Destination file for audit records."""
        return self._path

    @property
    def count(self) -> int:
        """Number of records written by this instance."""
        return self._count

    def _write(self, record: Dict[str, Any]) -> None:
        """Append one JSON object, followed by a newline.

        Args:
            record: The event payload.
        """
        if not self._enabled:
            return
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        if self._source:
            record["source"] = self._source
        line = json.dumps(_to_jsonable(record), separators=(",", ":"), sort_keys=True)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self._count += 1

    def identification(
        self,
        track_id: str,
        identity: str,
        similarity: float,
        margin: float,
        threshold: float,
        frame_number: int,
        media_seconds: Optional[float] = None,
        embeddings_fused: int = 0,
        **extra: Any,
    ) -> None:
        """Record an identification decision, including rejections.

        Rejections are recorded too.  A log containing only positive matches
        cannot answer whether the system considered and dismissed someone,
        which is precisely the question an audit asks.

        Args:
            track_id: UUID of the track being identified.
            identity: Resolved label, or the configured unknown sentinel.
            similarity: Cosine similarity of the best gallery match.
            margin: Gap between the best and second-best match.
            threshold: Acceptance threshold in force for this decision.
            frame_number: Frame at which the decision was made.
            media_seconds: Presentation time within the recording.
            embeddings_fused: Number of embeddings behind the query vector.
            **extra: Additional fields merged into the record.
        """
        self._write(
            {
                "event": "identification",
                "track_id": track_id,
                "identity": identity,
                "similarity": round(float(similarity), 6),
                "margin": round(float(margin), 6),
                "threshold": round(float(threshold), 6),
                "accepted": identity != "UNKNOWN",
                "frame_number": frame_number,
                "media_seconds": media_seconds,
                "embeddings_fused": embeddings_fused,
                **extra,
            }
        )

    def run_started(self, config_digest: Dict[str, Any], **extra: Any) -> None:
        """Record the start of a processing run and the settings in force.

        Args:
            config_digest: Summary of the configuration used.
            **extra: Additional fields merged into the record.
        """
        self._write({"event": "run_started", "config": config_digest, **extra})

    def run_finished(
        self, frames: int, tracks: int, identifications: int, **extra: Any
    ) -> None:
        """Record the completion of a processing run.

        Args:
            frames: Frames processed.
            tracks: Tracks created.
            identifications: Identification decisions made.
            **extra: Additional fields merged into the record.
        """
        self._write(
            {
                "event": "run_finished",
                "frames": frames,
                "tracks": tracks,
                "identifications": identifications,
                **extra,
            }
        )

    def read_all(self) -> list[Dict[str, Any]]:
        """Parse every record written so far.

        Returns:
            All events in write order; empty when the file does not exist.
        """
        if not self._path.is_file():
            return []
        with self._path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]