"""Shared pytest fixtures.

Sample clips are generated into a temporary directory rather than committed.
``.gitignore`` excludes ``*.mp4`` precisely because rendered video may contain
biometric data, and a test fixture should not be the exception that trains
people to commit video.  Generating them also means the tests assert against a
schedule this repository controls, not against whatever a particular camera
happened to emit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("av", reason="PyAV is required for video fixtures")

from scripts.make_test_video import write_clip  # noqa: E402


@pytest.fixture(scope="session")
def sample_clips(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Generate one clip per timing mode, once per test session.

    Args:
        tmp_path_factory: Pytest's session-scoped temporary directory factory.

    Returns:
        Mapping of mode name (``cfr``, ``vfr``, ``gap``) to the clip path.
    """
    directory = tmp_path_factory.mktemp("clips")
    return {
        mode: write_clip(directory / f"{mode}.mp4", mode, 4.0, 320, 240)
        for mode in ("cfr", "vfr", "gap")
    }


@pytest.fixture(scope="session")
def cfr_clip(sample_clips: Dict[str, Path]) -> Path:
    """Path to the constant-frame-rate control clip."""
    return sample_clips["cfr"]


@pytest.fixture(scope="session")
def vfr_clip(sample_clips: Dict[str, Path]) -> Path:
    """Path to the variable-frame-rate clip."""
    return sample_clips["vfr"]


@pytest.fixture(scope="session")
def gap_clip(sample_clips: Dict[str, Path]) -> Path:
    """Path to the clip containing a recorder stop/resume gap."""
    return sample_clips["gap"]