#!/usr/bin/env python3
"""Regenerate ``configs/default.yaml`` from the dataclass defaults.

The YAML file is a *derived artefact*, never hand-edited.  Generating it from
:class:`SurveillanceConfig` guarantees the two can never drift: if someone adds
a field to a config dataclass and forgets to update the YAML, the next run of
this script (or the test that invokes it) surfaces the difference immediately.

Paths are rewritten to repository-relative form before serialisation.  Without
that step the emitted file would hard-code the absolute paths of whichever
machine generated it, which breaks the moment anyone else clones the repo.

Usage:
    python scripts/generate_default_config.py
    python scripts/generate_default_config.py --check   # CI: fail if stale
    python scripts/generate_default_config.py --output configs/other.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Allow execution as a bare script from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import SurveillanceConfig  # noqa: E402

LOGGER = logging.getLogger("generate_default_config")

#: Repository-relative path defaults, substituted for the absolute paths that
#: ``PathConfig.__post_init__`` produces on the generating machine.
RELATIVE_PATHS: Dict[str, str] = {
    "root": ".",
    "weights_dir": "weights",
    "outputs_dir": "outputs",
    "cache_dir": ".cache",
    "log_dir": "outputs/logs",
    "gallery_dir": "datasets/gallery",
}

HEADER = """\
# Default configuration for the surveillance system.
#
# GENERATED FILE -- do not edit by hand.
# Regenerate with:  python scripts/generate_default_config.py
#
# Every key maps 1:1 to a field on a dataclass in configs/config.py. Override
# any value at runtime without editing this file:
#
#   SURV_DETECTION__BODY_CONF=0.5 SURV_RUNTIME__DEVICE=cuda:0 python main.py

"""


def build_payload() -> Dict[str, Any]:
    """Serialise the default configuration with portable, relative paths.

    Returns:
        A nested mapping ready for ``yaml.safe_dump``.
    """
    payload = SurveillanceConfig.default().validate().to_dict()
    payload["paths"] = dict(RELATIVE_PATHS)
    return payload


def render(payload: Dict[str, Any]) -> str:
    """Render the payload as a commented YAML document.

    Args:
        payload: Nested configuration mapping.

    Returns:
        The complete file contents, header included.

    Raises:
        SystemExit: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        LOGGER.error("PyYAML is required: pip install PyYAML")
        raise SystemExit(2) from None

    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return HEADER + body


def main() -> int:
    """Parse arguments and write or verify the generated configuration.

    Returns:
        Process exit code: ``0`` on success, ``1`` when ``--check`` finds the
        file stale or missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="destination file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the file is up to date without writing; exit 1 if stale",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    content = render(build_payload())

    if args.check:
        if not args.output.is_file():
            LOGGER.error("%s does not exist", args.output)
            return 1
        if args.output.read_text(encoding="utf-8") != content:
            LOGGER.error("%s is stale; rerun without --check", args.output)
            return 1
        LOGGER.info("%s is up to date", args.output)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    LOGGER.info("Wrote %s (%d bytes)", args.output, len(content))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())