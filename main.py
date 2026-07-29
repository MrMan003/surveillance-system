#!/usr/bin/env python3
"""Command-line entry point for the surveillance pipeline.

Examples:
    Detect, track and render, with no identification::

        python main.py --input footage.mkv --output outputs/annotated.mp4

    Identify against an enrolled gallery::

        python main.py --input footage.mkv --gallery datasets/gallery \\
            --output outputs/annotated.mp4 --manifest outputs/run.json

    Quick smoke test over the first 100 frames::

        python main.py --input footage.mkv --max-frames 100 --no-render

    Override any configuration field without editing YAML::

        SURV_DETECTION__BODY_CONF=0.5 python main.py --input footage.mkv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import ConfigError, SurveillanceConfig  # noqa: E402
from pipeline import SurveillancePipeline  # noqa: E402
from search.gallery import Gallery, GalleryError  # noqa: E402
from utils.log import AuditLogger, get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_argument_group("input")
    source.add_argument("--input", "-i", type=Path, required=True, help="source video")
    source.add_argument("--config", "-c", type=Path, help="YAML or JSON configuration")
    source.add_argument("--gallery", "-g", type=Path, help="enrolled gallery directory")

    output = parser.add_argument_group("output")
    output.add_argument("--output", "-o", type=Path, help="annotated video destination")
    output.add_argument("--manifest", "-m", type=Path, help="run manifest destination")
    output.add_argument(
        "--no-render",
        action="store_true",
        help="skip rendering; roughly halves wall clock when only the manifest is wanted",
    )

    limits = parser.add_argument_group("limits")
    limits.add_argument("--max-frames", type=int, help="stop after this many frames")
    limits.add_argument("--stride", type=int, help="process every Nth frame")
    limits.add_argument("--start", type=float, help="start offset in seconds")
    limits.add_argument("--end", type=float, help="stop time in seconds")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), help="compute device")
    runtime.add_argument("--seed", type=int, help="random seed")
    runtime.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    runtime.add_argument(
        "--no-audit", action="store_true", help="disable the identification audit log"
    )
    runtime.add_argument(
        "--allow-vfr-estimate",
        action="store_true",
        help=(
            "accept frames without a container timestamp. Disables the forensic "
            "guarantee: timing becomes approximate for any frame the muxer did "
            "not timestamp."
        ),
    )
    return parser


def load_configuration(args: argparse.Namespace) -> SurveillanceConfig:
    """Build the configuration from file, environment and command line.

    Precedence runs file, then environment, then command line, so an explicit
    flag always wins over an inherited setting.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the resulting configuration is invalid.
    """
    if args.config is not None:
        config = SurveillanceConfig.from_file(args.config)
    else:
        default = REPO_ROOT / "configs" / "default.yaml"
        config = (
            SurveillanceConfig.from_file(default)
            if default.is_file()
            else SurveillanceConfig.default()
        )

    config = config.apply_env_overrides()

    config.video.input_path = args.input
    if args.max_frames is not None:
        config.video.max_frames = args.max_frames
    if args.stride is not None:
        config.video.stride = args.stride
    if args.start is not None:
        config.video.start_seconds = args.start
    if args.end is not None:
        config.video.end_seconds = args.end
    if args.allow_vfr_estimate:
        config.video.strict_vfr = False
    if args.device is not None:
        config.runtime.device = args.device
    if args.seed is not None:
        config.runtime.seed = args.seed
    config.runtime.log_level = args.log_level

    # Re-validate: the assignments above bypass __post_init__.
    return SurveillanceConfig.from_dict(config.to_dict()).validate()


def load_gallery(directory: Path, config: SurveillanceConfig) -> Gallery:
    """Load and verify an enrolled gallery.

    Args:
        directory: Directory written by :meth:`Gallery.save`.
        config: The active configuration.

    Returns:
        The loaded, built gallery.

    Raises:
        GalleryError: If the gallery is missing, inconsistent, or lacks a
            consent manifest while one is required.
    """
    gallery = Gallery.load(directory, config.search, config.governance)
    gallery.verify_consent(directory)
    LOGGER.info(
        "Gallery: %d entries covering %d identities",
        gallery.size,
        len(gallery.identities),
    )
    return gallery


def main(argv: Optional[list] = None) -> int:
    """Parse arguments and run the pipeline.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        config = load_configuration(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    config.paths.ensure()
    setup_logging(level=config.runtime.log_level, log_dir=config.paths.log_dir)
    device = config.runtime.apply()

    if not args.input.is_file():
        LOGGER.error("Input not found: %s", args.input)
        return 2

    if args.allow_vfr_estimate:
        LOGGER.warning(
            "strict_vfr is disabled. Frames without a container timestamp will "
            "be dropped rather than timed, so the output timeline may be "
            "incomplete. Do not rely on this run for claims about timing."
        )

    LOGGER.info("Device: %s", device)
    LOGGER.info("%s", config.summary())

    gallery = None
    if args.gallery is not None:
        try:
            gallery = load_gallery(args.gallery, config)
        except GalleryError as exc:
            LOGGER.error("Gallery error: %s", exc)
            return 2
    else:
        LOGGER.warning(
            "No gallery supplied; running detection, tracking and rendering "
            "without identification"
        )

    output_video: Optional[Path] = None
    if not args.no_render:
        output_video = args.output or (
            config.paths.outputs_dir / config.rendering.output_name
        )

    audit = None
    if config.governance.audit_log and not args.no_audit:
        audit = AuditLogger(
            config.paths.log_dir / config.governance.audit_log_name,
            enabled=True,
            source=str(args.input),
        )

    pipeline = SurveillancePipeline(config, gallery)
    try:
        result = pipeline.run(args.input, output_video, audit)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; partial output may be incomplete")
        return 130
    except Exception as exc:  # noqa: BLE001 - top level must report, not traceback
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1

    print(pipeline.profiler.report())
    LOGGER.info(
        "Processed %d frame(s), %d track(s), %d identified, %d unknown",
        result.frames_processed,
        result.tracks_created,
        result.identified_count,
        len(result.identifications) - result.identified_count,
    )

    manifest = args.manifest or (config.paths.outputs_dir / "run_manifest.json")
    result.save_manifest(manifest)

    if output_video is not None:
        LOGGER.info("Annotated video: %s", output_video)
    if audit is not None:
        LOGGER.info("Audit log: %s (%d record(s))", audit.path, audit.count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())