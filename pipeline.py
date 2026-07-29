"""End-to-end orchestration of the surveillance pipeline.

This module owns sequencing and nothing else.  Every algorithm lives in its own
package; the pipeline's job is to move data between them, decide when a track
is ready to identify, and keep memory bounded.  If a change here requires
understanding how OC-SORT works, something has leaked.

Frame ordering
--------------
Frames are decoded and processed strictly in presentation order.  That is not
merely convenient: the tracker's motion model, the writer's monotonic timestamp
requirement and the audit log's chronology all assume it.

Identification timing
---------------------
A track is identified once it accumulates ``fusion.min_samples`` good
embeddings, and then re-identified as it accumulates more, up to a cap.  The
alternative -- identifying once and never revisiting -- locks in a decision
made from the fewest observations the system would accept, which is exactly
when it is least reliable.

Memory
------
Fusion buffers are dropped when their track dies.  Without that, memory grows
with the number of tracks *ever seen* rather than the number currently alive,
which on an hour of busy footage is thousands of stale 512-dimensional buffers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from alignment.umeyama import FaceAligner
from association.face_body import FaceBodyAssociator
from configs.config import SurveillanceConfig
from datasets.frame_stream import FrameStream, StreamInfo, TimedFrame, probe
from detection.combined import CombinedDetector
from recognition.encoder import FaceEncoder, build_encoder
from recognition.fusion import TemporalFusion
from recognition.quality import QualityGate
from rendering.annotator import Annotator, FrameAnnotation
from rendering.writer import AnnotatedVideoWriter
from search.gallery import Gallery
from search.open_set import Identification, OpenSetIdentifier
from tracking.ocsort import OCSort
from utils.log import AuditLogger, get_logger
from utils.profiling import PipelineProfiler

__all__ = ["PipelineResult", "SurveillancePipeline"]

LOGGER = get_logger(__name__)

#: Re-identify a track each time it gains this many new embeddings.
_REIDENTIFY_EVERY = 5


@dataclass
class PipelineResult:
    """Outcome of one processing run.

    Attributes:
        source: Path to the input video.
        output_video: Path to the rendered output, when rendering was enabled.
        frames_processed: Frames decoded and processed.
        tracks_created: Tracks created over the run.
        identifications: Final decision per track.
        stream_info: Metadata for the source stream.
        profile: Profiling summary.
    """

    source: Path
    output_video: Optional[Path]
    frames_processed: int
    tracks_created: int
    identifications: Dict[str, Identification] = field(default_factory=dict)
    stream_info: Optional[StreamInfo] = None
    profile: Dict[str, object] = field(default_factory=dict)

    @property
    def identified_count(self) -> int:
        """Tracks resolved to a named identity."""
        return sum(1 for i in self.identifications.values() if i.accepted)

    def to_dict(self) -> Dict[str, object]:
        """Serialise the result as a run manifest.

        Returns:
            A JSON-friendly mapping.
        """
        return {
            "source": str(self.source),
            "output_video": str(self.output_video) if self.output_video else None,
            "frames_processed": self.frames_processed,
            "tracks_created": self.tracks_created,
            "tracks_identified": self.identified_count,
            "tracks_unknown": len(self.identifications) - self.identified_count,
            "stream": {
                "codec": self.stream_info.codec if self.stream_info else None,
                "resolution": (
                    f"{self.stream_info.width}x{self.stream_info.height}"
                    if self.stream_info
                    else None
                ),
                "time_base": str(self.stream_info.time_base) if self.stream_info else None,
                "duration_s": self.stream_info.duration_seconds if self.stream_info else None,
            },
            "identifications": [i.to_dict() for i in self.identifications.values()],
            "profile": self.profile,
        }

    def save_manifest(self, path: Path) -> Path:
        """Write the run manifest to disk.

        Args:
            path: Destination JSON file.

        Returns:
            The resolved destination.
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        LOGGER.info("Wrote manifest to %s", path)
        return path


class SurveillancePipeline:
    """Runs the full detect-track-recognise pipeline over a video.

    Args:
        config: Complete pipeline configuration.
        gallery: Enrolled identities.  When ``None``, the pipeline detects,
            tracks and renders but performs no identification, which is the
            right mode for enrolment capture and for tuning the earlier stages.
    """

    def __init__(self, config: SurveillanceConfig, gallery: Optional[Gallery] = None) -> None:
        self._config = config
        self._gallery = gallery

        self._detector = CombinedDetector(config.detection, config.runtime, config.paths)
        self._tracker = OCSort(config.tracking)
        self._associator = FaceBodyAssociator(config.association)
        self._aligner = FaceAligner(config.alignment)
        self._encoder: FaceEncoder = build_encoder(
            config.recognition, config.runtime, config.paths
        )
        self._quality = QualityGate(config.quality)
        self._fusion = TemporalFusion(config.fusion)
        self._annotator = Annotator(config.rendering, config.search.unknown_label)

        self._identifier: Optional[OpenSetIdentifier] = (
            OpenSetIdentifier(gallery, config.search) if gallery is not None else None
        )

        self._identifications: Dict[str, Identification] = {}
        self._last_identified_at: Dict[str, int] = {}
        self._profiler = PipelineProfiler(
            enabled=config.runtime.profile, device=config.runtime.resolve_device()
        )

    # -- properties -------------------------------------------------------- #
    @property
    def profiler(self) -> PipelineProfiler:
        """The run profiler."""
        return self._profiler

    # -- main entry point -------------------------------------------------- #
    def run(
        self,
        source: Path,
        output_video: Optional[Path] = None,
        audit: Optional[AuditLogger] = None,
    ) -> PipelineResult:
        """Process a video end to end.

        Args:
            source: Input container.
            output_video: Destination for the annotated render; ``None``
                disables rendering, which roughly halves wall clock when only
                the manifest is wanted.
            audit: Audit logger for identification decisions.

        Returns:
            The run result, including the profiling summary.
        """
        source = Path(source).expanduser().resolve()
        info = probe(source, self._config.video.stream_index)
        LOGGER.info("Source: %s", info.summary().replace("\n", "\n  "))

        video_config = self._config.video
        if video_config.input_path != source:
            video_config.input_path = source

        if self._identifier is not None and self._config.search.calibrate:
            self._identifier.calibrate()

        if audit is not None:
            audit.run_started(
                {
                    "source": str(source),
                    "encoder": self._encoder.name,
                    "threshold": self._identifier.threshold if self._identifier else None,
                    "gallery_size": self._gallery.size if self._gallery else 0,
                }
            )

        writer: Optional[AnnotatedVideoWriter] = None
        self._profiler.start()

        # Model loading and warmup are one-time costs. Attributing them to a
        # named stage keeps them out of "unaccounted", where they would look
        # like unexplained overhead on a short run and vanish on a long one.
        with self._profiler.stage("startup"):
            self._detector.load()
            self._encoder.load()
            self._detector.warmup(iterations=1)

        with self._detector as detector, self._encoder as encoder:

            with FrameStream(video_config) as stream:
                if output_video is not None:
                    writer = AnnotatedVideoWriter.from_stream_info(
                        output_video, self._config.rendering, info
                    ).open()

                try:
                    # The decode itself must be timed. Iterating the stream
                    # directly would leave demux and decode outside every
                    # stage, showing up only as unaccounted wall clock.
                    iterator = iter(stream)
                    while True:
                        with self._profiler.stage("decode"):
                            timed = next(iterator, None)
                        if timed is None:
                            break
                        self._process_frame(timed, detector, encoder, writer, audit)
                        self._profiler.count_frame()
                finally:
                    if writer is not None:
                        writer.close()

        self._profiler.finish()
        self._finalise(audit)

        result = PipelineResult(
            source=source,
            output_video=output_video,
            frames_processed=self._profiler.frames,
            tracks_created=self._tracker.total_created,
            identifications=dict(self._identifications),
            stream_info=info,
            profile=self._profiler.summary(),
        )

        if audit is not None:
            audit.run_finished(
                frames=result.frames_processed,
                tracks=result.tracks_created,
                identifications=result.identified_count,
            )
        return result

    # -- per-frame work ---------------------------------------------------- #
    def _process_frame(
        self,
        timed: TimedFrame,
        detector: CombinedDetector,
        encoder: FaceEncoder,
        writer: Optional[AnnotatedVideoWriter],
        audit: Optional[AuditLogger],
    ) -> None:
        """Run every stage over one frame.

        Args:
            timed: The decoded frame with its container timing.
            detector: The loaded combined detector.
            encoder: The loaded face encoder.
            writer: Output writer, or ``None`` when rendering is disabled.
            audit: Audit logger, or ``None``.
        """
        height, width = timed.frame.shape[:2]

        with self._profiler.stage("detect"):
            detections = detector.detect(timed.frame, timed.frame_number)

        with self._profiler.stage("track"):
            tracks = self._tracker.update(detections.bodies, timed.frame_number)

        with self._profiler.stage("associate"):
            association = self._associator.associate(
                tracks, detections.faces, timed.frame_number
            )

        with self._profiler.stage("align", items=len(association.associations)):
            aligned = [
                (pairing, self._aligner.align(timed.frame, pairing.face))
                for pairing in association.associations
            ]
            usable = [(pairing, crop) for pairing, crop in aligned if crop is not None]

        if usable:
            with self._profiler.stage("encode", items=len(usable)):
                embeddings = encoder.encode(
                    [crop.image for _, crop in usable],
                    [pairing.face.score for pairing, _ in usable],
                )

            with self._profiler.stage("quality", items=len(usable)):
                for (pairing, crop), embedding in zip(usable, embeddings):
                    truncation = pairing.face.box.truncation(width, height)
                    verdict = self._quality.assess(embedding, crop, truncation)
                    if verdict.passed:
                        self._fusion.add(
                            pairing.track_id,
                            embedding,
                            self._fusion.apply_weighting(verdict.score),
                        )

        if self._identifier is not None:
            with self._profiler.stage("identify"):
                self._identify_ready_tracks(tracks, timed, audit)

        if writer is not None:
            with self._profiler.stage("render"):
                canvas = self._annotator.draw(
                    FrameAnnotation(
                        timed=timed,
                        tracks=tracks,
                        faces=detections.faces,
                        associations=association.associations,
                        identifications=self._identifications,
                    )
                )
            with self._profiler.stage("write"):
                writer.write(canvas, timed)

        self._release_dead_tracks()

    # -- identification ---------------------------------------------------- #
    def _identify_ready_tracks(
        self,
        tracks: Sequence[object],
        timed: TimedFrame,
        audit: Optional[AuditLogger],
    ) -> None:
        """Identify or re-identify tracks that have accumulated enough evidence.

        Args:
            tracks: Confirmed tracks for this frame.
            timed: The current frame, for audit timestamps.
            audit: Audit logger, or ``None``.
        """
        assert self._identifier is not None

        for track in tracks:
            track_id = getattr(track, "track_id")
            samples = self._fusion.sample_count(track_id)
            if samples < self._config.fusion.min_samples:
                continue

            previous = self._last_identified_at.get(track_id, 0)
            if track_id in self._identifications and samples - previous < _REIDENTIFY_EVERY:
                continue

            fused = self._fusion.fuse(track_id)
            if fused is None:
                continue

            decision = self._identifier.identify(fused)
            self._identifications[track_id] = decision
            self._last_identified_at[track_id] = samples

            if audit is not None:
                audit.identification(
                    track_id=track_id,
                    identity=decision.identity,
                    similarity=decision.similarity,
                    margin=0.0 if np.isinf(decision.margin) else decision.margin,
                    threshold=decision.threshold,
                    frame_number=timed.frame_number,
                    media_seconds=float(timed.seconds),
                    embeddings_fused=fused.sample_count,
                    coherence=round(fused.coherence, 4),
                    rejection_reason=decision.rejection_reason,
                )

    # -- housekeeping ------------------------------------------------------ #
    def _release_dead_tracks(self) -> None:
        """Drop fusion buffers for tracks the tracker has removed.

        Memory must scale with live tracks, not with tracks ever seen.
        """
        live = {track.track_id for track in self._tracker.tracks}
        for track_id in list(self._fusion.ready_tracks()):
            if track_id not in live:
                self._fusion.drop(track_id)

    def _finalise(self, audit: Optional[AuditLogger]) -> None:
        """Identify any track that never reached a re-identification point.

        A track that ends shortly after becoming eligible would otherwise leave
        the run with no decision recorded at all.

        Args:
            audit: Audit logger, or ``None``.
        """
        if self._identifier is None:
            return

        for track_id in self._fusion.ready_tracks():
            if track_id in self._identifications:
                continue
            fused = self._fusion.fuse(track_id)
            if fused is None:
                continue
            decision = self._identifier.identify(fused)
            self._identifications[track_id] = decision
            if audit is not None:
                audit.identification(
                    track_id=track_id,
                    identity=decision.identity,
                    similarity=decision.similarity,
                    margin=0.0 if np.isinf(decision.margin) else decision.margin,
                    threshold=decision.threshold,
                    frame_number=-1,
                    embeddings_fused=fused.sample_count,
                    finalised=True,
                )

    # -- diagnostics ------------------------------------------------------- #
    def statistics(self) -> Dict[str, object]:
        """Collect statistics from every stage.

        Returns:
            A mapping of per-stage summaries.
        """
        return {
            "detector": self._detector.stats_summary(),
            "tracker": self._tracker.summary(),
            "quality": self._quality.statistics(),
            "fusion": self._fusion.statistics(),
            "identifier": self._identifier.statistics() if self._identifier else None,
        }