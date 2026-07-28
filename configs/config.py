"""Central, strongly-typed configuration for the surveillance system.

This module is the single source of truth for every tunable parameter in the
pipeline.  It is deliberately dependency-light: only the standard library is
required at import time.  ``torch``, ``numpy`` and ``yaml`` are imported lazily
inside the functions that need them so that this module can be imported by
tooling (docs, CI linters, unit tests) on machines without a GPU stack.

Design notes
------------
* Every stage of the pipeline owns exactly one dataclass.  Adding a parameter
  never requires touching an unrelated stage (Open/Closed principle).
* ``SurveillanceConfig`` is a pure aggregate; it holds no behaviour beyond
  construction, validation and serialisation (Single Responsibility).
* Construction is total: ``SurveillanceConfig()`` yields a fully valid,
  runnable configuration tuned for a 16 GB Tesla T4.
* Validation happens in ``__post_init__`` of each node, so an invalid config
  can never be constructed -- failures are loud and immediate rather than
  surfacing 40 minutes into a decode job.

Example
-------
>>> from configs.config import SurveillanceConfig
>>> cfg = SurveillanceConfig.default()
>>> cfg.runtime.resolve_device()
'cpu'
>>> cfg.recognition.embedding_dim
512
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = [
    "ConfigError",
    "FusionStrategy",
    "IndexType",
    "InterpolationMode",
    "PathConfig",
    "VideoConfig",
    "DetectionConfig",
    "TrackingConfig",
    "AssociationConfig",
    "AlignmentConfig",
    "RecognitionConfig",
    "QualityConfig",
    "FusionConfig",
    "SearchConfig",
    "RenderingConfig",
    "RuntimeConfig",
    "GovernanceConfig",
    "SurveillanceConfig",
    "get_config",
    "set_config",
]

LOGGER = logging.getLogger(__name__)

#: Prefix for environment-variable overrides, e.g. ``SURV_DETECTION__BODY_CONF=0.4``
ENV_PREFIX = "SURV_"
#: Separator between the config section and the field name in env overrides.
ENV_SECTION_SEP = "__"

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration value is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class FusionStrategy(str, Enum):
    """Strategy used to collapse many per-frame embeddings into one per track."""

    WEIGHTED_MEAN = "weighted_mean"
    EMA = "ema"
    MEDIAN = "median"


class IndexType(str, Enum):
    """FAISS index families supported by the gallery search module."""

    FLAT_IP = "IndexFlatIP"
    FLAT_L2 = "IndexFlatL2"
    IVF_FLAT_IP = "IndexIVFFlatIP"


class InterpolationMode(str, Enum):
    """OpenCV interpolation flags, expressed symbolically for serialisation."""

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS4 = "lanczos4"

    def to_cv2(self) -> int:
        """Translate to the corresponding ``cv2.INTER_*`` integer flag.

        Returns:
            The OpenCV interpolation constant.

        Raises:
            ImportError: If OpenCV is not installed.
        """
        import cv2  # local import: keeps this module importable without OpenCV

        mapping = {
            InterpolationMode.NEAREST: cv2.INTER_NEAREST,
            InterpolationMode.LINEAR: cv2.INTER_LINEAR,
            InterpolationMode.CUBIC: cv2.INTER_CUBIC,
            InterpolationMode.LANCZOS4: cv2.INTER_LANCZOS4,
        }
        return mapping[self]


class ThreadType(str, Enum):
    """PyAV codec-context threading models."""

    NONE = "NONE"
    FRAME = "FRAME"
    SLICE = "SLICE"
    AUTO = "AUTO"


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    """Assert a configuration invariant.

    Args:
        condition: Invariant that must hold.
        message: Human-readable description used in the raised error.

    Raises:
        ConfigError: If ``condition`` is falsy.
    """
    if not condition:
        raise ConfigError(message)


def _require_range(value: float, low: float, high: float, name: str) -> None:
    """Assert that ``value`` lies within the inclusive interval ``[low, high]``.

    Args:
        value: Value under test.
        low: Inclusive lower bound.
        high: Inclusive upper bound.
        name: Field name used in the error message.

    Raises:
        ConfigError: If the value falls outside the interval.
    """
    _require(low <= value <= high, f"{name} must be in [{low}, {high}], got {value!r}")


def _require_positive(value: Union[int, float], name: str) -> None:
    """Assert that ``value`` is strictly greater than zero.

    Args:
        value: Value under test.
        name: Field name used in the error message.

    Raises:
        ConfigError: If the value is not strictly positive.
    """
    _require(value > 0, f"{name} must be > 0, got {value!r}")


# --------------------------------------------------------------------------- #
# Section: filesystem layout
# --------------------------------------------------------------------------- #
@dataclass
class PathConfig:
    """Filesystem layout for weights, outputs, caches and logs.

    Attributes:
        root: Repository root.  All other paths default to children of it.
        weights_dir: Destination for downloaded/converted model weights.
        outputs_dir: Destination for rendered video and JSON manifests.
        cache_dir: Scratch space for decoded index files and FAISS shards.
        log_dir: Destination for rotating log files.
        gallery_dir: Enrolled-identity images and serialised gallery indices.
    """

    root: Path = Path(".")
    weights_dir: Path = Path("weights")
    outputs_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    log_dir: Path = Path("outputs/logs")
    gallery_dir: Path = Path("datasets/gallery")

    def __post_init__(self) -> None:
        """Normalise every path to an absolute ``Path`` rooted at ``root``."""
        self.root = Path(self.root).expanduser().resolve()
        for f in fields(self):
            if f.name == "root":
                continue
            value = Path(getattr(self, f.name)).expanduser()
            if not value.is_absolute():
                value = self.root / value
            setattr(self, f.name, value)

    def ensure(self) -> "PathConfig":
        """Create every configured directory if it does not already exist.

        Returns:
            ``self``, to allow fluent chaining.
        """
        for f in fields(self):
            path: Path = getattr(self, f.name)
            path.mkdir(parents=True, exist_ok=True)
            LOGGER.debug("Ensured directory %s", path)
        return self


# --------------------------------------------------------------------------- #
# Section: video decoding (Phase 1)
# --------------------------------------------------------------------------- #
@dataclass
class VideoConfig:
    """Forensic, variable-frame-rate decoding parameters for :class:`FrameStream`.

    Attributes:
        input_path: Source container.  ``None`` until supplied at runtime.
        allowed_suffixes: Container extensions the loader will accept.
        stream_index: Index of the video stream inside the container.
        thread_type: PyAV threading model; ``AUTO`` maximises decode throughput.
        thread_count: Decoder worker threads; ``0`` lets FFmpeg choose.
        strict_vfr: When ``True`` the pipeline refuses to synthesise timestamps
            from ``frame_index / fps`` and raises instead.  Forensic runs must
            keep this enabled.
        start_seconds: Optional seek target in container time.
        end_seconds: Optional stop time in container time.
        max_frames: Optional hard cap on decoded frames, for smoke tests.
        stride: Decode-and-keep every ``stride``-th frame (``1`` keeps all).
        prefetch_depth: Frames buffered ahead of the consumer.
        pixel_format: Target pixel format for converted frames.
        allow_missing_dts: Tolerate ``None`` DTS (common in remuxed MKV).
    """

    input_path: Optional[Path] = None
    allowed_suffixes: Tuple[str, ...] = (".mp4", ".mkv")
    stream_index: int = 0
    thread_type: ThreadType = ThreadType.AUTO
    thread_count: int = 0
    strict_vfr: bool = True
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    max_frames: Optional[int] = None
    stride: int = 1
    prefetch_depth: int = 16
    pixel_format: str = "bgr24"
    allow_missing_dts: bool = True

    def __post_init__(self) -> None:
        """Coerce and validate decoding parameters."""
        if self.input_path is not None:
            self.input_path = Path(self.input_path).expanduser()
            _require(
                self.input_path.suffix.lower() in self.allowed_suffixes,
                f"Unsupported container {self.input_path.suffix!r}; "
                f"allowed: {self.allowed_suffixes}",
            )
        if isinstance(self.thread_type, str):
            self.thread_type = ThreadType(self.thread_type.upper())
        _require(self.stream_index >= 0, "stream_index must be >= 0")
        _require(self.thread_count >= 0, "thread_count must be >= 0")
        _require_positive(self.stride, "stride")
        _require_positive(self.prefetch_depth, "prefetch_depth")
        if self.max_frames is not None:
            _require_positive(self.max_frames, "max_frames")
        if self.start_seconds is not None:
            _require(self.start_seconds >= 0, "start_seconds must be >= 0")
        if self.start_seconds is not None and self.end_seconds is not None:
            _require(
                self.end_seconds > self.start_seconds,
                "end_seconds must be strictly greater than start_seconds",
            )
            
# --------------------------------------------------------------------------- #
# Section: detection (Phase 2)
# --------------------------------------------------------------------------- #
@dataclass
class DetectionConfig:
    """Body (YOLOv8s) and face (SCRFD) detector settings.

    Attributes:
        body_weights: Path or Ultralytics alias for the person detector.
        body_conf: Confidence floor for person boxes.
        body_iou: IoU threshold used by YOLO's internal NMS.
        body_imgsz: Square inference resolution for YOLO.
        body_classes: COCO class ids to retain (``0`` is ``person``).
        body_max_det: Upper bound on detections per frame.
        face_model_name: InsightFace model pack containing the SCRFD detector.
        face_det_size: SCRFD input resolution (width, height).
        face_conf: Confidence floor for face boxes.
        face_nms: IoU threshold for face NMS.
        min_body_area: Reject person boxes smaller than this many pixels².
        min_face_size: Reject faces whose shorter side is below this in pixels.
        batch_size: Frames per detector forward pass.
        half: Run detectors in FP16 when CUDA is available.
    """

    body_weights: str = "yolov8s.pt"
    body_conf: float = 0.35
    body_iou: float = 0.55
    body_imgsz: int = 640
    body_classes: Tuple[int, ...] = (0,)
    body_max_det: int = 300

    face_model_name: str = "buffalo_l"
    face_det_size: Tuple[int, int] = (640, 640)
    face_conf: float = 0.50
    face_nms: float = 0.40

    min_body_area: int = 512
    min_face_size: int = 24

    batch_size: int = 8
    half: bool = True

    def __post_init__(self) -> None:
        """Validate detector thresholds and geometry."""
        _require_range(self.body_conf, 0.0, 1.0, "body_conf")
        _require_range(self.body_iou, 0.0, 1.0, "body_iou")
        _require_range(self.face_conf, 0.0, 1.0, "face_conf")
        _require_range(self.face_nms, 0.0, 1.0, "face_nms")
        _require(self.body_imgsz % 32 == 0, "body_imgsz must be a multiple of 32")
        _require_positive(self.body_max_det, "body_max_det")
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.min_face_size, "min_face_size")
        _require(len(self.face_det_size) == 2, "face_det_size must be (w, h)")
        _require(
            all(v % 32 == 0 for v in self.face_det_size),
            "face_det_size components must be multiples of 32",
        )
        _require(len(self.body_classes) > 0, "body_classes must not be empty")
        self.body_classes = tuple(int(c) for c in self.body_classes)
        self.face_det_size = (int(self.face_det_size[0]), int(self.face_det_size[1]))


# --------------------------------------------------------------------------- #
# Section: tracking (Phase 3)
# --------------------------------------------------------------------------- #
@dataclass
class TrackingConfig:
    """OC-SORT hyper-parameters.

    Attributes:
        det_threshold: Confidence floor for detections entering the tracker.
        max_age: Frames a track may go unmatched before deletion.
        min_hits: Consecutive hits required before a track is published.
        iou_threshold: IoU floor for the primary association stage.
        delta_t: Temporal window (frames) for OC-SORT's observation-centric
            momentum term.
        inertia: Weight of the velocity-direction consistency cost.
        use_byte: Enable the low-confidence second association pass.
        low_threshold: Confidence band floor for the BYTE second pass.
        max_trajectory: Number of centroids retained per track for rendering.
        reid_enabled: Reserve appearance slots for re-identification.
    """

    det_threshold: float = 0.35
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.30
    delta_t: int = 3
    inertia: float = 0.20
    use_byte: bool = True
    low_threshold: float = 0.10
    max_trajectory: int = 64
    reid_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate tracker thresholds and lifecycle bounds."""
        _require_range(self.det_threshold, 0.0, 1.0, "det_threshold")
        _require_range(self.iou_threshold, 0.0, 1.0, "iou_threshold")
        _require_range(self.inertia, 0.0, 1.0, "inertia")
        _require_range(self.low_threshold, 0.0, 1.0, "low_threshold")
        _require(
            self.low_threshold < self.det_threshold,
            "low_threshold must be below det_threshold for BYTE association",
        )
        _require_positive(self.max_age, "max_age")
        _require_positive(self.min_hits, "min_hits")
        _require_positive(self.delta_t, "delta_t")
        _require_positive(self.max_trajectory, "max_trajectory")


# --------------------------------------------------------------------------- #
# Section: face-body association (Phase 4)
# --------------------------------------------------------------------------- #
@dataclass
class AssociationConfig:
    """Geometric face-to-body assignment parameters.

    Attributes:
        containment_threshold: Minimum fraction of the face box that must lie
            inside a body box for the pair to be assignable.
        head_region_ratio: Fraction of the body box height, measured from the
            top, treated as the plausible head region.
        centre_weight: Weight of the horizontal-centre alignment cost term.
        containment_weight: Weight of the containment cost term.
        scale_weight: Weight of the face/body area-ratio plausibility term.
        expected_face_ratio: Prior on ``face_area / body_area`` for an upright
            full-body detection.
        max_cost: Assignments above this cost are rejected after the Hungarian
            solve, leaving the face unassociated.
    """

    containment_threshold: float = 0.60
    head_region_ratio: float = 0.45
    centre_weight: float = 0.30
    containment_weight: float = 0.50
    scale_weight: float = 0.20
    expected_face_ratio: float = 0.035
    max_cost: float = 0.65

    def __post_init__(self) -> None:
        """Validate cost weights and geometric priors."""
        _require_range(self.containment_threshold, 0.0, 1.0, "containment_threshold")
        _require_range(self.head_region_ratio, 0.0, 1.0, "head_region_ratio")
        _require_range(self.max_cost, 0.0, 1.0, "max_cost")
        _require_positive(self.expected_face_ratio, "expected_face_ratio")
        total = self.centre_weight + self.containment_weight + self.scale_weight
        _require(
            abs(total - 1.0) < 1e-6,
            f"association cost weights must sum to 1.0, got {total:.6f}",
        )


# --------------------------------------------------------------------------- #
# Section: alignment (Phase 5)
# --------------------------------------------------------------------------- #
#: Canonical ArcFace/InsightFace 5-point template for a 112x112 crop.
ARCFACE_TEMPLATE_112: Tuple[Tuple[float, float], ...] = (
    (38.2946, 51.6963),  # left eye
    (73.5318, 51.5014),  # right eye
    (56.0252, 71.7366),  # nose tip
    (41.5493, 92.3655),  # left mouth corner
    (70.7299, 92.2041),  # right mouth corner
)


@dataclass
class AlignmentConfig:
    """Umeyama similarity-transform alignment settings.

    Attributes:
        output_size: Aligned crop resolution (width, height).
        reference_landmarks: Canonical 5-point template in output coordinates.
        template_size: Resolution the template was authored for; the template
            is rescaled if ``output_size`` differs.
        padding_ratio: Isotropic outward scale applied to the crop, useful for
            models trained with looser margins.
        interpolation: Resampling kernel for the affine warp.
        border_value: Constant fill colour for regions outside the source.
        max_roll_degrees: Reject faces whose estimated in-plane roll exceeds
            this magnitude; ``None`` disables the check.
    """

    output_size: Tuple[int, int] = (112, 112)
    reference_landmarks: Tuple[Tuple[float, float], ...] = ARCFACE_TEMPLATE_112
    template_size: Tuple[int, int] = (112, 112)
    padding_ratio: float = 0.0
    interpolation: InterpolationMode = InterpolationMode.LINEAR
    border_value: Tuple[int, int, int] = (0, 0, 0)
    max_roll_degrees: Optional[float] = 45.0

    def __post_init__(self) -> None:
        """Validate crop geometry and the landmark template."""
        if isinstance(self.interpolation, str):
            self.interpolation = InterpolationMode(self.interpolation.lower())
        _require(len(self.output_size) == 2, "output_size must be (w, h)")
        _require(len(self.template_size) == 2, "template_size must be (w, h)")
        self.output_size = (int(self.output_size[0]), int(self.output_size[1]))
        self.template_size = (int(self.template_size[0]), int(self.template_size[1]))
        _require_positive(self.output_size[0], "output_size[0]")
        _require_positive(self.output_size[1], "output_size[1]")
        _require(
            len(self.reference_landmarks) == 5,
            "reference_landmarks must contain exactly 5 points",
        )
        self.reference_landmarks = tuple(
            (float(x), float(y)) for x, y in self.reference_landmarks
        )
        _require(self.padding_ratio >= 0.0, "padding_ratio must be >= 0")
        if self.max_roll_degrees is not None:
            _require_range(self.max_roll_degrees, 0.0, 90.0, "max_roll_degrees")

    def scaled_template(self) -> Tuple[Tuple[float, float], ...]:
        """Return the landmark template rescaled to ``output_size``.

        The canonical ArcFace template is authored for 112x112.  When a
        different crop size is requested the template is scaled anisotropically
        and then shrunk by ``padding_ratio`` about the crop centre.

        Returns:
            Five ``(x, y)`` pairs in output-crop pixel coordinates.
        """
        sx = self.output_size[0] / self.template_size[0]
        sy = self.output_size[1] / self.template_size[1]
        cx = self.output_size[0] / 2.0
        cy = self.output_size[1] / 2.0
        shrink = 1.0 / (1.0 + self.padding_ratio)
        return tuple(
            (
                cx + (x * sx - cx) * shrink,
                cy + (y * sy - cy) * shrink,
            )
            for x, y in self.reference_landmarks
        )
        
# --------------------------------------------------------------------------- #
# Section: recognition (Phase 6)
# --------------------------------------------------------------------------- #
@dataclass
class RecognitionConfig:
    """AdaFace embedding-extractor settings.

    Attributes:
        architecture: Backbone identifier (``ir18`` or ``ir50``).
        weights: Checkpoint filename resolved against ``PathConfig.weights_dir``.
        embedding_dim: Dimensionality of the L2-normalised output embedding.
        batch_size: Aligned crops per forward pass.
        amp: Enable ``torch.autocast`` mixed precision on CUDA.
        input_mean: Per-channel mean applied after scaling to ``[0, 1]``.
        input_std: Per-channel standard deviation.
        input_bgr: AdaFace checkpoints expect BGR channel order.
        normalize: L2-normalise the emitted embedding.
        flip_tta: Average the embedding with its horizontally flipped twin.
    """

    architecture: str = "ir50"
    weights: str = "adaface_ir50_ms1mv2.ckpt"
    embedding_dim: int = 512
    batch_size: int = 64
    amp: bool = True
    input_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_bgr: bool = True
    normalize: bool = True
    flip_tta: bool = False

    #: Backbones this project ships implementations for.
    SUPPORTED_ARCHITECTURES: ClassVar[Tuple[str, ...]] = ("ir18", "ir50")

    def __post_init__(self) -> None:
        """Validate the backbone selection and normalisation statistics."""
        self.architecture = str(self.architecture).lower()
        _require(
            self.architecture in self.SUPPORTED_ARCHITECTURES,
            f"architecture must be one of {self.SUPPORTED_ARCHITECTURES}, "
            f"got {self.architecture!r}",
        )
        _require(self.embedding_dim == 512, "AdaFace emits 512-d embeddings")
        _require_positive(self.batch_size, "batch_size")
        _require(len(self.input_mean) == 3, "input_mean must have 3 channels")
        _require(len(self.input_std) == 3, "input_std must have 3 channels")
        _require(all(s > 0 for s in self.input_std), "input_std entries must be > 0")


# --------------------------------------------------------------------------- #
# Section: quality gating (Phase 7)
# --------------------------------------------------------------------------- #
@dataclass
class QualityConfig:
    """Embedding-quality gate driven by the pre-normalisation feature norm.

    AdaFace's raw feature norm correlates strongly with image quality, which
    makes it a cheap, model-native quality proxy.  Blur and illumination checks
    run on the aligned crop as inexpensive pre-filters.

    Attributes:
        min_norm: Absolute floor on the raw embedding L2 norm.
        max_norm: Ceiling above which a norm is treated as anomalous.
        adaptive: Enable percentile-based thresholding over observed norms.
        adaptive_percentile: Percentile of the running norm distribution used
            as the acceptance floor when ``adaptive`` is enabled.
        warmup_samples: Samples collected before the adaptive threshold engages.
        blur_threshold: Minimum variance of the Laplacian on the aligned crop.
        min_brightness: Minimum mean luminance in ``[0, 255]``.
        max_brightness: Maximum mean luminance in ``[0, 255]``.
        min_detection_score: Detector confidence floor for embedding extraction.
        max_edge_truncation: Maximum fraction of the face box permitted to fall
            outside the frame before the crop is discarded as partial.
        max_stored_per_track: Cap on retained embeddings per track.
    """

    min_norm: float = 18.0
    max_norm: float = 48.0
    adaptive: bool = True
    adaptive_percentile: float = 35.0
    warmup_samples: int = 64
    blur_threshold: float = 45.0
    min_brightness: float = 25.0
    max_brightness: float = 235.0
    min_detection_score: float = 0.60
    max_edge_truncation: float = 0.10
    max_stored_per_track: int = 32

    def __post_init__(self) -> None:
        """Validate quality bounds for internal consistency."""
        _require(self.min_norm < self.max_norm, "min_norm must be < max_norm")
        _require_range(self.adaptive_percentile, 0.0, 100.0, "adaptive_percentile")
        _require_positive(self.warmup_samples, "warmup_samples")
        _require(self.blur_threshold >= 0.0, "blur_threshold must be >= 0")
        _require_range(self.min_brightness, 0.0, 255.0, "min_brightness")
        _require_range(self.max_brightness, 0.0, 255.0, "max_brightness")
        _require(
            self.min_brightness < self.max_brightness,
            "min_brightness must be < max_brightness",
        )
        _require_range(self.min_detection_score, 0.0, 1.0, "min_detection_score")
        _require_range(self.max_edge_truncation, 0.0, 1.0, "max_edge_truncation")
        _require_positive(self.max_stored_per_track, "max_stored_per_track")


# --------------------------------------------------------------------------- #
# Section: temporal fusion (Phase 8)
# --------------------------------------------------------------------------- #
@dataclass
class FusionConfig:
    """Temporal aggregation of per-frame embeddings into one per track.

    Attributes:
        strategy: Aggregation rule.
        ema_alpha: Smoothing factor for the exponential moving average.
        min_samples: Embeddings required before a track is searchable.
        max_samples: Most recent embeddings retained for fusion.
        quality_power: Exponent applied to the quality weight; values above one
            sharpen the preference for high-quality observations.
        renormalize: L2-normalise the fused embedding.
    """

    strategy: FusionStrategy = FusionStrategy.WEIGHTED_MEAN
    ema_alpha: float = 0.30
    min_samples: int = 3
    max_samples: int = 32
    quality_power: float = 2.0
    renormalize: bool = True

    def __post_init__(self) -> None:
        """Validate fusion parameters."""
        if isinstance(self.strategy, str):
            self.strategy = FusionStrategy(self.strategy.lower())
        _require_range(self.ema_alpha, 0.0, 1.0, "ema_alpha")
        _require_positive(self.min_samples, "min_samples")
        _require_positive(self.max_samples, "max_samples")
        _require(
            self.min_samples <= self.max_samples,
            "min_samples must be <= max_samples",
        )
        _require(self.quality_power > 0, "quality_power must be > 0")


# --------------------------------------------------------------------------- #
# Section: open-set search (Phase 9)
# --------------------------------------------------------------------------- #
@dataclass
class SearchConfig:
    """FAISS gallery-search and open-set rejection settings.

    Attributes:
        index_type: FAISS index family.  Inner product over L2-normalised
            vectors is equivalent to cosine similarity.
        use_gpu: Attempt to move the index onto the GPU.
        top_k: Neighbours retrieved per query.
        similarity_threshold: Cosine similarity floor for accepting an identity.
        margin_threshold: Minimum gap between the best and second-best match;
            a small margin indicates an ambiguous, and therefore rejected, hit.
        unknown_label: Label emitted when no candidate clears the thresholds.
        nlist: Number of Voronoi cells for IVF indices.
        nprobe: Cells visited per query for IVF indices.
        calibrate: Estimate the threshold from gallery impostor statistics.
        target_far: Target false-accept rate driving threshold calibration.
    """

    index_type: IndexType = IndexType.FLAT_IP
    use_gpu: bool = False
    top_k: int = 5
    similarity_threshold: float = 0.35
    margin_threshold: float = 0.05
    unknown_label: str = "UNKNOWN"
    nlist: int = 100
    nprobe: int = 16
    calibrate: bool = True
    target_far: float = 1e-3

    def __post_init__(self) -> None:
        """Validate search and rejection parameters."""
        if isinstance(self.index_type, str):
            self.index_type = IndexType(self.index_type)
        _require_positive(self.top_k, "top_k")
        _require_range(self.similarity_threshold, -1.0, 1.0, "similarity_threshold")
        _require_range(self.margin_threshold, 0.0, 2.0, "margin_threshold")
        _require_positive(self.nlist, "nlist")
        _require_positive(self.nprobe, "nprobe")
        _require(0.0 < self.target_far < 1.0, "target_far must be in (0, 1)")
        _require(bool(self.unknown_label), "unknown_label must be non-empty")


# --------------------------------------------------------------------------- #
# Section: rendering (Phase 10)
# --------------------------------------------------------------------------- #
@dataclass
class RenderingConfig:
    """Annotated-video output settings.

    Attributes:
        codec: FFmpeg encoder name.
        pixel_format: Output pixel format; ``yuv420p`` maximises compatibility.
        crf: Constant rate factor; lower is higher quality.
        preset: x264 speed/compression trade-off.
        output_name: Filename written into ``PathConfig.outputs_dir``.
        draw_body: Draw person bounding boxes.
        draw_face: Draw face bounding boxes.
        draw_landmarks: Draw the SCRFD 5-point landmarks.
        draw_trajectory: Draw the track centroid history.
        draw_identity: Draw the resolved identity and similarity.
        draw_timestamp: Draw the decoded PTS and frame number.
        trajectory_length: Centroids drawn per track.
        box_thickness: Rectangle line width in pixels.
        font_scale: OpenCV Hershey font scale.
        font_thickness: Glyph stroke width in pixels.
        unknown_color: BGR colour for unidentified tracks.
        known_color: BGR colour for confidently identified tracks.
        face_color: BGR colour for face boxes.
        redact_unknown: Blur faces that resolve to ``UNKNOWN``.
        redaction_kernel: Gaussian kernel size used when redacting.
    """

    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = 20
    preset: str = "medium"
    output_name: str = "annotated.mp4"

    draw_body: bool = True
    draw_face: bool = True
    draw_landmarks: bool = True
    draw_trajectory: bool = True
    draw_identity: bool = True
    draw_timestamp: bool = True

    trajectory_length: int = 48
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1

    unknown_color: Tuple[int, int, int] = (60, 60, 220)
    known_color: Tuple[int, int, int] = (80, 200, 90)
    face_color: Tuple[int, int, int] = (230, 180, 60)

    redact_unknown: bool = False
    redaction_kernel: int = 31

    #: Presets accepted by the x264/x265 encoders.
    VALID_PRESETS: ClassVar[Tuple[str, ...]] = (
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    )

    def __post_init__(self) -> None:
        """Validate encoder and overlay parameters."""
        _require_range(self.crf, 0, 51, "crf")
        _require(
            self.preset in self.VALID_PRESETS,
            f"preset must be one of {self.VALID_PRESETS}, got {self.preset!r}",
        )
        _require_positive(self.trajectory_length, "trajectory_length")
        _require_positive(self.box_thickness, "box_thickness")
        _require_positive(self.font_thickness, "font_thickness")
        _require(self.font_scale > 0, "font_scale must be > 0")
        _require(
            self.redaction_kernel % 2 == 1 and self.redaction_kernel > 0,
            "redaction_kernel must be a positive odd integer",
        )
        for name in ("unknown_color", "known_color", "face_color"):
            colour = tuple(int(c) for c in getattr(self, name))
            _require(len(colour) == 3, f"{name} must be a BGR triplet")
            _require(
                all(0 <= c <= 255 for c in colour),
                f"{name} components must be in [0, 255]",
            )
            setattr(self, name, colour)

# --------------------------------------------------------------------------- #
# Section: runtime / performance (Phase 11)
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeConfig:
    """Device, determinism and performance-tuning knobs.

    Attributes:
        device: ``"cuda"``, ``"cuda:N"``, ``"cpu"`` or ``"auto"``.
        seed: Global RNG seed.
        deterministic: Request deterministic kernels; costs throughput.
        cudnn_benchmark: Let cuDNN autotune algorithms; mutually exclusive with
            ``deterministic``.
        allow_tf32: Permit TF32 matmuls.  Ampere and newer only; the T4 is
            Turing, so this is inert there but correct on newer hardware.
        pin_memory: Allocate page-locked host buffers for H2D transfers.
        num_workers: Host-side worker threads for pre/post-processing.
        cuda_streams: Concurrent CUDA streams for overlapping copy and compute.
        empty_cache_every: Call ``torch.cuda.empty_cache`` every N frames;
            ``0`` disables periodic cache clearing.
        max_vram_fraction: Upper bound on the GPU memory fraction this process
            may reserve.
        profile: Collect FPS, latency and memory statistics.
        log_level: Root logger level name.
    """

    device: str = "auto"
    seed: int = 1337
    deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    pin_memory: bool = True
    num_workers: int = 2
    cuda_streams: int = 2
    empty_cache_every: int = 0
    max_vram_fraction: float = 0.92
    profile: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate runtime settings and reconcile determinism flags."""
        self.device = str(self.device).lower()
        _require(
            self.device == "auto"
            or self.device == "cpu"
            or self.device == "cuda"
            or (self.device.startswith("cuda:") and self.device.split(":")[1].isdigit()),
            f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {self.device!r}",
        )
        _require(self.num_workers >= 0, "num_workers must be >= 0")
        _require_positive(self.cuda_streams, "cuda_streams")
        _require(self.empty_cache_every >= 0, "empty_cache_every must be >= 0")
        _require(0.0 < self.max_vram_fraction <= 1.0, "max_vram_fraction must be in (0, 1]")
        self.log_level = self.log_level.upper()
        _require(
            self.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
            f"invalid log_level {self.log_level!r}",
        )
        if self.deterministic and self.cudnn_benchmark:
            LOGGER.warning(
                "cudnn_benchmark disabled because deterministic execution was requested"
            )
            self.cudnn_benchmark = False

    def resolve_device(self) -> str:
        """Resolve ``device`` against the hardware actually present.

        ``auto`` becomes ``cuda`` when a CUDA device is visible and ``cpu``
        otherwise.  An explicit CUDA request on a machine without CUDA degrades
        to CPU with a warning rather than raising, so that CI can exercise the
        pipeline end to end.

        Returns:
            A concrete device string suitable for ``torch.device``.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard runtime dep
            LOGGER.warning("torch unavailable; falling back to CPU")
            return "cpu"

        cuda_available = torch.cuda.is_available()
        if self.device == "auto":
            resolved = "cuda" if cuda_available else "cpu"
        elif self.device.startswith("cuda") and not cuda_available:
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
            resolved = "cpu"
        else:
            resolved = self.device

        if resolved.startswith("cuda:"):
            index = int(resolved.split(":")[1])
            count = torch.cuda.device_count()
            if index >= count:
                LOGGER.warning(
                    "cuda:%d requested but only %d device(s) visible; using cuda:0",
                    index,
                    count,
                )
                resolved = "cuda:0"
        return resolved

    def use_half(self, requested: bool) -> bool:
        """Decide whether FP16 should actually be used.

        Args:
            requested: Whether the calling module asked for half precision.

        Returns:
            ``True`` only when half precision was requested *and* the resolved
            device is CUDA.  FP16 on CPU is slower than FP32 and numerically
            worse, so it is never enabled there.
        """
        return bool(requested) and self.resolve_device().startswith("cuda")

    def apply(self) -> str:
        """Seed every RNG and apply determinism and precision policies.

        Returns:
            The resolved device string.
        """
        os.environ.setdefault("PYTHONHASHSEED", str(self.seed))
        random.seed(self.seed)

        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:  # pragma: no cover
            LOGGER.warning("numpy unavailable; skipping numpy seeding")

        device = "cpu"
        try:
            import torch

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            device = self.resolve_device()

            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            torch.backends.cudnn.deterministic = self.deterministic
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

            if self.deterministic:
                # cuBLAS requires this workspace hint for deterministic GEMMs.
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except (AttributeError, RuntimeError) as exc:  # pragma: no cover
                    LOGGER.warning("Could not enable deterministic algorithms: %s", exc)

            if device.startswith("cuda") and self.max_vram_fraction < 1.0:
                index = int(device.split(":")[1]) if ":" in device else 0
                torch.cuda.set_per_process_memory_fraction(self.max_vram_fraction, index)
        except ImportError:  # pragma: no cover
            LOGGER.warning("torch unavailable; determinism applied to stdlib/numpy only")

        LOGGER.info(
            "Runtime ready: device=%s seed=%d deterministic=%s",
            device,
            self.seed,
            self.deterministic,
        )
        return device


# --------------------------------------------------------------------------- #
# Section: governance
# --------------------------------------------------------------------------- #
@dataclass
class GovernanceConfig:
    """Data-protection controls for biometric processing.

    Face embeddings are biometric identifiers and are regulated as sensitive
    personal data under GDPR Art. 9, India's DPDP Act 2023 and Illinois BIPA,
    among others.  These switches make retention and auditability explicit
    rather than incidental.

    Attributes:
        retention_days: Days after which persisted embeddings and crops should
            be purged; ``0`` means do not persist at all.
        persist_crops: Write aligned face crops to disk.
        persist_embeddings: Write fused embeddings to disk.
        audit_log: Append an immutable JSONL record of every identification.
        audit_log_name: Filename of the audit log inside ``log_dir``.
        require_consent_manifest: Refuse to build a gallery unless a consent
            manifest accompanies the enrolled images.
    """

    retention_days: int = 30
    persist_crops: bool = False
    persist_embeddings: bool = True
    audit_log: bool = True
    audit_log_name: str = "identification_audit.jsonl"
    require_consent_manifest: bool = True

    def __post_init__(self) -> None:
        """Validate retention settings."""
        _require(self.retention_days >= 0, "retention_days must be >= 0")
        _require(bool(self.audit_log_name), "audit_log_name must be non-empty")
# --------------------------------------------------------------------------- #
# Serialisation machinery
# --------------------------------------------------------------------------- #
def _is_optional(annotation: Any) -> bool:
    """Return ``True`` when ``annotation`` is ``Optional[X]``."""
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _unwrap_optional(annotation: Any) -> Any:
    """Return ``X`` from ``Optional[X]``, or ``annotation`` unchanged."""
    if not _is_optional(annotation):
        return annotation
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def _build(annotation: Any, value: Any) -> Any:
    """Recursively coerce a plain Python value into ``annotation``.

    Handles nested dataclasses, ``Optional``, ``Path``, ``Enum``, ``Tuple`` and
    scalars.  Unknown annotations pass the value through untouched.

    Args:
        annotation: Target type annotation.
        value: Raw value taken from JSON/YAML/dict input.

    Returns:
        The coerced value.
    """
    if value is None:
        return None
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)

    if is_dataclass(annotation) and isinstance(value, Mapping):
        return _from_mapping(annotation, value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is Path:
        return Path(value)
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], v) for v in value)
        return tuple(_build(a, v) for a, v in zip(args, value))
    if origin is list:
        (arg,) = get_args(annotation) or (Any,)
        return [_build(arg, v) for v in value]
    return value


def _from_mapping(cls: type, data: Mapping[str, Any]) -> Any:
    """Construct a dataclass instance from a mapping, ignoring unknown keys.

    Args:
        cls: Target dataclass.
        data: Mapping of field name to raw value.

    Returns:
        A validated instance of ``cls``.

    Raises:
        ConfigError: If a value cannot be coerced or fails validation.
    """
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    for key, raw in data.items():
        if key not in known:
            LOGGER.warning("Ignoring unknown config key %s.%s", cls.__name__, key)
            continue
        kwargs[key] = _build(hints.get(key, Any), raw)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Cannot construct {cls.__name__}: {exc}") from exc


def _to_plain(value: Any) -> Any:
    """Convert a config value into a JSON/YAML-serialisable primitive.

    Args:
        value: Any config value.

    Returns:
        A structure composed only of dicts, lists, strings, numbers and bools.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


def _coerce_scalar(text: str) -> Any:
    """Parse an environment-variable string into a bool, number or string.

    Args:
        text: Raw environment value.

    Returns:
        ``bool``, ``int``, ``float`` or the original ``str``.
    """
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on", "1"}:
        return True
    if lowered in {"false", "no", "off", "0"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


# --------------------------------------------------------------------------- #
# Aggregate root
# --------------------------------------------------------------------------- #
@dataclass
class SurveillanceConfig:
    """Root configuration aggregating every pipeline stage.

    Attributes:
        paths: Filesystem layout.
        video: Decoder settings.
        detection: Body and face detector settings.
        tracking: OC-SORT settings.
        association: Face-body assignment settings.
        alignment: Face alignment settings.
        recognition: Embedding extractor settings.
        quality: Embedding quality gate.
        fusion: Temporal fusion settings.
        search: Gallery search and open-set rejection.
        rendering: Annotated-video output.
        runtime: Device, determinism and performance.
        governance: Biometric data-protection controls.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def default(cls) -> "SurveillanceConfig":
        """Return the stock configuration tuned for a 16 GB Tesla T4.

        Returns:
            A fully populated, validated configuration.
        """
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurveillanceConfig":
        """Build a configuration from a nested mapping.

        Args:
            data: Nested mapping keyed by section name.

        Returns:
            A validated configuration.

        Raises:
            ConfigError: If any value fails coercion or validation.
        """
        return _from_mapping(cls, data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SurveillanceConfig":
        """Load a configuration from a ``.yaml``, ``.yml`` or ``.json`` file.

        Args:
            path: Path to the configuration file.

        Returns:
            A validated configuration.

        Raises:
            ConfigError: If the file is missing, has an unsupported extension,
                or contains invalid values.
        """
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("PyYAML is required to read YAML configs") from exc
            payload = yaml.safe_load(text) or {}
        elif suffix == ".json":
            payload = json.loads(text)
        else:
            raise ConfigError(f"Unsupported config extension {suffix!r}")

        if not isinstance(payload, Mapping):
            raise ConfigError(f"Config root must be a mapping, got {type(payload).__name__}")
        LOGGER.info("Loaded configuration from %s", path)
        return cls.from_dict(payload)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to plain Python primitives.

        Returns:
            A nested dictionary safe for ``json.dumps`` or ``yaml.safe_dump``.
        """
        return _to_plain(self)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the configuration to disk as YAML or JSON.

        Args:
            path: Destination file; the extension selects the format.

        Returns:
            The resolved destination path.

        Raises:
            ConfigError: If the extension is unsupported or PyYAML is missing.
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        suffix = path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("PyYAML is required to write YAML configs") from exc
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
        elif suffix == ".json":
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            raise ConfigError(f"Unsupported config extension {suffix!r}")

        LOGGER.info("Saved configuration to %s", path)
        return path

    # -- overrides --------------------------------------------------------- #
    def apply_env_overrides(self, environ: Optional[Mapping[str, str]] = None) -> "SurveillanceConfig":
        """Overlay ``SURV_SECTION__FIELD`` environment variables onto this config.

        This is how Kaggle notebooks and CI jobs retarget a run without editing
        YAML, e.g. ``SURV_DETECTION__BODY_CONF=0.5``.

        Args:
            environ: Mapping to read; defaults to ``os.environ``.

        Returns:
            A new, revalidated configuration with the overrides applied.
        """
        environ = os.environ if environ is None else environ
        payload = self.to_dict()
        applied = 0

        for raw_key, raw_value in environ.items():
            if not raw_key.startswith(ENV_PREFIX):
                continue
            body = raw_key[len(ENV_PREFIX) :]
            if ENV_SECTION_SEP not in body:
                LOGGER.warning("Ignoring malformed override %s", raw_key)
                continue
            section, _, field_name = body.partition(ENV_SECTION_SEP)
            section, field_name = section.lower(), field_name.lower()
            if section not in payload or not isinstance(payload[section], dict):
                LOGGER.warning("Ignoring override for unknown section %r", section)
                continue
            if field_name not in payload[section]:
                LOGGER.warning("Ignoring override for unknown field %s.%s", section, field_name)
                continue
            payload[section][field_name] = _coerce_scalar(raw_value)
            applied += 1
            LOGGER.debug("Override %s.%s = %r", section, field_name, raw_value)

        if applied:
            LOGGER.info("Applied %d environment override(s)", applied)
        return SurveillanceConfig.from_dict(payload)

    # -- cross-section validation ------------------------------------------ #
    def validate(self) -> "SurveillanceConfig":
        """Check invariants that span more than one section.

        Per-section invariants are enforced in each ``__post_init__``; this
        method covers only relationships between sections.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            ConfigError: If a cross-section invariant is violated.
        """
        _require(
            self.detection.min_face_size <= min(self.alignment.output_size),
            "detection.min_face_size exceeds the aligned crop size; every face "
            "would be upsampled beyond its native resolution",
        )
        _require(
            self.quality.max_stored_per_track >= self.fusion.min_samples,
            "quality.max_stored_per_track must be >= fusion.min_samples, "
            "otherwise no track can ever accumulate enough embeddings",
        )
        _require(
            self.rendering.trajectory_length <= self.tracking.max_trajectory,
            "rendering.trajectory_length exceeds tracking.max_trajectory; "
            "the requested history is never retained",
        )
        _require(
            self.tracking.det_threshold >= self.detection.body_conf,
            "tracking.det_threshold must be >= detection.body_conf, otherwise "
            "the tracker threshold is unreachable",
        )
        if self.governance.retention_days == 0:
            _require(
                not self.governance.persist_embeddings and not self.governance.persist_crops,
                "retention_days=0 forbids persisting embeddings or crops",
            )
        LOGGER.debug("Cross-section validation passed")
        return self

    # -- convenience ------------------------------------------------------- #
    def bootstrap(self) -> str:
        """Prepare the process: create directories, seed RNGs, configure logging.

        Returns:
            The resolved device string.
        """
        self.paths.ensure()
        logging.basicConfig(
            level=getattr(logging, self.runtime.log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.validate()
        return self.runtime.apply()

    def summary(self) -> str:
        """Render a compact, human-readable digest of the key settings.

        Returns:
            A multi-line string suitable for logging at start-up.
        """
        lines = [
            "SurveillanceConfig",
            f"  device        : {self.runtime.device} (seed={self.runtime.seed}, "
            f"deterministic={self.runtime.deterministic})",
            f"  video         : strict_vfr={self.video.strict_vfr} stride={self.video.stride}",
            f"  detection     : {self.detection.body_weights} @ {self.detection.body_imgsz} "
            f"+ SCRFD/{self.detection.face_model_name} @ {self.detection.face_det_size}",
            f"  tracking      : OC-SORT max_age={self.tracking.max_age} "
            f"min_hits={self.tracking.min_hits} iou={self.tracking.iou_threshold}",
            f"  alignment     : {self.alignment.output_size} via Umeyama",
            f"  recognition   : AdaFace-{self.recognition.architecture} "
            f"dim={self.recognition.embedding_dim} amp={self.recognition.amp}",
            f"  fusion        : {self.fusion.strategy.value} "
            f"(min={self.fusion.min_samples}, max={self.fusion.max_samples})",
            f"  search        : {self.search.index_type.value} top_k={self.search.top_k} "
            f"thr={self.search.similarity_threshold} margin={self.search.margin_threshold}",
            f"  rendering     : {self.rendering.codec} crf={self.rendering.crf} "
            f"-> {self.paths.outputs_dir / self.rendering.output_name}",
            f"  governance    : retention={self.governance.retention_days}d "
            f"audit={self.governance.audit_log}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Process-wide accessor
# --------------------------------------------------------------------------- #
_ACTIVE_CONFIG: Optional[SurveillanceConfig] = None


def get_config() -> SurveillanceConfig:
    """Return the process-wide configuration, creating a default on first use.

    Returns:
        The active :class:`SurveillanceConfig`.
    """
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        _ACTIVE_CONFIG = SurveillanceConfig.default()
        LOGGER.debug("Instantiated default configuration")
    return _ACTIVE_CONFIG


def set_config(config: SurveillanceConfig) -> SurveillanceConfig:
    """Install ``config`` as the process-wide configuration.

    Args:
        config: The configuration to activate.

    Returns:
        The installed configuration.

    Raises:
        ConfigError: If ``config`` is not a :class:`SurveillanceConfig`.
    """
    global _ACTIVE_CONFIG
    if not isinstance(config, SurveillanceConfig):
        raise ConfigError(f"Expected SurveillanceConfig, got {type(config).__name__}")
    _ACTIVE_CONFIG = config.validate()
    return _ACTIVE_CONFIG


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = SurveillanceConfig.default().apply_env_overrides().validate()
    print(cfg.summary())
    print(f"\nresolved device: {cfg.runtime.resolve_device()}")
    print(f"scaled template: {cfg.alignment.scaled_template()[0]}")