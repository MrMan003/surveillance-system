"""Face embedding extraction.

Two backends behind one interface:

:class:`AdaFaceEncoder`
    The intended encoder.  Loads an official AdaFace checkpoint into the
    from-scratch IResNet in :mod:`recognition.backbones`.

:class:`ArcFaceOnnxEncoder`
    An interim encoder using ``w600k_r50.onnx``, which is already on disk
    inside the ``buffalo_l`` pack downloaded for SCRFD.  It exists so the
    pipeline is runnable end to end before the AdaFace weights are obtained --
    they are hosted on Google Drive and cannot be fetched programmatically.

Both emit a 512-dimensional unit vector plus a scalar quality signal, so
:mod:`recognition` can be swapped without touching Phases 7 through 9.

On the quality signal
---------------------
AdaFace's pre-normalisation feature norm correlates strongly with image
quality, because the training objective scales its angular margin by that norm.
ArcFace has no such property -- its norms are not calibrated to quality in any
useful way.  Rather than pretend otherwise, :class:`ArcFaceOnnxEncoder` reports
``norm_is_quality_calibrated = False``, and Phase 7 falls back to pixel-domain
measures when that flag is unset.  Treating an uncalibrated norm as a quality
score would silently gate on noise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import List, Optional, Sequence, Type

import numpy as np

from configs.config import PathConfig, RecognitionConfig, RuntimeConfig
from utils.log import get_logger

__all__ = [
    "EncoderError",
    "Embedding",
    "FaceEncoder",
    "AdaFaceEncoder",
    "ArcFaceOnnxEncoder",
    "build_encoder",
]

LOGGER = get_logger(__name__)

#: Filename of the ArcFace recognition model inside an InsightFace pack.
ARCFACE_FILENAME = "w600k_r50.onnx"


class EncoderError(RuntimeError):
    """Raised when an encoder cannot be loaded or a forward pass fails."""


@dataclass(frozen=True, slots=True)
class Embedding:
    """A face embedding with its quality signal.

    Attributes:
        vector: L2-normalised ``float32`` array of shape ``(512,)``.  Unit
            length, so an inner product against another embedding is cosine
            similarity directly.
        norm: Pre-normalisation magnitude.  A quality proxy when the encoder
            reports it as calibrated, and meaningless otherwise.
        detection_score: Detector confidence of the source face, carried
            through for the Phase 7 gate.
        norm_is_quality_calibrated: Whether ``norm`` may be interpreted as a
            quality measure.
    """

    vector: np.ndarray
    norm: float
    detection_score: float = 1.0
    norm_is_quality_calibrated: bool = True

    def __post_init__(self) -> None:
        """Validate dimensionality and normalisation.

        Raises:
            ValueError: If the vector is the wrong shape or not unit length.
        """
        if self.vector.shape != (512,):
            raise ValueError(f"Embedding must have shape (512,), got {self.vector.shape}")
        magnitude = float(np.linalg.norm(self.vector))
        if not np.isclose(magnitude, 1.0, atol=1e-3):
            raise ValueError(f"Embedding must be unit length, got {magnitude:.6f}")

    def similarity(self, other: "Embedding") -> float:
        """Cosine similarity against another embedding.

        Args:
            other: The embedding to compare against.

        Returns:
            A value in ``[-1, 1]``; both vectors are unit length, so this is a
            plain inner product.
        """
        return float(np.dot(self.vector, other.vector))

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        flag = "" if self.norm_is_quality_calibrated else ", uncalibrated"
        return f"Embedding(dim=512, norm={self.norm:.2f}{flag})"


class FaceEncoder(ABC):
    """Base class for face embedding extractors.

    Args:
        config: Encoder architecture and batching settings.
        runtime: Device and precision policy.
        paths: Filesystem layout used to resolve checkpoints.
        name: Identifier used in logs.
    """

    #: Whether this encoder's feature norm is a usable quality signal.
    NORM_IS_QUALITY_CALIBRATED: bool = True

    def __init__(
        self,
        config: RecognitionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
        name: str = "encoder",
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._paths = paths or PathConfig()
        self._name = name
        self._device = runtime.resolve_device()
        self._loaded = False

    @property
    def name(self) -> str:
        """Identifier used in logs."""
        return self._name

    @property
    def device(self) -> str:
        """Resolved compute device."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether weights are resident."""
        return self._loaded

    @abstractmethod
    def load(self) -> "FaceEncoder":
        """Load weights onto the target device.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            EncoderError: If the checkpoint is missing or unloadable.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the model and free device memory."""

    @abstractmethod
    def _encode(self, crops: np.ndarray) -> tuple:
        """Run one forward pass.

        Args:
            crops: Aligned faces, ``uint8``, shape ``(B, 112, 112, 3)``, BGR.

        Returns:
            A tuple ``(vectors, norms)`` where ``vectors`` has shape
            ``(B, 512)`` with unit rows and ``norms`` has shape ``(B,)``.
        """

    def __enter__(self) -> "FaceEncoder":
        """Load on entering a ``with`` block."""
        return self.load()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Release on leaving a ``with`` block."""
        self.close()

    # -- public API -------------------------------------------------------- #
    def encode(
        self,
        crops: Sequence[np.ndarray],
        detection_scores: Optional[Sequence[float]] = None,
    ) -> List[Embedding]:
        """Embed a sequence of aligned face crops.

        Args:
            crops: Aligned faces, each ``uint8`` of shape ``(112, 112, 3)``.
            detection_scores: Detector confidence per crop, carried through to
                the resulting embeddings.

        Returns:
            One embedding per crop, in input order.

        Raises:
            EncoderError: If the encoder is not loaded or a crop is malformed.
            ValueError: If ``detection_scores`` length does not match.
        """
        if not self._loaded:
            raise EncoderError(f"{self._name} not loaded; call load() or use a with block")
        if not crops:
            return []

        if detection_scores is None:
            detection_scores = [1.0] * len(crops)
        elif len(detection_scores) != len(crops):
            raise ValueError(
                f"detection_scores has {len(detection_scores)} entries "
                f"for {len(crops)} crop(s)"
            )

        expected = (self._config.embedding_dim,)
        stacked = self._stack(crops)

        vectors: List[np.ndarray] = []
        norms: List[float] = []
        size = self._config.batch_size

        for start in range(0, len(stacked), size):
            batch = stacked[start : start + size]
            try:
                batch_vectors, batch_norms = self._encode(batch)
            except EncoderError:
                raise
            except Exception as exc:  # noqa: BLE001 - backends raise widely
                raise EncoderError(f"{self._name} forward pass failed: {exc}") from exc
            vectors.extend(np.asarray(batch_vectors, dtype=np.float32))
            norms.extend(float(v) for v in np.asarray(batch_norms).ravel())

        results: List[Embedding] = []
        for vector, norm, score in zip(vectors, norms, detection_scores):
            if vector.shape != expected:
                raise EncoderError(f"Expected {expected} embedding, got {vector.shape}")
            results.append(
                Embedding(
                    vector=np.ascontiguousarray(vector, dtype=np.float32),
                    norm=norm,
                    detection_score=float(score),
                    norm_is_quality_calibrated=self.NORM_IS_QUALITY_CALIBRATED,
                )
            )
        return results

    @staticmethod
    def _stack(crops: Sequence[np.ndarray]) -> np.ndarray:
        """Validate and stack crops into one array.

        Args:
            crops: Aligned face crops.

        Returns:
            A ``uint8`` array of shape ``(B, 112, 112, 3)``.

        Raises:
            EncoderError: If any crop has the wrong shape or dtype.
        """
        for index, crop in enumerate(crops):
            if crop.ndim != 3 or crop.shape[2] != 3:
                raise EncoderError(f"Crop {index} must be (H, W, 3), got {crop.shape}")
            if crop.dtype != np.uint8:
                raise EncoderError(f"Crop {index} must be uint8, got {crop.dtype}")
        return np.stack(crops).astype(np.uint8)

    def _resolve_weights(self, filename: str) -> Path:
        """Resolve a checkpoint filename against the weights directory.

        Args:
            filename: Absolute path or bare filename.

        Returns:
            The resolved path.

        Raises:
            EncoderError: If no file exists at the resolved location.
        """
        candidate = Path(filename)
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        local = self._paths.weights_dir / candidate.name
        if local.is_file():
            return local
        raise EncoderError(
            f"Checkpoint {candidate.name} not found in {self._paths.weights_dir}"
        )


class AdaFaceEncoder(FaceEncoder):
    """AdaFace encoder over the from-scratch IResNet backbone.

    Args:
        config: Encoder architecture and batching settings.
        runtime: Device and precision policy.
        paths: Filesystem layout used to resolve the checkpoint.
    """

    NORM_IS_QUALITY_CALIBRATED = True

    def __init__(
        self,
        config: RecognitionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
    ) -> None:
        super().__init__(config, runtime, paths, name=f"adaface-{config.architecture}")
        self._model = None
        self._amp = runtime.use_half(config.amp)

    def load(self) -> "AdaFaceEncoder":
        """Load the backbone and its checkpoint.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            EncoderError: If torch is unavailable or the checkpoint cannot be
                loaded into the architecture.
        """
        if self._loaded:
            return self

        try:
            import torch

            from recognition.backbones import build_backbone
        except ImportError as exc:
            raise EncoderError("torch is required for AdaFace") from exc

        checkpoint_path = self._resolve_weights(self._config.weights)
        model = build_backbone(self._config.architecture)

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:  # noqa: BLE001
            raise EncoderError(f"Cannot read {checkpoint_path}: {exc}") from exc

        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        # Official AdaFace checkpoints are PyTorch Lightning exports, so every
        # key is prefixed with the Lightning module attribute name.
        cleaned = {
            key.replace("model.", "", 1) if key.startswith("model.") else key: value
            for key, value in state.items()
        }

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            raise EncoderError(
                f"{checkpoint_path.name} is missing {len(missing)} parameter(s) "
                f"for {self._config.architecture}, e.g. {missing[:3]}. "
                "The checkpoint likely targets a different architecture."
            )
        if unexpected:
            LOGGER.debug("Ignoring %d unexpected checkpoint key(s)", len(unexpected))

        model.eval()
        model.to(self._device)
        self._model = model
        self._loaded = True

        LOGGER.info(
            "Loaded AdaFace-%s from %s on %s (amp=%s)",
            self._config.architecture,
            checkpoint_path.name,
            self._device,
            self._amp,
        )
        return self

    def close(self) -> None:
        """Release the model and free device memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        try:
            import torch

            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def _preprocess(self, crops: np.ndarray) -> "np.ndarray":
        """Convert uint8 crops to the tensor layout the backbone expects.

        Args:
            crops: ``uint8`` array of shape ``(B, 112, 112, 3)`` in BGR.

        Returns:
            A ``float32`` torch tensor of shape ``(B, 3, 112, 112)``.
        """
        import torch

        array = crops.astype(np.float32) / 255.0
        if not self._config.input_bgr:
            array = array[..., ::-1]

        mean = np.asarray(self._config.input_mean, dtype=np.float32)
        std = np.asarray(self._config.input_std, dtype=np.float32)
        array = (array - mean) / std

        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))
        return tensor.to(self._device)

    def _encode(self, crops: np.ndarray) -> tuple:
        """Run the backbone over one batch.

        Args:
            crops: ``uint8`` array of shape ``(B, 112, 112, 3)``.

        Returns:
            A tuple ``(vectors, norms)``.
        """
        import torch

        tensor = self._preprocess(crops)

        with torch.inference_mode():
            if self._amp:
                # float16 explicitly: the T4 is Turing and has no bfloat16
                # tensor cores, so bf16 autocast would silently run in software.
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings, norms = self._model(tensor)
            else:
                embeddings, norms = self._model(tensor)

            if self._config.flip_tta:
                flipped = torch.flip(tensor, dims=[3])
                if self._amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        flipped_embeddings, _ = self._model(flipped)
                else:
                    flipped_embeddings, _ = self._model(flipped)
                embeddings = embeddings + flipped_embeddings
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # AMP returns float16; downstream FAISS and numpy work in float32.
        return (
            embeddings.float().cpu().numpy(),
            norms.float().cpu().numpy().ravel(),
        )


class ArcFaceOnnxEncoder(FaceEncoder):
    """Interim encoder using the ArcFace model bundled with InsightFace packs.

    Uses ``w600k_r50.onnx``, already present from the SCRFD download.  Accuracy
    is competitive with AdaFace on clean frontal faces and worse on the low
    quality, extreme pose material CCTV produces -- which is precisely what
    AdaFace's quality-adaptive margin was designed for.

    Args:
        config: Encoder settings; ``architecture`` is ignored here.
        runtime: Device and precision policy.
        paths: Filesystem layout used to resolve the model pack.
        pack_name: InsightFace pack containing the model.
    """

    NORM_IS_QUALITY_CALIBRATED = False

    def __init__(
        self,
        config: RecognitionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
        pack_name: str = "buffalo_l",
    ) -> None:
        super().__init__(config, runtime, paths, name="arcface-onnx")
        self._pack_name = pack_name
        self._session = None
        self._input_name = ""

    def load(self) -> "ArcFaceOnnxEncoder":
        """Open an ONNXRuntime session for the ArcFace model.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            EncoderError: If ONNXRuntime is unavailable, the model is missing,
                or CUDA was requested without the CUDA provider.
        """
        if self._loaded:
            return self

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EncoderError("onnxruntime is required for ArcFaceOnnxEncoder") from exc

        model_path = (
            self._paths.weights_dir / "models" / self._pack_name / ARCFACE_FILENAME
        )
        if not model_path.is_file():
            raise EncoderError(
                f"{ARCFACE_FILENAME} not found at {model_path}. "
                "Load the face detector once to download the pack."
            )

        available = ort.get_available_providers()
        if self._device.startswith("cuda"):
            if "CUDAExecutionProvider" not in available:
                raise EncoderError(
                    "CUDA requested but CUDAExecutionProvider is unavailable "
                    f"(providers: {available})."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            session_options = ort.SessionOptions()
            # w600k_r50.onnx declares a dynamic input batch but a fixed (1, 512)
            # output shape. The computation is correct for any batch size --
            # verified by identical crops producing identical embeddings -- but
            # ORT warns on every batched call. Raising the severity floor keeps
            # errors visible while dropping that one spurious warning per batch.
            session_options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(model_path), sess_options=session_options, providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            raise EncoderError(f"Cannot open {model_path}: {exc}") from exc

        self._loaded = True
        LOGGER.info(
            "Loaded ArcFace %s on %s (interim encoder; norm is not quality calibrated)",
            model_path.name,
            self._device,
        )
        return self

    def close(self) -> None:
        """Release the ONNX session."""
        if self._session is not None:
            del self._session
            self._session = None
        self._loaded = False

    def _encode(self, crops: np.ndarray) -> tuple:
        """Run the ONNX model over one batch.

        InsightFace's ArcFace expects RGB scaled to roughly ``[-1, 1]`` via
        ``(x - 127.5) / 127.5``.  Crops arrive BGR from the aligner, so the
        channel order is reversed here.

        Args:
            crops: ``uint8`` array of shape ``(B, 112, 112, 3)`` in BGR.

        Returns:
            A tuple ``(vectors, norms)``.
        """
        rgb = crops[..., ::-1].astype(np.float32)
        blob = (rgb - 127.5) / 127.5
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))

        raw = self._session.run(None, {self._input_name: blob})[0]
        norms = np.linalg.norm(raw, axis=1)
        vectors = raw / np.maximum(norms[:, None], 1e-12)
        return vectors.astype(np.float32), norms.astype(np.float32)


def build_encoder(
    config: RecognitionConfig,
    runtime: RuntimeConfig,
    paths: Optional[PathConfig] = None,
    prefer_adaface: bool = True,
) -> FaceEncoder:
    """Construct an encoder, falling back to ArcFace when AdaFace is absent.

    The fallback is logged loudly rather than silently: the two encoders differ
    materially on low-quality faces, and a run that quietly used the interim
    model would be difficult to account for afterwards.

    Args:
        config: Encoder settings.
        runtime: Device and precision policy.
        paths: Filesystem layout.
        prefer_adaface: Try AdaFace first.  When ``False``, ArcFace is used
            unconditionally.

    Returns:
        An unloaded encoder.
    """
    paths = paths or PathConfig()
    checkpoint = paths.weights_dir / Path(config.weights).name

    if prefer_adaface and checkpoint.is_file():
        return AdaFaceEncoder(config, runtime, paths)

    if prefer_adaface:
        LOGGER.warning(
            "AdaFace checkpoint %s not found; falling back to the interim "
            "ArcFace encoder. Recognition accuracy on low-quality faces will "
            "be lower, and embedding norms will not be quality calibrated.",
            checkpoint.name,
        )
    return ArcFaceOnnxEncoder(config, runtime, paths)