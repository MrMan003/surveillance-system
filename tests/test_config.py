"""Unit tests for the configuration layer (Phase 12 coverage for Phase 1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from configs.config import (
    ConfigError,
    FusionStrategy,
    IndexType,
    SurveillanceConfig,
    get_config,
    set_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_default_is_valid() -> None:
    """The stock configuration must construct and validate without arguments."""
    assert SurveillanceConfig.default().validate() is not None


def test_dict_round_trip_is_lossless() -> None:
    """to_dict -> from_dict -> to_dict must be a fixed point."""
    original = SurveillanceConfig.default().to_dict()
    assert SurveillanceConfig.from_dict(original).to_dict() == original


def test_to_dict_is_json_serialisable() -> None:
    """Paths, tuples and enums must degrade to JSON primitives."""
    json.dumps(SurveillanceConfig.default().to_dict())


def test_yaml_round_trip(tmp_path) -> None:
    """A saved YAML config must reload identically."""
    cfg = SurveillanceConfig.default()
    path = cfg.save(tmp_path / "cfg.yaml")
    assert SurveillanceConfig.from_file(path).to_dict() == cfg.to_dict()


def test_json_round_trip(tmp_path) -> None:
    """A saved JSON config must reload identically."""
    cfg = SurveillanceConfig.default()
    path = cfg.save(tmp_path / "cfg.json")
    assert SurveillanceConfig.from_file(path).to_dict() == cfg.to_dict()


def test_unsupported_extension_rejected(tmp_path) -> None:
    """Only YAML and JSON are valid config formats."""
    with pytest.raises(ConfigError):
        SurveillanceConfig.default().save(tmp_path / "cfg.toml")


def test_missing_file_rejected(tmp_path) -> None:
    """A nonexistent config path must fail loudly."""
    with pytest.raises(ConfigError):
        SurveillanceConfig.from_file(tmp_path / "nope.yaml")


def test_enums_survive_deserialisation() -> None:
    """String enum values must be rehydrated into typed enum members."""
    cfg = SurveillanceConfig.from_dict(
        {"fusion": {"strategy": "median"}, "search": {"index_type": "IndexFlatIP"}}
    )
    assert cfg.fusion.strategy is FusionStrategy.MEDIAN
    assert cfg.search.index_type is IndexType.FLAT_IP


def test_env_overrides_are_typed() -> None:
    """Environment strings must be coerced to bool/int/float, not left as str."""
    cfg = SurveillanceConfig.default().apply_env_overrides(
        {
            "SURV_DETECTION__BODY_CONF": "0.5",
            "SURV_RUNTIME__DETERMINISTIC": "false",
            "SURV_TRACKING__MAX_AGE": "45",
        }
    )
    assert cfg.detection.body_conf == pytest.approx(0.5)
    assert cfg.runtime.deterministic is False
    assert cfg.tracking.max_age == 45


def test_unknown_env_override_is_ignored() -> None:
    """Unknown sections and fields must warn, not crash."""
    cfg = SurveillanceConfig.default().apply_env_overrides(
        {"SURV_NOPE__FIELD": "1", "SURV_MALFORMED": "1"}
    )
    assert cfg.to_dict() == SurveillanceConfig.default().to_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {"detection": {"body_conf": 1.5}},
        {"detection": {"body_imgsz": 641}},
        {"association": {"centre_weight": 0.9}},
        {"tracking": {"low_threshold": 0.9}},
        {"recognition": {"architecture": "ir101"}},
        {"rendering": {"preset": "turbo"}},
        {"rendering": {"redaction_kernel": 30}},
        {"quality": {"min_norm": 60.0}},
        {"runtime": {"device": "tpu"}},
        {"search": {"target_far": 0.0}},
    ],
)
def test_invalid_values_are_rejected(payload) -> None:
    """Every out-of-domain value must raise at construction time."""
    with pytest.raises(ConfigError):
        SurveillanceConfig.from_dict(payload)


def test_cross_section_invariant() -> None:
    """A track can never reach fusion.min_samples if the store cap is lower."""
    cfg = SurveillanceConfig.from_dict(
        {"fusion": {"min_samples": 40, "max_samples": 64}, "quality": {"max_stored_per_track": 8}}
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_unreachable_tracker_threshold_rejected() -> None:
    """A tracker threshold below the detector floor is dead configuration."""
    cfg = SurveillanceConfig.from_dict(
        {"tracking": {"det_threshold": 0.1, "low_threshold": 0.05},
         "detection": {"body_conf": 0.5}}
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_zero_retention_forbids_persistence() -> None:
    """retention_days=0 must be incompatible with writing biometrics to disk."""
    cfg = SurveillanceConfig.from_dict(
        {"governance": {"retention_days": 0, "persist_embeddings": True}}
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_determinism_disables_cudnn_benchmark() -> None:
    """Requesting both determinism and autotuning must resolve in favour of determinism."""
    cfg = SurveillanceConfig.from_dict(
        {"runtime": {"deterministic": True, "cudnn_benchmark": True}}
    )
    assert cfg.runtime.cudnn_benchmark is False


def test_scaled_template_matches_canonical_at_112() -> None:
    """The template must pass through unchanged at its authoring resolution."""
    cfg = SurveillanceConfig.default()
    flat = [v for point in cfg.alignment.scaled_template() for v in point]
    expected = [v for point in cfg.alignment.reference_landmarks for v in point]
    assert flat == pytest.approx(expected)


def test_scaled_template_is_centred_after_padding() -> None:
    """Padding must shrink the template about the crop centre, not the origin."""
    cfg = SurveillanceConfig.from_dict({"alignment": {"padding_ratio": 0.25}})
    base = SurveillanceConfig.default().alignment.scaled_template()
    padded = cfg.alignment.scaled_template()
    cx = cfg.alignment.output_size[0] / 2.0
    base_span = max(p[0] for p in base) - min(p[0] for p in base)
    padded_span = max(p[0] for p in padded) - min(p[0] for p in padded)
    assert padded_span < base_span
    assert sum(p[0] for p in padded) / 5 == pytest.approx(
        cx + (sum(p[0] for p in base) / 5 - cx) / 1.25
    )


def test_scaled_template_doubles_at_224() -> None:
    """Doubling the crop size must double every template coordinate."""
    cfg = SurveillanceConfig.from_dict({"alignment": {"output_size": [224, 224]}})
    base = SurveillanceConfig.default().alignment.scaled_template()
    scaled = cfg.alignment.scaled_template()
    for (bx, by), (sx, sy) in zip(base, scaled):
        assert sx == pytest.approx(bx * 2)
        assert sy == pytest.approx(by * 2)


def test_paths_are_absolute_after_construction() -> None:
    """Relative directories must be anchored to the configured root."""
    cfg = SurveillanceConfig.default()
    assert cfg.paths.outputs_dir.is_absolute()
    assert cfg.paths.outputs_dir.parent == cfg.paths.root


def test_paths_ensure_creates_directories(tmp_path) -> None:
    """ensure() must be idempotent and create every configured directory."""
    cfg = SurveillanceConfig.from_dict({"paths": {"root": str(tmp_path)}})
    cfg.paths.ensure().ensure()
    assert cfg.paths.weights_dir.is_dir()
    assert cfg.paths.log_dir.is_dir()


def test_unsupported_container_rejected() -> None:
    """Only the whitelisted container formats may be configured."""
    with pytest.raises(ConfigError):
        SurveillanceConfig.from_dict({"video": {"input_path": "clip.avi"}})


def test_half_never_enabled_on_cpu() -> None:
    """FP16 must be gated on an actual CUDA device."""
    cfg = SurveillanceConfig.from_dict({"runtime": {"device": "cpu"}})
    assert cfg.runtime.use_half(True) is False


def test_global_accessor_round_trip() -> None:
    """set_config must install and validate; get_config must return it."""
    cfg = SurveillanceConfig.default()
    assert set_config(cfg) is cfg
    assert get_config() is cfg
    with pytest.raises(ConfigError):
        set_config({"not": "a config"})  # type: ignore[arg-type]


def test_classvar_is_not_a_dataclass_field() -> None:
    """Validation whitelists must not be constructor arguments."""
    from dataclasses import fields

    from configs.config import RecognitionConfig, RenderingConfig

    assert "SUPPORTED_ARCHITECTURES" not in {f.name for f in fields(RecognitionConfig())}
    assert "VALID_PRESETS" not in {f.name for f in fields(RenderingConfig())}


def test_default_yaml_is_not_stale() -> None:
    """configs/default.yaml must match what the generator would emit.

    Guards against someone adding a dataclass field and forgetting to
    regenerate the YAML, which would otherwise fail silently.
    """
    result = subprocess.run(
        [sys.executable, "scripts/generate_default_config.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "configs/default.yaml is stale. "
        "Run: python scripts/generate_default_config.py"
    )


def test_default_yaml_loads_and_validates() -> None:
    """The shipped YAML must produce a valid configuration."""
    SurveillanceConfig.from_file(REPO_ROOT / "configs" / "default.yaml").validate()