"""Training configuration: merge, override and validate.

Later sources win: configs/train.yaml (shared), configs/backbones/<backbone>.yaml (that model's
differences), then `key=value` overrides read as YAML.

    from training import config
    settings, record = config.load_config(backbone="swin_v2_t", overrides=["jpeg_quality=90"])
"""
import hashlib
import json
from pathlib import Path

import yaml

from common import REPO_ROOT, is_int, is_name, is_num

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
BACKBONES = ("swin_v2_t", "convnext_v2_t")
SCHEDULERS = ("none", "step")
FOLD_NAMES = ("0", "1", "2", "3", "4", "final")
STAGE_KEYS = {"learning_rate", "epochs", "class_weighted"}

# `backbone` comes from the command line and is not allowed in a file.
FILE_KEYS = {
    "package_dir", "folds_csv", "labels_csv", "label_id_column", "label_status_column",
    "label_methylated_value", "label_unmethylated_value", "seed", "tiles_per_bag",
    "max_tiles_per_patient", "max_eval_tiles_per_patient", "jpeg_quality",
    "smoke_train_on_validation", "macenko_reference_patch_ids", "attention_hidden",
    "attention_dropout", "optimizer", "patients_per_update", "mixed_precision", "scheduler",
    "scheduler_step_size", "scheduler_gamma", "early_stopping_patience",
    "validation_tiles_per_patient", "stages_real", "stages_synthetic", "output_dir",
    "wandb_project",
}
KEYS = FILE_KEYS | {"backbone"}


def _is_bool(value):
    return isinstance(value, bool)


def _is_text(value):
    return isinstance(value, str) and value.strip() != ""


def _is_count_or_none(value):
    """A whole number >= 1, or None."""
    return value is None or (is_int(value) and value >= 1)


def _is_stage(value):
    return (isinstance(value, dict) and set(value) == STAGE_KEYS
            and is_num(value["learning_rate"]) and value["learning_rate"] > 0
            and is_int(value["epochs"]) and value["epochs"] >= 1
            and _is_bool(value["class_weighted"]))


def _is_stage_list(value, length):
    return isinstance(value, list) and len(value) == length and all(_is_stage(v) for v in value)


def _is_reference_map(value):
    return (isinstance(value, dict)
            and all(k in FOLD_NAMES and _is_text(v) for k, v in value.items()))


CHECKS = {
    "backbone": lambda v, s: v in BACKBONES,
    "package_dir": lambda v, s: _is_text(v),
    "folds_csv": lambda v, s: _is_text(v),
    "labels_csv": lambda v, s: _is_text(v),
    "label_id_column": lambda v, s: _is_text(v),
    "label_status_column": lambda v, s: _is_text(v),
    "label_methylated_value": lambda v, s: _is_text(v),
    "label_unmethylated_value": lambda v, s: _is_text(v) and v != s["label_methylated_value"],
    "seed": lambda v, s: is_int(v) and v >= 0,
    "tiles_per_bag": lambda v, s: is_int(v) and v >= 1,
    "max_tiles_per_patient": lambda v, s: _is_count_or_none(v),
    "max_eval_tiles_per_patient": lambda v, s: _is_count_or_none(v),
    "jpeg_quality": lambda v, s: v is None or (is_int(v) and 1 <= v <= 100),
    "smoke_train_on_validation": lambda v, s: _is_bool(v),
    "macenko_reference_patch_ids": lambda v, s: _is_reference_map(v),
    "attention_hidden": lambda v, s: is_int(v) and v >= 1,
    "attention_dropout": lambda v, s: is_num(v) and 0 <= v < 1,
    "optimizer": lambda v, s: v == "adam",
    "patients_per_update": lambda v, s: is_int(v) and v >= 1,
    "mixed_precision": lambda v, s: _is_bool(v),
    "scheduler": lambda v, s: v in SCHEDULERS,
    "scheduler_step_size": lambda v, s: (_is_count_or_none(v)
                                         and (s["scheduler"] != "step" or v is not None)),
    "scheduler_gamma": lambda v, s: ((v is None or (is_num(v) and 0 < v <= 1))
                                     and (s["scheduler"] != "step" or v is not None)),
    "early_stopping_patience": lambda v, s: _is_count_or_none(v),
    "validation_tiles_per_patient": lambda v, s: _is_count_or_none(v),
    "stages_real": lambda v, s: _is_stage_list(v, 1),
    "stages_synthetic": lambda v, s: _is_stage_list(v, 2),
    "output_dir": lambda v, s: _is_text(v),
    "wandb_project": lambda v, s: v is None or is_name(v),
}


def _read_mapping(path, what):
    """A YAML file as (dict, SHA-256); an empty file is {}."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{what} is not a YAML mapping: {path}")
    return data, hashlib.sha256(text.encode()).hexdigest()


def _parse_overrides(pairs):
    """Parse "key=value" strings, reading each value as YAML."""
    overrides = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or key not in FILE_KEYS:
            raise RuntimeError(f"override must be key=value with a known key: {pair!r}")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def _validate(settings):
    """Require exactly the known keys and that each value passes its check."""
    if set(settings) != KEYS:
        raise RuntimeError(f"config keys {sorted(set(settings) ^ KEYS)} missing or unexpected")
    bad = [key for key, check in CHECKS.items() if not check(settings[key], settings)]
    if bad:
        raise RuntimeError("bad config values: "
                           + ", ".join(f"{k}={settings[k]!r}" for k in bad))


def _settings_sha256(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def load_config(path=DEFAULT_TRAIN_CONFIG, backbone=None, overrides=(), backbone_dir=None):
    """Return (settings, record); `record` holds the file hashes, overrides and merged hash."""
    if backbone not in BACKBONES:
        raise RuntimeError(f"backbone must be one of {BACKBONES}: {backbone!r}")
    shared, shared_sha = _read_mapping(path, "config")
    if set(shared) != FILE_KEYS:
        raise RuntimeError(f"config keys {sorted(set(shared) ^ FILE_KEYS)} missing or unexpected "
                           f"in {path}")
    folder = Path(backbone_dir) if backbone_dir else Path(path).parent / "backbones"
    backbone_path = folder / f"{backbone}.yaml"
    specific, specific_sha = _read_mapping(backbone_path, "backbone config")
    unknown = sorted(set(specific) - FILE_KEYS)
    if unknown:
        raise RuntimeError(f"{backbone_path} has unknown keys: {unknown}")
    changes = _parse_overrides(overrides)
    settings = {**shared, **specific, **changes, "backbone": backbone}
    _validate(settings)
    record = {"train_config_sha256": shared_sha, "backbone_config_sha256": specific_sha,
              "backbone_config_keys": sorted(specific), "overrides": changes,
              "settings_sha256": _settings_sha256(settings)}
    return settings, record


def check_pool_limits(settings, tiles_per_patient):
    """Refuse a pool limit above the largest packaged tile count of any patient.

    A limit above one patient's count keeps all of that patient's tiles.
    """
    largest = max(tiles_per_patient.values())
    for key in ("max_tiles_per_patient", "max_eval_tiles_per_patient"):
        value = settings[key]
        if value is not None and value > largest:
            raise RuntimeError(f"{key}={value} is above the largest number of packaged tiles "
                               f"for any patient ({largest})")
