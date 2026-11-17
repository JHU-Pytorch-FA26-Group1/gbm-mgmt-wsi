"""Regression tests for training/config.py: the committed configs, merging, overrides and checks."""
import shutil
from pathlib import Path

import pytest
import yaml

from training import config

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "configs" / "train.yaml"


def load(**kwargs):
    """Load the committed config for swin_v2_t unless the test says otherwise."""
    kwargs.setdefault("backbone", "swin_v2_t")
    return config.load_config(**kwargs)


def copy_configs(tmp_path):
    """A writable copy of train.yaml and the backbone files, for tests that edit them."""
    shutil.copy(TRAIN, tmp_path / "train.yaml")
    shutil.copytree(REPO / "configs" / "backbones", tmp_path / "backbones")
    return tmp_path / "train.yaml"


@pytest.mark.parametrize("backbone", config.BACKBONES)
def test_committed_configs_load(backbone):
    """Each committed backbone file merges with train.yaml into a config that passes validation."""
    settings, _ = load(backbone=backbone)
    assert settings["backbone"] == backbone


@pytest.mark.parametrize("override", [
    "optimizer=sgd", "scheduler_gamma=1.5", "scheduler=step", "early_stopping_patience=0",
    "validation_tiles_per_patient=0", "attention_hidden=0", "smoke_train_on_validation=maybe",
    "wandb_project=a/b",
    "stages_real=[{learning_rate: 1.0e-5, epochs: 0, class_weighted: true}]",
    "stages_real=[{learning_rate: 1.0e-5, epochs: 1, class_weighted: 1}]",
])
def test_out_of_range_optional_values_and_stage_entries_are_refused(override):
    """Optional values and stage entries are checked too, not only the required numbers."""
    with pytest.raises(RuntimeError, match="bad config values"):
        load(overrides=[override])


def test_the_scheduler_needs_its_step_size_and_gamma_together():
    """A step scheduler with only one of its two settings must stop the run."""
    with pytest.raises(RuntimeError, match="bad config values"):
        load(overrides=["scheduler=step", "scheduler_step_size=10"])
    settings, _ = load(overrides=["scheduler=step", "scheduler_step_size=10",
                                  "scheduler_gamma=0.5"])
    assert settings["scheduler"] == "step"


def test_jpeg_quality_accepts_both_ends_of_its_range():
    """1 and 100 are valid qualities."""
    for quality in (1, 100):
        assert load(overrides=[f"jpeg_quality={quality}"])[0]["jpeg_quality"] == quality


def test_overrides_are_read_as_yaml():
    """Command-line overrides read their value as YAML and are recorded."""
    settings, record = load(overrides=["max_tiles_per_patient=1000", "jpeg_quality=90",
                                       "smoke_train_on_validation=true", "wandb_project=null"])
    assert settings["max_tiles_per_patient"] == 1000
    assert settings["jpeg_quality"] == 90
    assert settings["smoke_train_on_validation"] is True
    assert settings["wandb_project"] is None
    assert record["overrides"]["jpeg_quality"] == 90


def test_same_settings_same_hash_and_a_change_changes_it():
    """The settings hash is stable for equal settings and changes with any value."""
    first = load()[1]["settings_sha256"]
    assert load()[1]["settings_sha256"] == first
    assert load(overrides=["seed=1"])[1]["settings_sha256"] != first


@pytest.mark.parametrize("override", [
    "tiles_per_bag=0", "max_tiles_per_patient=0", "max_tiles_per_patient=true",
    "max_eval_tiles_per_patient=-5", "jpeg_quality=0", "jpeg_quality=101", "jpeg_quality=90.5",
    "attention_dropout=1", "patients_per_update=0", "seed=-1",
    "mixed_precision=maybe", "macenko_reference_patch_ids={'7': 'x'}",
    "label_unmethylated_value=Methylated",
])
def test_bad_values_are_refused(override):
    """Each out-of-range or wrong-type value stops the run."""
    with pytest.raises(RuntimeError, match="bad config values"):
        load(overrides=[override])


@pytest.mark.parametrize("override", ["not_a_key=1", "tiles_per_bag", "backbone=swin_v2_t"])
def test_unknown_or_malformed_overrides_are_refused(override):
    """An unknown key, a missing "=", or the backbone key is refused."""
    with pytest.raises(RuntimeError, match="override must be key=value"):
        load(overrides=[override])


def test_backbone_must_be_chosen_and_known():
    """Only the two backbones in the proposal are accepted."""
    with pytest.raises(RuntimeError, match="backbone must be one of"):
        load(backbone="resnet50")
    with pytest.raises(RuntimeError, match="backbone must be one of"):
        load(backbone=None)


def test_step_scheduler_needs_its_two_settings():
    """scheduler=step needs a step size and a gamma."""
    with pytest.raises(RuntimeError, match="scheduler_step_size"):
        load(overrides=["scheduler=step"])
    settings, _ = load(overrides=["scheduler=step", "scheduler_step_size=10",
                                  "scheduler_gamma=0.5"])
    assert settings["scheduler_gamma"] == 0.5


def test_stage_lists_have_the_proposal_shape():
    """Real-only has one stage and synthetic-then-real two, each with all three keys."""
    with pytest.raises(RuntimeError, match="stages_real"):
        load(overrides=["stages_real=[]"])
    with pytest.raises(RuntimeError, match="stages_synthetic"):
        load(overrides=["stages_synthetic=[{learning_rate: 1e-5, epochs: 1, "
                        "class_weighted: true}]"])
    with pytest.raises(RuntimeError, match="stages_real"):
        load(overrides=["stages_real=[{learning_rate: 0, epochs: 1, class_weighted: true}]"])
    with pytest.raises(RuntimeError, match="stages_real"):
        load(overrides=["stages_real=[{learning_rate: 1e-5, epochs: 1}]"])


def test_missing_or_extra_key_in_the_shared_file_is_refused(tmp_path):
    """train.yaml must hold exactly the known keys."""
    path = copy_configs(tmp_path)
    data = yaml.safe_load(path.read_text())
    del data["seed"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(RuntimeError, match="missing or unexpected"):
        config.load_config(path, backbone="swin_v2_t")
    data["seed"] = 0
    data["surprise"] = 1
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(RuntimeError, match="surprise"):
        config.load_config(path, backbone="swin_v2_t")


def test_backbone_file_overrides_only_known_keys(tmp_path):
    """A backbone file overrides shared values for its own model only, and only known keys."""
    path = copy_configs(tmp_path)
    (tmp_path / "backbones" / "swin_v2_t.yaml").write_text("attention_hidden: 64\n")
    settings, record = config.load_config(path, backbone="swin_v2_t")
    assert settings["attention_hidden"] == 64
    assert record["backbone_config_keys"] == ["attention_hidden"]
    other, _ = config.load_config(path, backbone="convnext_v2_t")
    assert other["attention_hidden"] == 128
    (tmp_path / "backbones" / "swin_v2_t.yaml").write_text("backbone: convnext_v2_t\n")
    with pytest.raises(RuntimeError, match="unknown keys"):
        config.load_config(path, backbone="swin_v2_t")


def test_command_line_beats_the_backbone_file(tmp_path):
    """Command-line overrides win over the backbone file."""
    path = copy_configs(tmp_path)
    (tmp_path / "backbones" / "swin_v2_t.yaml").write_text("attention_hidden: 64\n")
    settings, _ = config.load_config(path, backbone="swin_v2_t",
                                     overrides=["attention_hidden=32"])
    assert settings["attention_hidden"] == 32


def test_pool_limits_refuse_what_the_package_cannot_supply():
    """A pool above every patient's packaged count is refused; the largest count is allowed."""
    settings, _ = load(overrides=["max_tiles_per_patient=5000", "max_eval_tiles_per_patient=null"])
    counts = {"P1": 433, "P2": 38556}
    config.check_pool_limits(settings, counts)
    settings["max_tiles_per_patient"] = 38556
    config.check_pool_limits(settings, counts)
    settings["max_tiles_per_patient"] = 38557
    with pytest.raises(RuntimeError, match="38556"):
        config.check_pool_limits(settings, counts)
    settings["max_tiles_per_patient"] = None
    settings["max_eval_tiles_per_patient"] = 1_000_000
    with pytest.raises(RuntimeError, match="max_eval_tiles_per_patient"):
        config.check_pool_limits(settings, counts)
