"""Regression tests for the verify command in training/run.py."""
import hashlib
import json

import loader
from training import run

FILES = {"modeling_patches.csv": b"a,b\n", "tile.h5": b"x" * 64}


def make_package(folder, files):
    """A package folder holding `files` ({name: bytes}) and a manifest that records them."""
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    manifest = {"files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in files.items()}}
    (folder / loader.PACKAGE_MANIFEST).write_text(json.dumps(manifest))
    return folder


def test_an_intact_package_passes(tmp_path):
    """Nothing is reported when every file matches its manifest entry."""
    package = make_package(tmp_path / "pkg", FILES)
    assert run.main(["verify", "--package", str(package)]) == 0


def test_a_same_size_change_fails_verify_though_the_load_time_check_passes(tmp_path):
    """Corruption that keeps a tile file's size is found only by rehashing it."""
    package = make_package(tmp_path / "pkg", FILES)
    (package / "tile.h5").write_bytes(b"y" + b"x" * 63)
    assert not loader.package_problems(package)
    assert run.main(["verify", "--package", str(package)]) == 1


def test_a_missing_file_fails(tmp_path):
    """An interrupted copy that never created a file is a problem, not a skip."""
    package = make_package(tmp_path / "pkg", FILES)
    (package / "tile.h5").unlink()
    assert run.main(["verify", "--package", str(package)]) == 1


def test_a_folder_that_is_not_a_package_is_a_setup_problem(tmp_path):
    """A wrong path exits 2 with a message instead of a traceback or a false pass."""
    assert run.main(["verify", "--package", str(tmp_path / "nowhere")]) == 2
    (tmp_path / "empty").mkdir()
    assert run.main(["verify", "--package", str(tmp_path / "empty")]) == 2


def test_train_names_the_run_folder_after_backbone_fold_and_time(monkeypatch):
    """Without --name the folder says which model and fold it holds, so runs are not confused."""
    from training import experiment  # pylint: disable=import-outside-toplevel
    seen = []
    monkeypatch.setattr(experiment, "run_fold", lambda *args: seen.append(args[3]))
    assert run.main(["train", "--backbone", "swin_v2_t", "--fold", "2"]) == 0
    assert seen[0].name.startswith("swin_v2_t_fold2_")


def test_a_missing_config_file_is_a_setup_problem_not_a_traceback(tmp_path):
    """A wrong --config path must exit 2 like every other setup problem."""
    missing = tmp_path / "nowhere.yaml"
    assert run.main(["train", "--backbone", "swin_v2_t", "--fold", "0",
                     "--config", str(missing)]) == 2


def test_train_passes_the_fold_device_name_and_overrides_to_the_run(monkeypatch):
    """What the command line says is what run_fold receives, with the overrides recorded."""
    from training import experiment  # pylint: disable=import-outside-toplevel
    seen = {}

    def fake(settings, record, fold, run_dir, device):
        seen.update(tiles_per_bag=settings["tiles_per_bag"], record=record, fold=fold,
                    run_dir=run_dir, device=device)

    monkeypatch.setattr(experiment, "run_fold", fake)
    args = ["train", "--backbone", "swin_v2_t", "--fold", "3", "--name", "r", "--device", "cuda",
            "--set", "tiles_per_bag=8", "--set", "seed=4"]
    assert run.main(args) == 0
    assert seen["tiles_per_bag"] == 8 and seen["fold"] == 3 and seen["device"] == "cuda"
    assert seen["run_dir"].name == "r"
    assert seen["record"]["overrides"] == {"tiles_per_bag": 8, "seed": 4}


def test_a_refused_config_value_is_a_setup_problem():
    """An out-of-range override exits 2 with a message instead of a traceback."""
    assert run.main(["train", "--backbone", "swin_v2_t", "--fold", "0", "--set", "seed=-1"]) == 2


def test_an_unwritten_backbone_is_a_setup_problem(monkeypatch):
    """The ConvNeXt stub must stop the command with exit 2, not a traceback."""
    from training import experiment  # pylint: disable=import-outside-toplevel

    def stub(*_):
        raise NotImplementedError("convnext_v2_t backbone is not written yet")

    monkeypatch.setattr(experiment, "run_fold", stub)
    assert run.main(["train", "--backbone", "convnext_v2_t", "--fold", "0"]) == 2
