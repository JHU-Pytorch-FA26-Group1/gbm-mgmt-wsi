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
