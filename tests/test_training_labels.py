"""Regression tests for training/labels.py: label reading, the fold-map check and the weight."""
import csv
import os
from pathlib import Path

import pytest

from common import read_folds
from training import config, labels

REPO = Path(__file__).resolve().parent.parent


def settings_for(tmp_path, rows, header=("source", "patient_id", "mgmt_status")):
    """Settings pointing at a temporary labels CSV with the given rows."""
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    settings, _ = config.load_config(backbone="swin_v2_t", overrides=[f"labels_csv={path}"])
    return settings


def test_committed_labels_match_the_fold_map_and_the_proposal_counts():
    """The committed labels cover the fold map's patients, with the proposal's class counts."""
    settings, _ = config.load_config(backbone="swin_v2_t")
    found = labels.read_labels(settings)
    folds = read_folds(config.repo_path(settings["folds_csv"]))
    labels.check_against_folds(found, folds)
    tcga = [patient for source, patient in folds if source == "TCGA"]
    cptac = [patient for source, patient in folds if source == "CPTAC"]
    assert labels.class_counts(found, tcga) == (64, 87)
    assert labels.class_counts(found, cptac) == (32, 51)


def test_status_outside_the_two_configured_values_is_refused(tmp_path):
    """A mistyped status must not silently become unmethylated."""
    settings = settings_for(tmp_path, [("TCGA", "P1", "methylated")])
    with pytest.raises(RuntimeError, match="P1.*'methylated'"):
        labels.read_labels(settings)


def test_a_patient_listed_twice_is_refused(tmp_path):
    """Two rows for one patient could disagree, so the file is refused."""
    rows = [("TCGA", "P1", "Methylated"), ("TCGA", "P1", "Unmethylated")]
    settings = settings_for(tmp_path, rows)
    with pytest.raises(RuntimeError, match="P1 appears more than once"):
        labels.read_labels(settings)


def test_missing_status_column_is_refused(tmp_path):
    """A renamed column fails with its name instead of a KeyError."""
    settings = settings_for(tmp_path, [("TCGA", "P1", "Methylated")],
                            header=("source", "patient_id", "status"))
    with pytest.raises(RuntimeError, match="needs columns"):
        labels.read_labels(settings)


def test_a_patient_without_a_label_or_a_fold_is_named():
    """Both directions of disagreement between labels and the fold map are reported."""
    folds = {("TCGA", "P1"): "0", ("TCGA", "P2"): "1"}
    with pytest.raises(RuntimeError, match=r"1 without a label \['P2'\]"):
        labels.check_against_folds({"P1": 1}, folds)
    with pytest.raises(RuntimeError, match=r"1 without a fold \['P3'\]"):
        labels.check_against_folds({"P1": 1, "P2": 0, "P3": 1}, folds)


def test_weight_follows_the_patients_passed_so_an_empty_bag_can_be_left_out():
    """Dropping a patient with no tiles changes the weight, as proposal 3.5 requires."""
    found = {"M1": 1, "M2": 1, "U1": 0, "U2": 0, "U3": 0, "EMPTY": 0}
    everyone = ["M1", "M2", "U1", "U2", "U3", "EMPTY"]
    assert labels.methylated_weight(found, everyone) == pytest.approx(4 / 2)
    with_tiles = [patient for patient in everyone if patient != "EMPTY"]
    assert labels.methylated_weight(found, with_tiles) == pytest.approx(3 / 2)


def test_a_single_class_cannot_be_weighted():
    """No methylated patients would divide by zero; no unmethylated would give a weight of zero."""
    with pytest.raises(RuntimeError, match="single class"):
        labels.methylated_weight({"U1": 0, "U2": 0}, ["U1", "U2"])
    with pytest.raises(RuntimeError, match="single class"):
        labels.methylated_weight({"M1": 1}, ["M1"])


def test_counting_an_unlabelled_patient_is_refused():
    """A patient missing from the labels must not be counted as unmethylated."""
    with pytest.raises(RuntimeError, match="no label for 1 patients"):
        labels.class_counts({"P1": 1}, ["P1", "P2"])


def test_relative_config_paths_resolve_against_the_repo_not_the_working_directory(tmp_path):
    """Running from another folder must still find manifests/patient_labels.csv."""
    settings, _ = config.load_config(backbone="swin_v2_t")
    expected = labels.read_labels(settings)
    here = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert labels.read_labels(settings) == expected
    finally:
        os.chdir(here)
