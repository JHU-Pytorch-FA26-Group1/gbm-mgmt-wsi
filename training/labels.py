"""MGMT labels per patient: read them, check them against the fold map, and weight the loss."""
from common import read_rows
from training import config


def read_labels(settings):
    """{patient_id: 1 if methylated, 0 if unmethylated} from the labels CSV named in the config."""
    methylated = settings["label_methylated_value"]
    unmethylated = settings["label_unmethylated_value"]
    id_column = settings["label_id_column"]
    status_column = settings["label_status_column"]
    path = config.repo_path(settings["labels_csv"])
    labels = {}
    for row in read_rows(path):
        if id_column not in row or status_column not in row:
            raise RuntimeError(f"{path} needs columns {id_column!r} and {status_column!r}")
        patient, status = row[id_column], row[status_column]
        if patient in labels:
            raise RuntimeError(f"{path}: patient {patient} appears more than once")
        if status not in (methylated, unmethylated):
            raise RuntimeError(f"{path}: patient {patient} has status {status!r}, expected "
                               f"{methylated!r} or {unmethylated!r}")
        labels[patient] = int(status == methylated)
    return labels


def check_against_folds(labels, folds):
    """Require a label for every patient in the fold map and no label for anyone else."""
    patients = {patient for _, patient in folds}
    unlabelled = sorted(patients - set(labels))
    unfolded = sorted(set(labels) - patients)
    if unlabelled or unfolded:
        raise RuntimeError(f"labels and fold map disagree: {len(unlabelled)} without a label "
                           f"{unlabelled[:3]}, {len(unfolded)} without a fold {unfolded[:3]}")


def class_counts(labels, patients):
    """(methylated, unmethylated) counts over `patients`, each of whom must have a label."""
    missing = sorted(set(patients) - set(labels))
    if missing:
        raise RuntimeError(f"no label for {len(missing)} patients, for example {missing[:3]}")
    methylated = sum(labels[patient] for patient in patients)
    return methylated, len(patients) - methylated


def methylated_weight(labels, patients):
    """Loss multiplier for methylated patients: unmethylated count / methylated count.

    Pass only the patients that have tiles, so an empty bag does not skew it.
    """
    methylated, unmethylated = class_counts(labels, patients)
    if methylated == 0 or unmethylated == 0:
        raise RuntimeError(f"cannot weight a single class: {methylated} methylated, "
                           f"{unmethylated} unmethylated")
    return unmethylated / methylated
