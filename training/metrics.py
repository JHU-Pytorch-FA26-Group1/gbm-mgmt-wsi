"""Reporting metrics over patient predictions.

The proposal (3.4, 3.5) names the metrics: AUROC with confidence intervals, balanced accuracy, F1
and recall for each class, thresholds chosen on TCGA out-of-fold predictions and frozen before
CPTAC, and paired patient-level bootstrap intervals for AUROC differences. It does not fix the
threshold rule or the bootstrap settings, so those are stubs until the team decides.
"""
import numpy as np


def threshold_metrics(labels, scores, threshold):
    """Balanced accuracy, F1 and recall per class when `score >= threshold` predicts methylated."""
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        raise ValueError("recall per class needs both classes")
    predicted = np.asarray(scores) >= threshold
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    true_negative = int(np.sum(~predicted & ~labels))
    recall_methylated = true_positive / (true_positive + false_negative)
    recall_unmethylated = true_negative / (true_negative + false_positive)
    denominator = 2 * true_positive + false_positive + false_negative
    return {"balanced_accuracy": (recall_methylated + recall_unmethylated) / 2,
            "f1": 2 * true_positive / denominator,
            "recall_methylated": recall_methylated, "recall_unmethylated": recall_unmethylated}


def choose_threshold(labels, scores):
    """The decision threshold chosen on TCGA out-of-fold predictions."""
    # TODO: the rule is not decided (the proposal only says it is chosen on out-of-fold predictions)
    raise NotImplementedError("threshold rule is not decided")


def bootstrap_interval(labels, scores):
    """Confidence interval for AUROC, or for a paired AUROC difference, over patient resamples."""
    # TODO: resample count, level and interval type are not decided
    raise NotImplementedError("bootstrap settings are not decided")
