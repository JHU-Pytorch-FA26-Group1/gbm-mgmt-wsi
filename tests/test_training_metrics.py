"""Regression tests for training/metrics.py."""
import pytest

from training import metrics


def test_thresholded_metrics_on_a_hand_counted_case():
    """tp 1, fp 1, fn 1, tn 2 at a threshold of 0.5."""
    result = metrics.threshold_metrics([1, 1, 0, 0, 0], [0.9, 0.2, 0.8, 0.1, 0.1], 0.5)
    assert result == pytest.approx({"balanced_accuracy": (0.5 + 2 / 3) / 2, "f1": 0.5,
                                    "recall_methylated": 0.5, "recall_unmethylated": 2 / 3})


def test_balanced_accuracy_ignores_the_class_sizes():
    """Calling all nine unmethylated patients right and the one methylated patient wrong is 0.5."""
    result = metrics.threshold_metrics([1] + [0] * 9, [0.1] * 10, 0.5)
    assert result["balanced_accuracy"] == 0.5


def test_a_score_equal_to_the_threshold_is_called_methylated():
    """A threshold taken from a score in the data must put that patient on the positive side."""
    result = metrics.threshold_metrics([1, 0], [0.5, 0.1], 0.5)
    assert result["recall_methylated"] == 1.0


def test_a_single_class_is_refused_rather_than_divided_by_zero():
    """No per-class recall exists without both classes."""
    with pytest.raises(ValueError):
        metrics.threshold_metrics([0, 0], [0.1, 0.2], 0.5)
