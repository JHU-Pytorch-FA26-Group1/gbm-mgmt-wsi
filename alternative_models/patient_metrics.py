"""Patient-level predictions and confusion counts; no tile-level scoring."""
import math


def summarize(rows, threshold=0.5):
    if not rows or not 0 <= threshold <= 1:
        raise ValueError("Predictions and a threshold in [0, 1] required")
    if len({row["patient_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate patient prediction")
    counts = {name: 0 for name in ("tn", "fp", "fn", "tp")}
    for row in rows:
        score, label = row["score"], row["label"]
        if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid patient score/label")
        prediction = int(score >= threshold)
        row.update(predicted_label=prediction, correct=prediction == label, threshold=threshold)
        counts[("tn", "fp", "fn", "tp")[label * 2 + prediction]] += 1
    tn, fp, fn, tp = (counts[name] for name in ("tn", "fp", "fn", "tp"))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    sensitivity, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    return dict(patients=len(rows), threshold=threshold, positive_class="methylated", **counts,
                confusion_matrix=[[tn, fp], [fn, tp]],
                matrix_rows="true [unmethylated, methylated]", matrix_columns="predicted [unmethylated, methylated]",
                accuracy=(tn+tp)/len(rows),
                balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
                precision=ratio(tp, tp+fp), sensitivity=sensitivity, specificity=specificity,
                f1=ratio(2*tp, 2*tp+fp+fn))
