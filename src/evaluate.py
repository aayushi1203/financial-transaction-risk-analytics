import json
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

def top_k_capture(y_true, y_score, k_frac: float = 0.01):
    """
    Rank transactions by risk score; flag top k% and measure fraud capture.
    Returns:
      - k (number flagged)
      - precision among flagged
      - recall (capture rate) among flagged
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_true)
    k = max(1, int(np.floor(k_frac * n)))

    idx = np.argsort(-y_score)[:k]
    flagged_true = y_true[idx]

    tp = flagged_true.sum()
    precision = tp / k
    recall = tp / max(1, y_true.sum())
    return k, float(precision), float(recall)

def find_best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # thresholds is length n-1
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(precision[best]), float(recall[best]), float(f1[best])

def eval_binary(y_true, y_score, threshold: float = 0.5, out_path: str = "results/metrics.json"):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Add operational metrics: top-k review queue simulation
    topk = {}
    for k_frac in [0.001, 0.005, 0.01, 0.02, 0.05]:
        k, prec, rec = top_k_capture(y_true, y_score, k_frac=k_frac)
        topk[str(k_frac)] = {"k_flagged": k, "precision": prec, "recall": rec}
    metrics["top_k_capture"] = topk

    # Best F1 threshold (useful for reporting; ops may prefer top-k instead)
    best_t, p, r, f1 = find_best_f1_threshold(y_true, y_score)
    metrics["best_f1"] = {"threshold": best_t, "precision": p, "recall": r, "f1": f1}

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
