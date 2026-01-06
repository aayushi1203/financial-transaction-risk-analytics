import os
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def ensure_dir(path: str):
    # If something exists but isn't a directory, remove it (prevents your earlier issue)
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)

def plot_curves(model, X_test, y_test, fig_dir=os.path.join("reports", "figures")):
    ensure_dir(fig_dir)

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(fig_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(fig_dir, "pr_curve.png"), bbox_inches="tight")
    plt.close()
