import os
import pandas as pd

from src.preprocess import make_xy
from src.train import train_log_reg
from src.evaluate import eval_binary
from src.viz import plot_curves

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    train_path = "data/processed/train_sample.csv"
    test_path = "data/processed/test_sample.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = make_xy(train_df, target_col="isFraud")
    X_test, y_test = make_xy(test_df, target_col="isFraud")

    # Train baseline model
    model = train_log_reg(X_train, y_train)

    # Risk scores
    y_score = model.predict_proba(X_test)[:, 1]

    # Evaluate (PR-AUC emphasized + top-k capture)
    metrics = eval_binary(y_test, y_score, threshold=0.5, out_path="results/metrics.json")
    print("Done. Metrics (high level):")
    print({k: metrics[k] for k in ["roc_auc", "avg_precision", "threshold"]})
    print("Best F1 operating point:", metrics["best_f1"])
    print("Top-k capture (review queue simulation):")
    print(metrics["top_k_capture"])

    # Plots
    plot_curves(model, X_test, y_test)

if __name__ == "__main__":
    main()
