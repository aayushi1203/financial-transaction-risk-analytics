import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Balance consistency checks (classic PaySim logic)
    df["orig_balance_diff"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    ).abs()

    df["dest_balance_diff"] = (
        df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    ).abs()

    # 2. Flags for suspicious balance behavior
    df["orig_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["dest_balance_zero"] = (df["oldbalanceDest"] == 0).astype(int)

    # 3. Transaction amount relative to origin balance
    df["amount_over_orig_balance"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0
    )

    # 4. Transaction type encoding (keep interpretable)
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # 5. Log-transform amount (heavy-tailed)
    df["log_amount"] = np.log1p(df["amount"])

    return df
