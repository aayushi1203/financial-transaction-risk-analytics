import pandas as pd

def load_paysim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic sanity checks
    expected_cols = {
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud"
    }

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df
