import pandas as pd
import numpy as np
from typing import Tuple, List

DEFAULT_DROP_COLS = [
    "nameOrig",  # high-cardinality IDs (not stable across datasets)
    "nameDest",
]

def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing essentials
    df = df.dropna(subset=["step", "amount", "isFraud"])

    # Ensure target is int 0/1
    df["isFraud"] = df["isFraud"].astype(int)

    return df

def drop_leaky_or_id_cols(df: pd.DataFrame, drop_cols: List[str] = None) -> pd.DataFrame:
    df = df.copy()
    drop_cols = drop_cols or DEFAULT_DROP_COLS
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

def time_split(
    df: pd.DataFrame,
    time_col: str = "step",
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split: train on earlier timesteps, test on later timesteps.
    Prevents leakage that happens with random splits in temporal transaction data.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    cutoff = int((1 - test_size) * len(df))
    train_df = df.iloc[:cutoff].copy()
    test_df = df.iloc[cutoff:].copy()
    return train_df, test_df

def stratified_downsample(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    max_nonfraud: int = 200_000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Keep ALL fraud rows. Downsample non-fraud to a fixed cap.
    This preserves rare-event signal while keeping runtime reasonable.
    """
    fraud = df[df[target_col] == 1]
    nonfraud = df[df[target_col] == 0]

    if len(nonfraud) > max_nonfraud:
        nonfraud = nonfraud.sample(n=max_nonfraud, random_state=random_state)

    out = pd.concat([fraud, nonfraud], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out

def make_xy(df: pd.DataFrame, target_col: str = "isFraud"):
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y
