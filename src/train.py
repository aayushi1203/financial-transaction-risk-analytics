from sklearn.linear_model import LogisticRegression

def train_log_reg(X_train, y_train):
    """
    Baseline risk model. Class-weighting handles imbalance without oversampling.
    """
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None
    )
    model.fit(X_train, y_train)
    return model
