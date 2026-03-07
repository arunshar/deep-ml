import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier and return a predict function.
    """

    # Ensure numpy arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).astype(int)

    # Deterministic, strong baseline for standardized numeric data
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=0
    )

    clf.fit(X_train, y_train)

    def predict(X):
        X = np.asarray(X)
        preds = clf.predict(X)          # already 0/1 labels
        return preds.astype(int)

    return predict