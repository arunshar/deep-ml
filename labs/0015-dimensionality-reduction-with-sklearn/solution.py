import numpy as np
from sklearn.decomposition import PCA

class MyReducer:
    """
    Dimensionality reduction to 10 dimensions using PCA.
    Optimized for k-NN on MNIST (whitening helps distance-based classifiers).
    """

    def __init__(self):
        self.n_components = 10
        # 'full' is maximally deterministic; randomized is faster but needs random_state
        self.reducer = PCA(
            n_components=self.n_components,
            whiten=True,
            svd_solver="full"
        )

    def fit(self, X):
        X = np.asarray(X)
        self.reducer.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return self.reducer.transform(X)

    def fit_transform(self, X):
        X = np.asarray(X)
        return self.reducer.fit_transform(X)