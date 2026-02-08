import numpy as np
import torch
import torch.nn as nn

class FeatureDeconfounder(nn.Module):
    """
    Removes the linear influence of metadata/confounding variables from features.
    
    The key insight is that we can model features as:
        f = X @ beta + residual
    where X is the metadata and residual is the part orthogonal to metadata.
    
    For batch processing with precomputed statistics:
        - fit(): compute Sigma_inv = (X^T X + reg*I)^{-1} on full training metadata
        - transform(): compute beta = Sigma_inv @ X^T @ f, return residual = f - X @ beta
    """
    
    def __init__(self, reg=1e-5):
        super().__init__()
        self.reg = reg
        self.Sigma_inv = None
        self.is_fitted = False
    
    def fit(self, metadata):
        """
        Precompute inverse covariance from training metadata.
        
        Args:
            metadata: Tensor of shape (N, K) - all training metadata
        
        Hints:
            - Compute Sigma = X^T @ X
            - Add small regularization (1e-5) for numerical stability
            - Store Sigma_inv for use in transform()
        """
        with torch.no_grad():
            X = metadata
            n_meta = X.shape[1]
            XtX = X.T @ X
            reg_I = self.reg * torch.eye(n_meta, device=X.device)
            self.Sigma_inv = torch.linalg.inv(XtX + reg_I)
            self.is_fitted = True
        return self
    
    def transform(self, features, metadata):
        """
        Remove metadata influence from features.
        
        Args:
            features: Tensor of shape (M, D) - batch of features
            metadata: Tensor of shape (M, K) - corresponding metadata
        
        Returns:
            Residualized features of shape (M, D)
        
        Hints:
            - Compute: beta = Sigma_inv @ (X^T @ f)
            - Return residual: f - X @ beta
        """

        assert self.is_fitted, "FeatureDeconfounder must be fitted before transform()"

        # Compute beta = Sigma_inv @ X^T @ f
        beta = self.Sigma_inv @ (metadata.T @ features)

        # Remove metadata contribution
        f_clean = features - (metadata @ beta)

        return f_clean
