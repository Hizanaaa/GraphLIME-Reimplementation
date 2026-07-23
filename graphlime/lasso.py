"""
HSIC-Lasso optimization utilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoLars
from torch import Tensor


class HSICLasso:
    """
    Sparse feature selection using LassoLars.
    """

    def __init__(
        self,
        alpha: float = 0.01,
    ) -> None:
        self.alpha = alpha

    @staticmethod
    def _flatten_kernel(kernel: Tensor) -> np.ndarray:
        """
        Flatten an N×N kernel matrix into a vector.
        """
        return kernel.detach().cpu().numpy().reshape(-1)

    def fit(
        self,
        feature_kernels,
        target_kernel: Tensor,
    ):
        """
        Fit HSIC-Lasso.

        Parameters
        ----------
        feature_kernels
            Iterable of (feature_index, kernel_matrix).

        target_kernel
            Kernel of latent embeddings.

        Returns
        -------
        dict[int, float]
            Selected feature coefficients.
        """

        X = []
        feature_ids = []

        for feature_idx, kernel in feature_kernels:
            X.append(self._flatten_kernel(kernel))
            feature_ids.append(feature_idx)

        X = np.column_stack(X)

        y = self._flatten_kernel(target_kernel)

        model = LassoLars(
            alpha=self.alpha,
            fit_intercept=False,
        )

        model.fit(X, y)

        importance = {}

        for idx, coef in zip(feature_ids, model.coef_):
            if abs(coef) > 1e-10:
                importance[idx] = float(coef)

        return importance
