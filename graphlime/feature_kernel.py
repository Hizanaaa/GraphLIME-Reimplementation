"""
Feature-wise kernel generation for GraphLIME.

Each feature column is converted into its own kernel matrix.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

from graphlime.kernel import KernelComputer


class FeatureKernelBuilder:
    """
    Generate one kernel matrix for each feature.
    """

    def __init__(
        self,
        sigma: float = 1.0,
    ) -> None:
        self.kernel = KernelComputer(sigma=sigma)

    def feature_kernel(
        self,
        feature: Tensor,
    ) -> Tensor:
        """
        Compute the kernel for a single feature.

        Parameters
        ----------
        feature : Tensor
            Shape (N,)

        Returns
        -------
        Tensor
            Shape (N, N)
        """

        feature = feature.unsqueeze(1)

        return self.kernel.rbf(feature)

    def iter_feature_kernels(
        self,
        X: Tensor,
    ) -> Iterator[tuple[int, Tensor]]:
        """
        Lazily generate kernels for every feature.

        Parameters
        ----------
        X : Tensor
            Shape (N, D)

        Yields
        ------
        tuple[int, Tensor]
            Feature index and kernel matrix.
        """

        num_features = X.size(1)

        for feature_idx in range(num_features):

            feature = X[:, feature_idx]

            kernel = self.feature_kernel(feature)

            yield feature_idx, kernel

if __name__ == "__main__":

    torch.manual_seed(42)

    X = torch.randn(12, 5)

    builder = FeatureKernelBuilder(sigma=1.0)

    count = 0

    for feature_idx, kernel in builder.iter_feature_kernels(X):

        print(
            f"Feature {feature_idx}: {kernel.shape}"
        )

        assert kernel.shape == (12, 12)

        count += 1

    assert count == X.size(1)

    print("\nAll tests passed.")
