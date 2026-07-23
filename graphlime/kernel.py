"""
Kernel functions used by GraphLIME.

Implements linear and Gaussian (RBF) kernels from scratch.

Reviewer Note
-------------
Test gaps
1. Extremely high-dimensional inputs have not been benchmarked.
2. Sigma selection is currently user-defined.
3. Sparse feature matrices are not yet supported.
"""

from __future__ import annotations

import torch
from torch import Tensor


class KernelComputer:
    """
    Compute similarity kernels.
    """

    def __init__(
        self,
        sigma: float = 1.0,
    ) -> None:

        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        self.sigma = sigma

    def linear(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute linear kernel.

        Parameters
        ----------
        x : Tensor
            Shape (N, D)

        Returns
        -------
        Tensor
            Shape (N, N)
        """

        return x @ x.T

    def rbf(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute Gaussian RBF kernel.

        Parameters
        ----------
        x : Tensor
            Shape (N, D)

        Returns
        -------
        Tensor
            Shape (N, N)
        """

        squared_norm = (x ** 2).sum(dim=1).view(-1, 1)

        distances = (
            squared_norm
            + squared_norm.T
            - 2 * x @ x.T
        )

        kernel = torch.exp(
            -distances /
            (2 * self.sigma ** 2)
        )

        return kernel

    def pairwise_distance(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute pairwise squared Euclidean distances.
        """

        squared_norm = (x ** 2).sum(dim=1).view(-1, 1)

        return (
            squared_norm
            + squared_norm.T
            - 2 * x @ x.T
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(8, 5)

    kernel = KernelComputer(sigma=1.0)

    linear = kernel.linear(x)

    rbf = kernel.rbf(x)

    distances = kernel.pairwise_distance(x)

    print("Input Shape:", x.shape)

    print("Linear:", linear.shape)

    print("RBF:", rbf.shape)

    print("Distances:", distances.shape)

    assert linear.shape == (8, 8)
    assert rbf.shape == (8, 8)
    assert distances.shape == (8, 8)

    assert torch.allclose(
        torch.diag(rbf),
        torch.ones(8),
    )

    print("\nAll tests passed.")
