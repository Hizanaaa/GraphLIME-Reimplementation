"""
Hilbert-Schmidt Independence Criterion (HSIC).

Implements the kernel centering operations and empirical HSIC
estimator used by GraphLIME.
"""

from __future__ import annotations

import torch
from torch import Tensor


class HSIC:
    """
    Compute HSIC scores between kernel matrices.

    The centering matrix is cached based on its size to avoid
    rebuilding it repeatedly during multiple explanations.
    """

    def __init__(self) -> None:
        self._cache: dict[int, Tensor] = {}

    def centering_matrix(
        self,
        n: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Return the n×n centering matrix.

        H = I - (1/n)11ᵀ
        """

        cached = self._cache.get(n)

        if (
            cached is not None
            and cached.device == device
            and cached.dtype == dtype
        ):
            return cached

        identity = torch.eye(
            n,
            device=device,
            dtype=dtype,
        )

        ones = torch.ones(
            (n, n),
            device=device,
            dtype=dtype,
        ) / n

        H = identity - ones

        self._cache[n] = H

        return H

    def center(
        self,
        kernel: Tensor,
    ) -> Tensor:
        """
        Center a kernel matrix.
        """

        n = kernel.size(0)

        H = self.centering_matrix(
            n,
            device=kernel.device,
            dtype=kernel.dtype,
        )

        return H @ kernel @ H

    def compute(
        self,
        K: Tensor,
        L: Tensor,
    ) -> Tensor:
        """
        Compute empirical HSIC.

        Parameters
        ----------
        K : Tensor
            Feature kernel.

        L : Tensor
            Target kernel.

        Returns
        -------
        Tensor
            HSIC score.
        """

        if K.shape != L.shape:
            raise ValueError(
                "Kernel matrices must have identical shapes."
            )

        n = K.size(0)

        Kc = self.center(K)
        Lc = self.center(L)

        score = torch.trace(Kc @ Lc)

        score = score / ((n - 1) ** 2)

        return score

if __name__ == "__main__":

    torch.manual_seed(42)

    X = torch.randn(10, 6)
    Y = torch.randn(10, 6)

    from graphlime.kernel import KernelComputer

    kernel = KernelComputer()

    K = kernel.rbf(X)
    L = kernel.rbf(Y)

    hsic = HSIC()

    score = hsic.compute(K, L)

    print("HSIC Score:", score.item())

    assert score.ndim == 0
    assert torch.isfinite(score)

    centered = hsic.center(K)

    assert centered.shape == K.shape

    print("\nAll tests passed.")
