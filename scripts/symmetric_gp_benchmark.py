"""Stable square-root coordinates for the unchanged binned Gaussian prior."""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from binned_gp_benchmark import BinnedGPPotential, covariance_factors


class SymmetricSquareRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        symmetric = (matrix + matrix.transpose(-1, -2)) / 2
        values, vectors = torch.linalg.eigh(symmetric)
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError("The covariance square root requires a positive definite matrix")
        roots = values.sqrt()
        ctx.save_for_backward(roots, vectors)
        return (vectors * roots.unsqueeze(-2)) @ vectors.transpose(-1, -2)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        roots, vectors = ctx.saved_tensors
        symmetric = (gradient + gradient.transpose(-1, -2)) / 2
        transformed = vectors.transpose(-1, -2) @ symmetric @ vectors
        # Differentiate S*S=A through S*dS+dS*S=dA. Sums of root eigenvalues
        # remain nonzero at repeated eigenvalues; eigenvector-gap derivatives do not.
        solution = transformed / (roots.unsqueeze(-1) + roots.unsqueeze(-2))
        return vectors @ solution @ vectors.transpose(-1, -2)


def symmetric_square_root(matrix):
    """Principal root with the first derivative needed by HMC, including ties."""
    return SymmetricSquareRoot.apply(matrix)


def symmetric_correlated_field(white, log_lengths, sigma, centers, jitter):
    matrices = torch.stack(covariance_factors(log_lengths, centers, jitter))
    factors = symmetric_square_root(matrices)
    bins = len(centers)
    field = white.reshape(bins, bins, bins)
    field = torch.einsum("ia,abc->ibc", factors[0], field)
    field = torch.einsum("jb,ibc->ijc", factors[1], field)
    field = torch.einsum("kc,ijc->ijk", factors[2], field)
    return sigma*field.reshape(-1)


class SymmetricBinnedGPPotential(BinnedGPPotential):
    def field(self, coordinate):
        return symmetric_correlated_field(coordinate[4:], coordinate[1:4], coordinate[0].exp(),
                                          self.centers, self.jitter)
