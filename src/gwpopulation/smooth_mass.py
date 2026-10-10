"""Continuous mass densities with a proper Gaussian prior on log heights."""
from __future__ import annotations

import torch

from .core import BoundedMonotoneSpline


class SmoothLogMassDensity(BoundedMonotoneSpline):
    """Piecewise linear density with correlated, exponentiated knot heights.

    Standard normal ``coefficients`` induce a Gaussian prior on log heights.
    The length scale is measured in log10 solar mass, and the slope scale
    controls the log-density change across the complete support.
    """

    def __init__(self, coefficients, log10_min, log10_max, length_scale=.35,
                 amplitude=1.5, slope_scale=3., validate_args=None):
        if length_scale <= 0 or amplitude <= 0 or slope_scale < 0:
            raise ValueError('Invalid smooth mass prior scales')
        if not float(log10_max) > float(log10_min):
            raise ValueError('Mass bounds must be strictly ordered')
        count = coefficients.shape[-1]
        self.coefficients = coefficients
        self.length_scale = float(length_scale)
        self.amplitude = float(amplitude)
        self.slope_scale = float(slope_scale)
        knots = torch.linspace(float(log10_min), float(log10_max), count,
                               dtype=torch.float64, device=coefficients.device)
        separation = knots[:, None] - knots[None, :]
        trend = (knots - knots.mean()) / (float(log10_max) - float(log10_min))
        covariance = (1 + slope_scale**2 * torch.outer(trend, trend)
                      + amplitude**2 * torch.exp(-.5 * (separation / length_scale).square()))
        covariance = covariance + 1e-6 * torch.eye(count, dtype=knots.dtype, device=knots.device)
        self.log_height_factor = torch.linalg.cholesky(covariance).to(coefficients.dtype)
        log_heights = coefficients @ self.log_height_factor.T
        super().__init__(log_heights, coordinate='normalized', validate_args=validate_args)

    def _normalized_heights(self):
        shifted = self.logits - self.logits.amax(dim=-1, keepdim=True)
        heights = shifted.exp()
        area = (.5 * heights[..., 0] + heights[..., 1:-1].sum(-1)
                + .5 * heights[..., -1]) / self.num_bins
        return heights / area.unsqueeze(-1)

    def icdf(self, value):
        p, _ = torch.broadcast_tensors(value.clamp(0, 1), self.logits[..., 0])
        heights = self._normalized_heights()
        areas = .5 * (heights[..., :-1] + heights[..., 1:]) / self.num_bins
        edges = torch.cat([torch.zeros_like(areas[..., :1]), areas.cumsum(-1)], dim=-1)
        index = (p.unsqueeze(-1) >= edges[..., 1:]).sum(-1).clamp(max=self.num_bins - 1)
        expanded = heights.expand(*p.shape, self.num_knots)
        h0 = expanded.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        h1 = expanded.gather(-1, (index + 1).unsqueeze(-1)).squeeze(-1)
        left = edges.expand(*p.shape, self.num_knots).gather(-1, index.unsqueeze(-1)).squeeze(-1)
        target = ((p - left) * self.num_bins).clamp_min(0)
        root = (h0.square() + 2 * (h1 - h0) * target).clamp_min(0).sqrt()
        local = (2 * target / (h0 + root).clamp_min(torch.finfo(p.dtype).tiny)).clamp(0, 1)
        result = (index + local) / self.num_bins
        return torch.where(value <= 0, 0., torch.where(value >= 1, 1., result))
