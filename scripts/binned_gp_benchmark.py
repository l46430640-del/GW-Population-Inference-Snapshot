"""Binned GP shape model for the shared external-method experiment.

Equations (2)--(7) of Ray et al. (2023), with the common rate prior and
unit-coordinate measure described in README.md. The separable
covariance follows the later paper's Kronecker construction.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
import torch

FEATURE_NAMES = [
    "mean_mass", "mean_redshift", "mean_ratio", "variance_mass", "variance_redshift",
    "variance_ratio", "alpha", "mass_below_quarter", "mass_above_0p8",
    "cov_mass_redshift", "cov_mass_ratio", "cov_redshift_ratio",
]


def grid_geometry(bins):
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("A positive integer bin count is required")
    lower = np.stack(np.meshgrid(*[np.arange(bins) / bins] * 3, indexing="ij"), -1).reshape(-1, 3)
    upper = lower + 1 / bins
    return lower, upper


def controlled_selection(bins):
    """Integrate 0.05 + 0.95*x1**2 over every bin, in the common dx measure."""
    lower, upper = grid_geometry(bins)
    second = (lower[:, 0]**2 + lower[:, 0]*upper[:, 0] + upper[:, 0]**2) / 3
    return (.05 + .95*second) / bins**3


def controlled_log_event_weights(observed, sigma, bins, order=32):
    """Integrate selected event posteriors with positive Gaussian quadrature.

The Bernoulli detection indicator is part of each event likelihood. Its
factor depends only on x1, so the three one-dimensional integrals factor.
Rows are normalized under the uniform event prior, retaining all tiny tails
in log space. The row normalizer is independent of the population model.
"""
    observed = np.asarray(observed, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 3 or not np.isfinite(observed).all():
        raise ValueError("Finite three-dimensional measurements are required")
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Three positive measurement scales are required")
    grid_geometry(bins)
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("Quadrature order must be an integer at least two")
    node, weight = np.polynomial.legendre.leggauss(order)
    points = (np.arange(bins)[:, None] + (node + 1) / 2) / bins
    log_weight = np.log(weight / (2*bins))
    integrals = []
    for axis in range(3):
        difference = (points[None] - observed[:, axis, None, None]) / sigma[axis]
        log_density = -.5*difference**2 - np.log(sigma[axis]*np.sqrt(2*np.pi))
        if axis == 0:
            log_density += np.log(.05 + .95*points**2)
        integrals.append(logsumexp(log_density + log_weight, axis=-1))
    log_weights = (integrals[0][:, :, None, None] + integrals[1][:, None, :, None]
                   + integrals[2][:, None, None, :]).reshape(len(observed), -1)
    return log_weights - logsumexp(log_weights, axis=1, keepdims=True)


def posterior_log_event_weights(data, bins):
    """Bin event samples with their physical prior divided out exactly once."""
    data = np.asarray(data, dtype=float)
    grid_geometry(bins)
    if data.ndim != 3 or data.shape[-1] != 4 or min(data.shape[:2]) < 1:
        raise ValueError("Expected events by samples by (x1,x2,x3,prior)")
    if not np.isfinite(data).all() or np.any(data[..., 3] <= 0):
        raise ValueError("Finite samples and positive event priors are required")
    if np.any(data[..., :3] < 0) or np.any(data[..., :3] > 1):
        raise ValueError("Samples fall outside the common unit support")
    indices = np.minimum((data[..., :3]*bins).astype(int), bins-1)
    flat = (indices[..., 0]*bins + indices[..., 1])*bins + indices[..., 2]
    rows = np.stack([np.bincount(index, weights=1/prior, minlength=bins**3)
                     for index, prior in zip(flat, data[..., 3])]) / data.shape[1]
    with np.errstate(divide="ignore"):
        return np.log(rows)


def shape_log_likelihood(field, log_event_weights, log_selection):
    events = torch.logsumexp(log_event_weights + field, dim=-1)
    selected = torch.logsumexp(log_selection + field, dim=-1)
    return events.sum(-1) - log_event_weights.shape[0]*selected


def covariance_factors(log_lengths, centers, jitter):
    distance = centers[:, None] - centers[None, :]
    identity = torch.eye(len(centers), dtype=centers.dtype, device=centers.device)
    return [torch.exp(-.5*(distance / log_lengths[axis].exp()).square()) + jitter*identity
            for axis in range(3)]


def correlated_field(white, log_lengths, sigma, centers, jitter):
    """Apply three Cholesky factors without constructing a B**3 covariance."""
    factors = [torch.linalg.cholesky(matrix)
               for matrix in covariance_factors(log_lengths, centers, jitter)]
    bins = len(centers)
    field = white.reshape(bins, bins, bins)
    field = torch.einsum("ia,abc->ibc", factors[0], field)
    field = torch.einsum("jb,ibc->ijc", factors[1], field)
    field = torch.einsum("kc,ijc->ijk", factors[2], field)
    return sigma*field.reshape(-1)


class BinnedGPPotential:
    def __init__(self, bins, log_event_weights, selection, settings, jitter=1e-8):
        grid_geometry(bins)
        weights = np.asarray(log_event_weights, dtype=float)
        selection = np.asarray(selection, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != bins**3 or len(weights) < 1:
            raise ValueError("Event weights have the wrong grid shape")
        if np.isnan(weights).any() or np.isposinf(weights).any() or not np.isfinite(weights).any(axis=1).all():
            raise ValueError("Every event needs finite nonzero support")
        if selection.shape != (bins**3,) or not np.isfinite(selection).all():
            raise ValueError("Selection coefficients have the wrong shape or values")
        if np.any(selection < 0) or np.any(selection > (1+1e-12)/bins**3) or selection.sum() <= 0:
            raise ValueError("Selection must be a nonzero detection probability integral")
        if not np.isfinite(jitter) or jitter <= 0:
            raise ValueError("Positive covariance jitter is required")
        self.bins, self.jitter = bins, jitter
        self.centers = torch.tensor((np.arange(bins) + .5) / bins, dtype=torch.float64)
        self.log_event_weights = torch.tensor(weights, dtype=torch.float64)
        with np.errstate(divide="ignore"):
            self.log_selection = torch.tensor(np.log(selection), dtype=torch.float64)
        self.selection = torch.tensor(selection, dtype=torch.float64)
        self.sigma_scale = settings["sigma_prior"]["scale"]
        self.log_length_center = np.log(settings["length_prior"]["median"])
        self.log_length_scale = settings["length_prior"]["log_standard_deviation"]
        if min(self.sigma_scale, self.log_length_scale) <= 0:
            raise ValueError("Positive prior scales are required")

    @property
    def dimension(self):
        return self.bins**3 + 4

    def field(self, coordinate):
        return correlated_field(coordinate[4:], coordinate[1:4], coordinate[0].exp(),
                                self.centers, self.jitter)

    def log_prior(self, coordinate):
        log_sigma, log_lengths, white = coordinate[0], coordinate[1:4], coordinate[4:]
        # The half-normal amplitude is sampled in log coordinates: +log_sigma
        # is its Jacobian. Lognormal lengths become ordinary normal priors here.
        return (-.5*(log_sigma.exp()/self.sigma_scale).square() + log_sigma
                - .5*((log_lengths-self.log_length_center)/self.log_length_scale).square().sum()
                - .5*white.square().sum())

    def __call__(self, coordinate):
        field = self.field(coordinate)
        return -self.log_prior(coordinate) - shape_log_likelihood(
            field, self.log_event_weights, self.log_selection)

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        return torch.tensor(np.r_[0., np.full(3, self.log_length_center),
                                  .05*rng.standard_normal(self.bins**3)], dtype=torch.float64)


def population_functionals(field, selection, bins):
    """Exact intrinsic moments and tail probabilities for piecewise densities."""
    lower, upper = grid_geometry(bins)
    centers = torch.as_tensor((lower+upper)/2, dtype=field.dtype, device=field.device)
    second = torch.as_tensor((lower**2+lower*upper+upper**2)/3,
                             dtype=field.dtype, device=field.device)
    probability = torch.softmax(field, dim=-1)
    mean = probability @ centers
    variance = probability @ second - mean.square()
    alpha = probability @ (torch.as_tensor(selection, dtype=field.dtype, device=field.device)*bins**3)
    low = np.clip((.25-lower[:, 0])*bins, 0, 1)
    high = np.clip((upper[:, 0]-.8)*bins, 0, 1)
    low_mass = probability @ torch.as_tensor(low, dtype=field.dtype, device=field.device)
    high_mass = probability @ torch.as_tensor(high, dtype=field.dtype, device=field.device)
    covariance = torch.stack([probability @ (centers[:, i]*centers[:, j]) - mean[..., i]*mean[..., j]
                              for i, j in ((0, 1), (0, 2), (1, 2))], dim=-1)
    return torch.cat([mean, variance, alpha[..., None], low_mass[..., None],
                      high_mass[..., None], covariance], dim=-1)
