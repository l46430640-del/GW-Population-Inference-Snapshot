"""Noisy LISA summary likelihoods with the detection statistic retained."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp, log_ndtr
from scipy.stats import multivariate_normal, norm, truncnorm
import torch

from build_lisa_analysis_assets import LISASNR
from validate_population_morphologies import THRESHOLD, truth_sample

SIGMA = np.array([.005, .02, .025])


def log_normal_interval(low, high):
    use_survival = low > 0
    first = np.where(use_survival, log_ndtr(-low), log_ndtr(high))
    second = np.where(use_survival, log_ndtr(-high), log_ndtr(low))
    return first + np.log(-np.expm1(np.minimum(second - first, -1e-16)))


def inverse_snr_proposal(points, observed_rho, rho, y, rng=None):
    """Sample/evaluate a conditional redshift proposal along the SNR surface."""
    curves = rho.redshift_curve(points[:, [0, 2]])
    if np.any(np.diff(curves, axis=1) > 1e-10):
        raise ValueError('Conditional inverse requires monotone redshift SNR')
    low, high = np.expm1(curves[:, -1]), np.expm1(curves[:, 0])
    flat = high - low < 1e-12
    log_normalization = log_normal_interval(low - observed_rho, high - observed_rho)
    if rng is not None:
        value = np.zeros(len(points))
        value[~flat] = truncnorm.rvs(low[~flat] - observed_rho, high[~flat] - observed_rho,
                                    loc=observed_rho, random_state=rng)
        log_value = np.log1p(value)
        segment = np.clip((curves > log_value[:, None]).sum(axis=1) - 1, 0, curves.shape[1] - 2)
        row = np.arange(len(points))
        slope = curves[row, segment + 1] - curves[row, segment]
        fraction = (log_value - curves[row, segment]) / np.where(slope != 0, slope, 1.)
        points[~flat, 1] = ((segment + fraction) / (curves.shape[1] - 1))[~flat]
    position = np.clip(points[:, 1] * (curves.shape[1] - 1), 0, curves.shape[1] - 1 - 1e-12)
    segment = position.astype(int)
    row = np.arange(len(points))
    slope = curves[row, segment + 1] - curves[row, segment]
    log_value = curves[row, segment] + (position - segment) * slope
    derivative = np.abs(slope) * (curves.shape[1] - 1) * np.exp(log_value)
    with np.errstate(divide='ignore'):
        conditional = norm.logpdf(np.expm1(log_value) - observed_rho) + np.log(derivative) - log_normalization
    conditional[flat] = truncnorm.logpdf(points[flat, 1], -y[1] / SIGMA[1],
                                        (1-y[1]) / SIGMA[1], y[1], SIGMA[1])
    return conditional


def posterior_inverse_snr(y, observed_rho, rho, n_samples, rng):
    lo, hi = -y / SIGMA, (1 - y) / SIGMA
    all_points, all_weights = [], []
    size, ess = 4096, 0.
    while ess < max(1024, 4 * n_samples):
        if sum(len(x) for x in all_points) >= 262144:
            raise RuntimeError(f'Inverse-SNR importance ESS remains {ess:.1f}')
        points = truncnorm.rvs(lo, hi, loc=y, scale=SIGMA, size=(size, 3), random_state=rng)
        narrow = rng.uniform(size=size) < .85
        subset = points[narrow].copy()
        inverse_snr_proposal(subset, observed_rho, rho, y, rng)
        points[narrow] = subset
        log_y = truncnorm.logpdf(points, lo, hi, loc=y, scale=SIGMA)
        conditional = inverse_snr_proposal(points, observed_rho, rho, y)
        log_proposal = log_y[:, [0, 2]].sum(axis=1) + np.logaddexp(
            np.log(.85) + conditional, np.log(.15) + log_y[:, 1])
        all_points.append(points)
        all_weights.append(log_y.sum(axis=1) + norm.logpdf(rho(points)-observed_rho) - log_proposal)
        log_weights = np.concatenate(all_weights)
        normalized = np.exp(log_weights - logsumexp(log_weights))
        ess = 1 / np.sum(normalized**2)
        size = min(size * 2, 65536)
    points = np.concatenate(all_points)
    indices = rng.choice(len(points), n_samples, replace=True, p=normalized)
    return points[indices], np.sum(points * normalized[:, None], axis=0), {
        'proposal_ess': float(ess), 'proposal_points': len(points)}


def posterior_from_measurements(y, observed_rho, rho, n_samples, rng):
    """Importance-sample the nonlinear likelihood using a local Gaussian mixture."""
    if hasattr(rho, 'redshift_curve'):
        return posterior_inverse_snr(y, observed_rho, rho, n_samples, rng)
    def residual(x):
        return np.r_[(x - y) / SIGMA, float(rho(x[None])[0]) - observed_rho]

    fitted = least_squares(residual, np.clip(y, 1e-5, 1 - 1e-5),
                           bounds=(np.full(3, 1e-8), np.full(3, 1 - 1e-8)),
                           diff_step=1e-5, max_nfev=400)
    covariance = np.linalg.inv(fitted.jac.T @ fitted.jac)
    covariance *= 2.
    local = multivariate_normal(fitted.x, covariance, allow_singular=False)
    lo, hi = -y / SIGMA, (1 - y) / SIGMA
    all_points, all_weights = [], []
    size, ess = 4096, 0.
    while ess < max(1024, 4 * n_samples):
        if sum(len(x) for x in all_points) >= 262144:
            raise RuntimeError(f'Event importance ESS remains {ess:.1f} for rho={observed_rho:.2f}')
        narrow = rng.uniform(size=size) < .85
        points = np.empty((size, 3))
        points[narrow] = local.rvs(size=narrow.sum(), random_state=rng)
        points[~narrow] = truncnorm.rvs(lo, hi, loc=y, scale=SIGMA,
                                       size=((~narrow).sum(), 3), random_state=rng)
        inside = ((points > 0) & (points < 1)).all(axis=1)
        log_likelihood_y = truncnorm.logpdf(points, lo, hi, loc=y, scale=SIGMA).sum(axis=1)
        log_proposal = np.logaddexp(np.log(.85) + local.logpdf(points),
                                   np.log(.15) + log_likelihood_y)
        log_weight = np.full(size, -np.inf)
        log_weight[inside] = (log_likelihood_y[inside]
                              + norm.logpdf(observed_rho - rho(points[inside])) - log_proposal[inside])
        all_points.append(points)
        all_weights.append(log_weight)
        log_weights = np.concatenate(all_weights)
        normalized = np.exp(log_weights - logsumexp(log_weights))
        ess = 1 / np.sum(normalized**2)
        size = min(size * 2, 65536)
    points = np.concatenate(all_points)
    indices = rng.choice(len(points), n_samples, replace=True, p=normalized)
    center = np.sum(points * normalized[:, None], axis=0)
    return points[indices], center, {'proposal_ess': float(ess), 'proposal_points': len(points),
                                     'fitted_center': fitted.x.tolist(), 'covariance': covariance.tolist()}


def catalog(name, seed, mean_intrinsic=160., n_samples=256):
    rng = np.random.default_rng(seed)
    rho = LISASNR()
    truth = truth_sample(name, rng.poisson(mean_intrinsic), rng)
    optimal_rho = rho(truth)
    observed_rho = optimal_rho + rng.normal(size=len(truth))
    measurement = truth + rng.normal(size=truth.shape) * SIGMA
    detected = observed_rho > THRESHOLD
    samples, centers, diagnostics = [], [], []
    for index, (y, r) in enumerate(zip(measurement[detected], observed_rho[detected])):
        posterior, center, diagnostic = posterior_from_measurements(y, r, rho, n_samples, rng)
        samples.append(posterior)
        centers.append(center)
        diagnostics.append(diagnostic)
        if (index + 1) % 25 == 0:
            print('LISA event posteriors', index + 1, '/', detected.sum(), flush=True)
    samples = np.asarray(samples)
    data = np.concatenate([samples, np.ones((*samples.shape[:2], 1))], axis=-1)
    return torch.tensor(data, dtype=torch.float32), {
        'truth_intrinsic': truth, 'detected': detected, 'rho_observed': observed_rho,
        'rho_optimal': optimal_rho, 'measurement': measurement,
        'measurement_sigma': SIGMA,
        'posterior_centers': centers, 'mean_intrinsic': mean_intrinsic,
        'event_proposal_ess': [d['proposal_ess'] for d in diagnostics],
        'event_proposal_points': [d['proposal_points'] for d in diagnostics],
    }
