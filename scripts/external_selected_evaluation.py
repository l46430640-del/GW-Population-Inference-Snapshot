"""Bounded-memory numerical summaries for the controlled external comparison."""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, logsumexp, ndtr, ndtri
from scipy.stats import qmc

from binned_gp_benchmark import controlled_selection


def copula_log_density(parameters, points):
    """NumPy evaluation of the frozen unit-box Kumaraswamy copula density.

    The support and CDF clamps deliberately match core.py, including in the
    tails. This evaluator is checked against that implementation before reuse.
    """
    points = np.asarray(points, dtype=float)
    inside = ((points >= 0) & (points <= 1)).all(-1)
    log_x = np.log(np.clip(points, 1e-7, 1-1e-7))
    a = np.exp(parameters['log_k_a']) + .0001
    b = np.exp(parameters['log_k_b']) + .0001
    log_weight = parameters['weights_logits'] - logsumexp(parameters['weights_logits'])
    result = np.full(points.shape[:-1], -np.inf)
    for k in range(len(a)):
        inner = np.log(-np.expm1(a[k]*log_x))
        log_marginal = (np.log(a[k]*b[k]) + (a[k]-1)*log_x + (b[k]-1)*inner).sum(-1)
        y = ndtri(np.clip(-np.expm1(b[k]*inner), 1e-7, 1-1e-7))
        chol = parameters['k_L_cholesky'][k]
        inverse = np.linalg.inv(chol)
        precision_change = inverse.T @ inverse - np.eye(3)
        quadratic = sum(precision_change[i, i]*y[..., i]**2 for i in range(3))
        quadratic += sum(2*precision_change[i, j]*y[..., i]*y[..., j]
                         for i, j in ((0, 1), (0, 2), (1, 2)))
        log_copula = -.5*quadratic - np.log(np.diag(chol)).sum()
        result = np.logaddexp(result, log_weight[k] + log_marginal + log_copula)
    return np.where(inside, result, -np.inf)


def event_log_integrals(parameters, data, block_events=8):
    data = np.asarray(data, dtype=float)
    if data.ndim != 3 or data.shape[-1] != 4 or not np.isfinite(data).all() or np.any(data[..., 3] <= 0):
        raise ValueError('Finite event samples and positive prior densities are required')
    result = []
    for start in range(0, len(data), block_events):
        block = data[start:start+block_events]
        log_weights = copula_log_density(parameters, block[..., :3]) - np.log(block[..., 3])
        result.extend(logsumexp(log_weights, axis=-1) - np.log(block.shape[1]))
    return np.asarray(result)


def copula_exact_features(parameters):
    a = np.exp(parameters['log_k_a']) + .0001
    b = np.exp(parameters['log_k_b']) + .0001
    weight = np.exp(parameters['weights_logits'] - logsumexp(parameters['weights_logits']))
    mean = weight @ np.exp(np.log(b) + betaln(1+1/a, b))
    second = weight @ np.exp(np.log(b) + betaln(1+2/a, b))
    cdf = lambda x: -np.expm1(b*np.log(-np.expm1(a*np.log(x))))
    features = np.r_[mean, second-mean**2, .05+.95*second[0],
                     weight @ cdf(.25)[:, 0], weight @ (1-cdf(.8)[:, 0])]
    marginal_cdf = np.asarray([weight @ cdf(x) for x in (.25, .5, .75)]).T.ravel()
    return features, marginal_cdf


def normal_points(power, seed):
    return ndtri(qmc.Sobol(3, scramble=True, seed=seed).random_base2(power))


def coarsen(cells, fine, coarse):
    if fine % coarse:
        raise ValueError('Evaluation partitions must nest')
    ratio = fine // coarse
    return np.asarray(cells).reshape(coarse, ratio, coarse, ratio, coarse, ratio).sum((1, 3, 5)).ravel()


def copula_cells(parameters, normal, bins):
    """Transport in probability space retains integrable boundary mass."""
    a = np.exp(parameters['log_k_a']) + .0001
    b = np.exp(parameters['log_k_b']) + .0001
    weights = np.exp(parameters['weights_logits'] - logsumexp(parameters['weights_logits']))
    intrinsic, detected = np.zeros(bins**3), np.zeros(bins**3)
    products = np.zeros(3)
    for k, weight in enumerate(weights):
        gaussian = normal @ parameters['k_L_cholesky'][k].T
        uniform = np.clip(ndtr(gaussian), 1e-14, 1-1e-14)
        points = np.exp(np.log(-np.expm1(np.log1p(-uniform)/b[k]))/a[k])
        index = np.minimum((points*bins).astype(int), bins-1)
        flat = (index[:, 0]*bins+index[:, 1])*bins+index[:, 2]
        detection = .05 + .95*points[:, 0]**2
        intrinsic += weight*np.bincount(flat, minlength=bins**3)/len(points)
        detected += weight*np.bincount(flat, weights=detection, minlength=bins**3)/len(points)
        products += weight*np.asarray([np.mean(points[:, i]*points[:, j])
                                       for i, j in ((0, 1), (0, 2), (1, 2))])
    alpha = detected.sum()
    exact, cdf = copula_exact_features(parameters)
    covariance = products - exact[[0, 0, 1]]*exact[[1, 2, 2]]
    return {'intrinsic': intrinsic/intrinsic.sum(), 'detected': detected/alpha,
            'transport_alpha': alpha, 'features': np.r_[exact, covariance],
            'additional': np.r_[cdf, coarsen(intrinsic, bins, 2)]}


def expand_gp_cells(probability, bins, fine, detected=False):
    """Exact subdivision of a piecewise intrinsic density, including selection."""
    if fine % bins:
        raise ValueError('The evaluation partition must contain the GP boundaries')
    ratio = fine // bins
    result = np.asarray(probability).reshape(bins, bins, bins)
    for axis in range(3):
        result = np.repeat(result, ratio, axis=axis)
    result = result.ravel()/ratio**3
    if detected:
        result *= controlled_selection(fine)*fine**3
    return result


def gp_gradient(gradient, bins, fine, detected=False):
    factor = controlled_selection(fine)*fine**3 if detected else 1.
    return coarsen(gradient*factor, fine, bins)/(fine//bins)**3


def js_gradient(mean, truth):
    from scipy.spatial.distance import jensenshannon
    mean, truth = np.asarray(mean), np.asarray(truth)
    if np.any(mean < 0) or np.any(truth < 0):
        raise ValueError('Nonnegative cell probabilities are required')
    mean, truth = mean/mean.sum(), truth/truth.sum()
    distance = float(jensenshannon(mean, truth, base=2))
    gradient = np.zeros_like(mean)
    positive = mean > 0
    if distance > 1e-12:
        gradient[positive] = np.log2(2*mean[positive]/(mean[positive]+truth[positive]))/(4*distance)
    return distance, gradient


def interval_score(lower, upper, truth):
    if lower > upper:
        raise ValueError('Ordered interval endpoints are required')
    return upper-lower + 20*max(lower-truth, 0) + 20*max(truth-upper, 0)
