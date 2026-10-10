"""Population integration for the common LISA external-method comparison."""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, logsumexp, ndtr
from scipy.stats import beta, qmc

from binned_gp_benchmark import grid_geometry
from compare_lisa_posterior import exact_truth_probabilities
from external_selected_evaluation import copula_exact_features, expand_gp_cells
from validate_population_morphologies import BETA_A, BETA_B


def truth_features(alpha):
    weight = np.array([.55, .45])
    mean_components = BETA_A/(BETA_A+BETA_B)
    second = BETA_A*(BETA_A+1)/((BETA_A+BETA_B)*(BETA_A+BETA_B+1))
    mean = weight @ mean_components
    covariance = [weight @ (mean_components[:, i]*mean_components[:, j])-mean[i]*mean[j]
                  for i, j in ((0, 1), (0, 2), (1, 2))]
    return np.r_[mean, weight @ second-mean**2, alpha,
        weight @ beta.cdf(.25, BETA_A[:, 0], BETA_B[:, 0]),
        weight @ beta.sf(.8, BETA_A[:, 0], BETA_B[:, 0]), covariance]


def cell_integrals(response, bins, power, seed, check=lambda: None, block=1024):
    """Stratify in every evaluation cell, including tiny selected tails."""
    lower, _ = grid_geometry(bins)
    uniform = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    selected, truth = np.empty(bins**3), np.empty(bins**3)
    for start in range(0, len(lower), block):
        check()
        points = lower[start:start+block, None]+uniform[None]/bins
        pdet = ndtr(response(points.reshape(-1, 3))-8).reshape(points.shape[:2])
        density = np.zeros(points.shape[:2])
        for weight, a, b in zip((.55, .45), BETA_A, BETA_B):
            log_pdf = ((a-1)*np.log(points)+(b-1)*np.log1p(-points)-betaln(a, b)).sum(-1)
            density += weight*np.exp(log_pdf)
        selected[start:start+len(points)] = pdet.mean(1)/bins**3
        truth[start:start+len(points)] = (density*pdet).mean(1)/bins**3
    alpha = truth.sum()
    return {'selection': selected, 'intrinsic': exact_truth_probabilities(bins)[0],
        'detected': truth/alpha, 'features': truth_features(alpha)}


def component_transports(parameters, normal, response):
    a, b = np.exp(parameters['log_k_a'])+.0001, np.exp(parameters['log_k_b'])+.0001
    weights = np.exp(parameters['weights_logits']-logsumexp(parameters['weights_logits']))
    for k, weight in enumerate(weights):
        uniform = np.clip(ndtr(normal @ parameters['k_L_cholesky'][k].T), 1e-14, 1-1e-14)
        points = np.exp(np.log(-np.expm1(np.log1p(-uniform)/b[k]))/a[k])
        yield weight, points, ndtr(response(points)-8)


def cell_indices(points, bins):
    index = np.minimum((points*bins).astype(int), bins-1)
    return (index[:, 0]*bins+index[:, 1])*bins+index[:, 2]


def population_cells(parameters, normal, bins, response):
    intrinsic, detected = np.zeros(bins**3), np.zeros(bins**3)
    products = np.zeros(3)
    for weight, points, selection in component_transports(parameters, normal, response):
        flat = cell_indices(points, bins)
        intrinsic += weight*np.bincount(flat, minlength=bins**3)/len(points)
        detected += weight*np.bincount(flat, weights=selection, minlength=bins**3)/len(points)
        products += weight*np.array([np.mean(points[:, i]*points[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))])
    exact, _ = copula_exact_features(parameters)
    exact[6] = detected.sum()
    covariance = products-exact[[0, 0, 1]]*exact[[1, 2, 2]]
    return {'intrinsic': intrinsic, 'detected': detected/exact[6], 'features': np.r_[exact, covariance]}


def population_projection(parameters, normal, response, gradients):
    """Project transported points directly, avoiding large per-draw histograms."""
    values = {key: 0. for key in gradients}
    alpha = 0.
    for weight, points, selection in component_transports(parameters, normal, response):
        alpha += weight*selection.mean()
        indices = {bins: cell_indices(points, bins) for bins in {int(k.split('_')[1]) for k in gradients}}
        for label, gradient in gradients.items():
            kind, bins = label.split('_')
            factor = selection if kind == 'detected' else 1.
            values[label] += weight*np.mean(gradient[indices[int(bins)]]*factor)
    return {key: value/alpha if key.startswith('detected_') else value for key, value in values.items()}


def gp_population(probability, bins, fine, selection):
    from external_selected_evaluation import coarsen
    bin_selection = coarsen(selection, fine, bins)*bins**3
    alpha = probability @ bin_selection
    return {'intrinsic': expand_gp_cells(probability.mean(0), bins, fine),
        'detected': expand_gp_cells((probability/alpha[:, None]).mean(0), bins, fine)*selection*fine**3,
        'alpha': alpha}


def gp_projection(probability, bins, fine, selection, alpha, label, gradient):
    from external_selected_evaluation import coarsen
    kind, partition = label.split('_')
    partition = int(partition)
    factor = coarsen(selection, fine, partition)*partition**3 if kind == 'detected' else 1.
    coefficient = coarsen(gradient*factor, partition, bins)/(partition//bins)**3
    values = probability @ coefficient
    return values/alpha if kind == 'detected' else values
