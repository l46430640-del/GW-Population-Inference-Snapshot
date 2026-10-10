"""Label-invariant posterior and predictive comparisons for the selected mixture."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri
from scipy.stats import wasserstein_distance

from benchmark_selected_mixture import (
    ROOT, event_posterior, feature_draws, log_predictive, protocol, selected_model,
)
from gwpopulation.core import SafeKumaraswamy


def posterior_comparison(reference, approximate, names):
    rows = []
    for index, name in enumerate(names):
        exact, approx = reference[:, index], approximate[:, index]
        sd = exact.std()
        low, high = np.quantile(exact, [.05, .95])
        rows.append({'name': str(name),
                     'nuts_quantiles_05_50_95': np.quantile(exact, [.05, .5, .95]).tolist(),
                     'vi_quantiles_05_50_95': np.quantile(approx, [.05, .5, .95]).tolist(),
                     'sd_ratio': float(approx.std() / sd),
                     'mean_offset_in_nuts_sd': float((approx.mean() - exact.mean()) / sd),
                     'wasserstein_in_nuts_sd': float(wasserstein_distance(exact, approx) / sd),
                     'vi_probability_below_nuts_05': float(np.mean(approx < low)),
                     'vi_probability_above_nuts_95': float(np.mean(approx > high))})
    return rows


def paired_scores(correlated, independent):
    differences = np.asarray(correlated) - np.asarray(independent)
    count = len(differences)
    if count < 2:
        raise ValueError('At least two held-out events are required')
    return {'events': count, 'total_difference': float(differences.sum()),
            'total_standard_error': float(np.sqrt(count) * differences.std(ddof=1)),
            'per_event_difference': float(differences.mean()),
            'per_event_standard_error': float(differences.std(ddof=1) / np.sqrt(count)),
            'fraction_positive': float(np.mean(differences > 0))}


@torch.no_grad()
def conditional_bins(samples, axis=1, independent=False, order=64, n_bins=96,
                     mass_edges=(0., 1/3, 2/3, 1.), detected=False):
    """Conditional density from bivariate Gaussian copula CDF integrals.

    Integrating in the first marginal's probability coordinate resolves even
    narrow components. Only the relevant 2-by-2 correlation submatrix is needed.
    """
    if axis not in (1, 2):
        raise ValueError('The conditioned coordinate is the first marginal')
    a, b = (torch.tensor(np.exp(samples[key]) + .0001, dtype=torch.float64)
            for key in ('log_k_a', 'log_k_b'))
    weights = torch.softmax(torch.tensor(samples['weights_logits'], dtype=torch.float64), -1).numpy()
    cholesky = np.asarray(samples['k_L_cholesky'])
    matrix = cholesky @ np.swapaxes(cholesky, -1, -2)
    rho = np.zeros(matrix.shape[:2]) if independent else matrix[..., 0, axis]
    mass = SafeKumaraswamy(a[..., 0], b[..., 0])
    other = SafeKumaraswamy(a[..., axis], b[..., axis])
    edge_cdf = other.cdf(torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[:, None, None]).numpy()
    eta = ndtri(np.clip(edge_cdf, 1e-15, 1-1e-15))
    nodes, quad_weight = leggauss(order)
    densities, probabilities = [], []
    for low, high in zip(mass_edges[:-1], mass_edges[1:]):
        left = mass.cdf(torch.tensor(low, dtype=torch.float64)).numpy()
        width = mass.cdf(torch.tensor(high, dtype=torch.float64)).numpy() - left
        uniform = left[None] + .5 * (nodes[:, None, None] + 1) * width[None]
        standardized = ndtri(np.clip(uniform, 1e-15, 1-1e-15))
        integration_weight = .5 * quad_weight[:, None, None] * width[None]
        if detected:
            x = mass.icdf(torch.tensor(uniform)).numpy()
            integration_weight *= .05 + .95 * x*x
        denominator = (integration_weight.sum(0) * weights).sum(-1)
        cdf = np.zeros((n_bins + 1, len(a)))
        # Bounded batches avoid an edges x quadrature x draws x components array.
        for index in range(n_bins + 1):
            conditional = ndtr((eta[index][None] - rho[None] * standardized)
                               / np.sqrt(1-rho[None]**2))
            cdf[index] = ((conditional * integration_weight).sum(0) * weights).sum(-1) / denominator
        cdf[0], cdf[-1] = 0., 1.
        densities.append(np.diff(cdf, axis=0).T * n_bins)
        probabilities.append(denominator)
    return np.asarray(densities), np.asarray(probabilities)


def truth_samples():
    config = protocol()
    return {'log_k_a': np.log(np.asarray(config['kumaraswamy_a'])[None] - .0001),
            'log_k_b': np.log(np.asarray(config['kumaraswamy_b'])[None] - .0001),
            'weights_logits': np.log(np.asarray(config['component_weights'])[None]),
            'k_L_cholesky': np.linalg.cholesky(config['correlations'])[None]}


def selected_draws(samples, count):
    indices = np.linspace(0, len(samples['weights_logits'])-1,
                          min(count, len(samples['weights_logits'])), dtype=int)
    return {key: value[indices] for key, value in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default='results/analysis/selected_mixture')
    parser.add_argument('--draws', type=int, default=512)
    parser.add_argument('--fine-event-samples', type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    directory = ROOT / args.directory
    diagnostics = json.loads((directory / 'nuts_summary.json').read_text())
    nuts = np.load(directory / 'nuts_features.npz')
    reference = nuts['features'].reshape(-1, len(nuts['names']))
    catalog = np.load(directory / 'test_catalog.npz')
    rng = np.random.default_rng(526901)
    fine = np.asarray([event_posterior(y, args.fine_event_samples, rng) for y in catalog['observed']])
    fine_data = torch.tensor(np.concatenate([fine, np.ones((*fine.shape[:2], 1))], -1))
    data = torch.tensor(catalog['data'])
    predictions, fine_predictions, summaries, conditional, precision = {}, {}, {}, {}, {}
    source_paths = [directory / 'target.json', directory / 'nuts_summary.json',
                    directory / 'test_catalog.npz', ROOT / 'scripts/summarize_selected_mixture.py']
    required_starts = protocol()['vi_starts']
    for mode in ('vi', 'independent'):
        if len(list(directory.glob(f'{mode}_start_*.npz'))) < required_starts:
            raise ValueError(f'Missing declared {mode} starts')
    for mode in ('vi', 'independent'):
        for path in sorted(directory.glob(f'{mode}_start_*.npz')):
            name = path.stem
            source_paths.append(path)
            samples = dict(np.load(path))
            features_path = directory / f'{mode}_features_{name.rsplit("_", 1)[-1]}.npz'
            features = np.load(features_path)['features']
            metadata = json.loads(path.with_suffix('.json').read_text())
            summaries[name] = {'seconds': metadata['seconds'],
                               'last_1000_mean_negative_elbo': metadata['last_1000_mean_negative_elbo']}
            if mode == 'vi':
                summaries[name]['posterior_comparison'] = posterior_comparison(reference, features, nuts['names'])
            independent = mode == 'independent'
            model = selected_model(independent)
            predictions[name] = log_predictive(model, samples, data, args.draws)
            fine_predictions[name] = log_predictive(model, samples, fine_data, args.draws)
            half = log_predictive(model, samples, fine_data, args.draws // 2)
            precision[name] = {'coarse_minus_fine_total_score': float((predictions[name] - fine_predictions[name]).sum()),
                               'half_minus_full_draw_total_score': float((half - fine_predictions[name]).sum())}
            plot_draws = selected_draws(samples, 192)
            for axis in (1, 2):
                fine_density, probability = conditional_bins(plot_draws, axis, independent, order=64)
                coarse_density, _ = conditional_bins(plot_draws, axis, independent, order=32)
                conditional[f'{name}_axis{axis}'] = fine_density
                conditional[f'{name}_axis{axis}_mass_probabilities'] = probability
                precision[name][f'conditional_axis{axis}_max_tv_change'] = float(.5*np.abs(fine_density-coarse_density).sum(-1).max()/96)
            print(name, 'summarized', flush=True)
    for axis in (1, 2):
        conditional[f'truth_axis{axis}'] = conditional_bins(truth_samples(), axis, order=128)[0]
    chains = [dict(np.load(path)) for path in sorted(directory.glob('nuts_chain_*.npz'))]
    nuts_samples = {key: np.concatenate([chain[key] for chain in chains]) for key in chains[0]}
    check_samples = selected_draws(nuts_samples, 256)
    feature_names, coarse_features = feature_draws(check_samples, power=10)
    _, fine_features = feature_draws(check_samples, power=14)
    reference_sd = reference.std(0)
    functional_precision = {
        'draws': len(check_samples['weights_logits']),
        'coarse_points_per_component': 2**10,
        'fine_points_per_component': 2**14,
        'maximum_change_in_nuts_sd': {
            name: float(np.max(np.abs(fine_features[:, index]-coarse_features[:, index]))
                        / reference_sd[index])
            for index,name in enumerate(feature_names)}}
    fine_predictions['nuts'] = log_predictive(selected_model(), nuts_samples, fine_data, args.draws)
    prediction_validation = {name: paired_scores(values, fine_predictions['nuts'])
                             for name, values in fine_predictions.items() if name.startswith('vi_')}
    source_paths += sorted(directory.glob('nuts_chain_*.*'))
    comparisons = {}
    for corr in sorted(name for name in fine_predictions if name.startswith('vi_')):
        for indep in sorted(name for name in fine_predictions if name.startswith('independent_')):
            name = f'{corr}_vs_{indep}'
            comparisons[name] = paired_scores(fine_predictions[corr], fine_predictions[indep])
            coarse = paired_scores(predictions[corr], predictions[indep])
            comparisons[name]['coarse_event_total_difference'] = coarse['total_difference']
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    result = {'nuts_diagnostics': diagnostics, 'flow_starts': summaries,
              'vi_minus_nuts_test_scores': prediction_validation,
              'paired_test_scores': comparisons, 'numerical_precision': precision,
              'functional_precision': functional_precision,
              'configuration': vars(args), 'sha256': hashes,
              'interpretation': 'Predictive uncertainty is the paired test-event standard error conditional on the training catalog; no Bayes factor or repeated-training calibration is implied.'}
    np.savez_compressed(directory / 'conditional_densities.npz', **conditional)
    np.savez_compressed(directory / 'test_predictions.npz', **fine_predictions)
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'paired_test_scores': comparisons, 'numerical_precision': precision}, indent=2))


if __name__ == '__main__':
    main()
