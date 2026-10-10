"""Evaluate posterior importance correction using the completed mixture flows."""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import arviz as az
import numpy as np
import pyro
import torch
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from benchmark_selected_mixture import ROOT, feature_draws, runtime_metadata, selected_model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latent_sites(model, latent):
    params = model._unflatten_latent_vector(latent, apply_numerical_limits=False)
    return {
        'weights_logits': params['weights_logits'],
        'log_k_a': params['log_k_a'],
        'log_k_b': params['log_k_b'],
        'k_L_cholesky': model._build_cholesky_from_unconstrained(
            params['k_cholesky_params'], 3, 'cpu'),
    }


def swap_components(model, latent):
    """The two exchangeable components have identical priors and coordinates."""
    if model.K != 2 or model.num_corr_kumaraswamy != 2:
        raise ValueError('This benchmark requires two Kumaraswamy components')
    chunks, offset = [], 0
    for name, size in model.dims.items():
        if size:
            if name not in ('weights', 'k_marginals', 'k_cholesky'):
                raise ValueError(f'Unexpected active parameter block: {name}')
            block = latent[..., offset:offset + size]
            chunks.append(block.reshape(*latent.shape[:-1], 2, size // 2)
                          .flip(-2).reshape(*latent.shape[:-1], size))
        offset += size
    return torch.cat(chunks, -1)


@torch.no_grad()
def proposal_log_prob(models, latent):
    swapped = swap_components(models[0], latent)
    terms = [model.flow_dist.log_prob(points)
             for model in models for points in (latent, swapped)]
    return torch.logsumexp(torch.stack(terms), dim=0) - np.log(len(terms))


@torch.no_grad()
def draw_proposal(models, count, seed):
    pyro.set_rng_seed(seed)
    source = torch.randint(len(models), (count,))
    latent = torch.empty(count, models[0].latent_dim)
    for index, model in enumerate(models):
        selected = source == index
        latent[selected] = model.flow_dist.sample((int(selected.sum()),))
    permuted = torch.randint(2, (count,), dtype=torch.bool)
    latent[permuted] = swap_components(models[0], latent[permuted])
    return latent, source, permuted


@torch.no_grad()
def latent_log_target(model, latent, data):
    sites = latent_sites(model, latent)
    trace = pyro.poutine.trace(pyro.poutine.condition(model._pyro_model, data=sites)).get_trace(data)
    return trace.log_prob_sum() + model._latent_log_abs_det_jacobian(latent)


def importance_weights(log_ratios):
    values = np.asarray(log_ratios, dtype=float)
    if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
        raise ValueError('At least 20 finite scalar log ratios are required')
    raw_log = values - logsumexp(values)
    if np.ptp(values) < 1e-12:
        smooth_log, pareto_k = raw_log.copy(), 0.
    else:
        smooth_log, pareto_k = az.psislw(values)
    weights = {'raw': np.exp(raw_log), 'psis': np.exp(smooth_log)}
    diagnostics = {'pareto_k': float(pareto_k),
                   'pareto_k_limit': float(min(.7, 1 - 1 / np.log10(len(values)))),
                   'draws': len(values)}
    for name, w in weights.items():
        diagnostics[name] = {'ess': float(1 / np.sum(w * w)),
                             'maximum_weight': float(w.max())}
    return weights, diagnostics


def weighted_quantiles(values, weights, probabilities=(.05, .5, .95)):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    indices = np.searchsorted(cdf, probabilities).clip(0, len(order) - 1)
    return values[order[indices]]


def compare_features(reference, values, names, weights=None):
    weights = np.ones(len(values)) / len(values) if weights is None else np.asarray(weights)
    if (weights.shape != (len(values),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Weights must be finite, nonnegative, and match the draws')
    weights = weights / weights.sum()
    rows = []
    for index, name in enumerate(names):
        exact, approx = reference[:, index], values[:, index]
        scale = exact.std()
        low, high = np.quantile(exact, [.05, .95])
        mean = np.dot(weights, approx)
        sd = np.sqrt(np.dot(weights, (approx - mean)**2))
        rows.append({
            'name': str(name),
            'quantiles_05_50_95': weighted_quantiles(approx, weights).tolist(),
            'sd_ratio': float(sd / scale),
            'mean_offset_in_nuts_sd': float((mean - exact.mean()) / scale),
            'mean_mcse_in_nuts_sd': float(np.sqrt(np.sum(weights**2 * (approx - mean)**2)) / scale),
            'wasserstein_in_nuts_sd': float(wasserstein_distance(exact, approx, v_weights=weights) / scale),
            'probability_below_nuts_05': float(weights[approx < low].sum()),
            'probability_above_nuts_95': float(weights[approx > high].sum()),
        })
    distances = [row['wasserstein_in_nuts_sd'] for row in rows]
    tail_errors = [abs(row[key] - .05) for row in rows
                   for key in ('probability_below_nuts_05', 'probability_above_nuts_95')]
    return {'features': rows,
            'mean_wasserstein_in_nuts_sd': float(np.mean(distances)),
            'maximum_absolute_mean_offset_in_nuts_sd': max(abs(row['mean_offset_in_nuts_sd']) for row in rows),
            'mean_absolute_tail_probability_error': float(np.mean(tail_errors)),
            'sd_ratio_range': [min(row['sd_ratio'] for row in rows), max(row['sd_ratio'] for row in rows)]}


def checked_sources(config):
    source = ROOT / config['source']
    signature = json.loads((source / 'target.json').read_text())
    for name, expected in signature['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Original target source changed: {name}')
    paths = [source / 'target.json', source / 'train_catalog.npz', source / 'nuts_features.npz',
             source / 'nuts_summary.json', ROOT / 'configs/posterior_correction.json',
             ROOT / 'scripts/correct_selected_posterior.py']
    paths += [ROOT / name for name in signature['source_sha256']]
    paths += [source / f'vi_start_{index}.pt' for index in config['flow_starts']]
    paths += [source / f'vi_features_{index}.npz' for index in config['flow_starts']]
    paths += [ROOT / 'scripts/validate_population_morphologies.py',
              ROOT / 'scripts/benchmark_variational_posterior.py']
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config = json.loads((ROOT / 'configs/posterior_correction.json').read_text())['posterior_correction']
    source, output = ROOT / config['source'], ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    sources = checked_sources(config)
    models = []
    for index in config['flow_starts']:
        model = selected_model()
        checkpoint = torch.load(source / f'vi_start_{index}.pt', weights_only=True, map_location='cpu')
        model.flow_transforms.load_state_dict(checkpoint['state'])
        models.append(model)
    data = torch.tensor(np.load(source / 'train_catalog.npz')['data'])
    reference_file = np.load(source / 'nuts_features.npz')
    names = reference_file['names']
    reference = reference_file['features'].reshape(-1, len(names))
    baseline = {
        str(index): compare_features(reference, np.load(source / f'vi_features_{index}.npz')['features'], names)
        for index in config['flow_starts']}
    seeds = [config['pilot_seed']] if args.pilot else config['repeat_seeds']
    count = config['pilot_draws'] if args.pilot else config['draws_per_repeat']
    for seed in seeds:
        label = f'pilot_{seed}' if args.pilot else f'repeat_{seed}'
        path = output / f'{label}.json'
        if path.exists() or path.with_suffix('.npz').exists():
            raise FileExistsError(f'Completed calculation is retained: {path}')
        started = time.perf_counter()
        latent, source_index, permuted = draw_proposal(models, count, seed)
        log_q = proposal_log_prob(models, latent).numpy()
        log_p = np.empty(count)
        for index, point in enumerate(latent):
            log_p[index] = float(latent_log_target(models[0], point, data))
            if (index + 1) % 2048 == 0:
                print(label, index + 1, 'target evaluations', flush=True)
        evaluation_seconds = time.perf_counter() - started
        weights, diagnostics = importance_weights(log_p - log_q)
        with torch.no_grad():
            samples = {name: value.numpy() for name, value in latent_sites(models[0], latent).items()}
        actual_names, features = feature_draws(samples, power=config['feature_transport_power'])
        if list(names) != actual_names:
            raise ValueError('MCMC and corrected functionals differ')
        comparisons = {'unweighted_proposal': compare_features(reference, features, names)}
        comparisons.update({name: compare_features(reference, features, names, w) for name, w in weights.items()})
        stable = (np.isfinite(diagnostics['pareto_k'])
                  and diagnostics['pareto_k'] < min(config['maximum_pareto_k'], diagnostics['pareto_k_limit'])
                  and all(diagnostics[name]['ess'] >= config['minimum_effective_samples'] for name in weights))
        improved = {name: all(comparisons[name][metric] < comparisons['unweighted_proposal'][metric]
                             for metric in ('mean_wasserstein_in_nuts_sd', 'mean_absolute_tail_probability_error'))
                    for name in weights}
        result = {'configuration': config, 'pilot': args.pilot, 'seed': seed,
                  'target_evaluation_seconds': evaluation_seconds,
                  'total_seconds': time.perf_counter() - started, 'runtime': runtime_metadata(),
                  'source_sha256': sources, 'weight_diagnostics': diagnostics,
                  'stable_weights': bool(stable), 'improved_reference_agreement': improved,
                  'eligible_for_further_validation': bool(not args.pilot and stable and all(improved.values())),
                  'comparisons': comparisons, 'original_flow_comparisons': baseline,
                  'scope': 'Same finite-event-integral target as the archived MCMC. Two repeats measure importance sampling variability, not repeated-catalog coverage. Training data alone define the correction.'}
        np.savez_compressed(path.with_suffix('.npz'), latent=latent.numpy(), source_index=source_index.numpy(),
                            permuted=permuted.numpy(), log_target=log_p, log_proposal=log_q,
                            raw_weights=weights['raw'], psis_weights=weights['psis'], features=features, names=names)
        result['draw_product_sha256'] = sha256(path.with_suffix('.npz'))
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: result[key] for key in ('seed', 'total_seconds', 'weight_diagnostics',
                                                       'eligible_for_further_validation')}), flush=True)


if __name__ == '__main__':
    main()
