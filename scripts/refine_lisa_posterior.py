"""Refine the numerical LISA target while retaining correlated NUTS draws."""
from __future__ import annotations

import argparse
import json
import time

import arviz as az
import numpy as np
from scipy.special import logsumexp
import torch

from benchmark_selected_mixture import runtime_metadata
from build_lisa_analysis_assets import LISASNR
from correct_selected_posterior import sha256, weighted_quantiles
from lisa_noisy_catalog import posterior_from_measurements
from run_lisa_production import additional_functionals, verify_product, write_json
from sample_lisa_transport import differentiable_log_target, lisa_features, sampling_model
from validate_population_morphologies import ROOT


def ratio_weights(log_ratio, relative_efficiency=1.):
    values = np.asarray(log_ratio, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite chain-by-draw log ratios are required')
    flat = values.ravel()
    weights = np.exp(flat - logsumexp(flat))
    if np.ptp(flat) < 1e-12:
        k = 0.
    else:
        _, k = az.psislw(flat, reff=relative_efficiency)
    return weights.reshape(values.shape), {
        'pareto_k': float(k), 'diagnostic_relative_efficiency': float(relative_efficiency),
        'weight_ess': float(1 / (weights @ weights)),
        'weight_ess_fraction': float(1 / (weights @ weights) / weights.size),
        'maximum_weight': float(weights.max()), 'chain_weight': weights.reshape(values.shape).sum(1).tolist(),
        'weights_used': 'raw', 'interpretation': 'Weight concentration, not independent MCMC information.'}


def influence_diagnostics(values, batch_lengths):
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError('Finite chain-by-draw-by-functional influences are required')
    dataset = az.from_dict(posterior={'influence': values})
    spectral = np.asarray(az.mcse(dataset, method='mean')['influence'])
    rhat = np.asarray(az.rhat(dataset, method='rank')['influence'])
    batches = []
    chains, draws, columns = values.shape
    for length in batch_lengths:
        if length < 2 or draws % length or draws // length < 2:
            raise ValueError('Batch lengths must partition each chain into at least two blocks')
        means = values.reshape(chains, draws // length, length, columns).mean(2).reshape(-1, columns)
        batches.append(means.std(0, ddof=1) / np.sqrt(len(means)))
    methods = np.vstack([spectral, *batches])
    if not np.isfinite(methods).all() or not np.isfinite(rhat).all():
        raise ValueError('Non-finite influence diagnostics')
    return {'mcse': methods.max(0), 'mcse_arviz': spectral,
            'mcse_batches': np.asarray(batches), 'rhat': rhat}


def corrected_statistics(values, names, weights, population_columns, config):
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape[:2] != weights.shape or values.shape[2] != len(names):
        raise ValueError('Weights and named functional draws must have matching dimensions')
    if not np.isfinite(weights).all() or weights.min() < 0 or weights.sum() <= 0:
        raise ValueError('Weights must be finite and nonnegative')
    weights = weights / weights.sum()
    mean = np.einsum('ij,ijk->k', weights, values)
    sd = np.sqrt(np.einsum('ij,ijk->k', weights, (values - mean)**2))
    if np.any(sd <= 0):
        raise ValueError('Declared functionals must vary across posterior draws')
    scaled_weights = weights * weights.size
    influence = scaled_weights[:, :, None] * (values - mean)
    mean_check = influence_diagnostics(influence, config['batch_lengths'])
    rows = {}
    probabilities = config['cdf_probabilities']
    cdf_values, cdf_labels = [], []
    for j, name in enumerate(names):
        rows[name] = {'mean': float(mean[j]), 'sd': float(sd[j]),
                      'mean_mcse': float(mean_check['mcse'][j]),
                      'mean_mcse_arviz': float(mean_check['mcse_arviz'][j]),
                      'mean_mcse_batches': mean_check['mcse_batches'][:, j].tolist(),
                      'mean_mcse_over_sd': float(mean_check['mcse'][j] / sd[j]),
                      'mean_influence_rhat': float(mean_check['rhat'][j])}
        if j < population_columns:
            quantiles = weighted_quantiles(values[:, :, j].ravel(), weights.ravel(), probabilities)
            rows[name]['quantiles'] = quantiles.tolist()
            for p, q in zip(probabilities, quantiles):
                indicator = values[:, :, j] <= q
                estimated_cdf = float(np.sum(weights * indicator))
                cdf_values.append(scaled_weights * (indicator - estimated_cdf))
                cdf_labels.append((name, p, q, estimated_cdf))
    cdf_check = influence_diagnostics(np.stack(cdf_values, axis=-1), config['batch_lengths'])
    cdf_rows = []
    for j, (name, p, q, estimate) in enumerate(cdf_labels):
        cdf_rows.append({'name': name, 'probability': p, 'quantile': float(q),
                         'estimated_cdf': estimate, 'cdf_mcse': float(cdf_check['mcse'][j]),
                         'cdf_mcse_arviz': float(cdf_check['mcse_arviz'][j]),
                         'cdf_mcse_batches': cdf_check['mcse_batches'][:, j].tolist(),
                         'influence_rhat': float(cdf_check['rhat'][j]),
                         'mcse_equivalent_draws': float(p * (1-p) / cdf_check['mcse'][j]**2)})
    return {'functionals': rows, 'cdf_checks': cdf_rows,
            'maximum_influence_rhat': float(max(mean_check['rhat'].max(), cdf_check['rhat'].max())),
            'minimum_mean_precision_equivalent_draws': float(np.min((sd / mean_check['mcse'])**2)),
            'minimum_cdf_precision_equivalent_draws': min(r['mcse_equivalent_draws'] for r in cdf_rows),
            'interpretation': 'Delta-method MCSE of self-normalized ratios; original chain order is retained.'}


def decision(config, original_converged, weights, statistics, verification):
    groups = {*(f'chain_{i}' for i in range(4)), 'efficiency_extremes', 'largest_weights'}
    stable = (np.isfinite(weights['pareto_k']) and weights['pareto_k'] < config['maximum_pareto_k']
              and weights['weight_ess_fraction'] >= config['minimum_weight_ess_fraction']
              and weights['maximum_weight'] <= config['maximum_weight'])
    precise = (statistics['maximum_influence_rhat'] < config['maximum_influence_rhat']
               and statistics['minimum_mean_precision_equivalent_draws'] >= config['minimum_precision_equivalent_draws']
               and statistics['minimum_cdf_precision_equivalent_draws'] >= config['minimum_precision_equivalent_draws'])
    verified = (set(verification['groups']) == groups
                and all(r['log_target_change_sd'] < config['maximum_verification_log_target_sd']
                        for r in verification['groups'].values())
                and verification['uniform_weight_ess_fraction'] >= config['minimum_verification_weight_ess_fraction']
                and np.max(verification['uniform_mean_shift_in_sd']) <= config['maximum_verification_mean_shift_in_sd']
                and verification['maximum_alpha_change_in_sd'] <= config['maximum_verification_alpha_change_in_sd'])
    result = {'original_chains_converged': bool(original_converged), 'weights_passed': bool(stable),
              'monte_carlo_precision_passed': bool(precise), 'independent_integration_passed': bool(verified)}
    result['eligible_for_scientific_summaries'] = all(result.values())
    return result


def checked_inputs(config, config_path):
    source = ROOT / config['source']
    target = json.loads((source / 'target.json').read_text())
    report = json.loads((source / 'summary.json').read_text())
    for name, expected in target['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Production input changed: {name}')
    if (sha256(source / 'target.json') != report['target_product_sha256']
            or sha256(source / 'numerical.npz') != report['numerical_product_sha256']):
        raise ValueError('Production target or numerical product changed')
    converged = all(report['decision'][k] for k in ('complete', 'divergences_passed', 'mixing_passed',
                                                  'functional_quadrature_passed', 'precision_passed'))
    if not converged:
        raise ValueError('Original MCMC chains have not passed sampling checks')
    paths = [config_path, ROOT / 'scripts/refine_lisa_posterior.py', ROOT / 'scripts/lisa_noisy_catalog.py',
             source / 'target.json', source / 'summary.json', source / 'numerical.npz',
             ROOT / config['primary_event_catalog']]
    products = []
    for seed in target['configuration']['seeds']:
        path = source / f'chain_{seed}.npz'
        metadata = verify_product(path)
        if (sha256(path.with_suffix('.json')) != report['chain_reports'][path.with_suffix('.json').name]
                or metadata['target_product_sha256'] != report['target_product_sha256']):
            raise ValueError('A production chain has different provenance')
        products.append(dict(np.load(path)))
        paths.extend([path, path.with_suffix('.json')])
    sources = dict(target['source_sha256'])
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    source_config = json.loads((ROOT / target['configuration']['source_config']).read_text())
    directory = ROOT / source_config['sources'][0]
    model_config = json.loads((directory / 'summary.json').read_text())['configuration']
    catalog = dict(np.load(directory / 'catalog.npz'))
    fine = dict(np.load(ROOT / config['primary_event_catalog']))
    for key in ('measurement', 'rho_observed', 'detected'):
        np.testing.assert_array_equal(catalog[key], fine[key])
    return target, report, products, model_config, catalog, fine['data'], sources


def verification_catalog(config, output, catalog, sources):
    path = output / 'verification_catalog.npz'
    if path.with_suffix('.json').exists():
        metadata = verify_product(path)
        if metadata['source_sha256'] != sources:
            raise ValueError('Verification event samples have different inputs')
        return torch.tensor(np.load(path)['data'], dtype=torch.float64)
    if path.exists():
        raise FileExistsError('Unverified event product is retained')
    started = time.perf_counter()
    rng, snr = np.random.default_rng(config['verification_event_seed']), LISASNR()
    mask = catalog['detected']
    samples, diagnostics = [], []
    for i, (y, r) in enumerate(zip(catalog['measurement'][mask], catalog['rho_observed'][mask])):
        points, _, report = posterior_from_measurements(y, r, snr, config['verification_event_samples'], rng)
        samples.append(points)
        diagnostics.append(report)
        if (i+1) % 25 == 0:
            print('Verification events', i+1, '/', mask.sum(), flush=True)
    samples = np.asarray(samples)
    data = np.concatenate([samples, np.ones((*samples.shape[:2], 1))], -1)
    np.savez_compressed(path, data=data, measurement=catalog['measurement'],
                        rho_observed=catalog['rho_observed'], detected=mask)
    write_json(path.with_suffix('.json'), {'source_sha256': sources, 'configuration': config,
               'seconds': time.perf_counter()-started, 'event_diagnostics': diagnostics,
               'draw_product_sha256': sha256(path), 'runtime': runtime_metadata()})
    return torch.tensor(data, dtype=torch.float64)


@torch.no_grad()
def compute_primary(config, target, products, model, data, output, sources):
    path = output / 'posterior.npz'
    if path.with_suffix('.json').exists():
        metadata = verify_product(path)
        if metadata['source_sha256'] != sources:
            raise ValueError('Refined posterior has different inputs')
        return dict(np.load(path)), metadata
    if path.exists():
        raise FileExistsError('Unverified posterior product is retained')
    started = time.perf_counter()
    latent = np.stack([p['latent'] for p in products])
    log_coarse = np.stack([p['log_target'] for p in products])
    flat = torch.tensor(latent.reshape(-1, latent.shape[-1]))
    log_primary = np.full(len(flat), np.nan)
    original = target['configuration']
    old_seed = model.selection_mc_seed
    if (original['fine_selection_points'] == config['primary_selection_points']
            and old_seed + original['fine_selection_seed_offset'] == config['primary_selection_seed']
            and config['primary_event_catalog'] == json.loads((ROOT / original['source_config']).read_text())['fine_catalog']):
        cached = np.load(ROOT / config['source'] / 'numerical.npz')
        log_primary[cached['indices']] = (log_coarse.ravel()[cached['indices']]
                                          + cached['event_change'] + cached['selection_change'])
    reused = int(np.isfinite(log_primary).sum())
    model.lambda_det_mc_samples = config['primary_selection_points']
    model.selection_mc_seed = config['primary_selection_seed']
    for i, point in enumerate(flat):
        if not np.isfinite(log_primary[i]):
            log_primary[i] = float(differentiable_log_target(model, point, data))
        if (i+1) % 250 == 0:
            print('Refined target', i+1, '/', len(flat), flush=True)
    names, features = lisa_features(model, flat, config['feature_transport_power'])
    extra_names, extra = additional_functionals(model, flat, original['marginal_cdf_points'])
    values = np.column_stack([features, extra, log_primary]).reshape(*latent.shape[:2], -1)
    arrays = {'latent': latent, 'log_coarse': log_coarse, 'log_primary': log_primary.reshape(latent.shape[:2]),
              'feature_names': np.asarray(names), 'features': features.reshape(*latent.shape[:2], -1),
              'diagnostic_names': np.asarray(names + extra_names + ['log_target']), 'diagnostic_values': values}
    np.savez_compressed(path, **arrays)
    metadata = {'configuration': config, 'source_sha256': sources, 'reused_target_evaluations': reused,
                'seconds': time.perf_counter()-started, 'draw_product_sha256': sha256(path), 'runtime': runtime_metadata()}
    write_json(path.with_suffix('.json'), metadata)
    return arrays, metadata


@torch.no_grad()
def verify_refinement(config, model, arrays, weights, fine_data):
    chains, count = weights.shape
    groups = {f'chain_{c}': c * count + np.linspace(0, count-1, config['verification_draws_per_chain'], dtype=int)
              for c in range(chains)}
    uniform = np.concatenate(list(groups.values()))
    alpha_column = arrays['feature_names'].tolist().index('alpha')
    alpha = arrays['features'][:, :, alpha_column]
    tail = config['verification_extremes_per_chain']
    previous = np.load(ROOT / config['source'] / 'numerical.npz')['group_efficiency_extremes']
    groups['efficiency_extremes'] = np.unique(np.r_[previous, np.concatenate([
        c*count + np.r_[np.argsort(a)[:tail], np.argsort(a)[-tail:]] for c, a in enumerate(alpha)])])
    groups['largest_weights'] = np.argsort(weights.ravel())[-config['verification_largest_weights']:]
    indices = np.unique(np.concatenate(list(groups.values())))
    latent = torch.tensor(arrays['latent'].reshape(-1, arrays['latent'].shape[-1]))
    model.lambda_det_mc_samples = config['verification_selection_points']
    model.selection_mc_seed = config['verification_selection_seed']
    refined, refined_alpha = [], []
    for i, index in enumerate(indices):
        refined.append(float(differentiable_log_target(model, latent[index], fine_data)))
        params = model._unflatten_latent_vector(latent[index], apply_numerical_limits=False)
        refined_alpha.append(float(model._selection_fraction_for_params(params)))
        if (i+1) % 64 == 0:
            print('Independent verification', i+1, '/', len(indices), flush=True)
    difference = np.asarray(refined) - arrays['log_primary'].ravel()[indices]
    report = {'groups': {name: {'draws': len(selected),
              'log_target_change_sd': float(difference[np.searchsorted(indices, selected)].std()),
              'log_target_change_05_50_95': np.quantile(difference[np.searchsorted(indices, selected)], [.05, .5, .95]).tolist()}
              for name, selected in groups.items()}}
    offsets = np.searchsorted(indices, uniform)
    primary_weight = weights.ravel()[uniform]
    primary_weight = primary_weight / primary_weight.sum()
    next_weight = np.exp(np.log(primary_weight) + difference[offsets] - logsumexp(np.log(primary_weight) + difference[offsets]))
    values = arrays['features'].reshape(-1, len(arrays['feature_names']))[uniform]
    all_values = arrays['features'].reshape(-1, values.shape[-1])
    mean = weights.ravel() @ all_values
    sd = np.sqrt(weights.ravel() @ (all_values-mean)**2)
    next_values = values.copy()
    next_values[:, alpha_column] = np.asarray(refined_alpha)[offsets]
    report.update(uniform_weight_ess_fraction=float(1 / (next_weight @ next_weight) / len(next_weight)),
                  uniform_mean_shift_in_sd=(abs(next_weight @ next_values - primary_weight @ values) / sd).tolist(),
                  maximum_alpha_change_in_sd=float(abs(np.asarray(refined_alpha)-alpha.ravel()[indices]).max() / sd[alpha_column]))
    return report, {'indices': indices, 'uniform_indices': uniform, 'log_verification': np.asarray(refined),
                    'alpha_verification': np.asarray(refined_alpha), 'log_target_change': difference,
                    'uniform_primary_weights': primary_weight, 'uniform_verification_weights': next_weight,
                    **{f'group_{key}': value for key, value in groups.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lisa_refinement.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    path = ROOT / args.config
    config = json.loads(path.read_text())
    target, production, products, model_config, catalog, fine, sources = checked_inputs(config, path)
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed refinement is retained')
    model = sampling_model(model_config)
    arrays, metadata = compute_primary(config, target, products, model, torch.tensor(fine, dtype=torch.float64), output, sources)
    weight, weight_check = ratio_weights(arrays['log_primary'] - arrays['log_coarse'],
                                       production['mixing']['minimum_bulk_ess'] / arrays['log_coarse'].size)
    stats = corrected_statistics(arrays['diagnostic_values'], arrays['diagnostic_names'].tolist(), weight,
                                 len(arrays['feature_names']), config)
    verification_data = verification_catalog(config, output, catalog, sources)
    if (output / 'verification.npz').exists() or (output / 'weights.npz').exists():
        raise FileExistsError('Existing unverified refinement products are retained')
    started = time.perf_counter()
    verification, checked_arrays = verify_refinement(config, model, arrays, weight, verification_data)
    np.savez_compressed(output / 'verification.npz', **checked_arrays)
    np.savez_compressed(output / 'weights.npz', weights=weight, log_ratio=arrays['log_primary'] - arrays['log_coarse'])
    verdict = decision(config, True, weight_check, stats, verification)
    result = {'configuration': config, 'source_sha256': sources, 'weight_diagnostics': weight_check,
              'statistics': stats, 'verification': verification, 'decision': verdict,
              'primary_seconds': metadata['seconds'], 'verification_seconds': time.perf_counter()-started,
              'product_sha256': {p.name: sha256(p) for p in output.glob('*.npz')}, 'runtime': runtime_metadata()}
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': verdict, 'weights': weight_check,
                      'minimum_mean_precision_equivalent_draws': stats['minimum_mean_precision_equivalent_draws'],
                      'minimum_cdf_precision_equivalent_draws': stats['minimum_cdf_precision_equivalent_draws']}), flush=True)


if __name__ == '__main__':
    main()
