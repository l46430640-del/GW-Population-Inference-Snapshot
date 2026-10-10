"""Extend the LVK K=2 compression check to every saved variational draw."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import psutil
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon
import torch

from check_density_impact import ROOT, Sources, corrected_model, digest, write_json
from check_lvk_pe_uncertainty import log_weights, probability_features
from correct_selected_posterior import importance_weights, weighted_quantiles
from analyze_lvk_population import coordinates, estimator, load_inputs, reference_density, spin_density
from summarize_lvk_comparisons import common_marginals


def paired_means(features, log_ratio):
    features, log_ratio = np.asarray(features), np.asarray(log_ratio)
    n = len(features)
    if (features.ndim != 2 or n < 2 or log_ratio.shape != (n,)
            or not np.isfinite(log_ratio).all() or not np.isfinite(features).all()):
        raise ValueError('Finite paired ratios and at least two feature rows are required.')
    weight = np.exp(log_ratio-logsumexp(log_ratio))
    mean, sd = features.mean(0), features.std(0)
    updated = weight @ features
    influence = n*weight[:, None]*(features-updated)-(features-mean)
    mcse = influence.std(0, ddof=1)/np.sqrt(n)
    scale = lambda x: np.divide(x, sd, out=np.zeros_like(sd), where=sd>0)
    return {'old_mean': mean.tolist(), 'reweighted_mean': updated.tolist(), 'vi_sd': sd.tolist(),
            'mean_change': (updated-mean).tolist(), 'mean_change_over_vi_sd': scale(updated-mean).tolist(),
            'paired_change_mcse': mcse.tolist(), 'paired_change_mcse_over_vi_sd': scale(mcse).tolist(),
            'ratio_ess_fraction': float(1/(n*np.sum(weight*weight))), 'maximum_weight': float(weight.max())}


def quantiles(features, weights):
    weights = np.asarray(weights)
    return np.array([weighted_quantiles(values, weights) for values in np.asarray(features).T])


def paired_bootstrap(features, log_ratio, count, seed):
    rng = np.random.default_rng(seed)
    n = len(features)
    changes, quantile_changes = [], []
    for _ in range(count):
        indices = rng.integers(n, size=n)
        values, ratios = features[indices], log_ratio[indices]
        weights = np.exp(ratios-logsumexp(ratios))
        changes.append(weights @ values-values.mean(0))
        quantile_changes.append(quantiles(values, weights)-quantiles(values, np.ones(n)/n))
    changes, quantile_changes = np.array(changes), np.array(quantile_changes)
    return {'scope': 'paired resampling of independent saved VI draws; event integrals fixed',
            'replicates': count, 'seed': seed,
            'mean_change_mcse': changes.std(0, ddof=1).tolist(),
            'mean_change_percentiles_025_50_975': np.quantile(changes, [.025, .5, .975], axis=0).T.tolist(),
            'quantile_change_mcse': quantile_changes.std(0, ddof=1).tolist(),
            'quantile_change_percentiles_025_50_975': np.moveaxis(np.quantile(quantile_changes, [.025, .5, .975], axis=0), 0, -1).tolist()}


def distribution_comparison(comparison, reference, baseline, weights):
    marginal = np.einsum('d,adb->ab', weights, comparison['marginal_probabilities'])
    detected = weights @ comparison['detected_joint_probabilities']
    return {'marginal_distance_to_reference': jensenshannon(marginal, reference, base=2, axis=1).tolist(),
            'detected_distance_to_primary': float(jensenshannon(detected, baseline, base=2))}


def assess(products, comparison, reference, baseline, config):
    delta = -products['compression_log_delta'].sum(0)
    features = products['features']
    weights, diagnostics = importance_weights(delta)
    results = {}
    equal = np.ones(len(delta))/len(delta)
    old_quantiles = quantiles(features, equal)
    for name, w in weights.items():
        result = paired_means(features, np.log(w))
        result['quantiles_05_50_95'] = quantiles(features, w).tolist()
        result['quantile_change'] = (quantiles(features, w)-old_quantiles).tolist()
        result['distributions'] = distribution_comparison(comparison, reference, baseline, w)
        results[name] = result
    batches = []
    for indices in np.array_split(np.arange(len(delta)), 4):
        batches.append({'indices': indices.tolist(), 'means': paired_means(features[indices], delta[indices])})
    check_indices = products['reused_indices'][products['previous_representative']]
    initial = paired_means(features[check_indices], delta[check_indices])
    event_sd = products['compression_log_delta'].std(1)
    order = np.argsort(event_sd)[::-1]
    result = {'feature_names': config['feature_names'], 'draws': len(delta),
        'target_log_ratio_sd': float(delta.std()), 'target_log_ratio_range': [float(delta.min()), float(delta.max())],
        'ratio_diagnostics': diagnostics, 'raw': results['raw'], 'psis': results['psis'],
        'old_empirical_quantiles_05_50_95': old_quantiles.tolist(),
        'old_linear_quantiles_05_50_95': np.quantile(features, [.05, .5, .95], axis=0).T.tolist(),
        'old_distributions': distribution_comparison(comparison, reference, baseline, equal),
        'paired_bootstrap': paired_bootstrap(features, delta, config['bootstrap_replicates'], config['bootstrap_seed']),
        'four_disjoint_batches': batches, 'initial_16_point_assessment': initial,
        'events_by_compression_sd': [{'event': str(products['event_names'][i]), 'log_integral_delta_sd': float(event_sd[i])} for i in order],
        'full_event_ess_quantiles': np.quantile(products['full_ess'].min(0), [.05, .5, .95]).tolist(),
        'selection_held_fixed': True, 'full_posterior_importance_correction': False,
        'posterior_certified': False, 'historical_results_superseded': False,
        'finite_draw_uncertainty_excludes_pe_error': True}
    products.update(raw_weights=weights['raw'], psis_weights=weights['psis'], log_target_ratio=delta)
    return result


@torch.no_grad()
def evaluate(config, sources, check, products):
    directory = config['directory']
    saved = sources.json(directory+'/summary.json')
    setup = SimpleNamespace(**saved['configuration'])
    cache = ('external_data/lvk/analysis_inputs/importance_mixture_v2_'
             f'{setup.spin}_{setup.posterior_samples}_{setup.injections}_{setup.mmax:g}_372100.npz')
    sources.add(cache)
    arrays = load_inputs(setup.spin, setup.posterior_samples, setup.injections, setup.mmax)
    posterior = sources.arrays(directory+'/posterior.npz')
    density = sources.arrays(directory+'/density_summary.npz')
    if len(posterior['latent']) != config['expected_draws']:
        raise ValueError('Saved draw count changed.')
    previous = config['previous_assessment']
    sources.json(previous+'/sources.json')
    record = sources.json(previous+'/'+config['previous_case']+'.json')
    old_path = previous+'/'+config['previous_case']+'.npz'
    sources.add(old_path, record['product_sha256'])
    old = sources.arrays(old_path)
    names, reused = arrays['event_names'], old['indices']
    if not np.array_equal(names, old['event_names']) or not np.array_equal(posterior['latent'][reused], old['latent']):
        raise ValueError('Saved latent coordinates or event identities changed.')
    shape = (len(names), len(posterior['latent']))
    for key in ('full_log_integrals', 'full_ess', 'max_weight', 'compression_log_delta'):
        products[key] = np.full(shape, np.nan)
        products[key][:, reused] = old[key]
    features = probability_features(density)
    np.testing.assert_array_equal(features[reused], old['features'])
    products.update(features=features, indices=np.arange(shape[1]), event_names=names,
                    reused_indices=reused, previous_representative=old['representative'],
                    latent=posterior['latent'], completed_events=np.zeros(shape[0], dtype=bool))
    remaining = np.setdiff1d(np.arange(shape[1]), reused)
    with contextlib.redirect_stdout(io.StringIO()):
        historical = estimator(setup, arrays)
        model = corrected_model(historical)
    model.min_vals, model.max_vals = historical.min_vals.double(), historical.max_vals.double()
    parameters = [model._unflatten_latent_vector(torch.as_tensor(v, dtype=torch.float64)) for v in posterior['latent']]
    with sources.add(config['pe_input'], config['pe_sha256']).open('rb') as handle:
        events = pickle.load(handle)
    check()
    errors = []
    for event_index, name in enumerate(names):
        event = events[name]
        points, jacobian = coordinates(event['m1'], event['m2'], event['z'])
        prior = event['z_prior']*jacobian/np.maximum(spin_density(event['a1'], event['a2'], setup.spin), 1e-30)
        log_prior = np.log(prior)
        proposal = reference_density(points, setup.mmax)
        reference_mean = logsumexp(np.log(proposal, where=proposal>0,
                                  out=np.full(len(points), -np.inf))-log_prior)-np.log(len(points))
        compressed = arrays['data'][event_index].astype(np.float64)
        # Independently check two old rows in each declared event before reuse.
        verify = reused[[0, -1]] if name in config['reuse_check_events'] else np.array([], dtype=int)
        for index in np.r_[remaining, verify]:
            full = log_weights(model, parameters[index], points, log_prior, config['point_block'], check)
            log_sum = logsumexp(full)
            if not np.isfinite(log_sum):
                raise ValueError('Nonfinite full event integral.')
            full_mean = log_sum-np.log(len(full))
            p = np.exp(full-log_sum)
            comp = log_weights(model, parameters[index], compressed[:, :3], np.log(compressed[:, 3]), config['point_block'], check)
            delta = logsumexp(comp)-np.log(len(comp))+reference_mean-full_mean
            if index in verify:
                error = max(abs(full_mean-products['full_log_integrals'][event_index, index]),
                            abs(delta-products['compression_log_delta'][event_index, index]))
                errors.append(float(error))
                if error > config['reuse_tolerance']:
                    raise ValueError('Reused full-PE evaluation does not match.')
            else:
                products['full_log_integrals'][event_index, index] = full_mean
                products['compression_log_delta'][event_index, index] = delta
                products['full_ess'][event_index, index] = 1/np.sum(p*p)
                products['max_weight'][event_index, index] = p.max()
        products['completed_events'][event_index] = True
        if event_index % 10 == 0 or event_index == len(names)-1:
            print('Completed events', event_index+1, '/', len(names), flush=True)
    comparison = sources.arrays(directory+'/comparison_summary.npz')
    reference, _ = common_marginals(sources.arrays('results/analysis/lvk_reference/density_summary.npz'))
    analysis = sources.json('configs/analysis_runs.json')
    baseline = sources.arrays(analysis['lvk_baseline']+'/comparison_summary.npz')['detected_joint_probabilities'].mean(0)
    result = assess(products, comparison, reference.mean(1), baseline, config)
    result.update(reused_draws=len(reused), newly_evaluated_draws=len(remaining),
                  reuse_check_log_integral_errors=errors, event_count=len(names))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lvk_k2_compression.json')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    attempts = sorted(output.glob('attempt_*'))
    if any(not (p/'cost.json').is_file() for p in attempts):
        raise RuntimeError('Inspect the unfinished attempt before proceeding.')
    charged = sum(json.loads((p/'cost.json').read_text())['seconds'] for p in attempts)
    directory = output/f'attempt_{len(attempts):03d}'
    directory.mkdir(exist_ok=False)
    process, sources, products = psutil.Process(), Sources(), {}
    status, peak = 'failed', 0
    def check():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if time.perf_counter()-PROCESS_STARTED+charged > config['maximum_seconds']:
            raise RuntimeError('Shared calculation allowance exhausted.')
        if peak > config['maximum_resident_gib']*2**30:
            raise RuntimeError('Resident-memory allowance exhausted.')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > config['maximum_output_mib']*2**20:
            raise RuntimeError('Output allowance exhausted.')
    try:
        torch.set_num_threads(1)
        sources.add(args.config)
        sources.add(Path(__file__))
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).resolve().is_relative_to(ROOT/'scripts'):
                sources.add(Path(filename).resolve())
        result = evaluate(config, sources, check, products)
        check()
        status = 'complete'
    finally:
        np.savez_compressed(directory/'integrals.npz', **products)
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(directory/'cost.json', {'status': status, 'seconds': seconds, 'previous_seconds': charged,
                   'total_charge_seconds': charged+seconds, 'peak_resident_bytes': peak})
        write_json(directory/'sources.json', {'source_sha256': sources.hashes, 'configuration': config})
    result['product_sha256'] = digest(directory/'integrals.npz')
    write_json(directory/'summary.json', result)
    print(json.dumps({'status': status, 'seconds': seconds, 'reused_draws': result['reused_draws'],
                      'newly_evaluated_draws': result['newly_evaluated_draws']}))


if __name__ == '__main__':
    main()
