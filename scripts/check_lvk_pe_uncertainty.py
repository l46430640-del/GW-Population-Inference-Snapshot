"""Assess full-PE event integration noise on saved LVK population draws."""
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
from scipy.stats import spearmanr
import torch

from check_density_impact import ROOT, Sources, corrected_model, digest, write_json
from lvk_injections import covered_mass_pairs
from analyze_lvk_population import coordinates, estimator, load_inputs, reference_density, spin_density


def selected_indices(count, representative_count, old_full_ess, compressed_ess, stress_count):
    representative = np.linspace(0, count-1, min(count, representative_count), dtype=int)
    full = np.argsort(old_full_ess, kind='stable')[:stress_count]
    compressed = np.argsort(compressed_ess, kind='stable')[:stress_count]
    indices = np.unique(np.r_[representative, full, compressed])
    return indices, np.isin(indices, representative), full, compressed


def probability_features(product):
    x, density = product['marginal_x'], product['marginal_draws']
    result = []
    for axis, threshold, above in ((0, np.log10(10), False), (0, np.log10(50), True),
                                   (1, 1., True), (2, .5, False)):
        dx = float(x[axis, 1]-x[axis, 0])
        lower, upper = x[axis]-dx/2, x[axis]+dx/2
        width = np.clip(upper-threshold if above else threshold-lower, 0., dx)
        result.append(density[axis] @ width)
    return np.stack(result, -1)


def integral_statistics(log_weights, groups=None, group_counts=None):
    """Delta covariance for shared samples, optionally retaining exact repeats."""
    n = log_weights.shape[1]
    if n < 2 or not np.all(np.isfinite(logsumexp(log_weights, axis=1))):
        raise ValueError('Each integral needs at least two rows and positive finite mass.')
    log_sum = logsumexp(log_weights, axis=1)
    p = np.exp(log_weights-log_sum[:, None])
    covariance = n/(n-1)*(p @ p.T - 1/n)
    ess = 1/np.sum(p*p, axis=1)
    if groups is None:
        clustered = covariance.copy()
    else:
        sizes = np.asarray(group_counts)
        if len(sizes) < 2 or int(sizes.sum()) != n:
            raise ValueError('Invalid repeated-state groups.')
        centered = np.stack([np.bincount(groups, weights=row, minlength=len(sizes))-sizes/n for row in p])
        clustered = len(sizes)/(len(sizes)-1)*(centered @ centered.T)
    return dict(log_mean=log_sum-np.log(n), ess=ess, covariance=covariance,
                clustered_covariance=clustered, max_weight=p.max(1),
                dominant_index=p.argmax(1), probabilities=p)


def pair_variance(covariance):
    variance = np.diag(covariance)[:, None]+np.diag(covariance)[None, :]-2*covariance
    if variance.min() < -1e-10:
        raise ValueError('Negative difference variance.')
    return np.maximum(variance, 0.)


def mean_sensitivity(features, covariance, log_change=None):
    n = len(features)
    weights = np.ones(n)/n
    mean, sd = features.mean(0), features.std(0)
    influence = weights[:, None]*(features-mean)
    variance = np.einsum('af,ab,bf->f', influence, covariance, influence)
    if variance.min() < -1e-10:
        raise ValueError('Negative functional variance.')
    se = np.sqrt(np.maximum(variance, 0.))
    result = {'mean': mean.tolist(), 'vi_sd': sd.tolist(), 'conditional_mc_se': se.tolist(),
              'conditional_mc_se_over_vi_sd': np.divide(se, sd, out=np.zeros_like(sd), where=sd>0).tolist()}
    if log_change is not None:
        weight = np.exp(log_change-logsumexp(log_change))
        change = weight @ features-mean
        result.update(reweighted_mean=(weight @ features).tolist(), mean_change=change.tolist(),
                      mean_change_over_vi_sd=np.divide(change, sd, out=np.zeros_like(sd), where=sd>0).tolist(),
                      ratio_ess_fraction=float(1/(n*np.sum(weight*weight))))
    return result


def feature_associations(features, min_ess, threshold):
    low = min_ess < threshold
    correlation = [float(spearmanr(features[:, i], np.log(min_ess)).statistic)
                   if np.ptp(features[:, i]) and np.ptp(min_ess) else 0. for i in range(features.shape[1])]
    return {'spearman_with_log_minimum_ess': correlation, 'low_ess_points': int(low.sum()),
            'other_points': int((~low).sum()),
            'low_ess_feature_median': np.median(features[low], axis=0).tolist() if low.any() else None,
            'other_feature_median': np.median(features[~low], axis=0).tolist() if (~low).any() else None}


@torch.no_grad()
def log_weights(model, params, points, log_prior, block, check):
    parts = []
    for start in range(0, len(points), block):
        check()
        point = torch.as_tensor(points[start:start+block], dtype=torch.float64)
        unit = (point-model.min_vals)/(model.max_vals-model.min_vals)
        value = model._log_pdf_phys_from_params(params, unit)
        mass = 10**point[:, 0]
        support = covered_mass_pairs(mass, 3+(mass-3)*point[:, 2])
        value = torch.where(support, value, -torch.inf)
        parts.append(value.numpy()-log_prior[start:start+block])
    return np.concatenate(parts)


def prepare_events(events, names):
    prepared, metadata = [], []
    fields = ('m1', 'm2', 'z', 'a1', 'a2', 'cost1', 'cost2')
    for name in names:
        event = events[name]
        points, jacobian = coordinates(event['m1'], event['m2'], event['z'])
        states = np.column_stack([event[key] for key in fields])
        _, groups, sizes = np.unique(states, axis=0, return_inverse=True, return_counts=True)
        prepared.append({'points': points, 'jacobian': jacobian, 'z_prior': event['z_prior'],
                         'a1': event['a1'], 'a2': event['a2'], 'groups': groups, 'sizes': sizes,
                         'physical': states[:, :3].copy()})
        metadata.append({'name': str(name), 'rows': len(points), 'unique_physical_spin_states': len(sizes),
                         'maximum_repeat_count': int(sizes.max()), 'fields': sorted(event)})
    return prepared, metadata


def component_region(model, params):
    values = []
    for marginal, chol in model._components(params):
        corr = (chol @ chol.transpose(-1, -2)).numpy()
        values.extend([corr[:, 0, 1].tolist(), corr[:, 0, 2].tolist(), corr[:, 1, 2].tolist()])
    first = params.get('axis_sk_log_k_a', params.get('log_k_a')).exp().numpy()+.0001
    second = params.get('axis_sk_log_k_b', params.get('log_k_b')).exp().numpy()+.0001
    return {'mixture_weights': params['weights_logits'].clamp(-20, 20).softmax(-1).numpy().tolist(),
            'z_shape_a': first[:, -2].tolist(), 'z_shape_b': second[:, -2].tolist(),
            'u_shape_a': first[:, -1].tolist(), 'u_shape_b': second[:, -1].tolist(),
            'component_correlations_mz_mu_zu': values}


@torch.no_grad()
def evaluate_case(row, config, events, sources, check):
    directory = row['directory']
    saved = sources.json(directory+'/summary.json')
    setup = SimpleNamespace(**saved['configuration'])
    cache = ('external_data/lvk/analysis_inputs/importance_mixture_v2_'
             f'{setup.spin}_{setup.posterior_samples}_{setup.injections}_{setup.mmax:g}_372100.npz')
    sources.add(cache)
    arrays = load_inputs(setup.spin, setup.posterior_samples, setup.injections, setup.mmax)
    posterior = sources.arrays(directory+'/posterior.npz')
    old_check = sources.arrays(directory+'/monte_carlo_check.npz')
    density = sources.arrays(directory+'/density_summary.npz')
    count = config['primary_draws'] if directory == row['primary'] else config['sensitivity_draws']
    indices, representative, full_stress, compressed_stress = selected_indices(
        len(posterior['latent']), count, old_check['full_min_event_ess'], posterior['neff_event'],
        config['stress_draws_per_ranking'])
    with contextlib.redirect_stdout(io.StringIO()):
        historical = estimator(setup, arrays)
        model = corrected_model(historical)
    # Match the finite physical bounds whose repair sensitivity was measured.
    model.min_vals, model.max_vals = historical.min_vals.double(), historical.max_vals.double()
    parameters = [model._unflatten_latent_vector(torch.as_tensor(posterior['latent'][i], dtype=torch.float64)) for i in indices]
    features = probability_features(density)[indices]
    names = arrays['event_names'].tolist()
    if names != list(events):
        raise ValueError('Event identity/order differs between adopted LVK configurations.')
    logs, ess, maxima, covariance, clustered, compression, dominant, deletions = [], [], [], [], [], [], [], []
    event_records = []
    for event_index, (name, event) in enumerate(events.items()):
        check()
        points = event['points']
        prior = event['z_prior']*event['jacobian']/np.maximum(spin_density(event['a1'], event['a2'], setup.spin), 1e-30)
        full = np.stack([log_weights(model, params, points, np.log(prior), config['point_block'], check) for params in parameters])
        result = integral_statistics(full, event['groups'], event['sizes'])
        p = result['probabilities']
        data = arrays['data'][event_index].astype(np.float64)
        proposal = reference_density(points, setup.mmax)
        log_reference_mean = logsumexp(np.log(proposal,
                                      where=proposal>0,
                                      out=np.full(len(points), -np.inf))-np.log(prior))-np.log(len(points))
        compressed = np.array([logsumexp(log_weights(model, params, data[:, :3], np.log(data[:, 3]),
                                   config['point_block'], check))-np.log(len(data))+log_reference_mean for params in parameters])
        worst = int(result['ess'].argmin())
        chosen = int(result['dominant_index'][worst])
        # Delete the same original PE row at every population point; this is
        # a leverage stress, not a bootstrap estimate or changed likelihood.
        deletion = np.log1p(-p[:, chosen])+np.log(len(points)/(len(points)-1))
        logs.append(result['log_mean'])
        ess.append(result['ess'])
        maxima.append(result['max_weight'])
        covariance.append(result['covariance'])
        clustered.append(result['clustered_covariance'])
        compression.append(compressed-result['log_mean'])
        dominant.append(result['dominant_index'])
        deletions.append(deletion)
        pair = pair_variance(result['covariance'][np.ix_(representative, representative)])
        physical = event['physical'][chosen]
        low = result['ess'] < config['low_ess_diagnostic']
        event_records.append({'event': name, 'pe_rows': len(points), 'unique_states': len(event['sizes']),
            'ess_quantiles_representative': np.quantile(result['ess'][representative], config['quantiles']).tolist(),
            'minimum_ess_all_checked': float(result['ess'][worst]), 'worst_saved_draw': int(indices[worst]),
            'low_ess_representative_count': int(low[representative].sum()), 'low_ess_all_checked_count': int(low.sum()),
            'mean_pair_difference_variance_representative': float(pair.mean()),
            'worst_maximum_normalized_weight': float(result['max_weight'][worst]),
            'dominating_original_pe_row': chosen,
            'dominating_m1_z_q': [float(physical[0]), float(physical[2]), float(physical[1]/physical[0])],
            'feature_associations_all_checked': feature_associations(features, result['ess'], config['low_ess_diagnostic']),
            'worst_population_features': features[worst].tolist(),
            'worst_component_region': component_region(model, parameters[worst])})
    covariance, clustered = np.array(covariance), np.array(clustered)
    ess, compression = np.array(ess), np.array(compression)
    total, cluster_total = covariance.sum(0), clustered.sum(0)
    selected = np.ix_(representative, representative)
    pair = pair_variance(total[selected])
    delta = -compression.sum(0)
    reports = sorted(event_records, key=lambda r: r['mean_pair_difference_variance_representative'], reverse=True)
    summary = {'label': Path(directory).name, 'directory': directory, 'representative_draws': int(representative.sum()),
        'evaluated_draws': len(indices), 'event_count': len(names), 'configuration': saved['configuration'],
        'full_check_stress_indices': full_stress.tolist(), 'compressed_stress_indices': compressed_stress.tolist(),
        'feature_names': config['feature_names'], 'low_ess_diagnostic': config['low_ess_diagnostic'],
        'minimum_event_ess_representative_quantiles': np.quantile(ess[:, representative].min(0), config['quantiles']).tolist(),
        'minimum_event_ess_all_checked': float(ess.min()),
        'event_log_likelihood_conditional_sd_quantiles': np.quantile(np.sqrt(np.maximum(np.diag(total)[representative], 0)), config['quantiles']).tolist(),
        'mean_pair_difference_variance_representative': float(pair.mean()),
        'pair_difference_sd_quantiles': np.quantile(np.sqrt(pair[np.triu_indices(len(pair), 1)]), config['quantiles']).tolist(),
        'clustered_mean_pair_difference_variance': float(pair_variance(cluster_total[selected]).mean()),
        'compression_delta_sd_representative': float(delta[representative].std()),
        'scientific_sensitivity': mean_sensitivity(features[representative], total[selected], delta[representative]),
        'clustered_scientific_sensitivity': mean_sensitivity(features[representative], cluster_total[selected]),
        'feature_associations_representative': feature_associations(features[representative], ess[:, representative].min(0), config['low_ess_diagnostic']),
        'feature_associations_all_checked': feature_associations(features, ess.min(0), config['low_ess_diagnostic']),
        'events_by_pair_variance': reports, 'selection_held_fixed': True, 'posterior_certified': False,
        'pe_dependence_calibrated': False, 'covariance_scope': 'independent-row delta approximation; exact-duplicate cluster sensitivity',
        'historical_results_superseded': False}
    product = dict(indices=indices, representative=representative, features=features, event_names=np.array(names),
        full_log_integrals=np.array(logs), full_ess=ess, max_weight=np.array(maxima),
        covariance=covariance, clustered_covariance=clustered, compression_log_delta=compression,
        dominant_pe_indices=np.array(dominant), fixed_row_deletion_log_delta=np.array(deletions),
        latent=posterior['latent'][indices])
    return summary, product


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lvk_pe_uncertainty.json')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    attempts = sorted(output.glob('attempt_*'))
    if any(not (p/'cost.json').is_file() for p in attempts):
        raise RuntimeError('An earlier attempt has no terminal cost record; inspect it before continuing.')
    charged = sum(json.loads((p/'cost.json').read_text())['seconds'] for p in attempts)
    directory = output/f'attempt_{len(attempts):03d}'
    directory.mkdir(exist_ok=False)
    process, sources = psutil.Process(), Sources()
    peak, completed, status = 0, [], 'failed'
    def check():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if time.perf_counter()-PROCESS_STARTED+charged >= config['maximum_seconds']:
            raise RuntimeError('Declared shared time allowance exhausted.')
        if peak > config['maximum_resident_gib']*2**30:
            raise RuntimeError('Declared memory allowance exhausted.')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > config['maximum_output_mib']*2**20:
            raise RuntimeError('Declared output allowance exhausted.')
    try:
        torch.set_num_threads(1)
        for name in (args.config, 'scripts/check_lvk_pe_uncertainty.py', 'configs/analysis_runs.json',
                     'docs/analysis/lvk_input_manifest.json', 'docs/analysis/lvk_pe_source_metadata.json'):
            sources.add(name)
        sources.json('results/analysis/density_impact/attempt_000/sources.json')
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).resolve().is_relative_to(ROOT/'scripts'):
                sources.add(Path(filename).resolve())
        path = sources.add(config['pe_input'], config['pe_sha256'])
        with path.open('rb') as handle:
            raw = pickle.load(handle)
        analysis = sources.json('configs/analysis_runs.json')
        names = sources.json(analysis['lvk_baseline']+'/summary.json')['event_names']
        prepared, metadata = prepare_events(raw, names)
        del raw
        events = dict(zip(names, prepared))
        del prepared
        write_json(directory/'pe_metadata.json', {'events': metadata, 'chain_ids_available': False,
                   'temporal_order_verified': False, 'posterior_certified': False})
        for row in analysis['lvk_sensitivity']:
            check()
            print('Starting', Path(row['directory']).name, flush=True)
            began = time.perf_counter()
            report, arrays = evaluate_case(dict(row, primary=analysis['lvk_baseline']), config, events, sources, check)
            report['seconds'] = time.perf_counter()-began
            path = directory/(report['label']+'.npz')
            np.savez_compressed(path, **arrays)
            report['product_sha256'] = digest(path)
            write_json(path.with_suffix('.json'), report)
            completed.append(report)
            print('Finished', report['label'], 'seconds=', round(report['seconds'], 2),
                  'mean_pair_variance=', report['mean_pair_difference_variance_representative'], flush=True)
        check()
        status = 'complete'
    finally:
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(directory/'cost.json', {'status': status, 'seconds': seconds, 'previous_seconds': charged,
                   'total_charge_seconds': seconds+charged, 'peak_resident_bytes': peak,
                   'completed_cases': [r['label'] for r in completed]})
        write_json(directory/'sources.json', {'source_sha256': sources.hashes, 'configuration': config})
    write_json(directory/'summary.json', {'status': status, 'cases': completed, 'posterior_certified': False,
                'historical_results_superseded': False, 'pe_dependence_calibrated': False})
    print(json.dumps({'status': status, 'cases': len(completed), 'seconds': seconds}))


if __name__ == '__main__':
    main()
