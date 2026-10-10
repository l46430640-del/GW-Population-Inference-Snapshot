"""Sample and validate the bimodal LISA posterior with ordinary NUTS."""
from __future__ import annotations

import argparse
import json
import time

import arviz as az
import numpy as np
import pyro
from scipy.special import logsumexp
import torch

from benchmark_selected_mixture import kumaraswamy_moment, runtime_metadata
from correct_selected_posterior import compare_features, sha256, swap_components
from diagnose_lisa_correction import verified_inputs
from sample_lisa_transport import (
    TransportPotential, check_precision, differentiable_log_target, lisa_features,
    run_chain, sampling_model,
)
from validate_population_morphologies import ROOT
from gwpopulation.core import SafeKumaraswamy


def write_json(path, value):
    if path.exists():
        raise FileExistsError(f'Completed product is retained: {path}')
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_product(path):
    metadata = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != metadata['draw_product_sha256']:
        raise ValueError(f'Product differs from its recorded digest: {path}')
    return metadata


@torch.no_grad()
def initial_points(config, model):
    products = [dict(np.load(ROOT / name)) for name in config['initial_products']]
    weights = np.concatenate([p['raw_weights'] / p['raw_weights'].sum() / len(products) for p in products])
    alpha = np.concatenate([p['alpha'] for p in products])
    latent = torch.tensor(np.concatenate([p['latent'] for p in products]), dtype=torch.float64)
    order = np.argsort(alpha)
    cdf = np.cumsum(weights[order])
    indices = order[np.searchsorted(cdf, config['initial_efficiency_quantiles']).clip(0, len(order) - 1)]
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Declared initial quantiles do not give distinct starting points')
    points = latent[indices].clone()
    offsets = np.cumsum([0] + [len(p['latent']) for p in products])
    records = []
    for chain, index in enumerate(indices):
        source = int(np.searchsorted(offsets[1:], index, side='right'))
        local_index = int(index - offsets[source])
        if config['swap_initial_labels'][chain]:
            points[chain] = swap_components(model, points[chain])
        records.append({'source': config['initial_products'][source], 'index': local_index,
                        'efficiency_quantile': config['initial_efficiency_quantiles'][chain],
                        'alpha': float(alpha[index]), 'labels_swapped': config['swap_initial_labels'][chain],
                        'saved_log_target': float(products[source]['log_target'][local_index])})
    return points, records


@torch.no_grad()
def additional_functionals(model, latent, cdf_points):
    params = model._unflatten_latent_vector(latent, apply_numerical_limits=False)
    a, b = params['log_k_a'].exp() + .0001, params['log_k_b'].exp() + .0001
    means = kumaraswamy_moment(a, b, 1)
    swap = means[:, 0, 0] > means[:, 1, 0]
    canonical = latent.clone()
    canonical[swap] = swap_components(model, canonical[swap])
    names = [f'ordered_coordinate_{i:02d}' for i in range(latent.shape[1])]
    arrays = [canonical.numpy()]
    weights = torch.softmax(params['weights_logits'].clamp(-20., 20.), -1)
    marginal = SafeKumaraswamy(a, b)
    for x in cdf_points:
        cdf = (weights[:, :, None] * marginal.cdf(torch.tensor(x, dtype=latent.dtype))).sum(1)
        names.extend(f'cdf_x{axis}_{x:g}' for axis in range(3))
        arrays.append(cdf.numpy())
    return names, np.column_stack(arrays)


def mixing_summary(names, values):
    if values.ndim != 3 or values.shape[-1] != len(names) or not np.isfinite(values).all():
        raise ValueError('Finite chain-by-draw functional values are required')
    dataset = az.from_dict(posterior={name: values[:, :, j] for j, name in enumerate(names)})
    table = az.summary(dataset, kind='diagnostics', round_to='none')
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError('Non-finite MCMC diagnostics')
    return {'functionals': table.to_dict(orient='index'),
            'maximum_rhat': float(table.r_hat.max()),
            'minimum_bulk_ess': float(table.ess_bulk.min()),
            'minimum_tail_ess': float(table.ess_tail.min())}


def acceptance(config, reports, mixing, numerical):
    complete = (len(reports) == len(config['seeds']) == 4
                and all(r['status'] == 'complete' and r['samples_completed'] == config['samples']
                        and r['warmup_completed'] == config['warmup'] for r in reports))
    no_divergences = complete and sum(len(r['divergences']) for r in reports) <= config['maximum_divergences']
    mixing_ok = (mixing is not None and np.isfinite([mixing[k] for k in (
        'maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess')]).all()
        and mixing['maximum_rhat'] < config['maximum_rhat']
        and mixing['minimum_bulk_ess'] >= config['minimum_bulk_ess']
        and mixing['minimum_tail_ess'] >= config['minimum_tail_ess'])
    integration_ok = (numerical is not None
                      and set(numerical['groups']) == {*(f'chain_{i}' for i in range(4)), 'efficiency_extremes'}
                      and all(r['total_change_sd'] < config['maximum_refined_log_target_sd']
                              for r in numerical['groups'].values())
                      and numerical['uniform_weight_ess_fraction'] >= config['minimum_refinement_weight_ess_fraction']
                      and np.max(numerical['uniform_mean_shift_in_sd']) <= config['maximum_refined_mean_shift_in_sd'])
    features_ok = numerical is not None and np.max(numerical['feature_change_in_sd']) < config['maximum_feature_change_in_sd']
    precision_ok = numerical is not None and numerical['precision']['maximum_absolute_log_target_difference'] < config['precision_log_density_tolerance']
    decisions = {'complete': bool(complete), 'divergences_passed': bool(no_divergences),
                 'mixing_passed': bool(mixing_ok), 'integration_passed': bool(integration_ok),
                 'functional_quadrature_passed': bool(features_ok), 'precision_passed': bool(precision_ok)}
    decisions['eligible_for_scientific_summaries'] = all(decisions.values())
    return decisions


@torch.no_grad()
def numerical_check(model, latent, features, names, log_target, data, fine_data, config):
    count = latent.shape[1]
    groups = {f'chain_{chain}': chain * count + np.linspace(0, count - 1,
              min(count, config['refinement_draws_per_chain']), dtype=int) for chain in range(len(latent))}
    alpha = features[:, :, names.index('alpha')]
    tail_count = config['refinement_extremes_per_chain']
    groups['efficiency_extremes'] = np.concatenate([
        chain * count + np.concatenate([np.argsort(a)[:tail_count], np.argsort(a)[-tail_count:]])
        for chain, a in enumerate(alpha)])
    uniform = np.concatenate([groups[f'chain_{c}'] for c in range(len(latent))])
    indices = np.unique(np.concatenate(list(groups.values())))
    flat = latent.reshape(-1, latent.shape[-1])
    old_count, old_seed = model.lambda_det_mc_samples, model.selection_mc_seed
    event_change, selection_change = [], []
    try:
        for index in indices:
            point = flat[index]
            event_target = float(differentiable_log_target(model, point, fine_data))
            model.lambda_det_mc_samples = config['fine_selection_points']
            model.selection_mc_seed = old_seed + config['fine_selection_seed_offset']
            refined = float(differentiable_log_target(model, point, fine_data))
            model.lambda_det_mc_samples, model.selection_mc_seed = old_count, old_seed
            event_change.append(event_target - log_target.reshape(-1)[index])
            selection_change.append(refined - event_target)
    finally:
        model.lambda_det_mc_samples, model.selection_mc_seed = old_count, old_seed
    event_change, selection_change = np.asarray(event_change), np.asarray(selection_change)
    total = event_change + selection_change
    results = {}
    for name, selected in groups.items():
        offsets = np.searchsorted(indices, selected)
        results[name] = {'draws': len(selected), 'total_change_sd': float(total[offsets].std()),
                         'total_change_05_50_95': np.quantile(total[offsets], [.05, .5, .95]).tolist(),
                         'event_change_sd': float(event_change[offsets].std()),
                         'selection_change_sd': float(selection_change[offsets].std())}
    offsets = np.searchsorted(indices, uniform)
    weights = np.exp(total[offsets] - logsumexp(total[offsets]))
    values = features.reshape(-1, len(names))[uniform]
    scale = features.reshape(-1, len(names)).std(0)
    mean_shift = abs(weights @ values - values.mean(0)) / scale
    fine_names, fine_features = lisa_features(model, flat[uniform], config['fine_feature_transport_power'])
    if names != fine_names:
        raise ValueError('Fine and coarse functionals differ')
    feature_change = abs(fine_features - values).max(0) / scale
    precision = check_precision(model, data, flat[indices])
    summary = {'groups': results, 'uniform_draws': len(uniform),
               'uniform_weight_ess_fraction': float(1 / np.sum(weights**2) / len(weights)),
               'uniform_mean_shift_in_sd': mean_shift.tolist(),
               'feature_change_in_sd': feature_change.tolist(), 'precision': precision}
    arrays = {'indices': indices, 'uniform_indices': uniform, 'event_change': event_change,
              'selection_change': selection_change, 'uniform_refinement_weights': weights,
              'fine_features': fine_features, **{f'group_{k}': v for k, v in groups.items()}}
    return summary, arrays


def summarize(config, output, source_config, model, data, fine):
    target_hash = sha256(output / 'target.json')
    paths = [output / f'chain_{seed}.npz' for seed in config['seeds']]
    reports = [verify_product(p) for p in paths]
    if any(r['target_product_sha256'] != target_hash or r['seed'] != seed or r['configuration'] != config
           for r, seed in zip(reports, config['seeds'])):
        raise ValueError('Production chains have different targets')
    products = [dict(np.load(p)) for p in paths]
    minimum_length = min(len(p['latent']) for p in products)
    names = products[0]['feature_names'].tolist()
    full_names = products[0]['diagnostic_names'].tolist()
    if any(p['feature_names'].tolist() != names or p['diagnostic_names'].tolist() != full_names for p in products):
        raise ValueError('Production chains have different functionals')
    if minimum_length < 4:
        write_json(output / 'summary.json', {'configuration': config, 'decision': acceptance(config, reports, None, None),
                   'reason': 'Too few completed draws for diagnostics', 'target_product_sha256': target_hash})
        return
    latent = torch.tensor(np.stack([p['latent'][:minimum_length] for p in products]))
    features = np.stack([p['features'][:minimum_length] for p in products])
    diagnostic_values = np.stack([p['diagnostic_values'][:minimum_length] for p in products])
    log_target = np.stack([p['log_target'][:minimum_length] for p in products])
    mixing = mixing_summary(full_names, diagnostic_values)
    started = time.perf_counter()
    numerical, arrays = numerical_check(model, latent, features, names, log_target, data, fine, config)
    numerical_seconds = time.perf_counter() - started
    numerical_path = output / 'numerical.npz'
    if numerical_path.exists():
        raise FileExistsError(f'Completed numerical check is retained: {numerical_path}')
    np.savez_compressed(numerical_path, **arrays)
    decision = acceptance(config, reports, mixing, numerical)
    baseline = {}
    for directory in source_config['sources']:
        old = torch.tensor(np.load(ROOT / directory / 'posterior.npz')['latent'], dtype=torch.float64)
        old_names, old_features = lisa_features(model, old, config['feature_transport_power'])
        if old_names != names:
            raise ValueError('Baseline and MCMC functionals differ')
        baseline[directory] = compare_features(features.reshape(-1, len(names)), old_features, names)
    summaries = {name: {'mean': float(features[:, :, j].mean()), 'sd': float(features[:, :, j].std()),
                        'quantiles_05_50_95': np.quantile(features[:, :, j], [.05, .5, .95]).tolist()}
                 for j, name in enumerate(names)}
    result = {'configuration': config, 'decision': decision, 'mixing': mixing,
              'population_mixing': mixing_summary(names, features), 'numerical': numerical,
              'population_summaries': summaries, 'baseline_comparison': baseline,
              'numerical_seconds': numerical_seconds,
              'sampling_seconds': sum(r['sampling_seconds'] for r in reports),
              'compilation_seconds': sum(r['compilation_seconds'] for r in reports),
              'chain_reports': {p.with_suffix('.json').name: sha256(p.with_suffix('.json')) for p in paths},
              'numerical_product_sha256': sha256(numerical_path), 'target_product_sha256': target_hash,
              'runtime': runtime_metadata()}
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': decision, 'maximum_rhat': mixing['maximum_rhat'],
                      'minimum_bulk_ess': mixing['minimum_bulk_ess'],
                      'minimum_tail_ess': mixing['minimum_tail_ess']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lisa_production.json')
    parser.add_argument('--chain', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    source_path = ROOT / config['source_config']
    source_config = json.loads(source_path.read_text())
    configurations, coarse, fine, sources = verified_inputs(source_config, source_path)
    paths = [config_path, ROOT / 'scripts/run_lisa_production.py', ROOT / 'scripts/sample_lisa_transport.py',
             ROOT / 'configs/lisa_transport.json', ROOT / 'results/analysis/lisa_transport_pilot/summary.json',
             ROOT / 'results/analysis/lisa_transport_pilot/precision.json']
    for name in config['initial_products']:
        p = ROOT / name
        verify_product(p)
        paths.extend([p, p.with_suffix('.json')])
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    model = sampling_model(configurations[0])
    initial, records = initial_points(config, model)
    signature = {'configuration': config, 'source_sha256': sources,
                 'initial_points': initial.numpy().tolist(), 'initial_provenance': records}
    target_path = output / 'target.json'
    if target_path.exists():
        if json.loads(target_path.read_text()) != signature:
            raise ValueError('Existing production target differs from the requested calculation')
    else:
        write_json(target_path, signature)
    data, fine_data = torch.tensor(coarse, dtype=torch.float64), torch.tensor(fine, dtype=torch.float64)
    if args.summarize:
        if (output / 'summary.json').exists():
            raise FileExistsError('Completed production summary is retained')
        summarize(config, output, source_config, model, data, fine_data)
        return
    for chain in ([args.chain] if args.chain is not None else range(len(config['seeds']))):
        seed = config['seeds'][chain]
        path = output / f'chain_{seed}.npz'
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(f'Completed production chain is retained: {path}')
        pyro.clear_param_store()
        potential = TransportPotential(model, data, 'raw')
        report, arrays = run_chain(potential, initial[chain], config, seed)
        latent = torch.tensor(arrays['latent'])
        names, features = lisa_features(model, latent, config['feature_transport_power'])
        if len(latent):
            extra_names, extra_values = additional_functionals(model, latent, config['marginal_cdf_points'])
            with torch.no_grad():
                log_target = np.asarray([float(differentiable_log_target(model, z, data)) for z in latent])
            diagnostic_names = names + extra_names + ['log_target']
            diagnostic_values = np.column_stack([features, extra_values, log_target])
        else:
            log_target, diagnostic_names, diagnostic_values = np.empty(0), [], np.empty((0, 0))
        arrays.update(initial_latent=initial[chain].numpy(), feature_names=np.asarray(names), features=features,
                      diagnostic_names=np.asarray(diagnostic_names), diagnostic_values=diagnostic_values,
                      log_target=log_target)
        np.savez_compressed(path, **arrays)
        report.update(configuration=config, target_product_sha256=sha256(target_path),
                      draw_product_sha256=sha256(path), initial=records[chain], runtime=runtime_metadata())
        write_json(path.with_suffix('.json'), report)
        print('Saved', path.name, report['status'], round(report['sampling_seconds'], 2), 'seconds', flush=True)


if __name__ == '__main__':
    main()
