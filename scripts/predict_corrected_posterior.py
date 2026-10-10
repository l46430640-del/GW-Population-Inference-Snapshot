"""Held-out event predictions from the frozen posterior importance correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from scipy.special import logsumexp

from benchmark_selected_mixture import ROOT, event_posterior, exact_selection, runtime_metadata, selected_model
from check_selected_mixture_integrals import event_integrals
from correct_selected_posterior import latent_sites, sha256
from summarize_selected_mixture import paired_scores, selected_draws


def normalized_weights(weights, count):
    weights = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if (weights.shape != (count,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.max(initial=0) <= 0):
        raise ValueError('Weights must be finite, nonnegative, and match the draws')
    weights = weights / weights.max()
    return weights / weights.sum()


def weighted_prediction(log_integrals, weights=None):
    values = np.asarray(log_integrals)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError('Finite draw-by-event log integrals are required')
    weights = normalized_weights(weights, len(values))
    positive = weights > 0
    return logsumexp(values[positive] + np.log(weights[positive, None]), axis=0)


def systematic_indices(weights, count, seed):
    if count < 1:
        raise ValueError('A positive resample count is required')
    weights = normalized_weights(weights, len(weights))
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    points = (np.arange(count) + np.random.default_rng(seed).uniform()) / count
    return np.searchsorted(cdf, points, side='right')


@torch.no_grad()
def selected_integrals(model, samples, data):
    values, ess = event_integrals(model, samples, data)
    params = {key: torch.tensor(value, dtype=data.dtype) for key, value in samples.items()}
    alpha = exact_selection(params).numpy()
    return values - np.log(alpha[:, None]), ess


def array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def cached_integrals(output, label, model, samples, data, sources, require=False):
    path = output / f'{label}.npz'
    metadata_path = path.with_suffix('.json')
    signature = {'sources': sources, 'data_hash': array_hash(data.numpy()),
                 'sample_hashes': {key: array_hash(value) for key, value in samples.items()}}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['signature'] != signature or sha256(path) != metadata['product_sha256']:
            raise ValueError(f'Cached prediction has different inputs: {label}')
        return dict(np.load(path)), metadata
    if require:
        raise FileNotFoundError(f'Prediction is not complete: {label}')
    if path.exists():
        raise FileExistsError(f'Unverified output is retained for inspection: {path}')
    started = time.perf_counter()
    values, ess = [], []
    total = len(samples['weights_logits'])
    for offset in range(0, total, 64):
        chunk = {key: value[offset:offset + 64] for key, value in samples.items()}
        value, effective = selected_integrals(model, chunk, data)
        values.append(value)
        ess.append(effective)
        if (offset + 64) % 2048 == 0:
            print(label, min(offset + 64, total), 'draws evaluated', flush=True)
    product = {'log_integrals': np.concatenate(values), 'minimum_event_ess': np.concatenate(ess)}
    if not np.isfinite(product['log_integrals']).all():
        raise FloatingPointError(f'Nonfinite predictions: {label}')
    np.savez_compressed(path, **product)
    metadata = {'signature': signature, 'seconds': time.perf_counter() - started,
                'draws': total, 'events': data.shape[0], 'event_samples': data.shape[1],
                'product_sha256': sha256(path), 'runtime': runtime_metadata()}
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(label, f"{metadata['seconds']:.2f} seconds", flush=True)
    return product, metadata


def verified_sources(config, config_path):
    source, correction = ROOT / config['source'], ROOT / config['correction_source']
    paths = [config_path, ROOT / 'scripts/predict_corrected_posterior.py',
             ROOT / 'scripts/check_selected_mixture_integrals.py',
             source / 'comparison.json', source / 'test_predictions.npz']
    previous = json.loads((source / 'comparison.json').read_text())
    if (previous['configuration']['draws'] != config['baseline_draws']
            or previous['configuration']['fine_event_samples'] != config['fine_event_samples']):
        raise ValueError('Fine baseline predictions use a different resolution')
    for name, expected in previous['sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Baseline prediction input changed: {name}')
        paths.append(ROOT / name)
    for seed in config['repeat_seeds']:
        path = correction / f'repeat_{seed}.json'
        report = json.loads(path.read_text())
        if not report['eligible_for_further_validation']:
            raise ValueError(f'Correction did not pass its declared checks: {seed}')
        if sha256(path.with_suffix('.npz')) != report['draw_product_sha256']:
            raise ValueError(f'Correction product changed: {seed}')
        for name, expected in report['source_sha256'].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f'Correction input changed: {name}')
            paths.append(ROOT / name)
        paths.extend([path, path.with_suffix('.npz')])
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(set(paths))}


def compare_predictions(prediction, reference):
    result = paired_scores(prediction, reference)
    result.update({'event_log_score_rmse': float(np.sqrt(np.mean((prediction - reference)**2))),
                   'maximum_absolute_event_log_difference': float(np.max(np.abs(prediction - reference)))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/importance_prediction.json')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    sources = verified_sources(config, config_path)
    source, correction, output = (ROOT / config[key] for key in ('source', 'correction_source', 'output'))
    output.mkdir(parents=True, exist_ok=True)
    catalog = np.load(source / 'test_catalog.npz')
    coarse = torch.tensor(catalog['data'])
    rng = np.random.default_rng(config['fine_event_seed'])
    points = np.array([event_posterior(y, config['fine_event_samples'], rng) for y in catalog['observed']])
    fine = torch.tensor(np.concatenate([points, np.ones((*points.shape[:2], 1))], -1))
    model = selected_model()
    if args.pilot:
        latent = np.load(correction / f"repeat_{config['repeat_seeds'][0]}.npz")['latent'][:config['pilot_draws']]
        samples = {key: value.numpy() for key, value in latent_sites(model, torch.tensor(latent)).items()}
        for label, data in (('pilot_coarse', coarse), ('pilot_fine', fine)):
            cached_integrals(output, label, model, samples, data, sources, args.summarize_only)
        return
    chains = [dict(np.load(source / f'nuts_chain_{index}.npz')) for index in range(4)]
    nuts_samples = {key: np.concatenate([chain[key] for chain in chains]) for key in chains[0]}
    nuts, nuts_meta = cached_integrals(output, 'nuts_coarse', model, nuts_samples, coarse, sources, args.summarize_only)
    reference = weighted_prediction(nuts['log_integrals'])
    archived_fine = dict(np.load(source / 'test_predictions.npz'))
    predictions = {'nuts_coarse': reference, 'nuts_fine_subset': archived_fine['nuts']}
    baselines, cost = {}, {'nuts_coarse': nuts_meta['seconds']}
    for mode in ('vi', 'independent'):
        for start in (0, 1):
            label = f'{mode}_start_{start}'
            samples = selected_draws(dict(np.load(source / f'{label}.npz')), config['baseline_draws'])
            values, metadata = cached_integrals(output, label + '_coarse', selected_model(mode == 'independent'),
                                                samples, coarse, sources, args.summarize_only)
            predictions[label + '_coarse'] = weighted_prediction(values['log_integrals'])
            predictions[label + '_fine'] = archived_fine[label]
            baselines[label] = compare_predictions(predictions[label + '_coarse'], reference)
            cost[label] = metadata['seconds']
    results, resampling = {}, {}
    for repeat_index, seed in enumerate(config['repeat_seeds']):
        label = f'repeat_{seed}'
        product = dict(np.load(correction / f'{label}.npz'))
        with torch.no_grad():
            samples = {key: value.numpy() for key, value in latent_sites(model, torch.tensor(product['latent'])).items()}
        coarse_product, metadata = cached_integrals(output, label + '_coarse', model, samples, coarse,
                                                     sources, args.summarize_only)
        cost[label + '_coarse'] = metadata['seconds']
        values = coarse_product['log_integrals']
        weights = {'unweighted': np.ones(len(values)) / len(values),
                   'raw': product['raw_weights'], 'psis': product['psis_weights']}
        indices = {name: systematic_indices(w, config['fine_posterior_draws'],
                                           config['resampling_seed'] + 10 * repeat_index + index)
                   for index, (name, w) in enumerate(weights.items())}
        unique = np.unique(np.concatenate(list(indices.values())))
        fine_samples = {key: value[unique] for key, value in samples.items()}
        fine_product, metadata = cached_integrals(output, label + '_fine', model, fine_samples, fine,
                                                   sources, args.summarize_only)
        cost[label + '_fine'] = metadata['seconds']
        resampling[label + '_fine_unique'] = unique
        for name, w in weights.items():
            prefix = label + '_' + name
            selected = indices[name]
            resampling[prefix] = selected
            predictions[prefix + '_coarse'] = weighted_prediction(values, w)
            predictions[prefix + '_coarse_subset'] = weighted_prediction(values[selected])
            predictions[prefix + '_fine_subset'] = weighted_prediction(fine_product['log_integrals'][np.searchsorted(unique, selected)])
        results[label] = {}
        for name in weights:
            prefix = label + '_' + name
            row = {'versus_nuts_coarse': compare_predictions(predictions[prefix + '_coarse'], reference),
                   'coarse_subset_minus_all_draws': compare_predictions(predictions[prefix + '_coarse_subset'], predictions[prefix + '_coarse']),
                   'fine_minus_coarse_same_subset': compare_predictions(predictions[prefix + '_fine_subset'], predictions[prefix + '_coarse_subset']),
                   'versus_nuts_fine_subset': compare_predictions(predictions[prefix + '_fine_subset'], archived_fine['nuts'])}
            for resolution in ('coarse', 'coarse_subset', 'fine_subset'):
                row['versus_unweighted_' + resolution] = compare_predictions(predictions[prefix + '_' + resolution],
                                                                               predictions[label + '_unweighted_' + resolution])
            for start in (0, 1):
                row[f'versus_independent_{start}_coarse'] = compare_predictions(predictions[prefix + '_coarse'],
                                                                                 predictions[f'independent_start_{start}_coarse'])
                row[f'versus_independent_{start}_fine'] = compare_predictions(predictions[prefix + '_fine_subset'],
                                                                               archived_fine[f'independent_start_{start}'])
            results[label][name] = row
    nuts_indices = np.linspace(0, len(nuts_samples['weights_logits']) - 1, config['baseline_draws'], dtype=int)
    nuts_subset = weighted_prediction(nuts['log_integrals'][nuts_indices])
    np.savez_compressed(output / 'predictions.npz', **predictions)
    np.savez_compressed(output / 'resampling_indices.npz', **resampling)
    result = {'configuration': config, 'source_sha256': sources, 'events': len(coarse),
              'results': results, 'original_flow_comparisons': baselines,
              'reference_precision': {'coarse_subset_minus_all_draws': compare_predictions(nuts_subset, reference),
                                      'fine_minus_coarse_same_subset': compare_predictions(archived_fine['nuts'], nuts_subset)},
              'repeat_differences': {name: compare_predictions(predictions[f"repeat_{config['repeat_seeds'][1]}_{name}_coarse"],
                                                               predictions[f"repeat_{config['repeat_seeds'][0]}_{name}_coarse"])
                                     for name in ('unweighted', 'raw', 'psis')},
              'seconds_by_calculation': cost, 'prediction_seconds': sum(cost.values()),
              'product_sha256': {name: sha256(output / name) for name in ('predictions.npz', 'resampling_indices.npz')},
              'scope': 'Scores omit common event-evidence constants. Paired standard errors describe test-event variation conditional on one training catalog. Correction weights were fixed before evaluating the test events. Fine predictions use fixed weighted resamples solely to assess numerical error.'}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({label: {name: row['versus_unweighted_coarse'] for name, row in rows.items()}
                      for label, rows in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
