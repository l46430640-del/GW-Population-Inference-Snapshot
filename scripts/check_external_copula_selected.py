"""Refine the existing selected-catalog target before external comparison."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import torch

from benchmark_selected_mixture import event_posterior, selected_model
from check_selected_mixture_integrals import event_integrals
from compare_lisa_posterior import cached
from correct_selected_posterior import sha256
from external_selected_evaluation import copula_exact_features, event_log_integrals
from refine_lisa_posterior import ratio_weights
from run_lisa_production import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/external_selected_evaluation.json'


def settings():
    config = json.loads(CONFIG.read_text())
    return config, json.loads((ROOT / config['protocol']).read_text())


def selected_indices(chains, draws, count):
    return (np.arange(chains)[:, None]*draws + np.linspace(0, draws-1, count, dtype=int)).ravel()


def read_copula(config):
    directory = ROOT / config['copula_source']
    target = json.loads((directory / 'target.json').read_text())
    sources = dict(target['source_sha256'])
    for name, value in sources.items():
        if sha256(ROOT / name) != value:
            raise ValueError(f'Original copula source changed: {name}')
    paths = [directory / 'target.json', directory / 'nuts_summary.json',
             directory / 'train_catalog.npz', directory / 'test_catalog.npz']
    chains = []
    for i in range(4):
        path = directory / f'nuts_chain_{i}.npz'
        report = json.loads(path.with_suffix('.json').read_text())
        if report['divergences']:
            raise ValueError('Original copula chain has divergences')
        chains.append(dict(np.load(path)))
        paths.extend([path, path.with_suffix('.json')])
    samples = {name: np.concatenate([c[name] for c in chains]) for name in chains[0]}
    summary = json.loads((directory / 'nuts_summary.json').read_text())
    if max(summary['rhat']) > 1.01 or min(summary['ess_bulk']+summary['ess_tail']) < 400:
        raise ValueError('Original copula chains did not converge')
    paths.extend([CONFIG, ROOT / config['protocol'], Path(__file__),
                  ROOT / 'scripts/external_selected_evaluation.py',
                  ROOT / 'scripts/compare_lisa_posterior.py',
                  ROOT / 'scripts/check_selected_mixture_integrals.py',
                  ROOT / 'scripts/refine_lisa_posterior.py',
                  ROOT / 'scripts/correct_selected_posterior.py',
                  ROOT / 'scripts/run_lisa_production.py'])
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    return samples, (4, len(chains[0]['weights_logits'])), sources


def params_at(samples, index):
    return {name: value[index] for name, value in samples.items()}


def make_events(observed, count, seed):
    rng = np.random.default_rng(seed)
    points = np.asarray([event_posterior(y, count, rng) for y in observed])
    return {'data': np.concatenate([points, np.ones((*points.shape[:2], 1))], -1)}


def evaluate_events(samples, data, indices):
    values = []
    for i, index in enumerate(indices):
        values.append(event_log_integrals(params_at(samples, index), data))
        if (i+1) % 200 == 0:
            print('Event integrals', i+1, '/', len(indices), flush=True)
    return {'indices': np.asarray(indices), 'log_integrals': np.asarray(values)}


def target_decision(protocol, correction, verification, kernel_error):
    sampling, evaluation = protocol['sampling'], protocol['evaluation']
    result = {
        'kernel_equivalence_passed': bool(kernel_error < settings()[0]['maximum_kernel_total_log_likelihood_difference']),
        'ratio_weights_passed': bool(correction['pareto_k'] < sampling['maximum_pareto_k_for_reuse']
                                    and correction['weight_ess_fraction'] >= sampling['minimum_reuse_weight_ess_fraction']),
        'event_integration_passed': bool(max(verification.values()) <= evaluation['maximum_log_likelihood_difference_sd'])}
    result['target_reusable'] = all(result.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['primary', 'verify', 'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config, protocol = settings()
    evaluation = protocol['evaluation']
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    samples, shape, sources = read_copula(config)
    total = np.prod(shape)
    catalog = dict(np.load(ROOT / config['copula_source'] / 'train_catalog.npz'))
    def obtain(label, builder):
        return cached(output, label, sources, builder)[0]
    primary_events = obtain('train_events', lambda: make_events(catalog['observed'],
        evaluation['training_event_samples'], evaluation['evaluation_seed']))
    coarse = obtain('train_original', lambda: evaluate_events(samples, catalog['data'], np.arange(total)))
    primary = obtain('train_primary', lambda: evaluate_events(samples, primary_events['data'], np.arange(total)))
    log_ratio = (primary['log_integrals']-coarse['log_integrals']).sum(-1).reshape(shape)
    reff = min(1., float(np.asarray(az.ess(az.from_dict(posterior={'ratio': log_ratio}), method='mean')['ratio']))/total)
    weights, correction = ratio_weights(log_ratio, reff)
    obtain('copula_weights', lambda: {'weights': weights, 'log_ratio': log_ratio})
    if args.stage == 'primary':
        return
    verification_events = obtain('train_events_verification', lambda: make_events(catalog['observed'],
        evaluation['verification_training_event_samples'], evaluation['verification_seed']))
    uniform = selected_indices(*shape, config['verification_draws_per_chain'])
    alpha = np.asarray([copula_exact_features(params_at(samples, i))[0][6] for i in range(total)])
    extremes = config['verification_extreme_draws']
    groups = {f'chain_{c}': uniform.reshape(shape[0], -1)[c] for c in range(shape[0])}
    groups['largest_weights'] = np.argsort(weights.ravel())[-extremes:]
    groups['efficiency_extremes'] = np.r_[np.argsort(alpha)[:extremes], np.argsort(alpha)[-extremes:]]
    indices = np.unique(np.concatenate(list(groups.values())))
    fine = obtain('train_verification', lambda: evaluate_events(samples, verification_events['data'], indices))
    difference = (fine['log_integrals']-primary['log_integrals'][indices]).sum(-1)
    verification = {name: float(difference[np.searchsorted(indices, group)].std(ddof=1)) for name, group in groups.items()}
    kernel_indices = selected_indices(*shape, config['kernel_check_draws_per_chain'])
    def check_kernel():
        chosen = {name: value[kernel_indices] for name, value in samples.items()}
        exact, _ = event_integrals(selected_model(), chosen, torch.tensor(catalog['data']))
        return {'indices': kernel_indices, 'core_log_integrals': exact,
                'numpy_log_integrals': coarse['log_integrals'][kernel_indices]}
    checked = obtain('density_kernel_check', check_kernel)
    kernel_error = float(abs((checked['core_log_integrals']-checked['numpy_log_integrals']).sum(-1)).max())
    summary = {'configuration': config, 'source_sha256': sources, 'shape': list(shape),
               'correction': correction, 'refined_minus_original_log_likelihood_sd': float(log_ratio.std(ddof=1)),
               'verification_log_likelihood_difference_sd': verification,
               'maximum_kernel_total_log_likelihood_difference': kernel_error,
               'decision': target_decision(protocol, correction, verification, kernel_error),
               'product_sha256': {p.name: sha256(p) for p in output.glob('*.npz')},
               'scientific_scores_examined': False}
    write_json(output / 'copula_target_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('correction', 'verification_log_likelihood_difference_sd', 'decision')}), flush=True)


if __name__ == '__main__':
    main()
