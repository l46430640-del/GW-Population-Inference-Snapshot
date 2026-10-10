"""Resolve histogram error in known-truth cell masses for the first comparison."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import json
from pathlib import Path

import numpy as np
import torch

from binned_gp_benchmark import grid_geometry
from check_external_copula_selected import ROOT, params_at, settings
from compare_external_selected import (accepted_inputs, gp_product, population_influences,
    recovery_gradients)
from compare_lisa_posterior import cached, moment_check
from correct_selected_posterior import sha256
from external_selected_evaluation import copula_exact_features, copula_log_density
from run_lisa_production import write_json
from summarize_selected_mixture import truth_samples

CONFIG = ROOT / 'configs/external_selected_recovery_refinement.json'


def quadrature_truth(bins, order, block_size):
    node, weight = np.polynomial.legendre.leggauss(order)
    offsets = np.stack(np.meshgrid(*[(node+1)/(2*bins)]*3, indexing='ij'), -1).reshape(-1, 3)
    weights = np.prod(np.stack(np.meshgrid(*[weight/(2*bins)]*3, indexing='ij'), -1), axis=-1).ravel()
    lower, _ = grid_geometry(bins)
    intrinsic, detected = [], []
    parameters = params_at(truth_samples(), 0)
    for start in range(0, len(lower), block_size):
        points = lower[start:start+block_size, None, :] + offsets
        density = np.exp(copula_log_density(parameters, points))
        intrinsic.extend(density @ weights)
        detected.extend((density*(.05+.95*points[..., 0]**2)) @ weights)
    intrinsic, detected = np.asarray(intrinsic), np.asarray(detected)
    return {'intrinsic': intrinsic/intrinsic.sum(), 'detected': detected/detected.sum(),
            'normalization': np.asarray(intrinsic.sum()), 'alpha': np.asarray(detected.sum())}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    refinement = json.loads(CONFIG.read_text())
    config, protocol = settings()
    original_output = ROOT / config['output']
    original_summary = json.loads((original_output / 'summary.json').read_text())
    samples, shape, _, gp, sources = accepted_inputs(config)
    for name, expected in original_summary['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Original comparison source changed: {name}')
    for name, expected in original_summary['product_sha256'].items():
        path = original_output / name
        if sha256(path) != expected:
            raise ValueError(f'Original comparison product changed: {name}')
        sources[path.relative_to(ROOT).as_posix()] = expected
    paths = [CONFIG, Path(__file__), original_output / 'summary.json']
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    output = ROOT / refinement['output']
    output.mkdir(parents=True, exist_ok=True)
    def obtain(label, builder):
        product, metadata = cached(output, label, sources, builder)
        if metadata['runtime']['peak_working_set_bytes'] > protocol['resources']['maximum_resident_gib']*1024**3:
            raise RuntimeError('Refined comparison exceeded the memory limit')
        return product
    evaluation = protocol['evaluation']
    partitions = [evaluation['joint_cells_per_axis'], evaluation['verification_joint_cells_per_axis']]
    fine = max(partitions)
    truth = obtain('truth_quadrature', lambda: quadrature_truth(fine, refinement['truth_quadrature_order'], refinement['cell_block_size']))
    truth_fine = obtain('truth_quadrature_verification', lambda: quadrature_truth(fine, refinement['verification_truth_quadrature_order'], refinement['cell_block_size']))
    population = dict(np.load(original_output / 'copula_recovery_verification.npz'))
    population_coarse = dict(np.load(original_output / 'copula_recovery.npz'))
    distances, gradients = recovery_gradients(population, truth_fine, fine, partitions)
    # Use exactly the independent cloud that formed the refined posterior mean.
    influence_config = dict(config, population_transport_power=config['verification_population_transport_power'],
                            population_seed=config['verification_population_seed'])
    influences = obtain('copula_influences', lambda: population_influences(samples, influence_config, fine, partitions, gradients))
    weights = np.load(original_output / 'copula_weights.npz')['weights'].ravel()
    test = dict(np.load(ROOT / config['copula_source'] / 'test_catalog.npz'))
    products = {key: obtain('gp_'+key, lambda key=key: gp_product(config, protocol, gp, key, truth_fine, partitions, test)) for key in gp}
    result = json.loads(json.dumps(original_summary))
    result['source_sha256'] = sources
    result['recovery_refinement'] = refinement
    result['original_comparison_decision'] = original_summary['decision']
    numerical = {'truth_normalization_errors': [float(truth['normalization']-1), float(truth_fine['normalization']-1)],
                 'truth_alpha_change': float(truth_fine['alpha']-truth['alpha']),
                 'truth_alpha_error': float(truth_fine['alpha']-copula_exact_features(params_at(truth_samples(), 0))[0][6]),
                 'truth_quadrature_js_changes': {}, 'posterior_transport_js_changes': {}, 'partition_js_changes': {}}
    coarse_distances, _ = recovery_gradients(population_coarse, truth_fine, fine, partitions)
    numerical['posterior_transport_js_changes'] = {k: distances[k]-v for k, v in coarse_distances.items()}
    for key, product in [('copula', population), *products.items()]:
        row = result['copula'] if key == 'copula' else result['gp'][key]
        ds, _ = recovery_gradients(product, truth_fine, fine, partitions)
        ds_coarse, _ = recovery_gradients(product, truth, fine, partitions)
        numerical['truth_quadrature_js_changes'][key] = {k: ds[k]-v for k, v in ds_coarse.items()}
        numerical['partition_js_changes'][key] = {kind: ds[f'{kind}_{fine}']-ds[f'{kind}_{min(partitions)}'] for kind in ('intrinsic', 'detected')}
        this_shape = shape if key == 'copula' else product['features'].shape[:2]
        this_weights = weights if key == 'copula' else np.full(np.prod(this_shape), 1/np.prod(this_shape))
        row['joint'] = {label: {'distance': distance, **{k: v for k, v in moment_check(
            influences[label] if key == 'copula' else product['influence_'+label], this_weights,
            this_shape, config['batch_lengths']).items() if k != 'mean'}} for label, distance in ds.items()}
    changes = list(numerical['posterior_transport_js_changes'].values())
    for name in ('truth_quadrature_js_changes', 'partition_js_changes'):
        for group in numerical[name].values():
            changes.extend(group.values())
    result['numerical_recovery_refined'] = numerical
    result['decision']['recovery_precision_passed'] = bool(max(map(abs, changes)) < evaluation['maximum_joint_distance_numerical_change']
        and max(map(abs, numerical['truth_normalization_errors']+[numerical['truth_alpha_error']])) < refinement['maximum_truth_normalization_error'])
    joint_rhats = [v['influence_rhat'] for row in [result['copula'], *result['gp'].values()] for v in row['joint'].values()]
    for row in [result['copula'], *result['gp'].values()]:
        joint_rhats.extend([row['prediction_precision']['influence_rhat'], row['functionals']['maximum_influence_rhat']])
    result['maximum_derived_influence_rhat'] = max(joint_rhats)
    result['decision']['derived_mixing_passed'] = bool(max(joint_rhats) <= protocol['sampling']['maximum_functional_rhat'])
    result['decision']['eligible_for_scientific_comparison'] = all(v for k, v in result['decision'].items() if k != 'eligible_for_scientific_comparison')
    result['product_sha256'] = {p.name: sha256(p) for p in output.glob('*.npz')}
    result['refinement_costs'] = {p.stem: {'seconds': json.loads(p.read_text())['seconds'],
        'peak_bytes': json.loads(p.read_text())['runtime']['peak_working_set_bytes']} for p in output.glob('*.json')}
    result['recovery_scope'] = 'Joint recovery now uses deterministic truth cell probabilities and the declared finer posterior transport cloud. Prediction, intervals, original numerical checks and all model settings are retained from the original comparison.'
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': result['decision'], 'numerical': numerical,
                     'copula_joint': result['copula']['joint'], 'gp_joint': {k: v['joint'] for k, v in result['gp'].items()}}), flush=True)


if __name__ == '__main__':
    main()
