"""Common finer evaluation partition after the retained coarse-grid failure."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import json
from pathlib import Path

import numpy as np
import torch

from check_external_copula_selected import ROOT, params_at, settings
from compare_external_selected import (accepted_inputs, gp_product, population_product,
    population_influences, recovery_gradients)
from compare_lisa_posterior import cached, moment_check
from correct_selected_posterior import sha256
from external_selected_evaluation import coarsen, copula_exact_features
from refine_external_selected_recovery import quadrature_truth
from run_lisa_production import write_json
from summarize_selected_mixture import truth_samples

CONFIG = ROOT / 'configs/external_selected_partition_refinement.json'


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    refinement = json.loads(CONFIG.read_text())
    config, protocol = settings()
    old_path = ROOT / refinement['previous_summary']
    old = json.loads(old_path.read_text())
    samples, shape, _, gp, sources = accepted_inputs(config)
    for name, expected in old['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Previous source changed: {name}')
        sources[name] = expected
    for name, expected in old['product_sha256'].items():
        path = old_path.parent / name
        if sha256(path) != expected:
            raise ValueError(f'Previous product changed: {name}')
        sources[path.relative_to(ROOT).as_posix()] = expected
    for path in (CONFIG, Path(__file__), old_path):
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    output = ROOT / refinement['output']
    output.mkdir(parents=True, exist_ok=True)
    def obtain(label, builder):
        product, metadata = cached(output, label, sources, builder)
        if metadata['runtime']['peak_working_set_bytes'] > protocol['resources']['maximum_resident_gib']*1024**3:
            raise RuntimeError('Partition refinement exceeded the memory limit')
        return product
    partitions = [refinement['primary_cells_per_axis'], refinement['verification_cells_per_axis']]
    fine = max(partitions)
    truth = obtain('truth_quadrature', lambda: quadrature_truth(fine, refinement['truth_quadrature_order'], refinement['cell_block_size']))
    truth_fine = obtain('truth_quadrature_verification', lambda: quadrature_truth(fine, refinement['verification_truth_quadrature_order'], refinement['cell_block_size']))
    weights = np.load(ROOT / config['output'] / 'copula_weights.npz')['weights'].ravel()
    coarse = obtain('copula_transport', lambda: population_product(samples, weights, config, fine))
    population = obtain('copula_transport_verification', lambda: population_product(samples, weights, config, fine, True))
    previous_population = np.load(ROOT / config['output'] / 'copula_recovery_verification.npz')
    for kind in ('intrinsic', 'detected'):
        np.testing.assert_allclose(coarsen(population[kind], fine, 48), previous_population[kind], rtol=1e-12, atol=1e-14)
    distances, gradients = recovery_gradients(population, truth_fine, fine, partitions)
    influence_config = dict(config, population_transport_power=config['verification_population_transport_power'],
                            population_seed=config['verification_population_seed'])
    influences = obtain('copula_influences', lambda: population_influences(samples, influence_config, fine, partitions, gradients))
    test = dict(np.load(ROOT / config['copula_source'] / 'test_catalog.npz'))
    def gp_recovery(key):
        full = gp_product(config, protocol, gp, key, truth_fine, partitions, test)
        return {name: value for name, value in full.items() if name in ('intrinsic', 'detected', 'features') or name.startswith('influence_')}
    products = {key: obtain('gp_'+key, lambda key=key: gp_recovery(key)) for key in gp}
    result = json.loads(json.dumps(old))
    result['source_sha256'] = sources
    result['partition_refinement'] = refinement
    result['previous_partition_decision'] = old['decision']
    result['previous_joint_partitions'] = {'copula': old['copula']['joint'], **{key: row['joint'] for key, row in old['gp'].items()}}
    numerical = {'truth_normalization_errors': [float(truth['normalization']-1), float(truth_fine['normalization']-1)],
        'truth_alpha_error': float(truth_fine['alpha']-copula_exact_features(params_at(truth_samples(), 0))[0][6]),
        'truth_quadrature_js_changes': {}, 'posterior_transport_js_changes': {}, 'partition_js_changes': {}}
    coarse_distances, _ = recovery_gradients(coarse, truth_fine, fine, partitions)
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
    result['numerical_partition_refined'] = numerical
    result['decision']['recovery_precision_passed'] = bool(max(map(abs, changes)) < protocol['evaluation']['maximum_joint_distance_numerical_change']
        and max(map(abs, numerical['truth_normalization_errors']+[numerical['truth_alpha_error']])) < refinement['maximum_truth_normalization_error'])
    rhats = [v['influence_rhat'] for row in [result['copula'], *result['gp'].values()] for v in row['joint'].values()]
    for row in [result['copula'], *result['gp'].values()]:
        rhats.extend([row['prediction_precision']['influence_rhat'], row['functionals']['maximum_influence_rhat']])
    result['maximum_derived_influence_rhat'] = max(rhats)
    result['decision']['derived_mixing_passed'] = bool(max(rhats) <= protocol['sampling']['maximum_functional_rhat'])
    result['decision']['eligible_for_scientific_comparison'] = all(v for k, v in result['decision'].items() if k != 'eligible_for_scientific_comparison')
    result['product_sha256'] = {p.name: sha256(p) for p in output.glob('*.npz')}
    result['partition_refinement_costs'] = {p.stem: {'seconds': json.loads(p.read_text())['seconds'],
        'peak_bytes': json.loads(p.read_text())['runtime']['peak_working_set_bytes']} for p in output.glob('*.json')}
    result['recovery_scope'] = 'Joint recovery uses the common 48-cell primary and 96-cell verification partition. The declared 24-cell results and their failed precision test remain above and in the two prior summaries. All prediction and interval results are unchanged.'
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': result['decision'], 'numerical': numerical,
        'copula_joint': result['copula']['joint'], 'gp_joint': {k: v['joint'] for k, v in result['gp'].items()}}), flush=True)


if __name__ == '__main__':
    main()
