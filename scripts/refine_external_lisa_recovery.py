"""Verify the finer LISA recovery transport without changing any fitted model."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import time
PROCESS_STARTED = time.perf_counter()

import json
from pathlib import Path
import numpy as np
import torch

from benchmark_selected_mixture import runtime_metadata
from compare_external_lisa import accepted_inputs, influence_product, population_product
from compare_external_selected import posterior_functionals, recovery_gradients
from compare_lisa_posterior import cached, moment_check
from correct_selected_posterior import sha256
from pilot_external_gp_control_copula import enforce_budget
from run_lisa_production import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/external_lisa_recovery_refinement.json'


def refine(amendment, output, check):
    parent = ROOT / amendment['parent']
    report = json.loads((parent / 'summary.json').read_text())
    config = report['configuration']
    protocol = json.loads((ROOT / config['protocol']).read_text())
    if report['decision']['recovery_precision_passed'] or not all(v for k, v in report['decision'].items()
            if k not in ('recovery_precision_passed', 'eligible_for_scientific_comparison')):
        raise ValueError('This refinement requires the isolated recovery precision failure')
    for name, digest in report['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Parent source changed: {name}')
    for name, digest in report['product_sha256'].items():
        if sha256(parent / name) != digest:
            raise ValueError(f'Parent product changed: {name}')
    _, _, posterior, weights, _, _, samples, _, sources = accepted_inputs(config)
    sources.update({(parent / name).relative_to(ROOT).as_posix(): digest for name, digest in report['product_sha256'].items()})
    paths = [CONFIG, Path(__file__), parent / 'summary.json', parent / 'job_cost_000.json',
        ROOT / amendment['original_attempt'] / 'job_cost_000.json']
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    fine = config['verification_cells_per_axis']
    partitions = [config['primary_cells_per_axis'], fine]
    truth = dict(np.load(parent / 'truth.npz'))
    truth_coarse = dict(np.load(parent / 'truth_coarse.npz'))
    previous = dict(np.load(parent / 'copula_recovery.npz'))
    def obtain(name, builder):
        check()
        product, _ = cached(output, name, sources, builder)
        check()
        return product
    population = obtain('copula_recovery', lambda: population_product(samples, weights.ravel(),
        amendment['population_transport_power'], amendment['population_seed'], fine, check))
    distances, gradients = recovery_gradients(population, truth, fine, partitions)
    influences = obtain('copula_influences', lambda: influence_product(samples,
        amendment['population_transport_power'], amendment['population_seed'], gradients, check))
    features = population['features'].copy()
    alpha = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
    features[:, 6] = alpha
    row = report['copula']
    row['joint'] = {label: {'distance': value, **{k: v for k, v in moment_check(influences[label],
        weights.ravel(), weights.shape, config['batch_lengths']).items() if k != 'mean'}} for label, value in distances.items()}
    row['functionals'] = posterior_functionals(features, weights.ravel(), weights.shape, truth['features'], config)
    old_distances, _ = recovery_gradients(previous, truth, fine, partitions)
    truth_check, _ = recovery_gradients(population, truth_coarse, fine, partitions)
    old_numerical = report['numerical_recovery']
    numerical = dict(old_numerical,
        parent_failed_transport_and_truth_changes=old_numerical['copula_transport_and_truth_changes'],
        copula_transport_and_truth_changes={name: distances[name]-value for name, value in old_distances.items()},
        truth_integration_changes={name: distances[name]-value for name, value in truth_check.items()},
        maximum_transport_relative_alpha_error=float(np.max(abs(population['features'][:, 6]/alpha-1))),
        maximum_covariance_refinement_change=float(np.max(abs(population['features'][:, 9:]-previous['features'][:, 9:]))))
    numerical['partition_changes']['copula'] = {kind: distances[f'{kind}_{fine}']-distances[f'{kind}_{min(partitions)}'] for kind in ('intrinsic', 'detected')}
    changes = list(numerical['copula_transport_and_truth_changes'].values())+list(numerical['truth_integration_changes'].values())
    changes.extend(value for group in numerical['gp_cell_integration_changes'].values() for value in group.values())
    changes.extend(value for group in numerical['partition_changes'].values() for value in group.values())
    rhats = []
    for result in [row, *report['gp'].values()]:
        rhats.extend(value['influence_rhat'] for value in result['joint'].values())
        rhats.extend([result['prediction_precision']['influence_rhat'], result['functionals']['maximum_influence_rhat']])
    decision = dict(report['decision'])
    decision['recovery_precision_passed'] = max(map(abs, changes)) < protocol['evaluation']['maximum_joint_distance_numerical_change']
    decision['derived_mixing_passed'] = max(rhats) <= protocol['sampling']['maximum_functional_rhat']
    check()
    decision['eligible_for_scientific_comparison'] = all(v for k, v in decision.items() if k != 'eligible_for_scientific_comparison')
    report.update(source_sha256=sources, decision=decision, numerical_recovery=numerical,
        maximum_derived_influence_rhat=max(rhats), recovery_refinement=amendment,
        product_sha256={p.name: sha256(p) for p in output.glob('*.npz')},
        reused_parent_products={name: digest for name, digest in json.loads((parent / 'summary.json').read_text())['product_sha256'].items()
            if name not in ('copula_recovery.npz', 'copula_influences.npz')})
    write_json(output / 'summary.json', report)
    print(json.dumps({'decision': decision, 'recovery_checks': numerical,
        'copula_joint': row['joint']}), flush=True)


def main():
    amendment = json.loads(CONFIG.read_text())
    parent = ROOT / amendment['parent']
    prior = json.loads((parent / 'job_cost_000.json').read_text())
    initial = json.loads((ROOT / amendment['original_attempt'] / 'job_cost_000.json').read_text())
    already = prior['all_evaluation_attempts_seconds']+initial['all_evaluation_attempts_seconds']
    if already+amendment['maximum_seconds'] > amendment['maximum_all_evaluation_seconds']:
        raise ValueError('Refinement exceeds the original evaluation allocation')
    if prior['total_study_charge_seconds']+amendment['maximum_seconds'] > amendment['maximum_total_study_seconds']:
        raise ValueError('Refinement exceeds the declared cumulative ceiling')
    output = ROOT / amendment['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed refinement is retained')
    old = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('job_cost_*.json'))
    deadline = PROCESS_STARTED+amendment['maximum_seconds']-old
    check = lambda: enforce_budget(deadline, 2)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    status, error = 'complete', None
    try:
        refine(amendment, output, check)
        check()
    except BaseException as exc:
        status, error = 'stopped', str(exc)
        raise
    finally:
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(output / f'job_cost_{len(list(output.glob("job_cost_*.json"))):03d}.json',
            {'status': status, 'error': error, 'process_elapsed_seconds': seconds,
             'all_refinement_attempts_seconds': old+seconds,
             'all_lisa_evaluation_attempts_seconds': already+old+seconds,
             'total_study_charge_seconds': prior['total_study_charge_seconds']+old+seconds,
             'remaining_study_seconds': amendment['maximum_total_study_seconds']-prior['total_study_charge_seconds']-old-seconds,
             'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})


if __name__ == '__main__':
    main()
