"""Verify and time exact sparse evaluation on the retained LISA training states."""
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
from correct_selected_posterior import sha256
from pilot_external_gp_control_copula import enforce_budget
from run_external_gp_pilot import ROOT
from run_external_gp_selected import InitializedGPPotential
from run_lisa_production import verify_product, write_json
from sample_lisa_transport import compile_potential, value_gradient
from sparse_lisa_gp import SparseEventGPPotential

CONFIG = ROOT / 'configs/external_lisa_sparse_check.json'


def main():
    config = json.loads(CONFIG.read_text())
    source, output = ROOT / config['source'], ROOT / config['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Sparse likelihood check is retained')
    output.mkdir(parents=True, exist_ok=True)
    cost = json.loads((ROOT / config['previous_cost']).read_text())
    if cost['remaining_study_seconds'] < config['maximum_seconds']:
        raise ValueError('Check exceeds the remaining study budget')
    target = json.loads((source / 'target.json').read_text())
    sources = dict(target['source_sha256'])
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Original source changed: {name}')
    paths = [CONFIG, Path(__file__), ROOT / 'scripts/sparse_lisa_gp.py',
        source / 'target.json', ROOT / config['previous_cost']]
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    deadline = PROCESS_STARTED+config['maximum_seconds']
    check = lambda: enforce_budget(deadline, 2)
    rows = {}
    for bins in config['bins_per_axis']:
        check()
        path = source / f'inputs_{bins}.npz'
        verify_product(path)
        inputs = dict(np.load(path))
        path = source / f'bins_{bins}/chain_847201.npz'
        verify_product(path)
        data = np.load(path)
        indices = np.unique(np.r_[np.linspace(0, len(data['coordinate'])-1, config['checked_draws'], dtype=int),
            data['coordinate'][:, :4].argmin(0), data['coordinate'][:, :4].argmax(0)])
        points = np.vstack([data['initial_coordinate'], data['coordinate'][indices]])
        for suffix in ('npz', 'json'):
            p = path.with_suffix('.'+suffix)
            sources[p.relative_to(ROOT).as_posix()] = sha256(p)
        kwargs = (bins, inputs['log_event_weights'], inputs['selection_coefficients'], target['effective_protocol']['binned_gp'])
        dense, sparse = InitializedGPPotential(*kwargs), SparseEventGPPotential(*kwargs)
        initial = torch.tensor(points[0])
        dense_jit, dense_trace = compile_potential(dense, initial, config['maximum_gradient_error'])
        sparse_jit, sparse_trace = compile_potential(sparse, initial, config['maximum_gradient_error'])
        values, gradients = [], []
        for point in points:
            check()
            point = torch.tensor(point)
            reference, gradient = value_gradient(dense, point)
            row_value, row_gradient = [], []
            for evaluator in (sparse, dense_jit, sparse_jit):
                value, derivative = value_gradient(evaluator, point)
                row_value.append(float(abs(reference-value)))
                row_gradient.append(float(abs(gradient-derivative).max()))
            values.append(row_value)
            gradients.append(row_gradient)
        timings = {'dense': [], 'sparse': []}
        for index in np.linspace(0, len(points)-1, config['timing_points'], dtype=int):
            point = torch.tensor(points[index])
            for name, evaluator in (('dense', dense_jit), ('sparse', sparse_jit)):
                check()
                value_gradient(evaluator, point)
                started = time.perf_counter()
                for _ in range(config['timing_repetitions']):
                    value_gradient(evaluator, point)
                timings[name].append((time.perf_counter()-started)/config['timing_repetitions'])
        arrays = {'coordinate': points, 'original_retained_indices': indices,
            'absolute_value_error': np.asarray(values), 'maximum_gradient_error': np.asarray(gradients),
            'occupied_event_bin_counts': sparse.occupied_counts,
            'dense_seconds_per_gradient': np.asarray(timings['dense']), 'sparse_seconds_per_gradient': np.asarray(timings['sparse'])}
        np.savez_compressed(output / f'bins_{bins}.npz', **arrays)
        rows[str(bins)] = {'maximum_value_error': float(np.max(values)), 'maximum_gradient_error': float(np.max(gradients)),
            'checked_states': len(points), 'dense_trace_checks': dense_trace, 'sparse_trace_checks': sparse_trace,
            'original_event_matrix_entries': int(dense.log_event_weights.numel()),
            'sparse_padded_matrix_entries': int(sparse.sparse_log_event_weights.numel()),
            'full_population_bins_retained': sparse.dimension-4,
            'median_gradient_speedup': float(np.median(timings['dense'])/np.median(timings['sparse']))}
        print('Sparse LISA check', bins, rows[str(bins)], flush=True)
    check()
    seconds = time.perf_counter()-PROCESS_STARTED
    report = {'configuration': config, 'source_sha256': sources, 'models': rows,
        'equivalent': all(row['maximum_value_error'] <= config['maximum_value_error']
            and row['maximum_gradient_error'] <= config['maximum_gradient_error'] for row in rows.values()),
        'process_elapsed_seconds': seconds, 'total_study_charge_seconds': cost['total_study_charge_seconds']+seconds,
        'remaining_study_seconds': cost['remaining_study_seconds']-seconds,
        'product_sha256': {p.name: sha256(p) for p in output.glob('*.npz')},
        'scientific_scores_examined': False, 'new_posterior_created': False, 'runtime': runtime_metadata()}
    write_json(output / 'summary.json', report)
    print(json.dumps({k: report[k] for k in ('equivalent', 'process_elapsed_seconds', 'remaining_study_seconds')}), flush=True)


if __name__ == '__main__':
    status, error = 'complete', None
    try:
        main()
    except BaseException as exc:
        status, error = 'stopped', str(exc)
        raise
    finally:
        config = json.loads(CONFIG.read_text())
        output = ROOT / config['output']
        output.mkdir(parents=True, exist_ok=True)
        prior = json.loads((ROOT / config['previous_cost']).read_text())
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(output / f'job_cost_{len(list(output.glob("job_cost_*.json"))):03d}.json',
            {'status': status, 'error': error, 'process_elapsed_seconds': seconds,
             'total_study_charge_seconds': prior['total_study_charge_seconds']+seconds,
             'remaining_study_seconds': prior['remaining_study_seconds']-seconds,
             'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})
