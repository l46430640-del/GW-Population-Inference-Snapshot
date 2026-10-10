"""Complete the LISA high-resolution fit with a verified equivalent likelihood."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import time
PROCESS_STARTED = time.perf_counter()

import json
from pathlib import Path

import numpy as np
from pyro.ops import stats
import torch

from benchmark_selected_mixture import runtime_metadata
from correct_selected_posterior import sha256
from lisa_gp_inputs import integral_report
from pilot_external_gp_control_copula import enforce_budget
from run_external_control_production import freeze_target, install_adapter
from run_external_gp_pilot import ROOT
from run_external_gp_selected import checked_reports, summarize
import run_external_lisa_gp as production
from run_lisa_production import verify_product, write_json
from sparse_lisa_gp import SparseEventGPPotential

CONFIG = ROOT / 'configs/external_lisa_gp_sparse.json'


def settings(amendment):
    config = json.loads((ROOT / amendment['base_config']).read_text())
    config.update(output=amendment['output'], maximum_seconds=amendment['maximum_seconds'],
        retained_per_chain=amendment['retained_per_chain'], sparse_amendment=amendment)
    protocol = production.effective_protocol(config, json.loads((ROOT / config['protocol']).read_text()))
    return config, protocol


def sources_and_inputs(amendment):
    parent = ROOT / amendment['parent_output']
    target_path = parent / 'target.json'
    target = json.loads(target_path.read_text())
    sources = dict(target['source_sha256'])
    equivalence_path = ROOT / amendment['equivalence_summary']
    equivalence = json.loads(equivalence_path.read_text())
    if not equivalence['equivalent'] or equivalence['new_posterior_created']:
        raise ValueError('An accepted likelihood-equivalence check is required')
    sources.update(equivalence['source_sha256'])
    for name, digest in equivalence['product_sha256'].items():
        sources[(equivalence_path.parent / name).relative_to(ROOT).as_posix()] = digest
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen numerical source changed: {name}')
    bins = amendment['bins_per_axis']
    input_path = parent / f'inputs_{bins}.npz'
    verify_product(input_path)
    partial = parent / f'bins_{bins}/chain_847201.npz'
    if verify_product(partial)['status'] != 'time_limit':
        raise ValueError('This continuation must retain its resource-limited predecessor')
    paths = [CONFIG, Path(__file__), target_path, equivalence_path, ROOT / amendment['previous_cost'],
        ROOT / 'scripts/sparse_lisa_gp.py', parent / f'bins_{bins}/acceptance.json']
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    return sources, dict(np.load(input_path)), input_path


def run(amendment, output, deadline):
    config, protocol = settings(amendment)
    sources, inputs, input_path = sources_and_inputs(amendment)
    bins = amendment['bins_per_axis']
    directory = output / f'bins_{bins}'
    if (directory / 'summary.json').exists():
        raise FileExistsError('Completed sparse production is retained')
    target_hash = freeze_target(directory, {'configuration': config, 'effective_protocol': protocol,
        'source_sha256': sources, 'experiment': 'lisa_bimodal', 'bins_per_axis': bins,
        'inputs_path': input_path.relative_to(ROOT).as_posix(), 'inputs_product_sha256': sha256(input_path),
        'excluded_partial_chain': amendment['parent_output']+f'/bins_{bins}/chain_847201.npz'})
    check = lambda: enforce_budget(deadline, config['maximum_resident_gib'])
    production.InitializedGPPotential = SparseEventGPPotential
    original = install_adapter(config)
    try:
        previous = checked_reports(directory, target_hash, protocol)
        for chain, seed in enumerate(protocol['sampling']['chain_seeds']):
            if seed in {row['seed'] for row in previous}:
                continue
            check()
            report, arrays = production.run_chain(config, protocol, inputs, bins, chain, False, deadline, check)
            production.save_chain(directory / f'chain_{seed}.npz', report, arrays, target_hash)
            if report['status'] != 'complete':
                break
        summary = summarize(directory, protocol, target_hash)
        reports = checked_reports(directory, target_hash, protocol)
        names = ('event_log_likelihood_change', 'selection_log_likelihood_change', 'combined_log_likelihood_change')
        changes = {name: [] for name in names}
        for row in reports:
            with np.load(directory / f"chain_{row['seed']}.npz") as arrays:
                for name in names:
                    changes[name].extend(arrays[name])
        numerical = integral_report(changes, protocol['evaluation']['maximum_log_likelihood_difference_sd'])
        decision = dict(summary['decision'])
        decision['pooled_integrals_passed'] = numerical['passed']
        decision['posterior_accepted'] = all(value for key, value in decision.items() if key != 'posterior_accepted')
        check()
        write_json(directory / 'acceptance.json', {'decision': decision,
            'pooled_integral_verification': numerical, 'summary_sha256': sha256(directory / 'summary.json'),
            'target_product_sha256': target_hash, 'scientific_scores_examined': False})
        print('Sparse LISA GP acceptance', decision, flush=True)
    finally:
        stats._cummin = original


def main():
    amendment = json.loads(CONFIG.read_text())
    previous = json.loads((ROOT / amendment['previous_cost']).read_text())
    if amendment['maximum_seconds'] > previous['remaining_study_seconds']:
        raise ValueError('Sparse sampling allocation exceeds the remaining study allowance')
    output = ROOT / amendment['output']
    output.mkdir(parents=True, exist_ok=True)
    old = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('job_cost_*.json'))
    deadline = PROCESS_STARTED+amendment['maximum_seconds']-old
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    status, error = 'complete', None
    try:
        run(amendment, output, deadline)
    except BaseException as exc:
        status, error = 'stopped', str(exc)
        raise
    finally:
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(output / f'job_cost_{len(list(output.glob("job_cost_*.json"))):03d}.json',
            {'status': status, 'error': error, 'process_elapsed_seconds': seconds,
             'all_sparse_sampling_attempts_seconds': old+seconds,
             'total_study_charge_seconds': previous['total_study_charge_seconds']+old+seconds,
             'remaining_study_seconds': previous['remaining_study_seconds']-old-seconds,
             'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})


if __name__ == '__main__':
    main()
