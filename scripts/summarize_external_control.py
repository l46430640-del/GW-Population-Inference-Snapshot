"""Identify checked control posteriors and account for every sampling attempt."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import json
from pathlib import Path

import numpy as np

from correct_selected_posterior import sha256
from run_lisa_production import verify_product, write_json
from validate_population_morphologies import ROOT


def collect(directory, label, sources):
    target_path, summary_path = directory / 'target.json', directory / 'summary.json'
    target, summary = json.loads(target_path.read_text()), json.loads(summary_path.read_text())
    if sha256(target_path) != summary['target_product_sha256']:
        raise ValueError('Target digest differs from its summary')
    for name, digest in target['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed source: {name}')
    sources.update(target['source_sha256'])
    paths = [target_path, summary_path]
    reports, full_trees, halves = [], [], []
    diagnostic_names = None
    for seed in target['effective_protocol']['sampling']['chain_seeds']:
        path = directory / f'chain_{seed}.npz'
        report = verify_product(path)
        if (sha256(path.with_suffix('.json')) != summary['chain_reports_sha256'][path.with_suffix('.json').name]
                or report['target_product_sha256'] != summary['target_product_sha256']):
            raise ValueError('Chain provenance differs from its ensemble')
        with np.load(path) as data:
            progress = data['progress']
            calls = np.diff(np.r_[0, progress[:, 2]])[report['warmup_completed']:]
            full_trees.append(int(np.sum(calls >= 2**target['effective_protocol']['sampling']['maximum_tree_depth']-1)))
            names = data['diagnostic_names'].tolist()
            if diagnostic_names is not None and names != diagnostic_names:
                raise ValueError('Chain diagnostic names differ')
            diagnostic_names = names
            values = data['diagnostic_values']
            halves.append([values[:len(values)//2].mean(0).tolist(), values[len(values)//2:].mean(0).tolist()])
        reports.append(report)
        paths.extend([path, path.with_suffix('.json')])
    if label == 'copula':
        path = directory / 'posterior.npz'
        if sha256(path) != summary['posterior_product_sha256']:
            raise ValueError('Corrected posterior differs from its summary')
        paths.extend([path, directory / 'integration_reuse.json'])
    else:
        path = directory / 'prefix_verification.json'
        prefix = json.loads(path.read_text())
        if len(prefix['chains']) != 4 or not all(row['prefix_identical'] and row['compared_retained_draws'] == 800 for row in prefix['chains']):
            raise ValueError('The GP extension does not reproduce its original prefixes')
        paths.extend([path, directory / 'inputs.npz'])
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    mixing = summary['mixing']
    return {'directory': directory.relative_to(ROOT).as_posix(),
        'chains': len(reports), 'retained_per_chain': [r['samples_completed'] for r in reports],
        'divergences_by_chain': [r['divergences'] for r in reports],
        'decision': summary['decision'],
        'maximum_rhat': mixing['maximum_rhat'], 'minimum_bulk_ess': mixing['minimum_bulk_ess'],
        'minimum_tail_ess': mixing['minimum_tail_ess'],
        'retained_full_tree_evaluation_count_by_chain': full_trees,
        'diagnostic_names': diagnostic_names,
        'half_chain_means': halves,
        'summary_sha256': sha256(summary_path)}


def main():
    base = ROOT / 'results/analysis/external_control_production'
    extended = ROOT / 'results/analysis/external_control_gp_extended'
    repaired = ROOT / 'results/analysis/external_control_copula_repaired'
    sources = {Path(__file__).relative_to(ROOT).as_posix(): sha256(Path(__file__)),
        'scripts/run_lisa_production.py': sha256(ROOT / 'scripts/run_lisa_production.py'),
        'scripts/correct_selected_posterior.py': sha256(ROOT / 'scripts/correct_selected_posterior.py')}
    models = {label: collect(directory, label, sources) for label, directory in
        [('copula', repaired / 'copula'), ('gp_6', extended / 'gp_6'), ('gp_12', extended / 'gp_12')]}
    paths = [base / 'job_cost_000.json', repaired / 'job_cost_000.json',
        extended / 'outer_timing_gp_6.json', extended / 'outer_timing_gp_12.json',
        extended / 'job_cost_000.json', extended / 'job_cost_001.json']
    records = [json.loads(p.read_text()) for p in paths]
    initial_seconds, repair_seconds = records[0]['process_elapsed_seconds'], records[1]['process_elapsed_seconds']
    extension_seconds = sum(row['elapsed_upper_bound_seconds'] for row in records[2:4])
    for outer, inner in zip(records[2:4], records[4:]):
        if outer['elapsed_upper_bound_seconds'] < inner['process_elapsed_seconds']:
            raise ValueError('Outer timing cannot be shorter than the recorded inner process')
    total = initial_seconds+extension_seconds+repair_seconds
    base_config = json.loads((ROOT / 'configs/external_control_production.json').read_text())
    original = json.loads((ROOT / base_config['protocol']).read_text())
    inherited = base_config['resource_amendment']['prior_study_budget_charge_seconds']
    copula_initial = json.loads((base / 'copula/summary.json').read_text())['recorded_process_elapsed_seconds']
    resource_ok = (total <= base_config['resource_amendment']['maximum_seconds_control_production']
        and inherited+total <= original['resources']['maximum_total_cpu_hours']*3600
        and copula_initial+repair_seconds <= base_config['resource_amendment']['maximum_seconds_per_fit']
        and extension_seconds <= 2400
        and all(row['runtime']['peak_working_set_bytes'] <= original['resources']['maximum_resident_gib']*1024**3
                for row in [records[0], records[1], *records[4:]]))
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    report = {'experiment': 'selected_gp_truth', 'models': models, 'source_sha256': sources,
        'cost': {'initial_job_seconds': initial_seconds, 'gp_extension_outer_seconds': extension_seconds,
            'copula_repair_job_seconds': repair_seconds, 'all_attempts_seconds': total,
            'copula_original_and_repair_seconds': copula_initial+repair_seconds,
            'inherited_conservative_charge_seconds': inherited, 'total_study_charge_seconds': inherited+total,
            'remaining_study_seconds': original['resources']['maximum_total_cpu_hours']*3600-inherited-total},
        'resources_passed': bool(resource_ok),
        'posteriors_ready_for_comparison': bool(resource_ok and all(m['decision']['posterior_accepted'] for m in models.values())),
        'scientific_scores_examined': False,
        'ensemble_rule': 'Use only the four longer chains at each GP resolution and the repaired four-chain copula ensemble. Original GP prefixes and the rejected copula trajectory remain archived but are not pooled into these ensembles.'}
    write_json(ROOT / 'results/analysis/external_control_posteriors.json', report)
    print(json.dumps({k: report[k] for k in ('posteriors_ready_for_comparison', 'resources_passed', 'cost')}), flush=True)


if __name__ == '__main__':
    main()
