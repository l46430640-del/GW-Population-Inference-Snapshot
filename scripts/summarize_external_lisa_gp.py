"""Identify usable LISA GP ensembles and retain the complete numerical cost."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import json
from pathlib import Path

from correct_selected_posterior import sha256
from run_external_gp_pilot import ROOT
from run_lisa_production import verify_product, write_json


def collect(directory, inputs, sources):
    target_path = directory / 'target.json'
    summary_path = directory / 'summary.json'
    acceptance_path = directory / 'acceptance.json'
    target, summary, acceptance = [json.loads(p.read_text()) for p in (target_path, summary_path, acceptance_path)]
    if sha256(target_path) != acceptance['target_product_sha256'] or sha256(summary_path) != acceptance['summary_sha256']:
        raise ValueError('An accepted ensemble has different provenance')
    if summary['target_product_sha256'] != acceptance['target_product_sha256']:
        raise ValueError('Summary and acceptance targets differ')
    if sha256(inputs) != target['inputs_product_sha256']:
        raise ValueError('Numerical bin inputs changed')
    for name, digest in target['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    sources.update(target['source_sha256'])
    paths = [target_path, summary_path, acceptance_path, inputs, inputs.with_suffix('.json')]
    reports = []
    for name, digest in summary['chain_reports_sha256'].items():
        path = directory / name
        report = verify_product(path.with_suffix('.npz'))
        if sha256(path) != digest or report['target_product_sha256'] != acceptance['target_product_sha256']:
            raise ValueError('A chain differs from its ensemble')
        reports.append(report)
        paths.extend([path, path.with_suffix('.npz')])
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    return {'directory': directory.relative_to(ROOT).as_posix(), 'inputs': inputs.relative_to(ROOT).as_posix(),
        'decision': acceptance['decision'], 'chains': len(reports),
        'retained_per_chain': [r['samples_completed'] for r in reports],
        'divergences_by_chain': [r['divergences'] for r in reports],
        'mixing': None if summary['mixing'] is None else {key: summary['mixing'][key]
            for key in ('maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess')},
        'pooled_integrals': acceptance['pooled_integral_verification'],
        'retained_full_tree_evaluation_count': summary['retained_full_tree_evaluation_count']}


def main():
    original = ROOT / 'results/analysis/external_lisa_gp'
    sparse = ROOT / 'results/analysis/external_lisa_gp_sparse'
    completed = ROOT / 'results/analysis/external_lisa_gp_completed'
    check = ROOT / 'results/analysis/external_lisa_sparse_check'
    sources = {Path(__file__).relative_to(ROOT).as_posix(): sha256(Path(__file__))}
    prefix_path = completed / 'bins_12/prefix_verification.json'
    prefix = json.loads(prefix_path.read_text())
    if not prefix['prefix_identical'] or not prefix['prefix_not_counted_as_additional_draws']:
        raise ValueError('The fourth-chain replay must preserve and exclude its old prefix')
    if prefix['parent_chain_sha256'] != sha256(sparse / f"bins_12/chain_{prefix['seed']}.npz"):
        raise ValueError('The replay comparison refers to a different partial chain')
    sources[prefix_path.relative_to(ROOT).as_posix()] = sha256(prefix_path)
    models = {key: collect(directory, original / f'inputs_{key}.npz', sources)
        for key, directory in [('6', original / 'bins_6'), ('12', completed / 'bins_12')]}
    groups = {name: list(directory.glob('job_cost_*.json'))
        for name, directory in [('original', original), ('equivalence', check), ('sparse', sparse), ('completion', completed)]}
    seconds, records = {}, []
    for name, paths in groups.items():
        if not paths:
            raise ValueError('Every declared attempt needs its process cost')
        rows = [json.loads(p.read_text()) for p in paths]
        seconds[name] = sum(row['process_elapsed_seconds'] for row in rows)
        records.extend(rows)
        sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    prior_path = ROOT / 'results/analysis/external_control_comparison/job_cost_000.json'
    prior = json.loads(prior_path.read_text())
    sources[prior_path.relative_to(ROOT).as_posix()] = sha256(prior_path)
    total = prior['total_study_charge_seconds']+sum(seconds.values())
    latest = json.loads((completed / 'job_cost_000.json').read_text())
    if abs(total-latest['total_study_charge_seconds']) > 1e-7:
        raise ValueError('Complete cost ledger differs from the final process record')
    resource_ok = (total <= 43200 and seconds['original'] <= 4200 and seconds['equivalence'] <= 180
        and seconds['sparse'] <= 1400 and seconds['completion'] <= 450 and all(row['runtime']['threads'] == 1
            and row['runtime']['peak_working_set_bytes'] <= 2*1024**3 for row in records))
    result = {'experiment': 'lisa_bimodal', 'models': models, 'source_sha256': sources,
        'cost': {'original_preparation_and_attempts_seconds': seconds['original'],
            'equivalence_check_seconds': seconds['equivalence'], 'sparse_sampling_seconds': seconds['sparse'],
            'fourth_chain_completion_seconds': seconds['completion'],
            'all_lisa_gp_seconds': sum(seconds.values()), 'total_study_charge_seconds': total,
            'remaining_study_seconds': 43200-total},
        'resources_passed': bool(resource_ok), 'gp_posteriors_ready_for_comparison': bool(resource_ok
            and all(row['decision']['posterior_accepted'] for row in models.values())),
        'scientific_scores_examined': False,
        'fourth_chain_prefix_verification': prefix,
        'ensemble_rule': 'Use the four accepted original six-bin chains and the final four sparse twelve-bin chains only when all acceptance checks pass. Three complete sparse chains are copied byte for byte, and only the fourth is replayed to its original 800-draw endpoint. Preserve both pilots and both time-limited trajectories, excluding them as additional scientific samples. Charge every original and replayed computation once. The accepted copula posterior is separate; external predictive and recovery checks remain required.'}
    write_json(ROOT / 'results/analysis/external_lisa_gp_posteriors.json', result)
    print(json.dumps({key: result[key] for key in ('gp_posteriors_ready_for_comparison', 'resources_passed', 'cost')}), flush=True)


if __name__ == '__main__':
    main()
