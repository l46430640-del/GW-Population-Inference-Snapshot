import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from correct_selected_posterior import sha256
from lisa_gp_inputs import integral_report
from run_lisa_production import mixing_summary, verify_product

SPARSE = ROOT / 'results/analysis/external_lisa_gp_sparse'
COMPLETED = ROOT / 'results/analysis/external_lisa_gp_completed'


def test_completion_preserves_protocol_and_only_replaces_incomplete_chain():
    from run_external_lisa_gp_sparse import settings
    from run_lisa_production import verify_product
    amendment = json.loads((ROOT / 'configs/external_lisa_gp_completion.json').read_text())
    _, protocol = settings(amendment)
    parent = ROOT / amendment['completion']['parent']
    target = json.loads((parent / 'target.json').read_text())
    assert protocol == target['effective_protocol']
    completion = amendment['completion']
    assert completion['reuse_seeds']+[completion['replace_seed']] == protocol['sampling']['chain_seeds']
    for seed in completion['reuse_seeds']:
        report = verify_product(parent / f'chain_{seed}.npz')
        assert report['status'] == 'complete'
        assert report['samples_completed'] == 800
        assert report['divergences'] == []
    partial = verify_product(parent / f"chain_{completion['replace_seed']}.npz")
    assert partial['status'] == 'time_limit'
    assert partial['samples_completed'] == completion['expected_parent_retained'] == 444
    assert completion['require_identical_prefix']
    previous = json.loads((ROOT / amendment['previous_cost']).read_text())
    assert amendment['maximum_seconds'] == 450 < previous['remaining_study_seconds']


@pytest.mark.parametrize('bins', [6, 12])
def test_equivalence_summary_matches_archived_errors_and_keeps_full_population(bins):
    directory = ROOT / 'results/analysis/external_lisa_sparse_check'
    summary = json.loads((directory / 'summary.json').read_text())
    path = directory / f'bins_{bins}.npz'
    assert sha256(path) == summary['product_sha256'][path.name]
    arrays = np.load(path)
    row = summary['models'][str(bins)]
    assert row['maximum_value_error'] == arrays['absolute_value_error'].max() < 1e-7
    assert row['maximum_gradient_error'] == arrays['maximum_gradient_error'].max() < 1e-7
    assert row['checked_states'] == len(arrays['coordinate']) == 33
    assert row['full_population_bins_retained'] == bins**3
    assert row['sparse_padded_matrix_entries'] == 135*max(arrays['occupied_event_bin_counts'])
    assert row['median_gradient_speedup'] == pytest.approx(
        np.median(arrays['dense_seconds_per_gradient'])/np.median(arrays['sparse_seconds_per_gradient']))
    assert summary['equivalent'] and not summary['new_posterior_created']


def test_three_complete_chains_are_reused_byte_for_byte_without_recharging():
    for seed in (847201, 847202, 847203):
        parent = SPARSE / f'bins_12/chain_{seed}.npz'
        child = COMPLETED / f'bins_12/chain_{seed}.npz'
        old, new = verify_product(parent), verify_product(child)
        assert sha256(parent) == sha256(child)
        assert new['parent_report_sha256'] == sha256(parent.with_suffix('.json'))
        assert new['reused_from'] == parent.relative_to(ROOT).as_posix()
        assert new['parent_target_product_sha256'] == old['target_product_sha256']
        assert new['original_process_elapsed_seconds'] == old['process_elapsed_seconds'] > 0
        assert new['original_sampling_and_compilation_seconds'] == old['sampling_and_compilation_seconds'] > 0
        assert new['process_elapsed_seconds'] == new['sampling_and_compilation_seconds'] == 0


def test_replayed_fourth_chain_reproduces_warmup_and_prefix_without_pooling_them():
    parent = SPARSE / 'bins_12/chain_847204.npz'
    child = COMPLETED / 'bins_12/chain_847204.npz'
    old, new = np.load(parent), np.load(child)
    assert len(old['coordinate']) == 444
    assert len(new['coordinate']) == 800
    for name in ('initial_coordinate', 'warmup_coordinate'):
        np.testing.assert_array_equal(old[name], new[name])
    np.testing.assert_array_equal(old['coordinate'], new['coordinate'][:444])
    check = json.loads((COMPLETED / 'bins_12/prefix_verification.json').read_text())
    assert check['prefix_identical'] and check['prefix_not_counted_as_additional_draws']
    assert check['parent_chain_sha256'] == sha256(parent)
    assert check['compared_retained_draws'] == 444
    assert verify_product(parent)['status'] == 'time_limit'
    assert not json.loads((SPARSE / 'bins_12/acceptance.json').read_text())['decision']['posterior_accepted']


def test_completed_ensemble_reconstructs_mixing_integrals_and_acceptance():
    directory = COMPLETED / 'bins_12'
    summary = json.loads((directory / 'summary.json').read_text())
    final = json.loads((directory / 'acceptance.json').read_text())
    products = []
    for seed in range(847201, 847205):
        path = directory / f'chain_{seed}.npz'
        report = verify_product(path)
        assert report['status'] == 'complete'
        assert report['samples_completed'] == 800 and report['warmup_completed'] == 500
        assert report['divergences'] == []
        products.append(dict(np.load(path)))
    checked = mixing_summary(products[0]['diagnostic_names'].tolist(), np.stack([p['diagnostic_values'] for p in products]))
    for key in ('maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess'):
        assert checked[key] == pytest.approx(summary['mixing'][key], rel=1e-12)
    numerical = {name: np.concatenate([p[name] for p in products]) for name in final['pooled_integral_verification']['standard_deviations']}
    assert integral_report(numerical, .2) == final['pooled_integral_verification']
    decision = final['decision']
    assert decision['posterior_accepted'] == all(v for k, v in decision.items() if k != 'posterior_accepted')
    assert not final['scientific_scores_examined']


def test_final_registry_verifies_sources_and_charges_all_attempts_once():
    summary = json.loads((ROOT / 'results/analysis/external_lisa_gp_posteriors.json').read_text())
    for name, digest in summary['source_sha256'].items():
        assert sha256(ROOT / name) == digest
    jobs = []
    for name in ('external_lisa_gp', 'external_lisa_sparse_check', 'external_lisa_gp_sparse', 'external_lisa_gp_completed'):
        directory = ROOT / 'results/analysis' / name
        jobs.extend(json.loads(p.read_text()) for p in directory.glob('job_cost_*.json'))
        for path in directory.rglob('*.json'):
            record = json.loads(path.read_text())
            for source, digest in record.get('source_sha256', {}).items():
                assert sha256(ROOT / source) == digest
            if 'draw_product_sha256' in record:
                assert sha256(path.with_suffix('.npz')) == record['draw_product_sha256']
            if 'target_product_sha256' in record:
                assert sha256(path.parent / 'target.json') == record['target_product_sha256']
            for name, digest in record.get('chain_reports_sha256', {}).items():
                assert sha256(path.parent / name) == digest
    cost = sum(row['process_elapsed_seconds'] for row in jobs)
    prior = json.loads((ROOT / 'results/analysis/external_control_comparison/job_cost_000.json').read_text())
    assert summary['cost']['all_lisa_gp_seconds'] == pytest.approx(cost)
    assert summary['cost']['total_study_charge_seconds'] == pytest.approx(prior['total_study_charge_seconds']+cost)
    assert summary['cost']['remaining_study_seconds'] == pytest.approx(43200-prior['total_study_charge_seconds']-cost)
    assert summary['resources_passed']
    assert summary['gp_posteriors_ready_for_comparison'] == all(row['decision']['posterior_accepted'] for row in summary['models'].values())
    assert summary['models']['6']['retained_per_chain'] == [2400]*4
    assert summary['models']['12']['retained_per_chain'] == [800]*4
    assert not summary['scientific_scores_examined']
