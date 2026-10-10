import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from correct_selected_posterior import sha256
from run_lisa_production import verify_product


@pytest.mark.parametrize('bins', [6, 12])
def test_extended_gp_ensembles_pass_and_do_not_double_count_prefixes(bins):
    parent = ROOT / f'results/analysis/external_control_production/gp_{bins}'
    child = ROOT / f'results/analysis/external_control_gp_extended/gp_{bins}'
    summary = json.loads((child / 'summary.json').read_text())
    target = json.loads((child / 'target.json').read_text())
    assert summary['decision']['posterior_accepted']
    assert target['effective_protocol']['sampling']['retained_per_chain'] == 2400
    for name, digest in target['source_sha256'].items():
        assert sha256(ROOT / name) == digest
    for seed in target['effective_protocol']['sampling']['chain_seeds']:
        path = child / f'chain_{seed}.npz'
        report = verify_product(path)
        assert report['divergences'] == []
        assert report['samples_completed'] == 2400 and report['warmup_completed'] == 500
        assert report['target_product_sha256'] == sha256(child / 'target.json')
        before, after = np.load(parent / path.name), np.load(path)
        np.testing.assert_array_equal(before['initial_coordinate'], after['initial_coordinate'])
        np.testing.assert_array_equal(before['warmup_coordinate'], after['warmup_coordinate'])
        np.testing.assert_array_equal(before['coordinate'], after['coordinate'][:800])


def test_repair_reuses_only_three_unchanged_nondivergent_chains():
    parent = ROOT / 'results/analysis/external_control_production/copula'
    child = ROOT / 'results/analysis/external_control_copula_repaired/copula'
    target = json.loads((child / 'target.json').read_text())
    config = target['configuration']
    for name, digest in target['source_sha256'].items():
        assert sha256(ROOT / name) == digest
    for seed in config['sampling_repair']['reuse_seeds']:
        path = child / f'chain_{seed}.npz'
        report = verify_product(path)
        assert sha256(path) == sha256(parent / path.name)
        assert report['divergences'] == []
        assert report['process_elapsed_seconds'] == report['sampling_and_compilation_seconds'] == 0
        assert report['parent_report_sha256'] == sha256((parent / path.name).with_suffix('.json'))
    seed = config['sampling_repair']['replace_seed']
    path = child / f'chain_{seed}.npz'
    report = verify_product(path)
    assert 'reused_from' not in report
    assert sha256(path) != sha256(parent / path.name)
    assert target['effective_protocol']['sampling']['target_acceptance'] == .97
    assert json.loads((parent / path.name).with_suffix('.json').read_text())['divergences'] == [79, 86, 172, 245, 395, 507, 508]
    summary = json.loads((child / 'summary.json').read_text())
    assert summary['decision']['posterior_accepted'] == all(value for key, value in summary['decision'].items() if key != 'posterior_accepted')
    if summary['decision']['posterior_accepted']:
        assert report['divergences'] == []
        assert summary['mixing']['maximum_rhat'] <= 1.01
        assert min(summary['mixing']['minimum_bulk_ess'], summary['mixing']['minimum_tail_ess']) >= 400
        assert summary['correction']['weight_ess_fraction'] >= .9
        assert summary['correction']['pareto_k'] < .7
        assert max(summary['independent_event_log_likelihood_difference_sd'].values()) <= .2


def test_cached_evaluations_match_the_unchanged_states_exactly():
    parent = np.load(ROOT / 'results/analysis/external_control_production/copula/posterior.npz')
    path = ROOT / 'results/analysis/external_control_copula_repaired/copula'
    result = np.load(path / 'posterior.npz')
    summary = json.loads((path / 'summary.json').read_text())
    assert sha256(path / 'posterior.npz') == summary['posterior_product_sha256']
    keep = np.r_[0:1600, 2400:3200]
    np.testing.assert_array_equal(parent['log_integrals'][keep], result['log_integrals'][keep])
    for chain in [0, 1, 3]:
        np.testing.assert_array_equal(parent['coordinate'][chain], result['coordinate'][chain])
    assert not np.array_equal(parent['coordinate'][2], result['coordinate'][2])
    assert np.sum(result['weights']) == pytest.approx(1., abs=1e-14)
    reuse = json.loads((path / 'integration_reuse.json').read_text())
    assert reuse['reused_state_cloud_evaluations'] >= 4800
    assert reuse['fresh_state_cloud_evaluations'] >= 1600


def test_final_manifest_and_resource_accounting_match_the_evidence():
    report = json.loads((ROOT / 'results/analysis/external_control_posteriors.json').read_text())
    for name, digest in report['source_sha256'].items():
        assert sha256(ROOT / name) == digest
    assert not report['scientific_scores_examined']
    assert report['posteriors_ready_for_comparison'] == (report['resources_passed']
        and all(row['decision']['posterior_accepted'] for row in report['models'].values()))
    cost = report['cost']
    assert cost['all_attempts_seconds'] == pytest.approx(cost['initial_job_seconds']
        + cost['gp_extension_outer_seconds'] + cost['copula_repair_job_seconds'])
    assert cost['total_study_charge_seconds'] == pytest.approx(cost['all_attempts_seconds']+19800)
    assert cost['remaining_study_seconds'] == pytest.approx(43200-cost['total_study_charge_seconds'])
    if report['resources_passed']:
        assert cost['all_attempts_seconds'] <= 18000
        assert cost['copula_original_and_repair_seconds'] <= 14400
        assert cost['gp_extension_outer_seconds'] <= 2400
        assert cost['remaining_study_seconds'] >= 0
