"""Reconstruct the third comparison from its frozen numerical products."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from binned_gp_benchmark import FEATURE_NAMES
from compare_external_selected import difference_precision, prediction_diagnostics, recovery_gradients
from correct_selected_posterior import sha256, weighted_quantiles
from external_selected_evaluation import interval_score
from predict_corrected_posterior import weighted_prediction
from summarize_selected_mixture import paired_scores

PARENT = ROOT / 'results/analysis/external_lisa_comparison'
OUTPUT = ROOT / 'results/analysis/external_lisa_comparison_double'
OLD = ROOT / 'results/analysis/lisa_scientific_comparison'
COPULA = ROOT / 'results/analysis/lisa_refinement'


def summary():
    return json.loads((OUTPUT / 'summary.json').read_text())


def test_sources_and_every_retained_product_match_their_hashes():
    report = summary()
    for name, digest in report['source_sha256'].items():
        assert sha256(ROOT / name) == digest
    for name, digest in report['product_sha256'].items():
        path = OUTPUT / name
        assert sha256(path) == digest
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['product_sha256'] == digest
        assert metadata['source_sha256'] == report['source_sha256']
        assert metadata['runtime']['threads'] == 1
        assert metadata['runtime']['peak_working_set_bytes'] <= 2*1024**3


def test_double_precision_check_retains_failure_and_reuses_unchanged_training_products():
    original = json.loads((PARENT / 'copula_target_summary.json').read_text())
    updated = json.loads((OUTPUT / 'copula_target_summary.json').read_text())
    assert not original['decision']['kernel_passed']
    assert original['maximum_kernel_total_log_likelihood_difference'] > 1e-6
    assert updated['decision']['kernel_passed']
    assert updated['maximum_kernel_total_log_likelihood_difference'] <= 1e-6
    for name in ('training_primary', 'training_verification', 'training_changes'):
        assert sha256(PARENT / (name+'.npz')) == sha256(OUTPUT / (name+'.npz'))
        metadata = json.loads((OUTPUT / (name+'.json')).read_text())
        assert metadata['seconds'] == 0
        assert metadata['parent_metadata_sha256'] == sha256(PARENT / (name+'.json'))
        assert metadata['original_seconds'] == json.loads((PARENT / (name+'.json')).read_text())['seconds']


def test_external_target_check_recomputes_all_original_groups_at_8192_samples():
    report = summary()
    original = np.load(COPULA / 'verification.npz')
    arrays = np.load(OUTPUT / 'training_changes.npz')
    primary = np.load(OUTPUT / 'training_primary.npz')
    fine = np.load(OUTPUT / 'training_verification.npz')
    np.testing.assert_array_equal(arrays['indices'], original['indices'])
    np.testing.assert_array_equal(arrays['uniform_indices'], original['uniform_indices'])
    event = (fine['log_integrals']-primary['log_integrals']).sum(-1)
    np.testing.assert_allclose(arrays['event_change'], event, atol=1e-12)
    posterior = np.load(COPULA / 'posterior.npz')
    alpha = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
    selection = -135*np.log(original['alpha_verification']/alpha[arrays['indices']])
    np.testing.assert_array_equal(arrays['selection_change'], selection)
    np.testing.assert_allclose(arrays['combined_change'], event+selection, atol=1e-12)
    groups = {name.removeprefix('group_'): value for name, value in original.items() if name.startswith('group_')}
    for name, group in dict(groups, pooled=arrays['indices']).items():
        for kind in ('event', 'selection', 'combined'):
            values = arrays[kind+'_change'][np.searchsorted(arrays['indices'], group)]
            assert report['copula_target_checks'][name][kind] == pytest.approx(values.std())
            assert values.std() <= .2
    assert all(json.loads((OUTPUT / 'copula_target_summary.json').read_text())['decision'].values())
    assert np.load(ROOT / report['configuration']['training_verification'])['data'].shape == (135, 8192, 4)


def test_new_test_posteriors_preserve_batch_precision_and_common_event_count():
    data = np.load(OUTPUT / 'test_events_verification.npz')
    assert data['data'].shape == (192, 8192, 4)
    np.testing.assert_array_equal(data['data'][:, :, 3], 1.)
    assert data['batch_proposal_ess'].shape == (192, 4)
    assert data['batch_proposal_ess'].min() >= 8192
    np.testing.assert_allclose(data['proposal_ess'], 16/(1/data['batch_proposal_ess']).sum(1), rtol=1e-14)
    assert data['proposal_ess'].min() >= 32768


def test_primary_prediction_uses_all_draws_and_reuses_only_exact_existing_integrals():
    prediction = np.load(OUTPUT / 'copula_prediction.npz')
    uniform = prediction['reused_indices']
    np.testing.assert_array_equal(uniform, np.load(OLD / 'derived.npz')['nuts_verification_indices'])
    assert prediction['log_integrals'].shape == (2000, 192)
    posterior = np.load(COPULA / 'posterior.npz')
    alpha = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
    old_alpha = np.load(OLD / 'nuts_population_fine.npz')['alpha']
    expected = np.load(OLD / 'nuts_prediction_fine.npz')['log_integrals']+np.log(old_alpha[:, None])-np.log(alpha[uniform, None])
    np.testing.assert_array_equal(prediction['log_integrals'][uniform], expected)


def test_paired_predictive_scores_and_numerical_changes_reconstruct():
    report = summary()
    weight = np.load(COPULA / 'weights.npz')['weights']
    primary = np.load(OUTPUT / 'copula_prediction.npz')['log_integrals']
    score, precision = prediction_diagnostics(primary, weight.ravel(), weight.shape, report['configuration']['batch_lengths'])
    assert precision == report['copula']['prediction_precision']
    target = np.load(OUTPUT / 'training_changes.npz')
    uniform = target['uniform_indices']
    position = np.searchsorted(target['indices'], uniform)
    first_weight = weight.ravel()[uniform]
    first_weight /= first_weight.sum()
    fine_weight = np.exp(np.log(first_weight)+target['combined_change'][position]-logsumexp(np.log(first_weight)+target['combined_change'][position]))
    alpha = np.load(COPULA / 'verification.npz')['alpha_verification'][position]
    refined = np.load(OUTPUT / 'copula_prediction_verification.npz')['log_integrals']-np.log(alpha[:, None])
    change = difference_precision(primary[uniform], refined, first_weight, fine_weight, (4, 64))
    assert change == report['numerical_prediction']['event_selection_and_training_refinement']
    for key in ('6', '12'):
        gp = np.load(OUTPUT / f'gp_{key}.npz')
        shape = gp['features'].shape[:2]
        gp_score, gp_precision = prediction_diagnostics(gp['log_integrals'], np.full(np.prod(shape), 1/np.prod(shape)), shape, report['configuration']['batch_lengths'])
        expected = dict(paired_scores(score, gp_score), posterior_mcse=float(np.hypot(precision['mcse'], gp_precision['mcse'])))
        assert expected == report['copula_minus_gp_test_scores'][key]
        gp_change = (weighted_prediction(gp['fine_log_integrals'], gp['fine_weights'])-gp_score).sum()
        assert report['numerical_prediction']['gp_total_score_changes'][key] == pytest.approx(gp_change)
        assert report['numerical_prediction']['paired_total_score_changes'][key] == pytest.approx(change['total_score_change']-gp_change)


def test_reported_recovery_and_all_interval_scores_reconstruct():
    report = summary()
    truth = dict(np.load(OUTPUT / 'truth.npz'))
    weights = np.load(COPULA / 'weights.npz')['weights'].ravel()
    for label in ('copula', '6', '12'):
        product = dict(np.load(OUTPUT / ('copula_recovery.npz' if label == 'copula' else f'gp_{label}.npz')))
        row = report['copula'] if label == 'copula' else report['gp'][label]
        for kind in ('intrinsic', 'detected'):
            assert product[kind].sum() == pytest.approx(1., abs=1e-10)
        ds, _ = recovery_gradients(product, truth, 96, (48, 96))
        for name, distance in ds.items():
            assert row['joint'][name]['distance'] == distance
        features = product['features'].reshape(-1, 12).copy()
        weight = weights if label == 'copula' else np.full(len(features), 1/len(features))
        if label == 'copula':
            posterior = np.load(COPULA / 'posterior.npz')
            features[:, 6] = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
        for i, name in enumerate(FEATURE_NAMES):
            functional = row['functionals']['functionals'][name]
            lo, median, hi = weighted_quantiles(features[:, i], weight)
            np.testing.assert_allclose(functional['quantiles'], [lo, median, hi], atol=1e-14)
            assert functional['interval_score_90'] == pytest.approx(interval_score(lo, hi, truth['features'][i]))
            assert functional['truth_in_interval'] == bool(lo <= truth['features'][i] <= hi)


def test_all_attempts_remain_charged_and_failed_checks_cannot_be_accepted():
    report = summary()
    previous = json.loads((ROOT / report['configuration']['posterior_registry']).read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(OUTPUT.glob('job_cost_*.json'))]
    total = sum(job['process_elapsed_seconds'] for job in jobs)
    assert jobs[-1]['all_evaluation_attempts_seconds'] == pytest.approx(total)
    assert jobs[-1]['total_study_charge_seconds'] == pytest.approx(previous['cost']['total_study_charge_seconds']+total)
    assert jobs[-1]['remaining_study_seconds'] == pytest.approx(45000-jobs[-1]['total_study_charge_seconds'])
    assert total <= report['configuration']['maximum_seconds'] == 1780
    failed = json.loads((PARENT / 'job_cost_000.json').read_text())
    assert failed['status'] == 'stopped'
    assert total+failed['process_elapsed_seconds'] <= 2100
    original_registry = json.loads((ROOT / 'results/analysis/external_lisa_gp_posteriors.json').read_text())
    assert previous['cost']['total_study_charge_seconds'] == pytest.approx(
        original_registry['cost']['total_study_charge_seconds']+failed['process_elapsed_seconds'])
    assert jobs[-1]['total_study_charge_seconds'] <= 45000
    assert all(job['runtime']['threads'] == 1 and job['runtime']['peak_working_set_bytes'] <= 2*1024**3 for job in jobs)
    decision = report['decision']
    assert decision['eligible_for_scientific_comparison'] == all(v for k, v in decision.items() if k != 'eligible_for_scientific_comparison')
    assert decision['prediction_precision_passed'] == (max(map(abs, report['numerical_prediction']['paired_total_score_changes'].values())) < .25)
    assert decision['derived_mixing_passed'] == (report['maximum_derived_influence_rhat'] <= 1.01)
