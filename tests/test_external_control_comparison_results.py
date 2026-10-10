"""Reconstruct the control scores from retained arrays without new fits."""
import json
from pathlib import Path
import sys
import warnings

import arviz as az
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from binned_gp_benchmark import FEATURE_NAMES
from compare_external_control import refined_subset_weights
from compare_lisa_posterior import moment_check
from correct_selected_posterior import sha256, weighted_quantiles
from external_selected_evaluation import coarsen, interval_score

OUTPUT = ROOT / 'results/analysis/external_control_comparison'
POSTERIOR = ROOT / 'results/analysis/external_control_copula_repaired/copula/posterior.npz'


def report():
    return json.loads((OUTPUT / 'summary.json').read_text())


def test_sources_and_products_remain_identical_to_the_numerical_manifests():
    for path in OUTPUT.glob('*.json'):
        value = json.loads(path.read_text())
        for name, digest in value.get('source_sha256', {}).items():
            assert sha256(ROOT / name) == digest, (path.name, name)
        if 'product_sha256' in value:
            products = value['product_sha256']
            if isinstance(products, str):
                assert sha256(path.with_suffix('.npz')) == products
            else:
                for name, digest in products.items():
                    assert sha256(OUTPUT / name) == digest


def test_accepted_summary_has_no_nonfinite_or_missing_diagnostics():
    summary = report()
    assert all(summary['decision'].values())
    def check(value):
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, float):
            assert np.isfinite(value)
        else:
            assert value is not None
    check(summary)


def test_both_gp_resolutions_and_both_distances_reconstruct_from_cell_masses():
    summary = report()
    truth = np.load(OUTPUT / 'truth.npz')
    assert set(summary['gp']) == {'6', '12'}
    for key, name in [('copula', 'copula_recovery'), ('6', 'gp_6'), ('12', 'gp_12')]:
        product = np.load(OUTPUT / (name+'.npz'))
        row = summary['copula'] if key == 'copula' else summary['gp'][key]
        for kind in ('intrinsic', 'detected'):
            assert product[kind].sum() == pytest.approx(1., abs=2e-12)
            for bins in (48, 96):
                distance = jensenshannon(coarsen(product[kind], 96, bins), coarsen(truth[kind], 96, bins), base=2)
                assert row['joint'][f'{kind}_{bins}']['distance'] == pytest.approx(distance, abs=1e-14)
                assert row['joint'][f'{kind}_{bins}']['mcse'] > 0


def test_predictive_comparison_averages_densities_and_preserves_event_pairing():
    summary = report()
    weights = np.load(POSTERIOR)['weights'].ravel()
    alpha = np.load(OUTPUT / 'copula_recovery.npz')['features'][:, 6]
    events = np.load(OUTPUT / 'copula_prediction.npz')['log_integrals']
    assert events.shape == (3200, 192)
    copula = logsumexp(events-np.log(alpha[:, None])+np.log(weights[:, None]), axis=0)
    for bins in ('6', '12'):
        integrals = np.load(OUTPUT / f'gp_{bins}.npz')['log_integrals']
        assert integrals.shape == (9600, 192)
        gp = logsumexp(integrals, axis=0)-np.log(len(integrals))
        difference = copula-gp
        row = summary['copula_minus_gp_test_scores'][bins]
        assert row['total_difference'] == pytest.approx(difference.sum(), abs=1e-12)
        assert row['total_standard_error'] == pytest.approx(np.sqrt(192)*difference.std(ddof=1), abs=1e-12)
        assert row['posterior_mcse'] > 0


def test_finer_training_and_test_check_uses_the_same_draws():
    summary = report()
    posterior = dict(np.load(POSTERIOR))
    primary = np.load(OUTPUT / 'copula_prediction.npz')
    fine = np.load(OUTPUT / 'copula_prediction_verification.npz')
    indices = fine['indices']
    assert len(indices) == 512
    np.testing.assert_array_equal(indices.reshape(4, 128) % 800, np.tile(np.linspace(0, 799, 128, dtype=int), (4, 1)))
    a, b = refined_subset_weights(posterior, indices)
    alpha = np.load(OUTPUT / 'copula_recovery.npz')['features'][indices, 6]
    old = logsumexp(primary['log_integrals'][indices]-np.log(alpha[:, None])+np.log(a[:, None]), axis=0)
    new = logsumexp(fine['log_integrals']-np.log(alpha[:, None])+np.log(b[:, None]), axis=0)
    numerical = summary['numerical_prediction']
    assert numerical['training_target_and_event_integration']['total_score_change'] == pytest.approx((new-old).sum(), abs=1e-12)
    for key, change in numerical['gp_quadrature_total_score_changes'].items():
        assert numerical['paired_total_score_changes'][key] == pytest.approx((new-old).sum()-change, abs=1e-12)


@pytest.mark.parametrize('key', ['copula', '6', '12'])
def test_all_functional_intervals_and_scores_retain_truth_and_misses(key):
    summary = report()
    truth = np.load(OUTPUT / 'truth.npz')['features']
    name = 'copula_recovery' if key == 'copula' else 'gp_'+key
    values = np.load(OUTPUT / (name+'.npz'))['features'].reshape(-1, 12)
    weights = np.load(POSTERIOR)['weights'].ravel() if key == 'copula' else np.full(len(values), 1/len(values))
    rows = (summary['copula'] if key == 'copula' else summary['gp'][key])['functionals']['functionals']
    assert set(rows) == set(FEATURE_NAMES)
    for i, name in enumerate(FEATURE_NAMES):
        row = rows[name]
        q = weighted_quantiles(values[:, i], weights, [.05, .5, .95])
        np.testing.assert_allclose(row['quantiles'], q, atol=1e-14)
        assert row['truth'] == truth[i]
        assert row['error_of_mean'] == pytest.approx(weights @ values[:, i]-truth[i], abs=1e-14)
        assert row['interval_score_90'] == pytest.approx(interval_score(q[0], q[2], truth[i]), abs=1e-14)
        assert row['truth_in_interval'] == (q[0] <= truth[i] <= q[2])


def test_distance_uncertainty_preserves_mcmc_order_and_raw_weights():
    summary = report()
    weights = np.load(POSTERIOR)['weights'].ravel()
    influences = np.load(OUTPUT / 'copula_influences.npz')
    for label, row in summary['copula']['joint'].items():
        check = moment_check(influences[label], weights, (4, 800), [25, 50, 100])
        assert row['mcse'] == pytest.approx(check['mcse'], rel=1e-12)
        assert row['influence_rhat'] == pytest.approx(check['influence_rhat'], rel=1e-12)


def test_gp_cdf_rank_diagnostics_warn_only_for_constant_folded_indicators():
    for key in ('6', '12'):
        values = np.load(OUTPUT / f'gp_{key}.npz')['features']
        weights = np.full(values.shape[:2], 1/np.prod(values.shape[:2]))
        weights /= weights.sum()
        for column in range(12):
            quantiles = weighted_quantiles(values[:, :, column].ravel(), weights.ravel(), [.05, .5, .95])
            for quantile in quantiles:
                indicator = values[:, :, column] <= quantile
                cdf = np.sum(weights*indicator)
                influence = weights.size*weights*(indicator-cdf)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always', RuntimeWarning)
                    rhat = az.rhat(influence, method='rank')
                assert np.isfinite(rhat)
                runtime = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
                if runtime:
                    folded = abs(influence-np.median(influence))
                    assert np.ptp(folded) == 0
                    assert np.ptp(influence) > 0


def test_decision_and_cost_accounting_preserve_the_original_requirements():
    summary = report()
    decision = summary['decision']
    numerical = summary['numerical_recovery']
    changes = list(numerical['posterior_transport_js_changes'].values())
    changes.extend(value for group in numerical['partition_js_changes'].values() for value in group.values())
    assert decision['recovery_precision_passed'] == (max(map(abs, changes)) < .005)
    assert decision['prediction_precision_passed'] == (max(map(abs, summary['numerical_prediction']['paired_total_score_changes'].values())) < .25)
    assert decision['derived_mixing_passed'] == (summary['maximum_derived_influence_rhat'] <= 1.01)
    assert decision['eligible_for_scientific_comparison'] == all(value for key, value in decision.items() if key != 'eligible_for_scientific_comparison')
    jobs = [json.loads(path.read_text()) for path in OUTPUT.glob('job_cost_*.json')]
    assert jobs and jobs[-1]['status'] == 'complete'
    cost = sum(job['process_elapsed_seconds'] for job in jobs)
    assert jobs[-1]['all_evaluation_attempts_seconds'] == pytest.approx(cost)
    assert cost <= summary['configuration']['maximum_seconds']
    assert jobs[-1]['total_study_charge_seconds'] == pytest.approx(summary['posterior_costs']['total_study_charge_seconds']+cost)
    assert jobs[-1]['remaining_study_seconds'] == pytest.approx(43200-jobs[-1]['total_study_charge_seconds'])
    assert all(job['runtime']['threads'] == 1 and job['runtime']['peak_working_set_bytes'] <= 2*1024**3 for job in jobs)
