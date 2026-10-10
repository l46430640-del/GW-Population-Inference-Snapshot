"""Regressions for the retained first external comparison, without refitting."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from correct_selected_posterior import sha256, weighted_quantiles
from external_selected_evaluation import coarsen, interval_score

OUTPUT = ROOT / 'results/analysis/external_selected_comparison'


def read(name):
    return json.loads((OUTPUT / name).read_text())


def test_retained_sources_and_products_are_unchanged():
    for path in OUTPUT.rglob('*.json'):
        metadata = json.loads(path.read_text())
        for name, expected in metadata.get('source_sha256', {}).items():
            assert sha256(ROOT / name) == expected, (path.name, name)
        if 'product_sha256' in metadata:
            products = metadata['product_sha256']
            if isinstance(products, str):
                assert sha256(path.with_suffix('.npz')) == products
            else:
                for name, expected in products.items():
                    assert sha256(path.parent / name) == expected, name


def test_target_refinement_is_raw_and_preserves_all_chains():
    weights = np.load(OUTPUT / 'copula_weights.npz')
    primary = np.load(OUTPUT / 'train_primary.npz')['log_integrals']
    original = np.load(OUTPUT / 'train_original.npz')['log_integrals']
    ratio = (primary-original).sum(-1).reshape(4, 800)
    np.testing.assert_allclose(weights['log_ratio'], ratio, atol=1e-12)
    expected = np.exp(ratio-logsumexp(ratio))
    np.testing.assert_allclose(weights['weights'], expected, rtol=1e-13)
    summary = read('copula_target_summary.json')
    assert summary['decision']['target_reusable']
    assert summary['correction']['weight_ess_fraction'] >= .9
    assert summary['correction']['pareto_k'] < .7
    assert max(summary['verification_log_likelihood_difference_sd'].values()) <= .2


def test_both_resolutions_and_both_joint_outcomes_are_reported():
    summary = read('partition_refinement/summary.json')
    assert set(summary['gp']) == {'6', '12'}
    truth = np.load(OUTPUT / 'partition_refinement/truth_quadrature_verification.npz')
    products = {'copula': np.load(OUTPUT / 'partition_refinement/copula_transport_verification.npz')}
    products.update({key: np.load(OUTPUT / f'partition_refinement/gp_{key}.npz') for key in ('6', '12')})
    for key, product in products.items():
        rows = summary['copula']['joint'] if key == 'copula' else summary['gp'][key]['joint']
        for kind in ('intrinsic', 'detected'):
            np.testing.assert_allclose(product[kind].sum(), 1., atol=2e-12)
            for bins in (48, 96):
                expected = jensenshannon(coarsen(product[kind], 96, bins), coarsen(truth[kind], 96, bins), base=2)
                np.testing.assert_allclose(rows[f'{kind}_{bins}']['distance'], expected, atol=1e-14)
                assert rows[f'{kind}_{bins}']['mcse'] > 0


def test_predictive_scores_average_densities_before_taking_logs():
    summary = read('summary.json')
    weights = np.load(OUTPUT / 'copula_weights.npz')['weights'].ravel()
    alpha = np.load(OUTPUT / 'copula_recovery.npz')['features'][:, 6]
    events = np.load(OUTPUT / 'copula_test_primary.npz')['log_integrals']
    copula = logsumexp(events-np.log(alpha[:, None])+np.log(weights[:, None]), axis=0)
    assert copula.shape == (192,)
    for bins in ('6', '12'):
        log_integrals = np.load(OUTPUT / f'gp_{bins}_comparison.npz')['log_integrals']
        score = logsumexp(log_integrals, axis=0)-np.log(len(log_integrals))
        difference = copula-score
        row = summary['copula_minus_gp_test_scores'][bins]
        np.testing.assert_allclose(row['total_difference'], difference.sum(), atol=1e-12)
        np.testing.assert_allclose(row['total_standard_error'], np.sqrt(192)*difference.std(ddof=1), atol=1e-12)
        assert row['posterior_mcse'] > 0


def test_functional_intervals_and_scores_retain_misses():
    summary = read('summary.json')
    weights = np.load(OUTPUT / 'copula_weights.npz')['weights'].ravel()
    values = np.load(OUTPUT / 'copula_recovery.npz')['features']
    from binned_gp_benchmark import FEATURE_NAMES
    for i, name in enumerate(FEATURE_NAMES):
        row = summary['copula']['functionals']['functionals'][name]
        quantiles = weighted_quantiles(values[:, i], weights, [.05, .5, .95])
        np.testing.assert_allclose(row['quantiles'], quantiles, atol=1e-14)
        np.testing.assert_allclose(row['interval_score_90'], interval_score(quantiles[0], quantiles[2], row['truth']))
        assert row['truth_in_interval'] == (quantiles[0] <= row['truth'] <= quantiles[2])


def test_frozen_precision_and_sampling_requirements_are_met():
    summary = read('partition_refinement/summary.json')
    assert not read('summary.json')['decision']['recovery_precision_passed']
    assert not summary['original_comparison_decision']['eligible_for_scientific_comparison']
    assert not read('recovery_refinement/summary.json')['decision']['recovery_precision_passed']
    assert all(summary['decision'].values())
    numerical = summary['numerical_partition_refined']
    assert max(map(abs, numerical['posterior_transport_js_changes'].values())) < .005
    assert max(map(abs, numerical['truth_normalization_errors'])) < .0001
    assert abs(numerical['truth_alpha_error']) < .0001
    for name in ('truth_quadrature_js_changes', 'partition_js_changes'):
        assert all(max(map(abs, group.values())) < .005 for group in numerical[name].values())
    assert max(map(abs, summary['numerical_prediction']['paired_total_score_changes'].values())) < .25
    costs = summary['sampling_costs']
    assert len(costs['all_gp_selected_attempts']) == 4
    assert sum(row['accepted'] for row in costs['all_gp_selected_attempts'].values()) == 2
    assert all(row['peak_bytes'] <= 2*1024**3 for row in summary['evaluation_costs'].values())
    assert all(row['peak_bytes'] <= 2*1024**3 for row in summary['refinement_costs'].values())
    assert all(row['peak_bytes'] <= 2*1024**3 for row in summary['partition_refinement_costs'].values())
