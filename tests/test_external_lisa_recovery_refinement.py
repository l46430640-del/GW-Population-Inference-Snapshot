import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from binned_gp_benchmark import FEATURE_NAMES
from compare_external_selected import recovery_gradients
from compare_lisa_posterior import moment_check
from correct_selected_posterior import sha256, weighted_quantiles
from external_selected_evaluation import interval_score

PARENT = ROOT / 'results/analysis/external_lisa_comparison_double'
OUTPUT = ROOT / 'results/analysis/external_lisa_recovery_refinement'


def reports():
    return [json.loads((directory / 'summary.json').read_text()) for directory in (PARENT, OUTPUT)]


def test_refinement_preserves_failed_comparison_and_unchanged_prediction_gp_results():
    old, new = reports()
    assert not old['decision']['recovery_precision_passed']
    assert not old['decision']['eligible_for_scientific_comparison']
    assert new['recovery_refinement']['population_transport_power'] == 17
    assert old['configuration']['verification_population_transport_power'] == 16
    for key in ('gp', 'copula_minus_gp_test_scores', 'numerical_prediction', 'copula_target_checks'):
        assert new[key] == old[key]
    assert new['copula']['prediction_precision'] == old['copula']['prediction_precision']
    assert new['numerical_recovery']['parent_failed_transport_and_truth_changes'] == old['numerical_recovery']['copula_transport_and_truth_changes']


def test_all_sources_reused_products_and_new_products_are_frozen():
    _, new = reports()
    for name, digest in new['source_sha256'].items():
        assert sha256(ROOT / name) == digest
    for name, digest in new['reused_parent_products'].items():
        assert sha256(PARENT / name) == digest
    for name, digest in new['product_sha256'].items():
        path = OUTPUT / name
        assert sha256(path) == digest
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['source_sha256'] == new['source_sha256']
        assert metadata['product_sha256'] == digest


def test_independent_refinement_truth_and_partition_changes_reconstruct():
    _, new = reports()
    truth, coarse_truth = [dict(np.load(PARENT / name)) for name in ('truth.npz', 'truth_coarse.npz')]
    old_population = dict(np.load(PARENT / 'copula_recovery.npz'))
    population = dict(np.load(OUTPUT / 'copula_recovery.npz'))
    distances, _ = recovery_gradients(population, truth, 96, (48, 96))
    previous, _ = recovery_gradients(old_population, truth, 96, (48, 96))
    coarse, _ = recovery_gradients(population, coarse_truth, 96, (48, 96))
    for label, distance in distances.items():
        assert new['copula']['joint'][label]['distance'] == distance
        assert new['numerical_recovery']['copula_transport_and_truth_changes'][label] == pytest.approx(distance-previous[label])
        assert new['numerical_recovery']['truth_integration_changes'][label] == pytest.approx(distance-coarse[label])
    for kind in ('intrinsic', 'detected'):
        assert population[kind].sum() == pytest.approx(1., abs=1e-10)
        assert new['numerical_recovery']['partition_changes']['copula'][kind] == pytest.approx(distances[kind+'_96']-distances[kind+'_48'])
    changes = list(new['numerical_recovery']['copula_transport_and_truth_changes'].values())
    changes.extend(new['numerical_recovery']['truth_integration_changes'].values())
    for name in ('gp_cell_integration_changes', 'partition_changes'):
        changes.extend(value for group in new['numerical_recovery'][name].values() for value in group.values())
    assert new['decision']['recovery_precision_passed'] == (max(map(abs, changes)) < .005)


def test_reported_recovery_mcse_and_all_copula_intervals_use_finer_transport():
    _, new = reports()
    weights = np.load(ROOT / 'results/analysis/lisa_refinement/weights.npz')['weights']
    influences = np.load(OUTPUT / 'copula_influences.npz')
    for label, values in influences.items():
        expected = moment_check(values, weights.ravel(), weights.shape, new['configuration']['batch_lengths'])
        for key in ('mcse', 'influence_rhat'):
            assert new['copula']['joint'][label][key] == expected[key]
    features = np.load(OUTPUT / 'copula_recovery.npz')['features'].copy()
    posterior = np.load(ROOT / 'results/analysis/lisa_refinement/posterior.npz')
    features[:, 6] = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
    truth = np.load(PARENT / 'truth.npz')['features']
    for i, name in enumerate(FEATURE_NAMES):
        row = new['copula']['functionals']['functionals'][name]
        lo, median, hi = weighted_quantiles(features[:, i], weights.ravel())
        np.testing.assert_allclose(row['quantiles'], [lo, median, hi], atol=1e-14)
        assert row['interval_score_90'] == pytest.approx(interval_score(lo, hi, truth[i]))
        assert row['truth_in_interval'] == bool(lo <= truth[i] <= hi)


def test_final_acceptance_and_cost_include_every_failed_and_reused_stage():
    _, new = reports()
    initial = json.loads((ROOT / 'results/analysis/external_lisa_comparison/job_cost_000.json').read_text())
    parent = json.loads((PARENT / 'job_cost_000.json').read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(OUTPUT.glob('job_cost_*.json'))]
    seconds = sum(job['process_elapsed_seconds'] for job in jobs)
    all_seconds = seconds+initial['process_elapsed_seconds']+parent['process_elapsed_seconds']
    assert jobs[-1]['all_lisa_evaluation_attempts_seconds'] == pytest.approx(all_seconds)
    assert jobs[-1]['total_study_charge_seconds'] == pytest.approx(parent['total_study_charge_seconds']+seconds)
    assert jobs[-1]['remaining_study_seconds'] == pytest.approx(45000-jobs[-1]['total_study_charge_seconds'])
    assert seconds <= 600 and all_seconds <= 2100
    assert jobs[-1]['total_study_charge_seconds'] <= 45000
    assert all(job['runtime']['threads'] == 1 and job['runtime']['peak_working_set_bytes'] <= 2*1024**3 for job in jobs)
    assert new['decision']['eligible_for_scientific_comparison'] == all(v for k, v in new['decision'].items() if k != 'eligible_for_scientific_comparison')
    assert all(new['decision'].values())
