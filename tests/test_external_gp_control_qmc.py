"""Check the integration amendment and retained pilot without rerunning NUTS."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_selected_mixture import selected_model
from correct_selected_posterior import sha256
from pilot_external_gp_control_qmc import integration_summary, integrate_states
from run_lisa_production import verify_product
from selected_event_qmc import selected_event_cloud

DIRECTORY = ROOT / 'results/analysis/external_gp_control_qmc_pilot'


@pytest.fixture(scope='module', autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


@pytest.mark.parametrize('name', ['preparation', 'pilot'])
def test_all_sources_and_products_are_frozen(name):
    report = verify_product(DIRECTORY / f'{name}.npz')
    for path, expected in report['source_sha256'].items():
        assert sha256(ROOT / path) == expected
    for path, expected in report.get('product_sha256', {}).items():
        assert sha256(DIRECTORY / path) == expected
    assert not report['eligible_for_scientific_comparisons']
    assert not any('test_catalog' in path or path.endswith('/truth.npz') for path in report['source_sha256'])


def test_integration_amendment_preserves_initialization_and_sampling_settings():
    config = json.loads((ROOT / 'configs/external_gp_control_qmc_pilot.json').read_text())
    previous = json.loads((ROOT / 'configs/external_gp_control_copula_pilot.json').read_text())
    for name in ['seed', 'warmup', 'retained', 'target_acceptance', 'maximum_tree_depth',
                 'full_mass', 'maximum_seconds', 'maximum_resident_gib', 'cpu_threads',
                 'maximum_directional_derivative_error', 'maximum_precision_log_target_difference']:
        assert config[name] == previous[name]
    old = np.load(ROOT / config['previous_pilot'])
    prepared = np.load(DIRECTORY / 'preparation.npz')
    pilot = np.load(DIRECTORY / 'pilot.npz')
    np.testing.assert_array_equal(prepared['coordinate'], np.vstack([old['initial_coordinate'], old['warmup_coordinate']]))
    np.testing.assert_array_equal(pilot['initial_coordinate'], old['initial_coordinate'])
    assert config['primary_power'] == 9 and config['verification_power'] == 11
    assert config['primary_scramble_seed'] == 848101 and config['verification_scramble_seed'] == 848102
    catalog = np.load(ROOT / 'results/analysis/external_gp_control_catalog/train_catalog.npz')
    for label in ['primary', 'verification']:
        cloud = np.load(DIRECTORY / f'{label}_events.npz')
        np.testing.assert_array_equal(cloud['observed'], catalog['observed'])
        np.testing.assert_array_equal(cloud['sigma'], catalog['sigma'])
        repeated = selected_event_cloud(catalog['observed'][:2], catalog['sigma'],
            config[f'{label}_power'], config[f'{label}_scramble_seed'])
        np.testing.assert_array_equal(cloud['data'][:2], repeated['data'])


def test_preparation_records_numerical_agreement_and_paired_cost():
    report = verify_product(DIRECTORY / 'preparation.npz')
    arrays = np.load(DIRECTORY / 'preparation.npz')
    difference = (arrays['log_integrals'][:, 1:]-arrays['log_integrals'][:, :1]).sum(-1)
    np.testing.assert_allclose(difference.std(0, ddof=1), report['warmup_log_likelihood_difference_sd'], atol=0, rtol=0)
    elapsed = arrays['gradient_seconds']
    assert np.isfinite(elapsed).all() and (elapsed > 0).all()
    assert report['median_paired_gradient_speed_ratio'] == np.median(elapsed[:, 1]/elapsed[:, 0])
    error = abs(arrays['core_log_integrals']-arrays['log_integrals'][arrays['timing_indices'], 0]).sum(-1).max()
    assert error == report['maximum_numpy_core_event_log_likelihood_difference']
    assert error < report['configuration']['maximum_kernel_log_likelihood_difference']


def test_pilot_checks_every_retained_state_and_uses_conservative_readiness():
    report = verify_product(DIRECTORY / 'pilot.npz')
    arrays = np.load(DIRECTORY / 'pilot.npz')
    protocol = json.loads((ROOT / report['configuration']['protocol']).read_text())
    assert arrays['coordinate'].shape == (report['samples_completed'], 20)
    assert arrays['warmup_coordinate'].shape == (report['warmup_completed'], 20)
    assert len(arrays['all_log_integrals']) == report['integration_draws_checked']
    if report['integration_draws_checked']:
        summaries, weights = integration_summary(arrays['all_log_integrals'], protocol)
        assert summaries == report['integration_summaries']
        np.testing.assert_array_equal(weights, arrays['correction_weights'])
    integration_ok = (report['integration_status'] == 'complete'
        and report['integration_draws_checked'] == report['configuration']['retained']
        and len(report['integration_summaries']) == 2
        and all(item['passed'] for item in report['integration_summaries']))
    assert report['event_integration_passed'] == integration_ok
    verification = report['verification']
    if verification['draws_checked']:
        assert abs(arrays['derivative_automatic']-arrays['derivative_numeric']).max() == verification['maximum_directional_derivative_error']
    if report['integration_status'] == 'complete' and verification['status'] == 'complete' and len(arrays['coordinate']):
        indices = np.unique(np.linspace(0, len(arrays['coordinate'])-1,
            min(report['configuration']['derivative_check_draws'], len(arrays['coordinate'])), dtype=int))
        np.testing.assert_array_equal(arrays['checked_coordinate'][1:], arrays['coordinate'][indices])
        expected = (arrays['all_log_integrals'][indices, 1]-arrays['all_log_integrals'][indices, 0]).sum(-1)
        np.testing.assert_allclose(expected, arrays['event_log_likelihood_difference'][1:], rtol=0, atol=1e-7)
    ready = (report['status'] == 'complete' and report['divergences'] == []
        and integration_ok and report['resources_passed']
        and verification['derivatives_passed'] and verification['precision_passed'])
    assert report['ready_for_production_sampling'] == ready
    assert not report['eligible_for_scientific_comparisons']


def test_correction_checks_do_not_accept_large_target_changes_or_missing_tails():
    protocol = json.loads((ROOT / 'configs/external_benchmark.json').read_text())
    rng = np.random.default_rng(999)
    differences = rng.normal(0, .01, 100)
    values = np.stack([np.zeros((100, 3)), np.repeat(differences[:, None]/3, 3, axis=1)], axis=1)
    result, weights = integration_summary(values, protocol)
    assert result[0]['passed']
    assert weights.sum() == pytest.approx(1.)
    values[:, 1] *= 1000
    result, _ = integration_summary(values, protocol)
    assert not result[0]['passed']
    result, _ = integration_summary(values[:2], protocol)
    assert result[0]['pareto_k'] is None and not result[0]['passed']


def test_empty_or_expired_integration_is_explicit():
    import time
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    cloud = np.ones((2, 4, 4))*.5
    cloud[..., 3] = 1
    values, status = integrate_states(model, np.empty((0, 20)), [cloud, cloud],
        {'maximum_resident_gib': 2}, time.perf_counter()+5)
    assert values.shape == (0, 2, 2) and status == 'complete'
    values, status = integrate_states(model, np.zeros((1, 20)), [cloud, cloud],
        {'maximum_resident_gib': 2}, time.perf_counter()-1)
    assert values.shape == (0, 2, 2) and status == 'time_limit'
