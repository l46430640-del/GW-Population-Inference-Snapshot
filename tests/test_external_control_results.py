"""Verify the retained control posterior evidence without repeating sampling."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from correct_selected_posterior import sha256
from run_external_control_production import effective_protocol
from run_external_gp_selected import production_decision
from run_lisa_production import mixing_summary, verify_product

OUTPUT = ROOT / 'results/analysis/external_control_production'


@pytest.fixture(scope='module', autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


@pytest.mark.parametrize('label', ['gp_6', 'gp_12', 'copula'])
def test_sources_inputs_and_all_chains_are_consistent(label):
    directory = OUTPUT / label
    target = json.loads((directory / 'target.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    for path, expected in target['source_sha256'].items():
        assert sha256(ROOT / path) == expected
        assert 'test_catalog' not in path and not path.endswith('/truth.npz')
    assert target['effective_protocol'] == effective_protocol(target['configuration'],
        json.loads((ROOT / target['configuration']['protocol']).read_text()))
    assert summary['target_product_sha256'] == sha256(directory / 'target.json')
    assert not summary['comparison_results_available']
    if label != 'copula':
        assert target['inputs_product_sha256'] == sha256(directory / 'inputs.npz')
    for path, expected in summary['chain_reports_sha256'].items():
        assert sha256(directory / path) == expected
        report = verify_product((directory / path).with_suffix('.npz'))
        assert report['target_product_sha256'] == summary['target_product_sha256']
        arrays = np.load((directory / path).with_suffix('.npz'))
        assert arrays['coordinate'].shape[0] == report['samples_completed']
        assert arrays['warmup_coordinate'].shape[0] == report['warmup_completed']
        assert np.isfinite(arrays['coordinate']).all()
        assert arrays['diagnostic_values'].shape[0] == report['samples_completed']
        if label == 'copula':
            index = target['effective_protocol']['sampling']['chain_seeds'].index(report['seed'])
            np.testing.assert_array_equal(arrays['initial_coordinate'], target['initial_coordinates'][index])
        else:
            index = target['effective_protocol']['sampling']['chain_seeds'].index(report['seed'])
            np.testing.assert_allclose(np.exp(arrays['initial_coordinate'][:4]),
                target['initial_amplitude_and_lengths'][index], rtol=1e-14)


@pytest.mark.parametrize('label', ['gp_6', 'gp_12', 'copula'])
def test_mixing_summaries_are_reproducible(label):
    directory = OUTPUT / label
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['mixing'] is None:
        assert not summary['decision']['posterior_accepted']
        return
    products = [dict(np.load(directory / name.replace('.json', '.npz'))) for name in summary['chain_reports_sha256']]
    length = min(len(p['coordinate']) for p in products)
    values = np.stack([p['diagnostic_values'][:length] for p in products])
    mixing = mixing_summary(products[0]['diagnostic_names'].tolist(), values)
    for key in ['maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess']:
        assert mixing[key] == pytest.approx(summary['mixing'][key], rel=1e-12)


@pytest.mark.parametrize('bins', [6, 12])
def test_gp_acceptance_is_conservative_and_jitter_is_checked_in_sampled_coordinates(bins):
    directory = OUTPUT / f'gp_{bins}'
    target = json.loads((directory / 'target.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    reports = [json.loads((directory / name).read_text()) for name in summary['chain_reports_sha256']]
    decision = production_decision(target['effective_protocol'], reports, summary['mixing'])
    differences = []
    for name, report in zip(summary['chain_reports_sha256'], reports):
        data = np.load(directory / name.replace('.json', '.npz'))
        differences.extend(data['covariance_jitter_log_target_difference'])
        if report['samples_completed']:
            assert np.std(data['covariance_jitter_log_target_difference']) == report['covariance_jitter_log_target_difference_sd']
            assert abs(data['directional_derivative_automatic']-data['directional_derivative_finite_difference']).max() == report['coordinate_verification']['maximum_directional_derivative_error']
    if summary['pooled_jitter_log_target_difference_sd'] is None:
        decision['numerical_passed'] = decision['posterior_accepted'] = False
    else:
        assert np.std(differences) == summary['pooled_jitter_log_target_difference_sd']
        if np.std(differences) > target['effective_protocol']['evaluation']['maximum_log_likelihood_difference_sd']:
            decision['numerical_passed'] = decision['posterior_accepted'] = False
    assert decision == summary['decision']


def test_copula_correction_preserves_chain_order_and_checks_independent_integration():
    directory = OUTPUT / 'copula'
    summary = json.loads((directory / 'summary.json').read_text())
    decision = summary['decision']
    assert decision['posterior_accepted'] == all(v for k, v in decision.items() if k != 'posterior_accepted')
    if not decision['complete']:
        assert not decision['posterior_accepted']
        return
    data = np.load(directory / 'posterior.npz')
    assert sha256(directory / 'posterior.npz') == summary['posterior_product_sha256']
    assert data['coordinate'].shape == (4, 800, 20)
    assert data['features'].shape == (4, 800, 29)
    for i, name in enumerate(summary['chain_reports_sha256']):
        original = np.load(directory / name.replace('.json', '.npz'))
        np.testing.assert_array_equal(data['coordinate'][i], original['coordinate'])
        np.testing.assert_array_equal(data['features'][i], original['features'])
        checked = np.unique(np.linspace(0, 799, 3, dtype=int))
        np.testing.assert_array_equal(original['checked_coordinate'][1:], original['coordinate'][checked])
        expected_change = (data['log_integrals'][i*800+checked, 1]-data['log_integrals'][i*800+checked, 0]).sum(-1)
        np.testing.assert_allclose(original['event_log_likelihood_difference'][1:], expected_change, atol=1e-7, rtol=0)
    np.testing.assert_allclose(data['features'][:, :, 21:].sum(-1), 1., atol=1e-14)
    ratio = (data['log_integrals'][:, 1]-data['log_integrals'][:, 0]).sum(-1).reshape(4, 800)
    np.testing.assert_array_equal(ratio, data['log_ratio'])
    expected = np.exp(ratio-ratio.max())
    expected /= expected.sum()
    np.testing.assert_allclose(expected, data['weights'], rtol=1e-14)
    assert summary['correction']['weight_ess_fraction'] == pytest.approx(1/np.sum(expected**2)/expected.size)
    indices = data['verification_indices']
    difference = (data['verification_log_integrals']-data['log_integrals'][indices, 1]).sum(-1)
    for name, value in summary['independent_event_log_likelihood_difference_sd'].items():
        group = data['verification_group_'+name]
        assert np.std(difference[np.searchsorted(indices, group)], ddof=1) == value
    config = summary['configuration']
    if decision['posterior_accepted']:
        statistics = summary['corrected_statistics']
        assert statistics['maximum_influence_rhat'] <= 1.01
        assert statistics['minimum_mean_precision_equivalent_draws'] >= config['minimum_corrected_precision_equivalent_draws']
        assert statistics['minimum_cdf_precision_equivalent_draws'] >= config['minimum_corrected_precision_equivalent_draws']


def test_cost_record_and_allocation_are_preserved():
    config = json.loads((ROOT / 'configs/external_control_production.json').read_text())
    jobs = [json.loads(p.read_text()) for p in OUTPUT.glob('job_cost_*.json')]
    assert jobs and all(job['configuration_sha256'] == sha256(ROOT / 'configs/external_control_production.json') for job in jobs)
    total = sum(job['process_elapsed_seconds'] for job in jobs)
    if all(job['status'] == 'complete' for job in jobs):
        assert total <= config['resource_amendment']['maximum_seconds_control_production']
    assert config['resource_amendment']['prior_study_budget_charge_seconds'] + total <= 12*3600
