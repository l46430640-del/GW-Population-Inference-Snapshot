import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from refine_lisa_posterior import (
    checked_inputs, corrected_statistics, decision, influence_diagnostics, ratio_weights,
)
from correct_selected_posterior import sha256
from sample_lisa_transport import differentiable_log_target, lisa_features, sampling_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    old = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_num_threads(old[0])
    torch.set_default_dtype(old[1])


def config():
    return json.loads((ROOT / 'configs/lisa_refinement.json').read_text())


def test_ratio_weights_preserve_uniform_correction_and_shift_invariance():
    weights, report = ratio_weights(np.zeros((4, 100)))
    np.testing.assert_allclose(weights, 1/400)
    assert report['pareto_k'] == 0
    assert report['weight_ess_fraction'] == pytest.approx(1)
    rng = np.random.default_rng(42)
    log_ratio = rng.normal(scale=.1, size=(4, 500))
    first, _ = ratio_weights(log_ratio, .4)
    second, _ = ratio_weights(log_ratio + 900, .4)
    np.testing.assert_allclose(first, second, rtol=1e-12)
    assert first.sum() == pytest.approx(1)
    for bad in (np.zeros(10), np.full((4, 100), np.nan)):
        with pytest.raises(ValueError):
            ratio_weights(bad)


def test_corrected_mean_and_mcse_on_an_analytic_normal_tilt():
    values = np.random.default_rng(735200).normal(size=(4, 2000, 1))
    target_mean = .2
    weight, _ = ratio_weights(target_mean * values[:, :, 0] - target_mean**2/2)
    result = corrected_statistics(values, ['x'], weight, 1, config())
    row = result['functionals']['x']
    assert abs(row['mean'] - target_mean) < 4 * row['mean_mcse']
    assert row['sd'] == pytest.approx(1., abs=.06)
    rescaled = corrected_statistics(values, ['x'], 17 * weight, 1, config())
    assert rescaled['functionals']['x']['mean_mcse'] == pytest.approx(row['mean_mcse'], rel=1e-12)
    assert len(result['cdf_checks']) == 3
    assert all(r['estimated_cdf'] >= r['probability'] for r in result['cdf_checks'])


def test_influence_mcse_retains_serial_dependence():
    rng = np.random.default_rng(735201)
    independent = rng.normal(size=(4, 2000, 1))
    correlated = np.empty_like(independent)
    correlated[:, 0] = independent[:, 0]
    rho = .8
    for i in range(1, independent.shape[1]):
        correlated[:, i] = rho * correlated[:, i-1] + np.sqrt(1-rho*rho) * independent[:, i]
    first = influence_diagnostics(independent, [25, 50])
    second = influence_diagnostics(correlated, [25, 50])
    assert second['mcse'][0] > 2 * first['mcse'][0]
    expected = np.sqrt((1+rho)/(1-rho)/independent.size)
    assert second['mcse'][0] == pytest.approx(expected, rel=.35)
    assert np.all(second['mcse'] >= second['mcse_arviz'])
    assert np.all(second['mcse'] >= second['mcse_batches'].max(0))
    with pytest.raises(ValueError):
        influence_diagnostics(independent, [37])


def test_acceptance_requires_sampling_weights_mcse_and_independent_integration():
    settings = config()
    weight = {'pareto_k': .1, 'weight_ess_fraction': .98, 'maximum_weight': .001}
    stats = {'maximum_influence_rhat': 1.001, 'minimum_mean_precision_equivalent_draws': 600,
             'minimum_cdf_precision_equivalent_draws': 600}
    numerical = {'groups': {k: {'log_target_change_sd': .1}
                            for k in [*(f'chain_{i}' for i in range(4)), 'efficiency_extremes', 'largest_weights']},
                 'uniform_weight_ess_fraction': .98, 'uniform_mean_shift_in_sd': [.01],
                 'maximum_alpha_change_in_sd': .01}
    assert decision(settings, True, weight, stats, numerical)['eligible_for_scientific_summaries']
    assert not decision(settings, False, weight, stats, numerical)['eligible_for_scientific_summaries']
    for key, value in [('pareto_k', .8), ('weight_ess_fraction', .5), ('maximum_weight', .1)]:
        assert not decision(settings, True, dict(weight, **{key: value}), stats, numerical)['eligible_for_scientific_summaries']
    for key, value in [('maximum_influence_rhat', 1.02), ('minimum_mean_precision_equivalent_draws', 30),
                       ('minimum_cdf_precision_equivalent_draws', 30)]:
        assert not decision(settings, True, weight, dict(stats, **{key: value}), numerical)['eligible_for_scientific_summaries']
    for key, value in [('groups', {}), ('uniform_weight_ess_fraction', .5),
                       ('uniform_mean_shift_in_sd', [.01, np.nan]), ('maximum_alpha_change_in_sd', .2)]:
        assert not decision(settings, True, weight, stats, dict(numerical, **{key: value}))['eligible_for_scientific_summaries']


def test_refinement_inputs_require_the_converged_production_sources():
    settings = config()
    target, report, products, _, original, fine, sources = checked_inputs(settings, ROOT / 'configs/lisa_refinement.json')
    assert report['decision']['mixing_passed']
    assert not report['decision']['integration_passed']
    assert np.stack([p['latent'] for p in products]).shape == (4, 500, 20)
    assert fine.shape == (int(original['detected'].sum()), 2048, 4)
    assert target['configuration']['fine_selection_points'] == settings['primary_selection_points']
    assert 'scripts/refine_lisa_posterior.py' in sources


@pytest.fixture(scope='module')
def retained():
    settings = config()
    output = ROOT / settings['output']
    report = json.loads((output / 'summary.json').read_text())
    return settings, output, report, dict(np.load(output / 'posterior.npz')), dict(np.load(output / 'weights.npz'))


def test_retained_hashes_and_chain_order(retained):
    settings, output, report, posterior, weights = retained
    target, _, products, _, _, _, sources = checked_inputs(settings, ROOT / 'configs/lisa_refinement.json')
    assert report['source_sha256'] == sources
    assert report['configuration'] == settings
    for name, expected in report['product_sha256'].items():
        assert sha256(output / name) == expected
    for stem in ('posterior', 'verification_catalog'):
        metadata = json.loads((output / (stem + '.json')).read_text())
        assert metadata['source_sha256'] == sources
        assert metadata['configuration'] == settings
        assert metadata['draw_product_sha256'] == sha256(output / (stem + '.npz'))
    np.testing.assert_array_equal(posterior['latent'], np.stack([p['latent'] for p in products]))
    np.testing.assert_array_equal(posterior['log_coarse'], np.stack([p['log_target'] for p in products]))
    np.testing.assert_array_equal(weights['log_ratio'], posterior['log_primary'] - posterior['log_coarse'])
    assert posterior['diagnostic_values'].shape == (4, 500, 48)
    assert posterior['features'].shape == (4, 500, 12)
    np.testing.assert_array_equal(posterior['diagnostic_values'][:, :, :12], posterior['features'])
    np.testing.assert_array_equal(posterior['diagnostic_values'][:, :, -1], posterior['log_primary'])
    cache = np.load(ROOT / settings['source'] / 'numerical.npz')
    expected = posterior['log_coarse'].ravel()[cache['indices']] + cache['event_change'] + cache['selection_change']
    np.testing.assert_array_equal(posterior['log_primary'].ravel()[cache['indices']], expected)
    metadata = json.loads((output / 'posterior.json').read_text())
    assert metadata['reused_target_evaluations'] == len(cache['indices'])
    assert target['configuration']['fine_selection_points'] == settings['primary_selection_points']


def test_retained_statistics_and_decision_reconstruct(retained):
    settings, _, report, posterior, saved = retained
    production = json.loads((ROOT / settings['source'] / 'summary.json').read_text())
    weights, check = ratio_weights(saved['log_ratio'], production['mixing']['minimum_bulk_ess'] / saved['weights'].size)
    np.testing.assert_array_equal(weights, saved['weights'])
    assert check == report['weight_diagnostics']
    statistics = corrected_statistics(posterior['diagnostic_values'], posterior['diagnostic_names'].tolist(),
                                      weights, len(posterior['feature_names']), settings)
    assert statistics == report['statistics']
    converged = all(production['decision'][key] for key in (
        'complete', 'divergences_passed', 'mixing_passed', 'functional_quadrature_passed', 'precision_passed'))
    assert decision(settings, converged, check, statistics, report['verification']) == report['decision']


def test_retained_verification_indices_weights_and_changes(retained):
    settings, output, report, posterior, weights = retained
    check = np.load(output / 'verification.npz')
    indices, uniform = check['indices'], check['uniform_indices']
    alpha_column = posterior['feature_names'].tolist().index('alpha')
    alpha = posterior['features'][:, :, alpha_column]
    expected_uniform = np.concatenate([c * 500 + np.linspace(0, 499, settings['verification_draws_per_chain'], dtype=int)
                                       for c in range(4)])
    np.testing.assert_array_equal(uniform, expected_uniform)
    previous = np.load(ROOT / settings['source'] / 'numerical.npz')['group_efficiency_extremes']
    tail = settings['verification_extremes_per_chain']
    expected_extremes = np.unique(np.r_[previous, np.concatenate([
        c*500 + np.r_[np.argsort(a)[:tail], np.argsort(a)[-tail:]] for c, a in enumerate(alpha)])])
    np.testing.assert_array_equal(check['group_efficiency_extremes'], expected_extremes)
    np.testing.assert_array_equal(check['group_largest_weights'],
                                  np.argsort(weights['weights'].ravel())[-settings['verification_largest_weights']:])
    difference = check['log_verification'] - posterior['log_primary'].ravel()[indices]
    np.testing.assert_array_equal(check['log_target_change'], difference)
    for name, row in report['verification']['groups'].items():
        selected = check['group_' + name]
        offsets = np.searchsorted(indices, selected)
        assert row['draws'] == len(selected)
        assert row['log_target_change_sd'] == pytest.approx(difference[offsets].std())
        np.testing.assert_allclose(row['log_target_change_05_50_95'], np.quantile(difference[offsets], [.05, .5, .95]))
    first = weights['weights'].ravel()[uniform]
    first /= first.sum()
    log_next = np.log(first) + difference[np.searchsorted(indices, uniform)]
    second = np.exp(log_next - logsumexp(log_next))
    np.testing.assert_array_equal(check['uniform_primary_weights'], first)
    np.testing.assert_array_equal(check['uniform_verification_weights'], second)
    assert report['verification']['uniform_weight_ess_fraction'] == pytest.approx(1 / (second @ second) / len(second))
    values = posterior['features'].reshape(-1, 12)
    mean = weights['weights'].ravel() @ values
    sd = np.sqrt(weights['weights'].ravel() @ (values-mean)**2)
    next_values = values[uniform].copy()
    next_values[:, alpha_column] = check['alpha_verification'][np.searchsorted(indices, uniform)]
    shift = abs(second @ next_values - first @ values[uniform]) / sd
    np.testing.assert_allclose(report['verification']['uniform_mean_shift_in_sd'], shift)
    maximum_change = abs(check['alpha_verification'] - alpha.ravel()[indices]).max() / sd[alpha_column]
    assert report['verification']['maximum_alpha_change_in_sd'] == pytest.approx(maximum_change)


def test_retained_event_samples_use_same_observations_and_sufficient_proposal_ess(retained):
    settings, output, _, _, _ = retained
    *_, original, _, _ = checked_inputs(settings, ROOT / 'configs/lisa_refinement.json')
    events = np.load(output / 'verification_catalog.npz')
    for name in ('measurement', 'rho_observed', 'detected'):
        np.testing.assert_array_equal(events[name], original[name])
    data = events['data']
    assert data.shape == (135, settings['verification_event_samples'], 4)
    assert data.dtype == np.float64
    assert np.isfinite(data).all()
    assert data[:, :, :3].min() >= 0 and data[:, :, :3].max() <= 1
    np.testing.assert_array_equal(data[:, :, 3], 1)
    metadata = json.loads((output / 'verification_catalog.json').read_text())
    assert len(metadata['event_diagnostics']) == 135
    assert min(r['proposal_ess'] for r in metadata['event_diagnostics']) >= max(1024, 4 * settings['verification_event_samples'])


def test_retained_primary_and_verification_targets_recompute(retained):
    settings, output, _, posterior, _ = retained
    _, _, _, model_config, _, fine, _ = checked_inputs(settings, ROOT / 'configs/lisa_refinement.json')
    model = sampling_model(model_config)
    model.lambda_det_mc_samples = settings['primary_selection_points']
    model.selection_mc_seed = settings['primary_selection_seed']
    data = torch.tensor(fine, dtype=torch.float64)
    latent = posterior['latent'].reshape(-1, 20)
    with torch.no_grad():
        for index in [0, 1, len(latent)-1]:
            point = torch.tensor(latent[index], dtype=torch.float64)
            value = float(differentiable_log_target(model, point, data))
            assert value == pytest.approx(posterior['log_primary'].ravel()[index], rel=0, abs=1e-9)
            names, features = lisa_features(model, point[None], settings['feature_transport_power'])
            assert names == posterior['feature_names'].tolist()
            np.testing.assert_allclose(features[0], posterior['features'].reshape(-1, 12)[index], rtol=0, atol=1e-12)
        model.lambda_det_mc_samples = settings['verification_selection_points']
        model.selection_mc_seed = settings['verification_selection_seed']
        data = torch.tensor(np.load(output / 'verification_catalog.npz')['data'], dtype=torch.float64)
        check = np.load(output / 'verification.npz')
        for offset in [0, len(check['indices'])-1]:
            point = torch.tensor(latent[check['indices'][offset]], dtype=torch.float64)
            value = float(differentiable_log_target(model, point, data))
            assert value == pytest.approx(check['log_verification'][offset], rel=0, abs=1e-9)
            params = model._unflatten_latent_vector(point, apply_numerical_limits=False)
            assert float(model._selection_fraction_for_params(params)) == pytest.approx(check['alpha_verification'][offset], rel=0, abs=1e-12)
