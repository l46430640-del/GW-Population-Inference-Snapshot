import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import gammainc
from scipy.spatial.distance import jensenshannon
from scipy.stats import gamma
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_lisa_posterior import (
    baseline_draws, comparison_decision, distance_check, exact_truth_probabilities, moment_check, numerical_summary,
    prediction_product, rate_summary, selection_fractions, test_observations as make_observations,
    verified_sources,
)
from correct_selected_posterior import sha256
from predict_corrected_posterior import weighted_prediction
from summarize_selected_mixture import paired_scores
from sample_lisa_transport import sampling_model

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
    return json.loads((ROOT / 'configs/lisa_scientific_comparison.json').read_text())


def test_exact_truth_probabilities_normalize_and_have_consistent_marginals():
    cells, marginals = exact_truth_probabilities(12)
    assert cells.shape == (12**3,)
    assert cells.min() >= 0 and cells.sum() == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(marginals.sum(1), 1, atol=1e-14)
    for axis in range(3):
        reduced = cells.reshape(12, 12, 12).sum(tuple(d for d in range(3) if d != axis))
        np.testing.assert_allclose(reduced, marginals[axis], rtol=0, atol=1e-15)


def test_gamma_rate_mixture_matches_constant_efficiency_and_cdf():
    result = rate_summary(np.full(20, .4), np.ones(20), 135, 160)
    np.testing.assert_allclose(result['quantiles_05_50_95'], gamma.ppf([.05, .5, .95], 135, scale=2.5))
    assert result['mean'] == pytest.approx(135/.4)
    assert result['sd'] == pytest.approx(np.sqrt(135)/.4)
    alpha, weights = np.array([.2, .5, .8]), np.array([.2, .3, .5])
    result = rate_summary(alpha, weights, 135, 160)
    np.testing.assert_allclose([weights @ gammainc(135, alpha*r) for r in result['quantiles_05_50_95']], [.05, .5, .95])
    assert result['cdf_at_truth']['mean'] == pytest.approx(weights @ gammainc(135, alpha*160))
    with pytest.raises(ValueError):
        rate_summary(np.array([0., 1.]), np.ones(2), 135, 160)


def test_distance_mcse_matches_directional_derivative():
    rng = np.random.default_rng(736999)
    cells = rng.dirichlet(np.ones(5), size=100)
    truth = np.array([.05, .1, .15, .25, .45])
    weights = np.ones(100)/100
    result = distance_check(cells, truth, weights)
    mean = cells.mean(0)
    epsilon = 1e-5
    directions = cells-mean
    finite = np.array([(jensenshannon(mean+epsilon*d, truth, base=2)
                       - jensenshannon(mean-epsilon*d, truth, base=2))/(2*epsilon) for d in directions])
    assert result['mcse'] == pytest.approx(finite.std(ddof=1)/10, rel=1e-6)


def test_serial_mean_error_preserves_chain_shape():
    rng = np.random.default_rng(736998)
    draws = rng.normal(size=(4, 500))
    for i in range(1, 500):
        draws[:, i] += .8 * draws[:, i-1]
    flat, weight = draws.ravel(), np.ones(draws.size)
    correlated = moment_check(flat, weight, (4, 500))
    independent = moment_check(flat, weight)
    assert correlated['mcse'] > 2 * independent['mcse']
    assert correlated['mean'] == pytest.approx(independent['mean'])


def test_total_log_prediction_influence_matches_weight_perturbations():
    rng = np.random.default_rng(736997)
    log_values = rng.normal(size=(100, 8))
    weight = np.ones(100)/100
    score = weighted_prediction(log_values, weight)
    influence = np.exp(log_values-score).sum(1)-8
    epsilon = 1e-5
    for index in [0, 11, 99]:
        direction = -weight.copy()
        direction[index] += 1
        derivative = (weighted_prediction(log_values, weight+epsilon*direction).sum()
                      - weighted_prediction(log_values, weight-epsilon*direction).sum())/(2*epsilon)
        assert derivative == pytest.approx(influence[index], rel=1e-6, abs=1e-8)


def test_detected_test_catalog_is_independent_and_deterministic():
    settings = dict(config(), test_events=8)
    first, second = make_observations(settings), make_observations(settings)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert (first['rho_observed'] > 8).all()
    training = np.load(ROOT / settings['baselines'][1] / 'catalog.npz')
    assert not np.array_equal(first['measurement'], training['measurement'][:8])


def test_uniform_event_prediction_divides_by_efficiency_once():
    settings = config()
    original = json.loads((ROOT / settings['baselines'][1] / 'summary.json').read_text())['configuration']
    model = sampling_model(original)
    latent = np.zeros((2, 20))
    offset = 0
    for name, size in model.dims.items():
        if name == 'k_marginals':
            latent[:, offset:offset+size] = np.log(1-.0001)
        offset += size
    points = np.random.default_rng(19).uniform(.01, .99, size=(5, 64, 3))
    data = np.concatenate([points, np.ones((5, 64, 1))], -1)
    alpha = np.array([.4, .7])
    prediction = prediction_product(model, latent, alpha, data)
    np.testing.assert_allclose(prediction['log_integrals'], np.broadcast_to(-np.log(alpha[:, None]), (2, 5)), atol=1e-12)
    before = model.lambda_det_mc_samples, model.selection_mc_seed
    values = selection_fractions(model, latent, 128, 997)
    assert (model.lambda_det_mc_samples, model.selection_mc_seed) == before
    assert np.all((values > 0) & (values < 1))


def test_acceptance_retains_each_numerical_and_mixing_requirement():
    labels = ['vi_' + relative.rsplit('_s', 1)[1] for relative in config()['baselines']]
    numerical = {label: {'intrinsic_distance_change': .001, 'detected_distance_change': .001,
                         'hidden_change': .001, 'prediction_rmse_change': .001} for label in ['nuts', *labels]}
    comparisons = {label: {'fine_paired_score_change': .01} for label in labels}
    assert comparison_decision(config(), numerical, comparisons, 1.001)['eligible_for_scientific_comparison']
    for key in numerical['nuts']:
        changed = dict(numerical, nuts=dict(numerical['nuts'], **{key: 1.}))
        assert not comparison_decision(config(), changed, comparisons, 1.001)['eligible_for_scientific_comparison']
    changed = dict(comparisons, vi_91001={'fine_paired_score_change': 1.})
    assert not comparison_decision(config(), numerical, changed, 1.001)['eligible_for_scientific_comparison']
    assert not comparison_decision(config(), {}, {}, 1.001)['eligible_for_scientific_comparison']
    assert not comparison_decision(config(), numerical, comparisons, np.nan)['eligible_for_scientific_comparison']


def test_inputs_require_the_accepted_frozen_posterior():
    accepted, sources = verified_sources(config(), ROOT / 'configs/lisa_scientific_comparison.json')
    assert accepted['decision']['eligible_for_scientific_summaries']
    assert 'scripts/compare_lisa_posterior.py' in sources
    assert 'src/gwpopulation/core.py' in sources


@pytest.fixture(scope='module')
def retained():
    settings = config()
    output = ROOT / settings['output']
    report = json.loads((output / 'summary.json').read_text())
    arrays = dict(np.load(output / 'derived.npz'))
    return settings, output, report, arrays


def test_retained_products_and_sources_match(retained):
    settings, output, report, _ = retained
    _, sources = verified_sources(settings, ROOT / 'configs/lisa_scientific_comparison.json')
    assert report['configuration'] == settings and report['source_sha256'] == sources
    for name, expected in report['product_sha256'].items():
        path = output / name
        assert sha256(path) == expected
        if name != 'derived.npz':
            metadata = json.loads(path.with_suffix('.json').read_text())
            assert metadata['product_sha256'] == expected
            assert all(metadata['source_sha256'][k] == v for k, v in sources.items())


def test_retained_test_data_and_proposal_precision(retained):
    settings, output, _, _ = retained
    observations = np.load(output / 'test_observations.npz')
    expected = make_observations(settings)
    for key, value in expected.items():
        np.testing.assert_array_equal(observations[key], value)
    for label, count in [('events', settings['event_samples']), ('events_fine', settings['fine_event_samples'])]:
        product = np.load(output / (label+'.npz'))
        assert product['data'].shape == (settings['test_events'], count, 4)
        assert product['data'].dtype == np.float64
        assert np.all((product['data'][:, :, :3] >= 0) & (product['data'][:, :, :3] <= 1))
        np.testing.assert_array_equal(product['data'][:, :, 3], 1)
        assert product['proposal_ess'].min() >= max(1024, 4*count)


def test_retained_population_probability_and_gamma_mixture_summaries(retained):
    settings, output, report, arrays = retained
    truth = np.load(output / 'truth.npz')
    for label, row in report['results'].items():
        population = np.load(output / (label+'_population.npz'))
        weight = (np.load(ROOT / settings['source'] / 'weights.npz')['weights'].ravel()
                  if label == 'nuts' else np.full(settings['baseline_draws'], 1/settings['baseline_draws']))
        chain_shape = (4, 500) if label == 'nuts' else None
        for key in ('intrinsic', 'detected', 'marginals'):
            assert population[key].min() >= 0
            np.testing.assert_allclose(population[key].sum(-1), 1, rtol=0, atol=1e-12)
        for key in ('intrinsic', 'detected'):
            np.testing.assert_allclose(arrays[label+'_'+key+'_mean_cells'], weight @ population[key])
            assert distance_check(population[key], truth[key], weight, chain_shape) == row['joint'][key]
        rate = rate_summary(population['alpha'], weight, 135, 160, chain_shape)
        assert rate == row['rate']
        assert arrays[label+'_marginal_quantiles'].shape == (3, 96, 3)
        assert np.all(np.diff(arrays[label+'_marginal_quantiles'], axis=-1) >= 0)


def test_retained_prediction_and_independent_numerical_summary(retained):
    settings, output, report, arrays = retained
    truth, fine_truth = dict(np.load(output/'truth.npz')), dict(np.load(output/'truth_fine.npz'))
    for label in report['results']:
        population = dict(np.load(output / (label+'_population.npz')))
        fine_population = dict(np.load(output / (label+'_population_fine.npz')))
        prediction = dict(np.load(output / (label+'_prediction.npz')))
        fine_prediction = dict(np.load(output / (label+'_prediction_fine.npz')))
        weights = (np.load(ROOT / settings['source'] / 'weights.npz')['weights'].ravel()
                   if label == 'nuts' else np.full(settings['baseline_draws'], 1/settings['baseline_draws']))
        expected = weighted_prediction(prediction['log_integrals'], weights)
        np.testing.assert_array_equal(arrays[label+'_log_prediction'], expected)
        row, first, second = numerical_summary(population, fine_population, prediction, fine_prediction, weights,
                                             arrays[label+'_verification_indices'], truth, fine_truth)
        assert row == report['numerical'][label]
        np.testing.assert_array_equal(first, arrays[label+'_coarse_subset_prediction'])
        np.testing.assert_array_equal(second, arrays[label+'_fine_subset_prediction'])
    for label, row in report['comparisons'].items():
        expected = paired_scores(arrays['nuts_log_prediction'], arrays[label+'_log_prediction'])
        assert all(row[k] == v for k, v in expected.items())
        delta = (arrays['nuts_fine_subset_prediction']-arrays[label+'_fine_subset_prediction']).sum()
        delta -= (arrays['nuts_coarse_subset_prediction']-arrays[label+'_coarse_subset_prediction']).sum()
        assert row['fine_paired_score_change'] == delta
    assert comparison_decision(settings, report['numerical'], report['comparisons'], report['maximum_derived_rhat']) == report['decision']


def test_retained_integrals_recompute_for_new_points(retained):
    settings, output, _, arrays = retained
    original = json.loads((ROOT / settings['baselines'][1] / 'summary.json').read_text())['configuration']
    model = sampling_model(original)
    for label in ('nuts', 'vi_91001'):
        if label == 'nuts':
            latent = np.load(ROOT / settings['source'] / 'posterior.npz')['latent'].reshape(-1, 20)
        else:
            latent = np.load(output / (label+'_draws.npz'))['latent']
        population = np.load(output / (label+'_population.npz'))
        data = np.load(output / 'events.npz')['data'][:3]
        computed = prediction_product(model, latent[[0, -1]], population['alpha'][[0, -1]], data)
        expected = np.load(output / (label+'_prediction.npz'))['log_integrals'][[0, -1], :3]
        np.testing.assert_allclose(computed['log_integrals'], expected, rtol=0, atol=1e-12)
        accepted = json.loads((ROOT / settings['source'] / 'summary.json').read_text())['configuration']
        evaluated = selection_fractions(model, latent[[0, -1]], accepted['primary_selection_points'], accepted['primary_selection_seed'])
        np.testing.assert_allclose(evaluated, population['alpha'][[0, -1]], rtol=0, atol=1e-12)
        fine = np.load(output / (label+'_population_fine.npz'))
        indices = arrays[label+'_verification_indices'][[0, -1]]
        data = np.load(output / 'events_fine.npz')['data'][:3]
        computed = prediction_product(model, latent[indices], fine['alpha'][[0, -1]], data)
        expected = np.load(output / (label+'_prediction_fine.npz'))['log_integrals'][[0, -1], :3]
        np.testing.assert_allclose(computed['log_integrals'], expected, rtol=0, atol=1e-12)
        evaluated = selection_fractions(model, latent[indices], settings['fine_selection_points'], settings['fine_selection_seed'])
        np.testing.assert_allclose(evaluated, fine['alpha'][[0, -1]], rtol=0, atol=1e-12)


def test_retained_flow_draws_are_reproducible_without_refitting(retained):
    settings, output, _, _ = retained
    for index, relative in enumerate(settings['baselines']):
        label = 'vi_' + relative.rsplit('_s', 1)[1]
        expected = np.load(output / (label+'_draws.npz'))['latent']
        np.testing.assert_array_equal(baseline_draws(settings, index)['latent'], expected)
