import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_selected_mixture import log_predictive, selected_model
from predict_corrected_posterior import (
    cached_integrals, compare_predictions, selected_integrals,
    systematic_indices, weighted_prediction,
)
from summarize_selected_mixture import truth_samples


def test_prediction_averages_density_before_logarithm():
    integrals = np.array([[1., 3.], [9., 7.], [100., 200.]])
    weights = np.array([1., 3., 0.])
    actual = weighted_prediction(np.log(integrals), weights)
    np.testing.assert_allclose(actual, np.log([7., 6.]))
    assert not np.allclose(actual, np.average(np.log(integrals[:2]), axis=0, weights=weights[:2]))
    np.testing.assert_allclose(weighted_prediction(np.log(integrals) + 1000, weights), actual + 1000)


def test_systematic_resampling_preserves_discrete_probabilities_and_zero_weights():
    weights = np.array([0., .2, .3, .5])
    indices = systematic_indices(weights, 100, 98)
    np.testing.assert_array_equal(np.bincount(indices, minlength=4), [0, 20, 30, 50])
    np.testing.assert_array_equal(indices, systematic_indices(weights * 1e200, 100, 98))
    weights = np.array([.23, .31, .46])
    counts = np.array([np.bincount(systematic_indices(weights, 17, seed), minlength=3)
                       for seed in range(1000)])
    np.testing.assert_allclose(counts.mean(0) / 17, weights, atol=.003)


def test_selected_integral_matrix_matches_original_predictor():
    model = selected_model()
    samples = {key: np.repeat(value, 3, axis=0) for key, value in truth_samples().items()}
    samples['log_k_a'][1] += .2
    samples['log_k_b'][2] -= .1
    data = torch.tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]],
                         [[.6, .5, .7, 1.], [.7, .6, .8, 1.]]], dtype=torch.float64)
    values, _ = selected_integrals(model, samples, data)
    np.testing.assert_allclose(weighted_prediction(values), log_predictive(model, samples, data), atol=1e-12)
    repeated = np.repeat(values, [1, 3, 2], axis=0)
    np.testing.assert_allclose(weighted_prediction(values, [1, 3, 2]), weighted_prediction(repeated), atol=1e-12)


@pytest.mark.parametrize('weights', [[-1., 2.], [0., 0.], [1.], [np.nan, 1.]])
def test_prediction_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_prediction(np.zeros((2, 3)), weights)


def test_prediction_rejects_invalid_integrals():
    for values in (np.zeros(3), np.zeros((0, 3)), np.array([[0., np.nan]])):
        with pytest.raises(ValueError):
            weighted_prediction(values)


def test_integral_cache_reuses_verified_values_and_rejects_changed_inputs(tmp_path):
    model = selected_model()
    samples = truth_samples()
    data = torch.tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]]], dtype=torch.float64)
    product, metadata = cached_integrals(tmp_path, 'example', model, samples, data, {'source': 'fixed'})
    repeated, repeated_metadata = cached_integrals(tmp_path, 'example', model, samples, data,
                                                  {'source': 'fixed'}, require=True)
    np.testing.assert_array_equal(product['log_integrals'], repeated['log_integrals'])
    assert metadata == repeated_metadata
    with pytest.raises(ValueError, match='different inputs'):
        cached_integrals(tmp_path, 'example', model, samples, data, {'source': 'changed'})
    changed_data = data.clone()
    changed_data[..., 0] += .01
    with pytest.raises(ValueError, match='different inputs'):
        cached_integrals(tmp_path, 'example', model, samples, changed_data, {'source': 'fixed'})
    with pytest.raises(FileNotFoundError, match='not complete'):
        cached_integrals(tmp_path, 'missing', model, samples, data, {}, require=True)


@pytest.mark.parametrize('seed', [731001, 731002])
def test_retained_prediction_products_reproduce_weighting_and_comparisons(seed):
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/analysis/importance_prediction'
    report = json.loads((output / 'comparison.json').read_text())
    for name, expected in report['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for name, expected in report['product_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    label = f'repeat_{seed}'
    draws = np.load(root / f'results/analysis/importance_correction/{label}.npz')
    coarse = np.load(output / f'{label}_coarse.npz')['log_integrals']
    fine = np.load(output / f'{label}_fine.npz')['log_integrals']
    predictions = np.load(output / 'predictions.npz')
    resampling = np.load(output / 'resampling_indices.npz')
    assert coarse.shape == (8192, 192)
    for name, weights in (('unweighted', None), ('raw', draws['raw_weights']), ('psis', draws['psis_weights'])):
        prefix = label + '_' + name
        actual = weighted_prediction(coarse, weights)
        np.testing.assert_allclose(predictions[prefix + '_coarse'], actual, atol=1e-12)
        indices = resampling[prefix]
        fine_indices = np.searchsorted(resampling[label + '_fine_unique'], indices)
        np.testing.assert_allclose(predictions[prefix + '_fine_subset'], weighted_prediction(fine[fine_indices]), atol=1e-12)
        expected = compare_predictions(actual, predictions[label + '_unweighted_coarse'])
        assert expected == report['results'][label][name]['versus_unweighted_coarse']
