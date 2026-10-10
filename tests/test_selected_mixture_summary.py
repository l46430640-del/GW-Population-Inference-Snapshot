import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_selected_mixture import conditional_bins, paired_scores, truth_samples
from check_selected_mixture_integrals import event_integrals
from benchmark_selected_mixture import selected_model


def test_conditional_density_integrates_to_one_and_retains_partition_probabilities():
    samples = truth_samples()
    density, probability = conditional_bins(samples, order=96)
    np.testing.assert_allclose(density.sum(-1)/96, 1., atol=1e-12)
    np.testing.assert_allclose(probability.sum(0), 1., atol=1e-12)
    assert density.min() >= -1e-12


def test_independent_conditional_density_uses_mass_dependent_mixture_weights():
    samples = truth_samples()
    density, probability = conditional_bins(samples, independent=True)
    a, b = np.exp(samples['log_k_a'][0]) + .0001, np.exp(samples['log_k_b'][0]) + .0001
    weight = np.exp(samples['weights_logits'][0])
    edges = np.linspace(0, 1, 97)
    other_cdf = 1-(1-edges[:, None]**a[None, :, 1])**b[None, :, 1]
    for index, (low, high) in enumerate(zip([0, 1/3, 2/3], [1/3, 2/3, 1])):
        interval = (1-low**a[:, 0])**b[:, 0] - (1-high**a[:, 0])**b[:, 0]
        expected = np.diff(other_cdf @ (weight*interval/(weight@interval))) * 96
        np.testing.assert_allclose(density[index, 0], expected, atol=2e-11)


def test_gaussian_copula_conditional_probability_matches_analytic_orthant():
    rho = .5
    correlation = np.array([[1., rho, 0.], [rho, 1., 0.], [0., 0., 1.]])
    samples = {'log_k_a': np.full((1, 2, 3), np.log(1-.0001)),
               'log_k_b': np.full((1, 2, 3), np.log(1-.0001)),
               'weights_logits': np.zeros((1, 2)),
               'k_L_cholesky': np.tile(np.linalg.cholesky(correlation), (1, 2, 1, 1))}
    density, probability = conditional_bins(samples, order=256, mass_edges=(0., .5, 1.))
    assert density[0, 0, :48].sum()/96 == pytest.approx(.5 + np.arcsin(rho)/np.pi, abs=2e-7)
    np.testing.assert_allclose(probability, .5, atol=1e-12)
    ablated, _ = conditional_bins(samples, independent=True, mass_edges=(0., .5, 1.))
    np.testing.assert_allclose(ablated, 1., atol=1e-12)


def test_paired_predictive_score_uncertainty_uses_event_differences():
    first = np.array([10., -100., 2., 40.])
    difference = np.array([.1, .3, -.2, .5])
    result = paired_scores(first, first-difference)
    assert result['total_difference'] == pytest.approx(difference.sum())
    assert result['total_standard_error'] == pytest.approx(2*difference.std(ddof=1))
    assert result['per_event_standard_error'] == pytest.approx(difference.std(ddof=1)/2)


def test_replicated_pe_points_leave_event_integral_invariant():
    model = selected_model()
    samples = truth_samples()
    data = torch.tensor([[[.2, .3, .4, 1.], [.25, .4, .5, 1.]],
                         [[.6, .5, .7, 1.], [.7, .6, .8, 1.]]], dtype=torch.float64)
    original, original_ess = event_integrals(model, samples, data)
    repeated, repeated_ess = event_integrals(model, samples, data.repeat(1, 3, 1))
    np.testing.assert_allclose(original, repeated, atol=1e-12)
    np.testing.assert_allclose(3*original_ess, repeated_ess, atol=1e-12)
