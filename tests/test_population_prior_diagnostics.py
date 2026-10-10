import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import ndtri
from scipy.stats import qmc
from torch.distributions import Beta, LKJCholesky
from torch.distributions.transforms import CorrCholeskyTransform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from population_prior_diagnostics import draw_prior_latents, scientific_features
from validate_population_morphologies import build_model


def test_prior_draws_use_lkj_correlations_and_preserve_rng_state():
    model = build_model('kumaraswamy', 2, 64, 7)
    state = torch.random.get_rng_state().clone()
    latents = draw_prior_latents(model, 10000, 991)
    assert torch.equal(state, torch.random.get_rng_state())
    np.testing.assert_array_equal(latents, draw_prior_latents(model, 10000, 991))
    params = model._unflatten_latent_vector(torch.tensor(latents), apply_numerical_limits=False)
    cholesky = model._build_cholesky_from_unconstrained(params['k_cholesky_params'], 3, 'cpu')
    correlations = (cholesky @ cholesky.transpose(-1, -2)).numpy()[..., [1, 2, 2], [0, 0, 1]]
    assert abs(correlations.mean()) < .01
    assert np.max(abs(np.mean(correlations**2, axis=(0, 1)) - .2)) < .01
    assert abs(params['log_k_a'].numpy().mean()) < .02
    assert abs(params['log_k_b'].numpy().std() - 1) < .02


def test_normal_prior_centers_are_uniform_not_sigmoid_gaussians():
    model = build_model('normal', 2, 64, 7)
    latents = draw_prior_latents(model, 10000, 992)
    params = model._unflatten_latent_vector(torch.tensor(latents), apply_numerical_limits=False)
    centers = params['tn_locs'].numpy()
    assert abs(centers.mean() - .5) < .01
    assert abs(centers.var() - 1 / 12) < .003


def test_partial_correlation_prior_matches_target_density_with_jacobian():
    raw = torch.tensor([[.2, -.6, .8], [-1., .1, -.4]], dtype=torch.float64)
    transform = CorrCholeskyTransform()
    partial = raw.tanh()
    shape = torch.tensor([2., 2., 1.5], dtype=torch.float64)
    independent_log_density = (Beta(shape, shape).log_prob((partial + 1) / 2)
                               - np.log(2) + torch.log1p(-partial.square())).sum(-1)
    target = LKJCholesky(3, torch.tensor(1.5, dtype=torch.float64))
    transformed_log_density = target.log_prob(transform(raw)) + transform.log_abs_det_jacobian(raw, transform(raw))
    torch.testing.assert_close(independent_log_density, transformed_log_density, atol=1e-12, rtol=1e-12)


def test_population_functionals_preserve_probability_and_selection():
    model = build_model('kumaraswamy', 2, 64, 7)
    latent = np.zeros(model.latent_dim)
    normal = ndtri(qmc.Sobol(3, scramble=True, seed=33).random_base2(14))
    values, changes = scientific_features(model, latent, normal, 'lisa', lambda points: np.full(len(points), 8.))
    assert values['alpha'] == pytest.approx(.5)
    assert values['domain_probability'] == pytest.approx(1.)
    assert values['weak_detection_fraction'] == 0
    assert values['mass_above_1e8'] == pytest.approx(.25, abs=.001)
    assert max(changes.values()) < .001


def test_prior_sampler_rejects_unimplemented_roughness_measure():
    model = build_model('kumaraswamy', 2, 64, 7)
    model.m1_roughness_lambda = 1.
    with pytest.raises(ValueError, match='roughness prior'):
        draw_prior_latents(model, 2, 0)
