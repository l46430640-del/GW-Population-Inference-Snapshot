import sys
from pathlib import Path

import numpy as np
import pyro
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_selected_mixture import (
    detection_probability, event_posterior, exact_selection,
    kumaraswamy_moment, log_predictive, selected_model,
)


def test_exact_selection_agrees_with_direct_integral_and_has_finite_gradient():
    a = torch.tensor([2., 5.], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([8., 2.], dtype=torch.float64, requires_grad=True)
    actual = kumaraswamy_moment(a, b, 2)
    expected = [quad(lambda x: x*x*ai*bi*x**(ai-1)*(1-x**ai)**(bi-1), 0, 1)[0]
                for ai, bi in zip(a.detach().numpy(), b.detach().numpy())]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    gradients = torch.autograd.grad(actual.sum(), (a, b))
    assert all(torch.isfinite(value).all() for value in gradients)


def test_selected_event_posterior_includes_bernoulli_detection_factor():
    observation = np.array([.35, .65, .4])
    draws = event_posterior(observation, 20000, np.random.default_rng(711))
    density = lambda x: norm.pdf(x, .35, .04) * (.05 + .95*x*x)
    expected = quad(lambda x: x*density(x), 0, 1)[0] / quad(density, 0, 1)[0]
    assert expected > observation[0] + .005
    assert draws[:, 0].mean() == pytest.approx(expected, abs=.001)
    assert draws[:, 1].mean() == pytest.approx(observation[1], abs=.001)
    assert np.all((draws > 0) & (draws < 1))


def test_independent_ablation_removes_only_within_component_dependence():
    model = selected_model(independent=True)
    latent = torch.linspace(-.4, .4, model.latent_dim, dtype=torch.float64)
    params = model._unflatten_latent_vector(latent)
    points = torch.tensor([[.2, .3, .4], [.6, .7, .8]], dtype=torch.float64)
    before = model._log_pdf_phys_from_params(params, points)
    params['k_cholesky_params'] = params['k_cholesky_params'] + 1
    after = model._log_pdf_phys_from_params(params, points)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    correlated = selected_model()
    assert not torch.allclose(after, correlated._log_pdf_phys_from_params(params, points))


def test_selected_shape_likelihood_and_prediction_share_normalization():
    model = selected_model()
    sites = {'weights_logits': torch.tensor([.2, -.2], dtype=torch.float64),
             'log_k_a': torch.tensor([[.3, .1, .2], [.5, -.1, .6]], dtype=torch.float64),
             'log_k_b': torch.tensor([[.1, .3, .5], [.2, .4, -.3]], dtype=torch.float64),
             'k_L_cholesky': torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)}
    data = torch.tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]],
                         [[.6, .5, .7, 1.], [.7, .6, .8, 1.]]], dtype=torch.float64)
    trace = pyro.poutine.trace(pyro.poutine.condition(model._pyro_model, data=sites)).get_trace(data)
    trace.compute_log_prob()
    expected = torch.logsumexp(model._log_pdf_phys_from_params(sites, data[..., :3].reshape(-1, 3)).reshape(2, 2), -1)
    expected = expected - np.log(2) - exact_selection(sites).log()
    torch.testing.assert_close(trace.nodes['obs_log_prob']['log_prob'].sum(), expected.sum())
    samples = {name: value.numpy()[None] for name, value in sites.items()}
    np.testing.assert_allclose(log_predictive(model, samples, data), expected.detach().numpy(), atol=1e-12)


def test_traced_nuts_potential_and_gradients_match_eager_at_displaced_points():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        pyro.set_rng_seed(7167)
        model = selected_model()
        generator = torch.Generator().manual_seed(191)
        points = .05 + .9 * torch.rand((4, 16, 3), generator=generator)
        data = torch.cat([points, torch.ones(4, 16, 1)], -1)
        initial, eager, _, _ = initialize_model(model._pyro_model, model_args=(data,))
        _, traced, _, _ = initialize_model(model._pyro_model, model_args=(data,),
                                            initial_params=initial, jit_compile=True)
        for scale in (0., .1, .4):
            params = {key: value + scale * torch.randn(value.shape, generator=generator)
                      for key, value in initial.items()}
            eager_gradient, eager_value = potential_grad(eager, params)
            traced_gradient, traced_value = potential_grad(traced, params)
            torch.testing.assert_close(eager_value, traced_value, atol=1e-10, rtol=1e-12)
            for key in params:
                torch.testing.assert_close(eager_gradient[key], traced_gradient[key], atol=1e-9, rtol=1e-10)
    finally:
        torch.set_default_dtype(old_dtype)
