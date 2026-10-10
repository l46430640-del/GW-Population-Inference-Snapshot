import numpy as np
import pyro
import pytest
import torch
from scipy.stats import gamma
from torch.distributions.transforms import CorrCholeskyTransform

from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated
from gwpopulation.rate import conditional_rate_draws


def estimator(**kwargs):
    return BayesianMixtureEstimator_NF_Correlated(
        3, kwargs.pop('num_corr_kumaraswamy', 1),
        kwargs.pop('num_corr_tnormal', 0), 0,
        num_flow_transforms=2, hidden_dim_flow=16, device='cpu',
        physical_bounds=([0, 0, 0], [1, 1, 1]), **kwargs)


def test_correlation_jacobian_against_autograd():
    x = torch.tensor([0.2, -0.7, 0.4], dtype=torch.float64, requires_grad=True)
    transform = CorrCholeskyTransform()
    indices = torch.tril_indices(3, 3, -1)
    jacobian = torch.autograd.functional.jacobian(
        lambda y: transform(y)[indices[0], indices[1]], x)
    expected = torch.linalg.slogdet(jacobian).logabsdet
    torch.testing.assert_close(transform.log_abs_det_jacobian(x, transform(x)), expected)
    model = estimator()
    latent = torch.zeros(model.latent_dim, dtype=torch.float64)
    latent[-3:] = x.detach()
    torch.testing.assert_close(model._latent_log_abs_det_jacobian(latent), expected)


def test_guide_factor_is_log_q_in_model_coordinates():
    model = estimator(num_corr_tnormal=1)
    latent = torch.linspace(-.5, .5, model.latent_dim)

    class FixedDraw:
        def rsample(self):
            return latent

        def log_prob(self, value):
            return torch.tensor(-7.5)

    model.flow_dist = FixedDraw()
    pyro.clear_param_store()
    trace = pyro.poutine.trace(model._pyro_guide).get_trace(None)
    expected = -7.5 - model._latent_log_abs_det_jacobian(latent)
    torch.testing.assert_close(trace.log_prob_sum(), expected)


def test_samples_outside_support_contribute_zero_not_boundary_density():
    model = estimator()
    model.obs_neff_threshold = 0
    sites = {'weights_logits': torch.zeros(1), 'log_k_a': torch.zeros(1, 3),
             'log_k_b': torch.zeros(1, 3), 'k_L_cholesky': torch.eye(3)[None]}
    inside = torch.tensor([[[.4, .5, .6, 1.]]])
    padded = torch.cat([inside, torch.tensor([[[1.2, .5, .6, 1.]]])], dim=1)
    loglike = lambda x: pyro.poutine.trace(
        pyro.poutine.condition(model._pyro_model, data=sites)).get_trace(x).log_prob_sum()
    torch.testing.assert_close(loglike(padded) - loglike(inside), torch.tensor(-np.log(2), dtype=torch.float32))
    params = model._unflatten_latent_vector(torch.zeros(model.latent_dim))
    assert model._log_pdf_phys_from_params(params, padded[0, :, :3])[1] == -torch.inf


def test_additional_joint_support_retains_original_sample_denominator():
    model = estimator()
    model.obs_neff_threshold = 0
    model.event_support_function = lambda points: points[...,0]+points[...,2] < 1
    sites = {'weights_logits': torch.zeros(1), 'log_k_a': torch.zeros(1,3),
             'log_k_b': torch.zeros(1,3), 'k_L_cholesky': torch.eye(3)[None]}
    inside = torch.tensor([[[.2,.5,.3,1.]]])
    padded = torch.cat([inside,torch.tensor([[[.9,.5,.8,1.]]])],dim=1)
    score = lambda data: pyro.poutine.trace(pyro.poutine.condition(
        model._pyro_model,data=sites)).get_trace(data).log_prob_sum()
    torch.testing.assert_close(score(padded)-score(inside),torch.tensor(-np.log(2),dtype=torch.float32))
    with pytest.raises(ValueError,match='no posterior samples'):
        score(padded[:,1:])


def test_guide_keeps_prior_coordinates_beyond_likelihood_saturation_limits():
    model = estimator(num_corr_tnormal=1)
    latent = torch.full((model.latent_dim,), -11., dtype=torch.float64)
    latent[:model.K] = 25.
    prior_coordinates = model._unflatten_latent_vector(latent, apply_numerical_limits=False)
    likelihood_parameters = model._unflatten_latent_vector(latent)
    assert torch.all(prior_coordinates['weights_logits'] == 25)
    assert torch.all(prior_coordinates['log_tn_scales'] == -11)
    assert torch.all(prior_coordinates['tn_locs'] < 1e-4)
    assert torch.all(likelihood_parameters['weights_logits'] == 20)
    assert torch.all(likelihood_parameters['log_tn_scales'] == -8)


def test_outside_injections_have_zero_selection_weight():
    model = estimator()
    model.correct_selection_effects = True
    model.obs_neff_threshold = 0
    model.selection_neff_multiplier = 0
    model.injection_data = {'log_m1': torch.tensor([.5, 1.2]), 'z': torch.tensor([.5, .5]),
                            'q': torch.tensor([.5, .5]), 'log_p_draw': torch.zeros(2), 'Ndraw': 2}
    sites = {'weights_logits': torch.zeros(1), 'log_k_a': torch.zeros(1, 3),
             'log_k_b': torch.zeros(1, 3), 'k_L_cholesky': torch.eye(3)[None]}
    pyro.poutine.trace(pyro.poutine.condition(model._pyro_model, data=sites)).get_trace(
        torch.tensor([[[.4, .5, .6, 1.]]]))
    assert model._last_selection_log_weights[1] == -torch.inf


@pytest.mark.parametrize('components', [1, 2, 3])
def test_coupling_layers_transform_every_latent_coordinate(components):
    model = estimator(num_corr_kumaraswamy=components)
    identities = torch.arange(model.latent_dim)
    transformed = torch.zeros(model.latent_dim, dtype=torch.int64)
    for part in model.flow_transforms.parts:
        if hasattr(part, 'permutation'):
            identities = identities[part.permutation]
        else:
            transformed[identities[model.latent_dim // 2:]] += 1
    assert transformed.min() >= 1


def test_population_selection_quadrature_and_gradient():
    model = estimator()
    model.correct_selection_effects = True
    model.selection_function = lambda x: torch.special.ndtr(4 * x[:, 0] - 2)
    model.selection_mc_mode = 'fixed_sobol'
    model.lambda_det_mc_samples = 16384
    model.selection_integration = 'population_qmc'
    latent = torch.zeros(model.latent_dim, requires_grad=True)
    value = model._selection_fraction_for_params(model._unflatten_latent_vector(latent))
    assert float(value) == pytest.approx(.5, abs=.001)
    gradient = torch.autograd.grad(value, latent)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[1] > .1
    step = torch.zeros_like(latent)
    step[1] = 1e-3
    plus = model._selection_fraction_for_params(model._unflatten_latent_vector(latent + step))
    minus = model._selection_fraction_for_params(model._unflatten_latent_vector(latent - step))
    assert float(gradient[1]) == pytest.approx(float((plus - minus) / .002), rel=.002)


def test_rate_api_reconstructs_gamma_without_mcmc():
    model = estimator()
    model.is_trained = True
    model.observation_years = 4
    model.infer_rate(10, num_mcmc_samples=32)
    assert model.rate_diagnostics['method'] == 'conditional_gamma'
    assert model.rate_shape_latent_samples.shape == (32, model.latent_dim)
    torch.testing.assert_close(model.yearly_rate_posterior_samples * 4,
                               model.expected_detected_count_samples)


@pytest.mark.parametrize('family', ['normal', 'spline', 'two_resolution', 'bernstein', 'logitnormal_axis', 'decoupled'])
def test_location_and_correlation_jacobians_at_origin(family):
    kwargs = {'num_corr_kumaraswamy': 0}
    if family == 'normal':
        kwargs['num_corr_tnormal'] = 2
        n_locations = 6
    elif family == 'bernstein':
        kwargs['num_corr_axis_bernstein_tnormal'] = 2
        n_locations = 4
    elif family == 'logitnormal_axis':
        kwargs['num_corr_axis_lntnormal'] = 2
        n_locations = 4
    elif family == 'decoupled':
        kwargs['num_corr_decoupled_spline_tnormal_copula'] = 2
        n_locations = 2
    else:
        kwargs['num_corr_axis_spline_tnormal'] = 2
        n_locations = 4
        if family == 'two_resolution':
            kwargs['m1_basis_type'] = 'two_resolution'
    model = estimator(**kwargs)
    latent = torch.zeros(4, model.latent_dim, dtype=torch.float64)
    actual = model._latent_log_abs_det_jacobian(latent)
    torch.testing.assert_close(actual, torch.full((4,), -n_locations * np.log(4), dtype=torch.float64))


def test_rate_draws_use_same_prior_and_paired_alpha():
    torch.manual_seed(521)
    alpha = torch.linspace(0.2, 0.95, 60000, dtype=torch.float64)
    result = conditional_rate_draws(20, 4, alpha)
    torch.testing.assert_close(result['rate'] * 4 * alpha, result['detected_mean'])
    quantiles = torch.quantile(result['detected_mean'], torch.tensor([.05, .5, .95], dtype=torch.float64))
    np.testing.assert_allclose(quantiles.numpy(), gamma.ppf([.05, .5, .95], 20), rtol=.015)
    assert abs(result['replicated_detected_count'].mean().item() - 20) < .12
    assert abs(result['replicated_detected_count'].var().item() - 40) < 1
    assert torch.corrcoef(torch.stack([result['rate'], alpha]))[0, 1] < -.7


@pytest.mark.parametrize('n,t,a', [(0, 4, [.5]), (-1, 4, [.5]), (1.5, 4, [.5]),
                                      (5, 0, [.5]), (5, 4, [0]), (5, 4, [1.1])])
def test_invalid_rate_inputs(n, t, a):
    with pytest.raises(ValueError):
        conditional_rate_draws(n, t, a)
def test_kumaraswamy_is_finite_close_to_float32_upper_boundary():
    from gwpopulation.core import SafeKumaraswamy
    x = torch.tensor([.9999999, .999999, .5], dtype=torch.float32)
    a = torch.tensor(.12, dtype=torch.float32)
    b = torch.tensor(.88, dtype=torch.float32)
    distribution = SafeKumaraswamy(a, b)
    assert torch.isfinite(distribution.log_prob(x)).all()
    reference = SafeKumaraswamy(a.double(), b.double()).log_prob(x.double())
    torch.testing.assert_close(distribution.log_prob(x).double(), reference, atol=1e-5, rtol=1e-5)
    probability = torch.tensor([1e-7, .1, .9, .9999], dtype=torch.float32)
    recovered = distribution.cdf(distribution.icdf(probability))
    torch.testing.assert_close(recovered, probability, atol=2e-6, rtol=2e-5)


def test_spline_gather_matches_piecewise_linear_density_and_integral():
    from gwpopulation.core import BoundedMonotoneSpline
    logits = torch.tensor([[.2, -.5, 1.3, .4], [-.1, .9, .1, -.3]], dtype=torch.float64, requires_grad=True)
    distribution = BoundedMonotoneSpline(logits)
    x = torch.tensor([.01, .15, .36, .73, .99], dtype=torch.float64)[:, None]
    heights = distribution._normalized_heights().detach().numpy()
    expected_pdf, expected_cdf = [], []
    for row in heights:
        expected_pdf.append(np.interp(x.numpy().ravel(), np.linspace(0, 1, 4), row))
        index = (x.numpy().ravel()*3).astype(int)
        t = x.numpy().ravel()*3-index
        areas = (row[:-1]+row[1:])/6
        prefix = np.r_[0, np.cumsum(areas)]
        expected_cdf.append(prefix[index] + (row[index]*t + .5*(row[index+1]-row[index])*t*t)/3)
    np.testing.assert_allclose(distribution.log_prob(x).exp().detach(), np.array(expected_pdf).T, atol=1e-12)
    np.testing.assert_allclose(distribution.cdf(x).detach(), np.array(expected_cdf).T, atol=1e-12)
    assert torch.autograd.gradcheck(lambda p: BoundedMonotoneSpline(p).log_prob(x), (logits,))

