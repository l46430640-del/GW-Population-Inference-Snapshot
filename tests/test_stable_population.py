import contextlib
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import pyro
import pytest
import torch
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from scipy.integrate import quad
from scipy.stats import truncnorm

from gwpopulation.bounded_density import log1mexp, numpy_kumaraswamy_mixture_log_prob
from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated as HistoricalEstimator
from gwpopulation.stable_marginals import (
    PiecewiseLinearMarginal, TruncatedNormalMarginal, smooth_mass_marginal,
    two_resolution_mass_marginal,
)
from gwpopulation.stable_model import CorrectedPopulationEstimator


def tensor(value, **kwargs):
    return torch.tensor(value, dtype=torch.float64, **kwargs)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()
    torch.set_num_threads(previous)


def model(family='kumaraswamy', historical=False, bounds=None, components=1):
    options = dict(data_dim=3, num_corr_kumaraswamy=0, num_corr_tnormal=0, num_corr_tpareto=0,
                   physical_bounds=bounds or ([0., 0., 0.], [1., 1., 1.]),
                   hidden_dim_flow=8, num_flow_transforms=1, device='cpu')
    if family in ('smooth', 'local', 'single'):
        options['physical_bounds'] = bounds or ([math.log10(3), 0., 0.], [math.log10(150), 2.3, 1.])
        options['num_corr_axis_spline_kumaraswamy'] = components
        options['m1_basis_type'] = {'local': 'two_resolution'}.get(family, family)
        options['axis_spline_num_bins'] = 16
    else:
        options['num_corr_' + family] = components
    with contextlib.redirect_stdout(io.StringIO()):
        instance = (HistoricalEstimator if historical else CorrectedPopulationEstimator)(**options)
    instance.obs_neff_threshold = instance.selection_neff_multiplier = instance.q_monotonicity_lambda = 0.
    instance.kumaraswamy_log_param_clip = None
    return instance


def parameters(instance):
    params = instance._unflatten_latent_vector(torch.zeros(instance.latent_dim, dtype=torch.float64))
    for family, prefix, *_ in CorrectedPopulationEstimator._families:
        count = getattr(instance, 'num_corr_' + family)
        if count:
            params[prefix + '_L_cholesky'] = torch.eye(3, dtype=torch.float64).expand(count, 3, 3).clone()
    return params


def marginal(kind, parameter=None):
    p = tensor([-.5, .7, -1., .4, 1.5]) if parameter is None else parameter
    if kind == 'single':
        return PiecewiseLinearMarginal.from_logits(p)
    if kind == 'smooth':
        return smooth_mass_marginal(p, math.log10(3), math.log10(150), .35)
    return two_resolution_mass_marginal(p[:3], p[2:], p[0], .13, .71)


@pytest.mark.parametrize('kind', ['single', 'smooth', 'local'])
def test_piecewise_density_cdf_and_normalization(kind):
    m = marginal(kind)
    edges = m.edges.numpy()
    area = sum(quad(lambda x: float(m.log_prob(tensor(x)).exp()), a, b, epsabs=1e-11)[0]
               for a, b in zip(edges[:-1], edges[1:]))
    assert area == pytest.approx(1., abs=1e-12)
    x = tensor([1e-200, 1e-10, .08, .23, .48, .8, 1-1e-12], requires_grad=True)
    cdf, sf = m.log_tails(x)
    chosen = torch.where(cdf <= sf, cdf, sf)
    derivative = torch.autograd.grad(chosen.sum(), x)[0]
    expected = (m.log_prob(x) - chosen).exp() * torch.where(cdf <= sf, 1., -1.)
    torch.testing.assert_close(derivative, expected, rtol=2e-10, atol=1e-10)


@pytest.mark.parametrize('kind', ['single', 'smooth', 'local'])
def test_piecewise_inverse_extreme_tails_and_gradients(kind):
    lp = tensor([-10000., -1000., -100., -3., -math.log(2)])[:, None]
    cdf = torch.cat((lp, log1mexp(lp)), 1).flatten()
    sf = torch.cat((log1mexp(lp), lp), 1).flatten()
    m = marginal(kind)
    coordinates = m.icdf_log_coordinates(cdf, sf)
    actual = m.log_tails_from_log_coordinates(*coordinates)
    torch.testing.assert_close(actual[0], cdf, atol=2e-10, rtol=1e-12)
    torch.testing.assert_close(actual[1], sf, atol=2e-10, rtol=1e-12)
    p = tensor([-.5, .7, -1., .4, 1.5], requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: marginal(kind, v).icdf(tensor([.05, .31, .63, .91])), (p,))


@pytest.mark.parametrize('loc,scale', [(.3, .1), (.0001, .00044), (.9999, .00044), (.5, 150.), (.7, 1.)])
def test_normal_normalization_and_scipy_agreement(loc, scale):
    m = TruncatedNormalMarginal(tensor(loc), tensor(scale))
    a, b = (.0001-loc)/scale, (.9999-loc)/scale
    p = tensor([1e-10, .01, .3, .5, .9, 1-1e-10])
    x = tensor(truncnorm.ppf(p.numpy(), a, b, loc=loc, scale=scale))
    np.testing.assert_allclose(m.log_prob(x).numpy(), truncnorm.logpdf(x.numpy(), a, b, loc=loc, scale=scale), atol=2e-11)
    torch.testing.assert_close(m.cdf(x), p, rtol=2e-7, atol=2e-12)
    # SciPy's ordinary-probability inverse loses several digits in the upper
    # tail here. The log-tail inverse is tested separately below.
    torch.testing.assert_close(m.icdf(p), x, atol=1e-9, rtol=2e-11)
    integral = quad(lambda z: float(m.log_prob(tensor(loc + scale*z)).exp()) * scale,
                    max(a, -12), min(b, 12), epsabs=2e-11)[0]
    assert integral == pytest.approx(1., abs=2e-11)


@pytest.mark.parametrize('loc,scale', [(.3, .1), (.0001, .00044), (.9999, .00044), (.5, 150.)])
def test_normal_log_tail_inverse_and_gradient(loc, scale):
    m = TruncatedNormalMarginal(tensor(loc), tensor(scale))
    lp = tensor([-1e6, -10000., -1000., -100., -5., -math.log(2)])
    cdf = torch.cat((lp, log1mexp(lp)))
    sf = torch.cat((log1mexp(lp), lp))
    coordinates = m.icdf_log_coordinates(cdf, sf)
    actual = m.log_tails_from_log_coordinates(*coordinates)
    torch.testing.assert_close(actual[0], cdf, atol=2e-9, rtol=2e-11)
    torch.testing.assert_close(actual[1], sf, atol=2e-9, rtol=2e-11)
    parameters = tensor([loc, math.log(scale)], requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: TruncatedNormalMarginal(v[0], v[1].exp()).icdf(tensor([.1, .6, .9])),
                                    (parameters,), atol=2e-6)


@pytest.mark.parametrize('family', ['kumaraswamy', 'tnormal', 'single', 'smooth', 'local'])
def test_pyro_target_uses_same_density_and_priors(family):
    stable = model(family)
    legacy = model(family, historical=True)
    params = parameters(stable)
    unit = tensor([[[.12, .35, .78], [.36, .72, .57]], [[.61, .2, .6], [.79, .1, .36]]])
    physical = stable.min_vals + (stable.max_vals - stable.min_vals) * unit
    data = torch.cat((physical, torch.ones(2, 2, 1, dtype=torch.float64)), -1)
    stable_trace = pyro.poutine.trace(pyro.poutine.condition(stable._pyro_model, data=params)).get_trace(data)
    legacy_trace = pyro.poutine.trace(pyro.poutine.condition(legacy._pyro_model, data=params)).get_trace(data)
    stable_trace.compute_log_prob()
    legacy_trace.compute_log_prob()
    stable_sites = {k for k, v in stable_trace.nodes.items() if v['type'] == 'sample' and k != 'obs_log_prob'}
    legacy_sites = {k for k, v in legacy_trace.nodes.items() if v['type'] == 'sample' and k not in ('obs_log_prob', 'data')}
    assert stable_sites == legacy_sites
    for site in stable_sites:
        torch.testing.assert_close(stable_trace.nodes[site]['log_prob'].double(),
                                   legacy_trace.nodes[site]['log_prob'].double(), atol=1e-6, rtol=1e-6)
    expected = (torch.logsumexp(stable._log_pdf_phys_from_params(params, unit), -1) - math.log(2)).sum()
    torch.testing.assert_close(stable_trace.nodes['obs_log_prob']['log_prob'], expected)


def test_endpoint_counterexample_reaches_likelihood_injection_and_numpy():
    m = model(bounds=([-2., 0., 0.], [3., 2., 1.]))
    p = parameters(m)
    p['log_k_a'] = (tensor([[.02, 1., 1.]]) - .0001).log()
    p['log_k_b'] = torch.full((1, 3), math.log(1-.0001), dtype=torch.float64)
    points = tensor([[1e-10, .5, .5], [.2, .4, .6], [-.1, .5, .5]])
    physical = m.min_vals + (m.max_vals-m.min_vals) * points
    data = torch.cat((physical, tensor([[2., 1.], [3., 2.], [1., 4.]])), -1)[None]
    recovered = (physical - m.min_vals) / (m.max_vals - m.min_vals)
    direct = numpy_kumaraswamy_mixture_log_prob({k: v.numpy() for k, v in p.items()}, recovered.numpy()) - math.log(10)
    logpdf = m._log_pdf_phys_from_params(p, recovered)
    np.testing.assert_allclose(logpdf.numpy(), direct, rtol=1e-11, atol=1e-10)
    expected = torch.logsumexp(logpdf - data[0, :, 3].log() + data[0, :, 4].log(), 0) - math.log(7)
    torch.testing.assert_close(m.event_log_likelihoods(p, data)[0], expected)
    m.correct_selection_effects = True
    m.injection_data = dict(log_m1=physical[:, 0], z=physical[:, 1], q=physical[:, 2],
                            log_p_draw=torch.zeros(3, dtype=torch.float64), Ndraw=100)
    torch.testing.assert_close(m._log_alpha(p), torch.logsumexp(logpdf, 0) - math.log(100))
    assert m._log_pdf_phys_from_params(p, tensor([[1e-10, .5, .5]]))[0].exp() > 1e7


@pytest.mark.parametrize('family', ['kumaraswamy', 'tnormal', 'smooth', 'local'])
def test_transport_selection_and_svi_use_corrected_path(family):
    m = model(family)
    params = parameters(m)
    m.correct_selection_effects = True
    m.selection_function = lambda x: .2 + .5 * (x[:, 1] - m.min_vals[1]) / (m.max_vals[1] - m.min_vals[1])
    m.lambda_det_mc_samples = 32
    alpha = m._selection_fraction_for_params(params)
    assert float(alpha) == pytest.approx(.45, abs=.006)
    unit = tensor([[[.2, .3, .4], [.6, .7, .8]], [[.15, .45, .65], [.7, .6, .5]]])
    data = torch.cat((m.min_vals + (m.max_vals - m.min_vals)*unit, torch.ones(2, 2, 1, dtype=torch.float64)), -1)
    pyro.set_rng_seed(28)
    svi = SVI(m._pyro_model, m._pyro_guide, pyro.optim.Adam({'lr': 1e-4}), loss=Trace_ELBO())
    losses = [svi.step(data) for _ in range(2)]
    assert np.isfinite(losses).all()
    assert all(torch.isfinite(p).all() for p in m.flow_transforms.parameters())
    m.is_trained = True
    assert m.sample(5).shape == (5, 3)
    assert torch.isfinite(m.predict(data[0, :, :3], num_samples=2, mode='pdf')[0]).all()


def test_historical_core_is_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root / 'src/gwpopulation/core.py').read_bytes()).hexdigest() == (
        'e732b91dcbea39c37f838b4f0f8495a12595eb0bf3198864dfe4fcfd198fd3cb')


@pytest.mark.parametrize('kind', ['single', 'smooth', 'local'])
def test_piecewise_internal_knot_derivatives(kind):
    m = marginal(kind)
    x = m.edges[1:-1].detach().clone().requires_grad_()
    cdf = m.cdf(x)
    derivative = torch.autograd.grad(cdf.sum(), x)[0]
    torch.testing.assert_close(derivative, m.log_prob(x).exp(), rtol=1e-10, atol=1e-12)
    p = cdf.detach().requires_grad_()
    inverse = m.icdf(p)
    torch.testing.assert_close(inverse, x, atol=1e-12, rtol=1e-12)
    inverse_derivative = torch.autograd.grad(inverse.sum(), p)[0]
    assert torch.isfinite(inverse_derivative).all()


def test_normal_upper_tail_against_high_precision_reference():
    # 60-digit Gaussian interval integration at the returned root gives an
    # SF relative residual below 9e-15; SciPy ppf's residual is 5.8e-9.
    m = TruncatedNormalMarginal(tensor(.3), tensor(.1))
    result = m.icdf(tensor(1-1e-10))
    assert float(result) == pytest.approx(.9359579269110401, abs=2e-15)


@pytest.mark.parametrize('kind', ['single', 'smooth', 'local'])
def test_mass_prior_density_matches_historical_representation(kind):
    new, old = model(kind), model(kind, historical=True)
    # Use identical bounds to isolate the density definition from float32
    # rounding of the old physical coordinates.
    old.min_vals, old.max_vals = new.min_vals, new.max_vals
    logits = torch.linspace(-2, 2, new.m1_spline_num_params, dtype=torch.float64)[None]
    x = torch.linspace(.001, .999, 200, dtype=torch.float64)[:, None]
    stable = new._m1_spline_dist(logits)
    historical = old._m1_spline_dist(logits)
    torch.testing.assert_close(stable.log_prob(x), historical.log_prob(x), atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(stable.cdf(x), historical.cdf(x), atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('family', ['kumaraswamy', 'tnormal', 'smooth', 'local'])
def test_correlated_transport_density_jacobian_and_gradcheck(family):
    m = model(family, components=2)
    p = parameters(m)
    correlation = tensor([[1., .4, -.2], [.4, 1., .1], [-.2, .1, 1.]])
    for key in p:
        if key.endswith('_L_cholesky'):
            p[key] = torch.linalg.cholesky(correlation).expand(2, 3, 3).clone()
    z = tensor([-.6, .2, .7], requires_grad=True)
    transform = lambda normal: m.component_transport(p, normal[None])[0, 0]
    x = transform(z)
    jacobian = torch.autograd.functional.jacobian(transform, z)
    marginal_dist, chol = next(m._components(p))
    logp = m._copula_log_prob(marginal_dist, chol, x[None])[0, 0]
    normal_logp = -.5*z.square().sum() - 1.5*math.log(2*math.pi)
    torch.testing.assert_close(logp + torch.linalg.slogdet(jacobian)[1], normal_logp, atol=2e-10, rtol=2e-10)
    assert torch.autograd.gradcheck(transform, (z,), atol=2e-6)


def test_no_silent_unsupported_family_or_penalty():
    with pytest.raises(ValueError, match='Unmigrated'):
        model('logitnormal')
    m = model()
    m.obs_neff_threshold = 1
    with pytest.raises(ValueError, match='no ESS'):
        m._log_pdf_phys_from_params(parameters(m), tensor([[.5, .5, .5]]))


@pytest.mark.parametrize('family', ['kumaraswamy', 'tnormal', 'smooth', 'local'])
def test_outside_samples_and_zero_weights_have_finite_parameter_gradients(family):
    m = model(family)
    latent = torch.zeros(m.latent_dim, dtype=torch.float64, requires_grad=True)
    params = m._unflatten_latent_vector(latent)
    unit = tensor([[[-.1, .5, .5], [.2, .3, .4], [.3, .5, .6], [0., .4, .7]]])
    data = torch.cat((m.min_vals + (m.max_vals - m.min_vals)*unit,
                      tensor([[[1., 1.], [1., 1.], [1., 0.], [1., 1.]]])), -1)
    value = m.event_log_likelihoods(params, data)
    derivative = torch.autograd.grad(value.sum(), latent)[0]
    assert torch.isfinite(derivative).all()
    assert torch.isfinite(value).all()


def test_normal_random_parameter_log_tail_stress():
    generator = torch.Generator().manual_seed(918)
    loc = torch.rand(128, generator=generator, dtype=torch.float64) * .9998 + .0001
    scale = (torch.rand(128, generator=generator, dtype=torch.float64)*13-8).exp()+.0001
    m = TruncatedNormalMarginal(loc, scale)
    lp = tensor([-1e6, -1000., -50., -3., -math.log(2)])[:, None]
    for cdf, sf in [(lp, log1mexp(lp)), (log1mexp(lp), lp)]:
        actual = m.log_tails_from_log_coordinates(*m.icdf_log_coordinates(cdf, sf))
        torch.testing.assert_close(actual[0], cdf.expand_as(actual[0]), rtol=2e-10, atol=2e-8)
        torch.testing.assert_close(actual[1], sf.expand_as(actual[1]), rtol=2e-10, atol=2e-8)


@pytest.mark.parametrize('family', ['kumaraswamy', 'tnormal', 'smooth', 'local'])
def test_traced_density_at_displaced_inputs_and_parameters(family):
    m = model(family)
    latent = torch.zeros(m.latent_dim, dtype=torch.float64)
    points = tensor([[.15, .2, .8], [.55, .4, .3]])
    evaluate = lambda l, x: m._log_pdf_phys_from_params(m._unflatten_latent_vector(l), x)
    with pytest.warns(torch.jit.TracerWarning):
        traced = torch.jit.trace(evaluate, (latent, points), check_trace=False)
    displaced = (latent + .13).requires_grad_()
    boundary = tensor([[1e-10, .31, .73], [.67, 1-1e-12, .26]])
    eager, compiled = evaluate(displaced, boundary), traced(displaced, boundary)
    torch.testing.assert_close(compiled, eager, atol=2e-10, rtol=2e-10)
    d1 = torch.autograd.grad(eager.sum(), displaced)[0]
    d2 = torch.autograd.grad(compiled.sum(), displaced)[0]
    torch.testing.assert_close(d1, d2, atol=2e-10, rtol=2e-10)


def test_bounded_mcmc_smoke_uses_the_same_pyro_target():
    m = model()
    data = tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]], [[.6, .7, .8, 1.], [.5, .6, .7, 1.]]])
    pyro.set_rng_seed(419)
    sampler = MCMC(NUTS(m._pyro_model, max_tree_depth=2), warmup_steps=2, num_samples=2, disable_progbar=True)
    sampler.run(data)
    assert all(torch.isfinite(value).all() for value in sampler.get_samples().values())


def test_independent_control_removes_copula_from_density_and_transport():
    m = model(components=2)
    p = parameters(m)
    p['k_L_cholesky'] = torch.linalg.cholesky(tensor([[1., .7, .2], [.7, 1., .1], [.2, .1, 1.]])).expand(2, 3, 3)
    p['log_k_a'] = tensor([[.1, 1., -.4], [.5, -.3, .6]])
    x = tensor([[.1, .4, .7], [.6, .3, .2]])
    z = tensor([[.2, -.7, .6], [-.8, .5, .4]])
    m.independent_components = True
    independent_pdf = m._log_pdf_phys_from_params(p, x)
    independent_transport = m.component_transport(p, z)
    m.independent_components = False
    p['k_L_cholesky'] = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    torch.testing.assert_close(independent_pdf, m._log_pdf_phys_from_params(p, x), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(independent_transport, m.component_transport(p, z), atol=1e-12, rtol=1e-12)


def test_mixed_families_and_axis_normal_use_consistent_component_order():
    with contextlib.redirect_stdout(io.StringIO()):
        m = CorrectedPopulationEstimator(3, 1, 1, 0, num_corr_axis_spline_tnormal=1,
                                          num_corr_axis_spline_kumaraswamy=1, axis_spline_num_bins=4,
                                          physical_bounds=([0., 0., 0.], [1., 1., 1.]),
                                          hidden_dim_flow=8, num_flow_transforms=1, device='cpu')
    p = parameters(m)
    x = tensor([[.11, .2, .8], [.64, .4, .1]])
    for i in range(4):
        logits = torch.full((4,), -20., dtype=torch.float64)
        logits[i] = 20.
        p['weights_logits'] = logits
        expected = m._copula_log_prob(*list(m._components(p))[i], x).squeeze(-1)
        torch.testing.assert_close(m._log_pdf_phys_from_params(p, x), expected, atol=2e-12, rtol=2e-12)
    assert m.component_transport(p, tensor([[.1, .3, .5], [-.7, .4, .2]])).shape == (2, 4, 3)


@pytest.mark.parametrize('independent', [True, False])
def test_numpy_entrypoint_applies_the_same_parameter_mapping(independent):
    m = model(components=2, bounds=([-2., 0., 0.], [1., 2., 1.]))
    m.independent_components = independent
    p = parameters(m)
    p['weights_logits'] = tensor([45., -32.])
    p['log_k_a'] = tensor([[-3., .5, .8], [.2, 1., -.2]])
    p['log_k_b'] = tensor([[.3, -.6, .1], [0., .3, .2]])
    p['k_L_cholesky'] = torch.linalg.cholesky(tensor([[1., .7, .2], [.7, 1., .1], [.2, .1, 1.]])).expand(2, 3, 3)
    x = tensor([[1e-200, .2, .5], [.4, 1-1e-12, .2], [-1., .5, .5]])
    np.testing.assert_allclose(m.numpy_log_pdf_from_params(p, x),
                               m._log_pdf_phys_from_params(p, x).detach().numpy(), atol=2e-10, rtol=2e-11)
