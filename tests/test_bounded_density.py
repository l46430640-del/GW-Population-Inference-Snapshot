import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln, erfcx, log_ndtr, ndtri, ndtri_exp
from scipy.stats import qmc
import torch

from gwpopulation.bounded_density import (
    KumaraswamyMarginal, gaussian_copula_log_prob, kumaraswamy_component_transport,
    kumaraswamy_mixture_log_prob, log1mexp, normal_quantile_from_log_tails,
    numpy_kumaraswamy_component_transport, numpy_kumaraswamy_mixture_log_prob,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tensor(value, **kwargs):
    return torch.tensor(value, dtype=torch.float64, **kwargs)


def parameters(dtype=torch.float64):
    correlation = tensor([[1., .7, -.2], [.7, 1., .15], [-.2, .15, 1.]])
    return {
        'weights_logits': tensor([-.7, .4]).to(dtype),
        'log_k_a': tensor([[.02, 3., .4], [2., .7, 20.]]).sub(.0001).log().to(dtype),
        'log_k_b': tensor([[1., .02, 50.], [.3, 4., .8]]).sub(.0001).log().to(dtype),
        'k_L_cholesky': torch.stack([torch.linalg.cholesky(correlation), torch.eye(3)]).to(dtype),
    }


@pytest.mark.parametrize('a,b', [(.02, 1.), (1., .02), (.3, 5.), (12., .7), (1., 1.)])
def test_cdf_derivative_matches_density_without_a_boundary_strip(a, b):
    marginal = KumaraswamyMarginal(a, b)
    x = tensor([1e-300, 1e-10, 1e-8, .2, .8, 1-1e-12], requires_grad=True)
    cdf, survival = marginal.log_tails(x)
    use_cdf = cdf < survival
    smaller = torch.where(use_cdf, cdf, survival)
    derivative = torch.autograd.grad(smaller.sum(), x)[0]
    expected = torch.exp(marginal.log_prob(x) - smaller)
    torch.testing.assert_close(derivative, torch.where(use_cdf, expected, -expected), rtol=2e-11, atol=1e-10)


@pytest.mark.parametrize('a,b', [(.02, 1.), (1., .02), (.3, 5.), (12., .7)])
def test_normalization_and_moment_by_unclipped_log_coordinate_quadrature(a, b):
    marginal = KumaraswamyMarginal(a, b)

    def integrand(t, moment):
        log_x = tensor(-t)
        density = marginal.log_prob_from_log_coordinates(log_x, log1mexp(log_x))
        return float((density - (moment + 1)*t).exp())

    normalization, _ = quad(integrand, 0., np.inf, args=(0,), epsabs=2e-9)
    mean, _ = quad(integrand, 0., np.inf, args=(1,), epsabs=2e-9)
    assert normalization == pytest.approx(1., abs=2e-8)
    assert mean == pytest.approx(np.exp(np.log(b) + betaln(1+1/a, b)), abs=2e-8)


def test_diagnostic_counterexample_is_repaired_in_the_joint_kernel():
    p = {'weights_logits': tensor([0.]), 'log_k_a': tensor([[.02, 1., 1.]]).sub(.0001).log(),
         'log_k_b': tensor([[1., 1., 1.]]).sub(.0001).log(), 'k_L_cholesky': torch.eye(3)[None]}
    x = tensor([1e-300, 1e-12, 1e-10, 1e-8, 1e-7, .2])
    points = torch.stack([x, torch.full_like(x, .5), torch.full_like(x, .5)], dim=-1)
    actual = kumaraswamy_mixture_log_prob(p, points)
    torch.testing.assert_close(actual, math.log(.02) - .98*x.log(), rtol=2e-13, atol=2e-12)
    assert actual[2].exp().item() == pytest.approx(126191468.89603859, rel=1e-12)


@pytest.mark.parametrize('a,b', [(.0001, 1.), (1., .0001), (.02, 5.), (100., .01)])
def test_transport_roundtrip_retains_tails_after_physical_coordinates_round(a, b):
    marginal = KumaraswamyMarginal(a, b)
    gaussian = tensor([-1000., -40., -8., -1., 0., 1., 8., 40., 1000.])
    cdf, survival = torch.special.log_ndtr(gaussian), torch.special.log_ndtr(-gaussian)
    coordinates = marginal.icdf_log_coordinates(cdf, survival)
    recovered = marginal.log_tails_from_log_coordinates(*coordinates)
    for actual, expected in zip(recovered, (cdf, survival)):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=3e-10)
    normal = normal_quantile_from_log_tails(*recovered)
    torch.testing.assert_close(normal, gaussian, rtol=2e-12, atol=2e-11)
    assert all(torch.isfinite(item).all() for item in coordinates)


def test_marginal_endpoints_and_support_are_explicit():
    marginal = KumaraswamyMarginal(.02, 1.)
    x = tensor([-1., 0., .3, 1., 2., torch.nan])
    cdf = marginal.cdf(x)
    torch.testing.assert_close(cdf[:2], tensor([0., 0.]))
    torch.testing.assert_close(cdf[3:5], tensor([1., 1.]))
    assert cdf[-1].isnan()
    density = marginal.log_prob(x)
    assert density[0].isneginf() and density[1].isposinf()
    assert density[3] == pytest.approx(math.log(.02)) and density[4].isneginf()
    assert density[5].isnan()
    torch.testing.assert_close(marginal.icdf(tensor([0., 1.])), tensor([0., 1.]))
    assert marginal.icdf(tensor([-1., 2., torch.nan])).isnan().all()
    p = parameters()
    assert kumaraswamy_mixture_log_prob(p, tensor([[0., .5, .5], [1., .5, .5], [-1., .5, .5]])).isneginf().all()


def test_normal_log_quantile_matches_scipy_and_has_correct_gradients():
    log_p = tensor([-1e6, -10000., -800., -600.01, -600., -50., -2., -.7], requires_grad=True)
    log_q = log1mexp(log_p)
    actual = normal_quantile_from_log_tails(log_p, log_q)
    reference = ndtri_exp(log_p.detach().numpy())
    np.testing.assert_allclose(actual.detach().numpy(), reference, rtol=1e-12, atol=1e-12)
    # The installed SciPy inverse has a ~1e-6 log-CDF residual at log(p)=-1e6.
    # Refine its independent result instead of treating that residual as truth.
    reference -= ((log_ndtr(reference) - log_p.detach().numpy())
                  / np.exp(-.5*reference**2 - .5*np.log(2*np.pi) - log_ndtr(reference)))
    np.testing.assert_allclose(actual.detach().numpy(), reference, rtol=2e-14, atol=1e-12)
    np.testing.assert_allclose(log_ndtr(actual.detach().numpy()), log_p.detach().numpy(), rtol=2e-15, atol=1e-12)
    gradient = torch.autograd.grad(actual.sum(), log_p)[0]
    expected = np.sqrt(np.pi/2)*erfcx(-reference/np.sqrt(2))
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=2e-13)
    assert torch.autograd.gradcheck(lambda v: normal_quantile_from_log_tails(v, log1mexp(v)),
                                   (tensor([-800., -2.], requires_grad=True),), eps=1e-4, atol=1e-7)
    torch.testing.assert_close(normal_quantile_from_log_tails(tensor([-torch.inf, 0.]),
                                                            tensor([0., -torch.inf])), tensor([-torch.inf, torch.inf]))


def test_normal_quantile_gradient_does_not_subtract_huge_log_probabilities():
    log_p = tensor([-1e20, -1e100, -1e200], requires_grad=True)
    quantile = normal_quantile_from_log_tails(log_p, torch.zeros_like(log_p))
    gradient = torch.autograd.grad(quantile.sum(), log_p)[0]
    reference = ndtri_exp(log_p.detach().numpy())
    np.testing.assert_allclose(quantile.detach().numpy(), reference, rtol=5e-14)
    np.testing.assert_allclose(gradient.numpy(), np.sqrt(np.pi/2)*erfcx(-reference/np.sqrt(2)), rtol=5e-14)


def test_distribution_shape_and_coordinate_gradients():
    a, b = tensor([.02, 5., .3], requires_grad=True), tensor([1.2, .2, 4.], requires_grad=True)
    x = tensor([.1, .8, .6], requires_grad=True)

    def calculation(first, second, point):
        marginal = KumaraswamyMarginal(first, second)
        return torch.cat([marginal.log_prob(point), *marginal.log_tails(point), marginal.icdf(point)])

    assert torch.autograd.gradcheck(calculation, (a, b, x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_numpy_and_torch_agree_for_boundary_correlation_and_transport(dtype):
    p = parameters(dtype)
    points = tensor([[1e-300, 1-1e-15, .999], [1e-10, .4, 1e-12], [.8, .2, .6], [1., 2., -1.]])
    npp = {name: value.numpy() for name, value in p.items()}
    actual = kumaraswamy_mixture_log_prob(p, points).numpy()
    expected = numpy_kumaraswamy_mixture_log_prob(npp, points.numpy())
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=5e-10)
    normal = tensor([[-1000., -20., 30.], [10., 3., -7.], [0., 1., -1.], [1000., 20., -30.]])
    torch_logs = kumaraswamy_component_transport(p, normal, True)
    numpy_logs = numpy_kumaraswamy_component_transport(npp, normal.numpy(), True)
    for actual, expected in zip(torch_logs, numpy_logs):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-11, atol=2e-8)
    np.testing.assert_allclose(kumaraswamy_component_transport(p, normal).numpy(),
                               numpy_kumaraswamy_component_transport(npp, normal.numpy()), rtol=2e-12, atol=1e-14)


def test_correlated_copula_normalization_and_transport_jacobian():
    p = parameters()
    cholesky = p['k_L_cholesky'][0]
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    normal = tensor(np.stack(np.meshgrid(*([nodes*np.sqrt(2)]*3), indexing='ij'), -1).reshape(-1, 3))
    product_weights = np.prod(np.stack(np.meshgrid(*([weights/np.sqrt(np.pi)]*3), indexing='ij'), -1), -1).ravel()
    log_density = gaussian_copula_log_prob(torch.special.log_ndtr(normal), torch.special.log_ndtr(-normal), cholesky)
    assert np.dot(log_density.exp().numpy(), product_weights) == pytest.approx(1., abs=2e-7)

    z = tensor([.3, -.2, .1], requires_grad=True)
    single = {name: value[:1] for name, value in p.items()}
    logs = kumaraswamy_component_transport(single, z, True)
    marginal = KumaraswamyMarginal(single['log_k_a'].exp()+.0001, single['log_k_b'].exp()+.0001)
    density = marginal.log_prob_from_log_coordinates(*logs).sum(-1) + gaussian_copula_log_prob(
        *marginal.log_tails_from_log_coordinates(*logs), single['k_L_cholesky'])

    def logarithmic_transport(v):
        first, second = kumaraswamy_component_transport(single, v, True)
        return torch.where(first <= second, first, second).squeeze(0)

    log_jacobian = torch.autograd.functional.jacobian(logarithmic_transport, z)
    normal_density = -.5*z.square().sum() - 1.5*math.log(2*np.pi)
    torch.testing.assert_close((density + logarithmic_transport(z).sum()
                               + torch.linalg.slogdet(log_jacobian)[1]).squeeze(), normal_density, rtol=2e-11, atol=2e-11)
    single = {name: value[1:] for name, value in p.items()}
    point = kumaraswamy_component_transport(single, z).squeeze(0)
    jacobian = torch.autograd.functional.jacobian(lambda v: kumaraswamy_component_transport(single, v).squeeze(0), z)
    density = kumaraswamy_mixture_log_prob(single, point)
    torch.testing.assert_close(density + torch.linalg.slogdet(jacobian)[1], normal_density, rtol=2e-11, atol=2e-11)


def test_transport_moments_use_the_same_untruncated_law():
    p = parameters()
    normal = tensor(ndtri(qmc.Sobol(3, scramble=True, seed=4192).random_base2(15)))
    points = kumaraswamy_component_transport(p, normal).numpy()
    a, b = p['log_k_a'].exp().numpy()+.0001, p['log_k_b'].exp().numpy()+.0001
    expected = np.exp(np.log(b) + betaln(1+1/a, b))
    np.testing.assert_allclose(points.mean(0), expected, atol=1e-4, rtol=0)


def test_traced_density_matches_values_and_gradients_on_displaced_points():
    p = parameters()
    chol, logits = p['k_L_cholesky'], p['weights_logits']

    def density(a, b, x):
        return kumaraswamy_mixture_log_prob(dict(log_k_a=a, log_k_b=b, k_L_cholesky=chol, weights_logits=logits), x)

    initial = tensor([[.2, .3, .4], [.7, .8, .6]])
    traced = torch.jit.trace(density, (p['log_k_a'], p['log_k_b'], initial), check_trace=False)
    displaced = tensor([[1e-300, 1e-12, .9], [1e-7, 1-1e-15, 1e-10]])
    first = ((p['log_k_a']+.2).requires_grad_(), (p['log_k_b']-.1).requires_grad_(), displaced)
    second = tuple(v.detach().clone().requires_grad_(v.requires_grad) for v in first)
    original, compiled = density(*first), traced(*second)
    torch.testing.assert_close(compiled, original)
    for expected, actual in zip(torch.autograd.grad(original.sum(), first[:2]), torch.autograd.grad(compiled.sum(), second[:2])):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=1e-9)
