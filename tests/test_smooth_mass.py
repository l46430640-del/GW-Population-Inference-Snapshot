import numpy as np
import pytest
import torch

from gwpopulation.smooth_mass import SmoothLogMassDensity


def distribution(coefficients, length=.35):
    return SmoothLogMassDensity(coefficients, np.log10(3), np.log10(150), length_scale=length)


def test_smooth_mass_density_normalization_and_cdf_derivative():
    coefficients = torch.linspace(-1, 1, 17, dtype=torch.float64)
    dist = distribution(coefficients)
    x = torch.linspace(0, 1, 16001, dtype=torch.float64)
    density = dist.log_prob(x).exp()
    assert torch.trapezoid(density, x).item() == pytest.approx(1, abs=1e-9)
    assert dist.cdf(torch.tensor(0.)).item() == 0
    assert dist.cdf(torch.tensor(1.)).item() == 1
    inside = torch.tensor([.17, .44, .81], dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(dist.cdf(inside).sum(), inside)[0]
    torch.testing.assert_close(derivative, dist.log_prob(inside).exp(), atol=1e-10, rtol=1e-10)


def test_smooth_mass_has_no_density_jumps_at_old_local_boundaries():
    dist = distribution(torch.linspace(-2, 2, 17, dtype=torch.float64))
    x = (torch.log10(torch.tensor([5., 50.], dtype=torch.float64)) - np.log10(3)) / np.log10(50)
    left, right = dist.log_prob(x - 1e-8).exp(), dist.log_prob(x + 1e-8).exp()
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


def test_smooth_mass_inverse_cdf_is_stable_and_batched():
    coefficients = torch.stack([torch.zeros(17), torch.linspace(-2, 2, 17)]).double()
    dist = distribution(coefficients)
    p = torch.linspace(.0001, .9999, 101, dtype=torch.float64)[:, None].expand(-1, 2)
    torch.testing.assert_close(dist.cdf(dist.icdf(p)), p, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(dist.icdf(p)[:, 0], p[:, 0], atol=1e-12, rtol=1e-12)
    assert torch.isneginf(dist.log_prob(torch.tensor(-.1))).all()


def test_longer_length_scale_reduces_prior_curvature():
    short = distribution(torch.zeros(17, dtype=torch.float64), .2)
    long = distribution(torch.zeros(17, dtype=torch.float64), .6)
    second_difference = torch.diff(torch.eye(17, dtype=torch.float64), n=2, dim=0)
    short_energy = (second_difference @ short.log_height_factor).square().sum()
    long_energy = (second_difference @ long.log_height_factor).square().sum()
    assert long_energy < short_energy / 5


def test_density_gradient_propagates_to_whitened_coefficients():
    coefficients = torch.linspace(-.5, .5, 17, dtype=torch.float64, requires_grad=True)
    dist = distribution(coefficients)
    gradient = torch.autograd.grad(dist.log_prob(torch.tensor([.12, .53, .86])).sum(), coefficients)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > .1
