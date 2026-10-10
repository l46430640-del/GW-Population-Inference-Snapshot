import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from lisa_noisy_catalog import SIGMA, posterior_from_measurements
from build_lisa_analysis_assets import LISASNR


def test_importance_posterior_matches_known_gaussian_measurement_experiment():
    slope = np.array([20., -8., 2.])
    y, observed = np.array([.5, .5, .5]), 15.
    rho = lambda points: 10 + points @ slope
    covariance = np.linalg.inv(np.diag(SIGMA**-2) + np.outer(slope, slope))
    expected = covariance @ (y / SIGMA**2 + slope * (observed - 10))
    draws, center, diagnostic = posterior_from_measurements(y, observed, rho, 2000, np.random.default_rng(335))
    assert diagnostic['proposal_ess'] >= 8000
    np.testing.assert_allclose(center, expected, atol=.001)
    np.testing.assert_allclose(draws.mean(axis=0), expected, atol=.0015)
    np.testing.assert_allclose(draws.std(axis=0), np.sqrt(np.diag(covariance)), rtol=.06)


@pytest.mark.skipif(not (ROOT / 'model_assets/lisa_analysis/phenom_d_grid.npz').exists(), reason='Grid not built')
def test_torch_and_scipy_snr_interpolation_agree_and_are_differentiable():
    model = LISASNR()
    points = torch.tensor(np.random.default_rng(1).uniform(.1, .9, (32, 3)), requires_grad=True)
    value = model.torch(points)
    np.testing.assert_allclose(value.detach().numpy(), model(points.detach().numpy()), rtol=1e-10, atol=1e-10)
    gradient = torch.autograd.grad(value.sum(), points)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


def test_inverse_snr_proposal_resolves_a_curved_high_snr_likelihood():
    from scipy.stats import norm

    class MonotoneSNR:
        def __call__(self, x):
            return np.expm1(10 - 4 * x[:, 1])

        def redshift_curve(self, points):
            return np.broadcast_to(10 - 4 * np.linspace(0, 1, 97), (len(points), 97))

    rho = MonotoneSNR()
    y = np.array([.5, .5, .5])
    observed = np.expm1(10 - 4 * .52)
    r = observed + np.linspace(-10, 10, 20001)
    z = (10 - np.log1p(r)) / 4
    weights = norm.pdf(r-observed) * norm.pdf((z-y[1])/SIGMA[1]) / (4*(1+r))
    weights /= weights.sum()
    expected = np.sum(weights*z)
    std = np.sqrt(np.sum(weights*(z-expected)**2))
    draws, center, diagnostic = posterior_from_measurements(y, observed, rho, 3000, np.random.default_rng(85))
    assert diagnostic['proposal_ess'] >= 12000
    assert abs(center[1] - expected) < std * .08
    np.testing.assert_allclose(draws[:, 1].std(), std, rtol=.08)
    np.testing.assert_allclose(draws[:, [0, 2]].mean(axis=0), [.5, .5], atol=.0015)


@pytest.mark.skipif(not (ROOT / 'model_assets/lisa_analysis/phenom_d_grid.npz').exists(), reason='Grid not built')
def test_redshift_curve_matches_full_interpolator():
    rho = LISASNR()
    points = np.random.default_rng(44).uniform(.01, .99, (50, 3))
    curves = rho.redshift_curve(points[:, [0, 2]])
    position = points[:, 1] * (curves.shape[1]-1)
    lower = position.astype(int)
    log_rho = (curves[np.arange(50), lower] * (1-position+lower)
               + curves[np.arange(50), lower+1] * (position-lower))
    np.testing.assert_allclose(np.expm1(log_rho), rho(points), rtol=1e-11, atol=1e-11)
