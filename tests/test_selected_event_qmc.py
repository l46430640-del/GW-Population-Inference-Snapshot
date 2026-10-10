from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, qmc, truncnorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from selected_event_qmc import selected_event_cloud, selected_normal_mass, selected_normal_ppf


@pytest.mark.parametrize('observed', [-.2, 0., .3, .99, 1.2])
def test_analytic_selected_cdf_and_inverse_against_independent_quadrature(observed):
    sigma = .04
    def kernel(x):
        return (.05+.95*x*x)*norm.pdf((x-observed)/sigma)/sigma
    normalizer = quad(kernel, 0, 1, epsabs=1e-16, epsrel=1e-11)[0]
    assert selected_normal_mass(1, observed, sigma) == pytest.approx(normalizer, rel=1e-10)
    probabilities = np.array([1e-7, .001, .2, .5, .9, .999, 1-1e-7])
    points = selected_normal_ppf(probabilities, observed, sigma)
    assert np.all(np.diff(points) > 0)
    recovered = np.array([quad(kernel, 0, x, epsabs=1e-16, epsrel=1e-11)[0]/normalizer for x in points])
    np.testing.assert_allclose(recovered, probabilities, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(selected_normal_ppf([0, 1], observed, sigma), [0, 1])


def test_cloud_preserves_all_sobol_points_and_uses_independent_scrambles():
    observed = np.array([[.2, .5, .8], [.01, 1.04, .6]])
    sigma = np.array([.04, .05, .04])
    cloud = selected_event_cloud(observed, sigma, 9, 848101)
    duplicate = selected_event_cloud(observed, sigma, 9, 848101)
    independent = selected_event_cloud(observed, sigma, 11, 848102)
    np.testing.assert_array_equal(cloud['data'], duplicate['data'])
    assert not np.array_equal(cloud['data'], independent['data'][:, :512])
    assert len(set(cloud['event_seeds'])) == 2
    for i, y in enumerate(observed):
        expected = qmc.Sobol(3, scramble=True, seed=int(cloud['event_seeds'][i])).random_base2(9)
        points = cloud['data'][i, :, :3]
        recovered = np.column_stack([
            selected_normal_mass(points[:, 0], y[0], sigma[0])/selected_normal_mass(1, y[0], sigma[0]),
            truncnorm.cdf(points[:, 1:], -y[1:]/sigma[1:], (1-y[1:])/sigma[1:], loc=y[1:], scale=sigma[1:])])
        np.testing.assert_allclose(recovered, expected, rtol=0, atol=2e-14)
        for axis in range(3):
            np.testing.assert_array_equal(np.sort((expected[:, axis]*512).astype(int)), np.arange(512))
    assert np.all(cloud['data'][..., 3] == 1)


@pytest.mark.parametrize('y', [[.3, .5, .8], [-.02, .97, 1.01]])
def test_event_moments_against_quadrature(y):
    sigma = np.array([.04, .05, .04])
    data = selected_event_cloud(np.array([y]), sigma, 11, 848102)['data'][0]
    for axis in range(3):
        def kernel(x):
            return norm.pdf((x-y[axis])/sigma[axis])/sigma[axis]*(.05+.95*x*x if axis == 0 else 1.)
        mass = quad(kernel, 0, 1, epsabs=1e-12)[0]
        for power in (1, 2):
            exact = quad(lambda x: x**power*kernel(x), 0, 1, epsabs=1e-12)[0]/mass
            assert np.mean(data[:, axis]**power) == pytest.approx(exact, abs=1e-4)


@pytest.mark.parametrize('probability', [[-.1], [1.1], [np.nan]])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        selected_normal_ppf(probability, .5, .1)


@pytest.mark.parametrize('sigma,power', [([0, .1, .1], 9), ([.1, .1, .1], 2.5), ([.1, .1], 9)])
def test_invalid_cloud_configuration_is_rejected(sigma, power):
    with pytest.raises(ValueError):
        selected_event_cloud([[.5, .5, .5]], sigma, power, 1)
