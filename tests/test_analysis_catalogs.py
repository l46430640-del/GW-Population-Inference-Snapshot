import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.spatial.distance import jensenshannon
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_population_morphologies import (
    CORR, OFFSET, SIGMA, SLOPE, THRESHOLD, catalog, event_posterior, snr, truth_pdf,
)


@pytest.mark.parametrize('name', ['correlated', 'bimodal', 'curved'])
def test_morphology_density_normalization(name):
    points = qmc.Sobol(3, scramble=True, seed=73).random_base2(15)
    density = truth_pdf(name, points)
    assert abs(density.mean() - 1) < .003
    assert (density >= 0).all()


def test_event_posterior_uses_realized_detection_statistic():
    rng = np.random.default_rng(82)
    y = np.array([.5, .5, .5])
    _, mean, covariance = event_posterior(y, snr(y) + 1., 100, rng)
    assert not np.allclose(mean, y)
    expected = np.diag(SIGMA**-2) + np.outer(SLOPE, SLOPE)
    np.testing.assert_allclose(np.linalg.inv(covariance), expected)
    # The conditional Gaussian exactly reproduces the joint likelihood ratio.
    a, b = np.array([.45, .5, .48]), np.array([.5, .46, .48])
    rho = snr(y) + 1.
    log_like = lambda x: -.5 * np.sum(((y - x) / SIGMA)**2) - .5 * (rho - snr(x))**2
    log_posterior = lambda x: -.5 * (x - mean) @ expected @ (x - mean)
    assert log_like(a) - log_like(b) == pytest.approx(log_posterior(a) - log_posterior(b))


def test_detection_probability_matches_gaussian_threshold():
    rng = np.random.default_rng(22)
    point = np.array([.4, .7, .5])
    detection = (snr(point) + rng.normal(size=100000) > THRESHOLD).mean()
    assert detection == pytest.approx(ndtr(OFFSET + point @ SLOPE - THRESHOLD), abs=.005)


def test_catalog_measurements_are_noisy_and_independent_of_truth_center():
    _, data = catalog('correlated', 17, n_samples=16)
    standardized_noise = (data['measurement'] - data['truth_intrinsic']) / SIGMA
    assert standardized_noise.std() > .8
    assert not np.allclose(data['posterior_centers'], data['truth_intrinsic'][data['detected']])


def test_joint_distance_detects_dependence_with_identical_marginals():
    diagonal, antidiagonal = np.eye(5), np.fliplr(np.eye(5))
    assert jensenshannon(diagonal.sum(axis=0), antidiagonal.sum(axis=0), base=2) == 0
    assert jensenshannon(diagonal.ravel(), antidiagonal.ravel(), base=2) > .8
