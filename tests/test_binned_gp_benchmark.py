import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from scipy.stats import norm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from binned_gp_benchmark import (
    BinnedGPPotential, controlled_log_event_weights, controlled_selection,
    correlated_field, covariance_factors, grid_geometry, population_functionals,
    posterior_log_event_weights, shape_log_likelihood,
)
from sample_lisa_transport import compile_potential, value_gradient
from check_external_gp_jitter import symmetric_field


@pytest.fixture(scope="module", autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def example(bins=3):
    setting = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    observed = [[.13, .8, .5], [.6, .4, .92]]
    weights = controlled_log_event_weights(observed, [.04, .05, .04], bins)
    return BinnedGPPotential(bins, weights, controlled_selection(bins), setting)


def test_selection_coefficients_match_independent_integration():
    lower, upper = grid_geometry(6)
    direct = [quad(lambda x: .05 + .95*x*x, lo[0], hi[0])[0] * np.prod(hi[1:]-lo[1:])
              for lo, hi in zip(lower, upper)]
    np.testing.assert_allclose(controlled_selection(6), direct, rtol=1e-14)
    assert sum(direct) == pytest.approx(.05 + .95/3, abs=2e-15)


def test_positive_quadrature_matches_adaptive_event_integrals():
    observed = np.array([[.13, .8, .5], [-.08, .01, 1.07]])
    sigma = np.array([.04, .05, .04])
    bins = 6
    log_weights = controlled_log_event_weights(observed, sigma, bins, order=32)
    fine = controlled_log_event_weights(observed, sigma, bins, order=64)
    np.testing.assert_allclose(log_weights, fine, atol=1e-10, rtol=0)
    for event, measurement in enumerate(observed):
        integrals = []
        for axis in range(3):
            def density(x):
                selection = .05 + .95*x*x if axis == 0 else 1.
                return selection * norm.pdf(x, measurement[axis], sigma[axis])
            values = np.array([quad(density, i/bins, (i+1)/bins, epsabs=1e-100,
                                    epsrel=1e-11)[0] for i in range(bins)])
            integrals.append(values/values.sum())
        expected = np.einsum("i,j,k->ijk", *integrals).ravel()
        np.testing.assert_allclose(np.exp(log_weights[event]), expected, rtol=1e-10, atol=1e-15)


def test_posterior_weights_remove_the_event_prior_once_and_include_boundaries():
    data = np.array([[[0., .2, .2, 2.], [1., 1., 1., 4.], [.2, .2, .2, 1.]]])
    actual = np.exp(posterior_log_event_weights(data, 2))[0]
    expected = np.zeros(8)
    expected[[0, 7]] = [.5, 1/12]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(np.exp(posterior_log_event_weights(np.repeat(data, 2, axis=1), 2)),
                               actual[None], atol=1e-15)


def test_shape_likelihood_is_the_poisson_likelihood_marginalized_over_rate():
    potential = example(2)
    field = torch.linspace(-.4, .7, 8)
    direct = shape_log_likelihood(field, potential.log_event_weights, potential.log_selection)
    log_z = torch.logsumexp(field, 0) - np.log(8)
    event = torch.logsumexp(potential.log_event_weights + field, -1) - log_z
    alpha = torch.exp(torch.logsumexp(potential.log_selection + field, 0) - log_z)
    count = len(event)
    integrated = quad(lambda r: np.exp(-r*float(alpha))*r**(count-1), 0, np.inf)[0]
    assert float(direct) == pytest.approx(float(event.sum()) + np.log(integrated) - gammaln(count), abs=1e-11)
    shifted = shape_log_likelihood(field+100., potential.log_event_weights, potential.log_selection)
    assert float(shifted) == pytest.approx(float(direct), abs=1e-12)


def test_zero_detection_bins_remain_in_intrinsic_support():
    potential = example(2)
    selection = controlled_selection(2)
    selection[0] = 0.
    setting = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    potential = BinnedGPPotential(2, potential.log_event_weights, selection, setting)
    assert potential.dimension == 12
    assert torch.isneginf(potential.log_selection[0])
    assert torch.isfinite(potential(potential.initial(1)))
    feature = population_functionals(torch.zeros(8), selection, 2)
    assert feature[0] == pytest.approx(.5)
    assert feature[6] == pytest.approx(selection.sum())


def test_kronecker_field_and_gradient_match_full_covariance():
    bins = 3
    centers = torch.arange(bins)/bins + .5/bins
    coordinate = torch.tensor(np.r_[np.log(1.4), np.log([.2, .4, .7]),
                                    np.random.default_rng(182).normal(size=bins**3)])
    def structured(x):
        return correlated_field(x[4:], x[1:4], x[0].exp(), centers, 1e-8)
    def dense(x):
        matrices = covariance_factors(x[1:4], centers, 1e-8)
        covariance = x[0].exp().square()*torch.kron(torch.kron(matrices[0], matrices[1]), matrices[2])
        return torch.linalg.cholesky(covariance) @ x[4:]
    np.testing.assert_allclose(structured(coordinate), dense(coordinate), rtol=1e-10, atol=1e-10)
    a, ga = value_gradient(lambda x: (structured(x)**2).sum(), coordinate)
    b, gb = value_gradient(lambda x: (dense(x)**2).sum(), coordinate)
    assert float(a) == pytest.approx(float(b), abs=1e-10)
    np.testing.assert_allclose(ga, gb, rtol=1e-8, atol=1e-8)


def test_log_coordinate_prior_includes_amplitude_jacobian():
    potential = example()
    first, second = potential.initial(1), potential.initial(2)
    second[:4] += torch.tensor([.4, -.1, .2, .3])
    def direct(x):
        return (torch.distributions.HalfNormal(potential.sigma_scale).log_prob(x[0].exp()) + x[0]
                + torch.distributions.Normal(potential.log_length_center, potential.log_length_scale).log_prob(x[1:4]).sum()
                + torch.distributions.Normal(0., 1.).log_prob(x[4:]).sum())
    assert float(potential.log_prior(second)-potential.log_prior(first)) == pytest.approx(
        float(direct(second)-direct(first)), abs=1e-12)


def test_potential_directional_derivative_and_trace_agree():
    potential = example()
    point = potential.initial(83)
    value, gradient = value_gradient(potential, point)
    direction = torch.linspace(-.2, .3, potential.dimension)
    step = 1e-5
    difference = (potential(point+step*direction)-potential(point-step*direction))/(2*step)
    assert float(gradient@direction) == pytest.approx(float(difference), abs=1e-7)
    traced, checks = compile_potential(potential, point, 1e-7)
    assert len(checks) == 3
    assert float(traced(point)) == pytest.approx(float(value), abs=1e-10)


def test_uniform_population_moments_and_partition_tails_are_exact():
    for bins in (2, 6, 12):
        features = population_functionals(torch.zeros(bins**3), controlled_selection(bins), bins)
        expected = [.5]*3 + [1/12]*3 + [.05+.95/3, .25, .2] + [0.]*3
        np.testing.assert_allclose(features, expected, atol=1e-14)


def test_symmetric_root_candidate_has_the_same_gaussian_covariance():
    bins, jitter, sigma = 2, 1e-8, 1.4
    lengths = np.array([.2, .4, .7])
    centers = torch.tensor((np.arange(bins)+.5)/bins)
    matrices = [x.numpy() for x in covariance_factors(torch.tensor(np.log(lengths)), centers, jitter)]
    covariance = sigma**2*np.kron(np.kron(matrices[0], matrices[1]), matrices[2])
    columns = []
    for white in np.eye(bins**3):
        field, residual = symmetric_field(np.r_[np.log(sigma), np.log(lengths), white], bins, jitter)
        columns.append(field)
        assert residual < 1e-13
    root = np.stack(columns, axis=1)
    np.testing.assert_allclose(root@root.T, covariance, rtol=1e-12, atol=1e-12)


def test_invalid_event_information_and_selection_are_rejected():
    with pytest.raises(ValueError, match="positive event priors"):
        posterior_log_event_weights(np.zeros((1, 2, 4)), 2)
    with pytest.raises(ValueError, match="unit support"):
        posterior_log_event_weights(np.array([[[1.01, .1, .3, 1.]]]), 2)
    with pytest.raises(ValueError, match="positive measurement scales"):
        controlled_log_event_weights([[.2, .3, .4]], [0., .1, .1], 2)
    potential = example(2)
    setting = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    with pytest.raises(ValueError, match="nonzero detection"):
        BinnedGPPotential(2, potential.log_event_weights, np.zeros(8), setting)


@pytest.mark.parametrize("bins", [6, 12])
def test_retained_pilot_reconstructs_from_its_declared_training_information(bins):
    directory = ROOT / "results/analysis/external_gp_pilot"
    path = directory / f"bins_{bins}.npz"
    if not path.exists():
        pytest.skip("Bounded pilot has not been run at this resolution")
    report = json.loads(path.with_suffix(".json").read_text())
    signature = json.loads((directory / "target.json").read_text())
    assert signature["input_fields"] == ["observed", "sigma"]
    assert signature["source_sha256"] == report["source_sha256"]
    for name, expected in signature["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        assert "test_catalog" not in name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["draw_product_sha256"]
    assert report["status"] == "complete"
    assert report["warmup_completed"] == report["samples_completed"] == 100
    assert report["divergences"] == []
    assert not report["eligible_for_scientific_comparisons"]
    assert report["process_elapsed_seconds"] < report["configuration"]["maximum_seconds"]
    assert report["runtime"]["peak_working_set_bytes"] < 2*1024**3
    setting = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    with np.load(path) as product, np.load(ROOT / report["configuration"]["training_catalog"]) as catalog:
        expected = controlled_log_event_weights(catalog["observed"], catalog["sigma"], bins, 64)
        np.testing.assert_allclose(product["log_event_weights"], expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(product["selection_coefficients"], controlled_selection(bins), rtol=0, atol=1e-15)
        potential = BinnedGPPotential(bins, product["log_event_weights"], product["selection_coefficients"],
                                     setting, report["configuration"]["covariance_axis_jitter"])
        reference = BinnedGPPotential(bins, product["log_event_weights"], product["selection_coefficients"],
                                     setting, report["configuration"]["verification_covariance_axis_jitter"])
        assert np.isfinite(product["coordinate"]).all()
        assert np.all(np.diff(product["progress"][:, 2]) > 0)
        for index in (0, 49, 99):
            coordinate = torch.tensor(product["coordinate"][index])
            field = potential.field(coordinate)
            recomputed = population_functionals(field, potential.selection, bins)
            np.testing.assert_allclose(product["features"][index, :12], recomputed, atol=1e-12)
            assert product["features"][index, -1] == pytest.approx(float(-potential(coordinate)), abs=1e-10)
            assert product["covariance_jitter_log_target_difference"][index] == pytest.approx(
                float(potential(coordinate)-reference(coordinate)), abs=1e-10)
        assert report["covariance_jitter_log_target_difference_sd"] == pytest.approx(
            np.std(product["covariance_jitter_log_target_difference"]), abs=1e-15)
