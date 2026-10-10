import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import solve_sylvester, sqrtm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from binned_gp_benchmark import BinnedGPPotential, controlled_log_event_weights, controlled_selection, covariance_factors, population_functionals
from check_external_gp_jitter import symmetric_field
from sample_lisa_transport import compile_potential, value_gradient
from run_symmetric_gp_pilot import check_coordinates
from symmetric_gp_benchmark import SymmetricBinnedGPPotential, symmetric_correlated_field, symmetric_square_root


@pytest.fixture(scope="module", autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def example(bins):
    settings = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    weights = controlled_log_event_weights([[.13, .8, .5], [.6, .4, .92]], [.04, .05, .04], bins)
    return SymmetricBinnedGPPotential(bins, weights, controlled_selection(bins), settings)


@pytest.mark.parametrize("matrix", [np.eye(3), np.diag([1., 1., 1.000001]),
                                  np.array([[2., .5, .2], [.5, 2., .5], [.2, .5, 1.]])])
def test_principal_root_and_gradient_match_independent_sylvester_solution(matrix):
    argument = torch.tensor(matrix, requires_grad=True)
    weight = torch.arange(9.).reshape(3, 3)/7
    root = symmetric_square_root(argument)
    gradient, = torch.autograd.grad((root*weight).sum(), argument)
    expected_root = np.real(sqrtm(matrix))
    expected_gradient = solve_sylvester(expected_root, expected_root, (weight+weight.T).numpy()/2)
    np.testing.assert_allclose(root.detach(), expected_root, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(symmetric_square_root, (argument,), atol=1e-7, rtol=1e-6)


def test_batched_root_covariance_and_gradients():
    matrix = torch.stack([torch.eye(3), torch.eye(3)*2]).requires_grad_()
    root = symmetric_square_root(matrix)
    gradient, = torch.autograd.grad(root.sum(), matrix)
    np.testing.assert_allclose(root.detach() @ root.detach().transpose(-1, -2), matrix.detach(), atol=1e-14)
    np.testing.assert_allclose(gradient[0], np.full((3, 3), .5), atol=1e-14)
    np.testing.assert_allclose(gradient[1], np.full((3, 3), 1/(2*np.sqrt(2))), atol=1e-14)


def test_root_rejects_indefinite_matrices():
    with pytest.raises(ValueError, match="positive definite"):
        symmetric_square_root(torch.diag(torch.tensor([1., 0., -1.])))


def test_field_prior_and_coordinate_rotation_match_cholesky_model():
    potential = example(3)
    point = potential.initial(251)
    point[:4] = torch.tensor([np.log(1.4), np.log(.2), np.log(.4), np.log(.7)])
    point[4:] *= 20
    matrices = covariance_factors(point[1:4], potential.centers, potential.jitter)
    rotations = [torch.linalg.solve_triangular(torch.linalg.cholesky(matrix), symmetric_square_root(matrix), upper=False)
                 for matrix in matrices]
    rotation = torch.kron(torch.kron(rotations[0].contiguous(), rotations[1].contiguous()), rotations[2].contiguous())
    np.testing.assert_allclose(rotation.T@rotation, torch.eye(27), rtol=0, atol=1e-10)
    rotated = point.clone()
    rotated[4:] = rotation@point[4:]
    settings = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    original = BinnedGPPotential(3, potential.log_event_weights, potential.selection, settings)
    np.testing.assert_allclose(potential.field(point), original.field(rotated), rtol=0, atol=1e-11)
    assert float(potential.log_prior(point)) == pytest.approx(float(original.log_prior(rotated)), abs=1e-10)
    assert float(potential(point)) == pytest.approx(float(original(rotated)), abs=1e-10)


@pytest.mark.parametrize("bins,lengths", [(3, [.01, .02, .03]), (6, [.2, .4, .6]), (12, [.3, .5, 1.])])
def test_field_values_gradients_and_trace_on_degenerate_or_smooth_kernels(bins, lengths):
    potential = example(bins)
    point = potential.initial(256)
    point[1:4] = torch.tensor(np.log(lengths))
    point[4:] *= 20
    expected, _ = symmetric_field(point.numpy(), bins, potential.jitter)
    np.testing.assert_allclose(potential.field(point), expected, rtol=0, atol=2e-9)
    value, gradient = value_gradient(potential, point)
    assert torch.isfinite(gradient).all()
    direction = torch.sin(torch.arange(potential.dimension))/np.sqrt(potential.dimension)
    step = 1e-4
    derivative = (potential(point+step*direction)-potential(point-step*direction))/(2*step)
    assert float(gradient@direction) == pytest.approx(float(derivative), abs=2e-5)
    traced, checks = compile_potential(potential, point, 1e-7)
    assert len(checks) == 3
    assert float(traced(point)) == pytest.approx(float(value), abs=1e-10)


def test_coordinate_check_uses_the_symmetric_reference_and_extreme_hyperparameters():
    potential = example(6)
    settings = json.loads((ROOT / "configs/external_benchmark.json").read_text())["binned_gp"]
    config = json.loads((ROOT / "configs/external_gp_symmetric_pilot.json").read_text())
    reference = SymmetricBinnedGPPotential(6, potential.log_event_weights, potential.selection, settings, jitter=1e-10)
    coordinates = np.stack([potential.initial(i).numpy() for i in range(10)])
    coordinates[:, 2] = np.linspace(np.log(.1), np.log(.9), 10)
    report, arrays = check_coordinates(potential, reference, coordinates, config)
    assert report["derivatives_passed"]
    assert {0, 9} <= set(report["draws_checked"])
    assert report["jitter_difference_sd"] < .2
    assert arrays["directional_derivative_automatic"].shape[1] == 5
    for i, point in enumerate(coordinates):
        assert arrays["covariance_jitter_log_target_difference"][i] == pytest.approx(
            float(potential(torch.tensor(point))-reference(torch.tensor(point))), abs=1e-12)


def test_empty_coordinate_check_cannot_pass_validation():
    potential = example(3)
    config = json.loads((ROOT / "configs/external_gp_symmetric_pilot.json").read_text())
    report, arrays = check_coordinates(potential, potential, np.empty((0, potential.dimension)), config)
    assert not report["derivatives_passed"]
    assert report["jitter_difference_sd"] is None
    assert arrays["covariance_jitter_log_target_difference"].size == 0


@pytest.mark.parametrize("bins", [6, 12])
def test_saved_symmetric_pilot_reconstructs_in_its_sampled_coordinates(bins):
    directory = ROOT / "results/analysis/external_gp_symmetric_pilot"
    path = directory / f"bins_{bins}.npz"
    report = json.loads(path.with_suffix(".json").read_text())
    signature = json.loads((directory / "target.json").read_text())
    config = json.loads((ROOT / "configs/external_gp_symmetric_pilot.json").read_text())
    protocol = json.loads((ROOT / config["protocol"]).read_text())
    assert report["configuration"] == config
    assert signature["input_fields"] == ["observed", "sigma"]
    assert signature["source_sha256"] == report["source_sha256"]
    assert signature["coordinate_basis"] == report["coordinate_basis"] == config["coordinate_basis"]
    for name, expected in signature["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        assert "test_catalog" not in name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["draw_product_sha256"]
    assert report["status"] == "complete" and report["error"] is None
    assert report["warmup_completed"] == config["warmup"] == 100
    assert report["samples_completed"] == config["retained"] == 100
    assert report["divergences"] == []
    assert report["stability_check_passed"] and report["ready_for_production_sampling"]
    assert not report["eligible_for_scientific_comparisons"]
    assert report["process_elapsed_seconds"] < config["maximum_seconds"]
    assert report["runtime"]["peak_working_set_bytes"] < config["maximum_resident_gib"]*1024**3
    with np.load(path) as product, np.load(ROOT / config["training_catalog"]) as catalog:
        expected = controlled_log_event_weights(catalog["observed"], catalog["sigma"], bins, 64)
        np.testing.assert_allclose(product["log_event_weights"], expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(product["selection_coefficients"], controlled_selection(bins), rtol=0, atol=1e-15)
        potential = SymmetricBinnedGPPotential(bins, product["log_event_weights"], product["selection_coefficients"],
                                              protocol["binned_gp"], config["covariance_axis_jitter"])
        reference = SymmetricBinnedGPPotential(bins, product["log_event_weights"], product["selection_coefficients"],
                                              protocol["binned_gp"], config["verification_covariance_axis_jitter"])
        assert product["coordinate"].shape == (config["retained"], potential.dimension)
        assert np.isfinite(product["coordinate"]).all()
        assert np.all(np.diff(product["progress"][:, 2]) > 0)
        for index in (0, 49, 99):
            coordinate = torch.tensor(product["coordinate"][index])
            expected_field, _ = symmetric_field(coordinate.numpy(), bins, potential.jitter)
            np.testing.assert_allclose(potential.field(coordinate), expected_field, rtol=0, atol=2e-8)
            features = population_functionals(potential.field(coordinate), potential.selection, bins)
            np.testing.assert_allclose(product["features"][index, :12], features, rtol=0, atol=1e-12)
            assert product["features"][index, -1] == pytest.approx(float(-potential(coordinate)), abs=1e-10)
        verification, arrays = check_coordinates(potential, reference, product["coordinate"], config)
        for name, reconstructed in arrays.items():
            np.testing.assert_allclose(product[name], reconstructed, rtol=0, atol=1e-9)
        assert verification["draws_checked"] == report["coordinate_verification"]["draws_checked"]
        assert verification["derivatives_passed"]
        assert verification["jitter_difference_sd"] < protocol["evaluation"]["maximum_log_likelihood_difference_sd"]
        assert report["covariance_jitter_log_target_difference_sd"] == pytest.approx(
            verification["jitter_difference_sd"], abs=1e-10)
        error = np.abs(product["directional_derivative_automatic"]-product["directional_derivative_finite_difference"])
        assert error.shape == (len(verification["draws_checked"]), 5)
        assert report["coordinate_verification"]["maximum_directional_derivative_error"] == pytest.approx(error.max(), abs=1e-14)
        assert error.max() < config["maximum_directional_derivative_error"]
