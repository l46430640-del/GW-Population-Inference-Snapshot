import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_selected_mixture import selected_model
from external_selected_evaluation import (coarsen, copula_cells, copula_exact_features,
    copula_log_density, event_log_integrals, expand_gp_cells, gp_gradient,
    interval_score, js_gradient, normal_points)


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    old = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_num_threads(old[0])
    torch.set_default_dtype(old[1])


def uniform_parameters():
    return {'log_k_a': np.log(np.full((2, 3), .9999)),
            'log_k_b': np.log(np.full((2, 3), .9999)),
            'weights_logits': np.log([.4, .6]),
            'k_L_cholesky': np.broadcast_to(np.eye(3), (2, 3, 3)).copy()}


def test_numpy_density_matches_frozen_core():
    torch.set_default_dtype(torch.float64)
    model = selected_model()
    catalog = np.load(ROOT / 'results/analysis/selected_mixture/train_catalog.npz')
    points = np.vstack([catalog['data'][:4, :16, :3].reshape(-1, 3), [[0, 1, .5], [-.1, .5, .5]]])
    for chain in range(4):
        samples = np.load(ROOT / f'results/analysis/selected_mixture/nuts_chain_{chain}.npz')
        for i in (0, 399, 799):
            params = {name: samples[name][i] for name in samples.files}
            expected = model._log_pdf_phys_from_params({k: torch.tensor(v) for k, v in params.items()}, torch.tensor(points)).detach().numpy()
            actual = copula_log_density(params, points)
            # The frozen model rounds Gaussianized coordinates to float32.
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)


def test_uniform_density_event_prior_and_selection():
    parameters = uniform_parameters()
    data = np.ones((2, 32, 4))
    data[..., :3] = np.random.default_rng(8).uniform(size=(2, 32, 3))
    data[..., 3] = 2.
    np.testing.assert_allclose(event_log_integrals(parameters, data), -np.log(2.), atol=1e-13)
    exact, cdf = copula_exact_features(parameters)
    np.testing.assert_allclose(exact[:3], .5, atol=1e-14)
    assert exact[6] == pytest.approx(.05+.95/3)
    np.testing.assert_allclose(cdf, np.tile([.25, .5, .75], 3))
    data[..., 3] = 0
    with pytest.raises(ValueError):
        event_log_integrals(parameters, data)


def test_population_cells_and_nested_partitions():
    product = copula_cells(uniform_parameters(), normal_points(16, 10), 24)
    assert product['intrinsic'].sum() == pytest.approx(1.)
    assert product['detected'].sum() == pytest.approx(1.)
    assert product['transport_alpha'] == pytest.approx(.05+.95/3, abs=1e-5)
    np.testing.assert_allclose(coarsen(product['intrinsic'], 24, 2), 1/8, atol=1e-4)


@pytest.mark.parametrize('detected', [False, True])
def test_gp_subdivision_and_gradient_are_exact_duals(detected):
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(8))
    g = rng.normal(size=12**3)
    fine = expand_gp_cells(p, 2, 12, detected)
    assert fine @ g == pytest.approx(p @ gp_gradient(g, 2, 12, detected))
    if not detected:
        np.testing.assert_allclose(coarsen(fine, 12, 2), p)


def test_distance_influence_and_proper_interval_score():
    p, t = np.array([.2, .3, .5]), np.array([.4, .4, .2])
    distance, gradient = js_gradient(p, t)
    direction = np.array([.3, -.5, .2])
    h = 1e-6
    derivative = (js_gradient(p+h*direction, t)[0]-js_gradient(p-h*direction, t)[0])/(2*h)
    assert derivative == pytest.approx(gradient @ direction, abs=1e-8)
    assert distance > 0
    assert interval_score(0, 1, .5) == 1
    assert interval_score(0, 1, 1.2) == pytest.approx(5)


def test_frozen_gp_prediction_uses_selected_event_and_divides_alpha_once():
    from binned_gp_benchmark import controlled_log_event_weights, controlled_selection
    from compare_external_selected import gp_product
    from check_external_copula_selected import settings
    config, protocol = settings()
    bins = 2
    rng = np.random.default_rng(10)
    coordinates = np.zeros((4, bins**3+4))
    coordinates[:, 1:4] = np.log(.3)
    chains = [{'coordinate': coordinates, 'features': np.ones((4, 17))} for _ in range(4)]
    test = {'observed': rng.uniform(size=(8, 3)), 'sigma': np.array([.04, .05, .04])}
    inputs = {'log_event_weights': controlled_log_event_weights(test['observed'], test['sigma'], bins),
              'selection_coefficients': controlled_selection(bins)}
    fine = 12
    truth = {'intrinsic': np.full(fine**3, 1/fine**3),
             'detected': controlled_selection(fine)/controlled_selection(fine).sum()}
    result = gp_product(config, protocol, {'2': ({}, {}, chains, inputs)}, '2', truth, [6, 12], test)
    np.testing.assert_allclose(result['log_integrals'], -np.log(.05+.95/3), atol=1e-13)
    np.testing.assert_allclose(result['detected'], truth['detected'], atol=1e-14)


def test_prediction_difference_preserves_paired_draws_and_weights():
    from compare_external_selected import difference_precision
    rng = np.random.default_rng(18)
    values = rng.normal(size=(512, 8))
    weights = rng.dirichlet(np.full(512, 10.))
    # A constant factor changes each log prediction by the same known amount.
    refined = values + .03 + .0001*rng.normal(size=values.shape)
    result = difference_precision(values, refined, weights, weights, (4, 128))
    assert result['total_score_change'] == pytest.approx(.24, abs=1e-4)
    assert result['mcse_of_change'] < .001


def test_truth_quadrature_recovers_exact_uniform_selection(monkeypatch):
    import refine_external_selected_recovery as refinement
    from binned_gp_benchmark import controlled_selection
    parameters = {name: value[None] for name, value in uniform_parameters().items()}
    monkeypatch.setattr(refinement, 'truth_samples', lambda: parameters)
    result = refinement.quadrature_truth(4, 4, 8)
    np.testing.assert_allclose(result['intrinsic'], 1/4**3, atol=1e-14)
    np.testing.assert_allclose(result['detected'], controlled_selection(4)/controlled_selection(4).sum(), atol=1e-14)
    assert result['normalization'] == pytest.approx(1.)
    assert result['alpha'] == pytest.approx(.05+.95/3)
