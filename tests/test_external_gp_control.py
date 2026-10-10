import json
from pathlib import Path
import sys
import time

import numpy as np
import pyro
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_selected_mixture import selected_model
from binned_gp_benchmark import controlled_log_event_weights, controlled_selection, shape_log_likelihood
from build_external_gp_control import gp_truth, selected_catalog, settings, verify_manifest
from correct_selected_posterior import latent_sites
from pilot_external_gp_control_copula import derivative_directions, run_bounded
from sample_lisa_transport import TransportPotential, compile_potential, value_gradient


@pytest.fixture(scope='module', autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def test_control_preserves_the_original_predeclared_population_and_seeds():
    config, protocol, experiment = settings()
    assert experiment['truth_seed'] == 847001
    assert experiment['catalog_seed'] == 847002 and experiment['test_seed'] == 847003
    assert experiment['truth_sigma'] == 1.2
    assert experiment['truth_length_scales'] == [.25, .25, .25]
    assert config['training_event_samples'] == protocol['evaluation']['training_event_samples']
    assert config['verification_training_event_samples'] == protocol['evaluation']['verification_training_event_samples']
    generated = gp_truth(experiment)
    n = experiment['truth_grid_per_axis']
    center = np.array(list(np.ndindex(n, n, n)), dtype=float)/n+.5/n
    np.testing.assert_allclose(generated['centers'], center, rtol=0, atol=2e-16)
    covariance = np.empty((n**3, n**3))
    for i, x in enumerate(center):
        covariance[i] = 1.2**2*np.exp(-.5*np.sum(((center-x)/.25)**2, axis=1))
    covariance += 1e-10*np.eye(n**3)
    np.testing.assert_allclose(generated['covariance'], covariance, rtol=2e-15, atol=2e-15)
    expected = np.linalg.cholesky(covariance) @ np.random.default_rng(847001).standard_normal(n**3)
    np.testing.assert_allclose(generated['field'], expected, rtol=1e-10, atol=1e-10)


def test_first_catalogs_and_independent_verification_cloud_are_retained():
    config, _, experiment = settings()
    directory = ROOT / config['output']
    manifest = verify_manifest(directory)
    assert not manifest['eligible_for_scientific_comparisons']
    truth = dict(np.load(directory / 'truth.npz'))
    generated = gp_truth(experiment)
    for name in generated:
        if name in ('white', 'centers', 'covariance'):
            np.testing.assert_array_equal(truth[name], generated[name])
        else:
            # Cholesky's last bits may depend on the importing BLAS thread pool.
            np.testing.assert_allclose(truth[name], generated[name], rtol=1e-9, atol=1e-9)
    for label, count, seed in [('train', 96, 847002), ('test', 192, 847003)]:
        catalog = dict(np.load(directory / f'{label}_catalog.npz'))
        expected = selected_catalog(truth, 6, count, seed, np.array([.04, .05, .04]), config['proposal_batch_size'])
        for name in expected:
            np.testing.assert_array_equal(catalog[name], expected[name])
        assert catalog['observed'].shape == (count, 3)
        index = np.minimum((catalog['truth']*6).astype(int), 5)
        np.testing.assert_array_equal(catalog['truth_bin'], (index[:, 0]*6+index[:, 1])*6+index[:, 2])
    train = np.load(directory / 'train_catalog.npz')
    fine = np.load(directory / 'train_verification.npz')
    assert train['data'].shape == (96, 2048, 4)
    assert fine['data'].shape == (96, 8192, 4)
    np.testing.assert_array_equal(train['observed'], fine['observed'])
    assert not np.array_equal(train['data'], fine['data'][:, :2048])
    assert np.all(train['data'][..., 3] == 1) and np.all(fine['data'][..., 3] == 1)


def test_selected_uniform_catalog_obeys_the_detection_weighted_law():
    truth = {'probabilities': np.full(8, 1/8)}
    catalog = selected_catalog(truth, 2, 20000, 991, np.array([.04, .05, .04]), 512)
    mean_x1 = (.05/2+.95/4)/(.05+.95/3)
    expected = np.array([mean_x1, .5, .5])
    for name in ('truth', 'observed'):
        se = catalog[name].std(0)/np.sqrt(len(catalog[name]))
        assert np.max(abs(catalog[name].mean(0)-expected)/se) < 5


def test_event_posterior_cloud_matches_independent_one_dimensional_integrals():
    config, _, _ = settings()
    catalog = np.load(ROOT / config['output'] / 'train_catalog.npz')
    for event in np.linspace(0, 95, 8, dtype=int):
        for axis in range(3):
            y, sigma = catalog['observed'][event, axis], catalog['sigma'][axis]
            def kernel(x):
                return norm.pdf((x-y)/sigma)/sigma * (.05+.95*x*x if axis == 0 else 1.)
            mass = quad(kernel, 0, 1, epsabs=1e-11)[0]
            mean = quad(lambda x: x*kernel(x), 0, 1, epsabs=1e-11)[0]/mass
            points = catalog['data'][event, :, axis]
            assert abs(points.mean()-mean) < 5*points.std()/np.sqrt(len(points))


def test_uniform_copula_and_gp_have_the_same_selected_catalog_likelihood():
    config, _, _ = settings()
    catalog = np.load(ROOT / config['output'] / 'train_catalog.npz')
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    point = torch.zeros(20)
    point[2:14] = np.log(.9999)
    data = torch.tensor(catalog['data'][:8, :64])
    trace = pyro.poutine.trace(pyro.poutine.condition(model._pyro_model, data=latent_sites(model, point))).get_trace(data)
    trace.compute_log_prob()
    copula = float(trace.nodes['obs_log_prob']['log_prob'].sum())
    expected = -len(data)*np.log(.05+.95/3)
    assert copula == pytest.approx(expected, abs=1e-10)
    for bins in (6, 12):
        weights = controlled_log_event_weights(catalog['observed'][:8], catalog['sigma'], bins)
        gp = shape_log_likelihood(torch.zeros(bins**3), torch.tensor(weights), torch.tensor(np.log(controlled_selection(bins))))
        assert float(gp) == pytest.approx(expected, abs=1e-10)


def test_raw_copula_target_jit_and_directional_derivatives():
    config, _, _ = settings()
    catalog = np.load(ROOT / config['output'] / 'train_catalog.npz')
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    potential = TransportPotential(model, torch.tensor(catalog['data'][:3, :32]), 'raw')
    point = torch.linspace(-.2, .2, 20)
    _, checks = compile_potential(potential, point, 1e-7)
    assert len(checks) == 3
    _, gradient = value_gradient(potential, point)
    for direction in derivative_directions(20):
        step = 1e-5
        numerical = (potential(point+step*direction)-potential(point-step*direction))/(2*step)
        assert float(gradient @ direction) == pytest.approx(float(numerical), abs=1e-4)


class Gaussian:
    def __call__(self, point):
        return .5*point.square().sum()


@pytest.mark.parametrize('seconds,status', [(20, 'complete'), (0, 'time_limit')])
def test_bounded_pilot_retains_samples_or_an_explicit_stop(seconds, status):
    config = {'maximum_resident_gib': 2, 'jit_absolute_tolerance': 1e-7, 'seed': 94,
        'full_mass': True, 'target_acceptance': .8, 'maximum_tree_depth': 3, 'warmup': 10, 'retained': 10}
    report, arrays = run_bounded(Gaussian(), torch.tensor([.1, -.2]), config, time.perf_counter()+seconds)
    assert report['status'] == status
    assert arrays['coordinate'].shape == (report['samples_completed'], 2)
    assert arrays['warmup_coordinate'].shape == (report['warmup_completed'], 2)
    assert np.isfinite(arrays['progress']).all()
