import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from binned_gp_benchmark import grid_geometry, posterior_log_event_weights
from lisa_gp_inputs import integral_changes, integral_report, selection_coefficients
from run_external_gp_selected import InitializedGPPotential
from run_external_lisa_gp import CONFIG, effective_protocol, source_manifest


@pytest.fixture(scope='module', autouse=True)
def precision():
    old = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(old[0])
    torch.set_num_threads(old[1])


def test_every_selection_bin_is_retained_and_constant_response_is_exact():
    for rho in (0., 8., 30.):
        result = selection_coefficients(lambda x: np.full(len(x), rho), 6, 4, 72)
        np.testing.assert_allclose(result, ndtr(rho-8)/6**3, atol=1e-16)
        assert len(result) == 6**3 and np.all(result > 0)


def test_selection_subdivision_matches_separable_analytic_integral():
    lower, upper = grid_geometry(6)
    primitive = lambda x: x*ndtr(x)+np.exp(-x*x/2)/np.sqrt(2*np.pi)
    expected = (primitive(6*upper[:, 0]-3)-primitive(6*lower[:, 0]-3))/6/6**2
    coarse = selection_coefficients(lambda x: 5+6*x[:, 0], 6, 10, 81)
    fine = selection_coefficients(lambda x: 5+6*x[:, 0], 6, 12, 82)
    np.testing.assert_allclose(coarse, expected, atol=1e-8)
    np.testing.assert_allclose(fine, expected, atol=1e-8)


def test_event_binning_divides_the_event_prior_and_does_not_apply_detection_twice():
    data = np.ones((2, 3, 4))
    data[..., :3] = [[[.1, .1, .1], [.6, .1, .1], [.6, .1, .1]],
                    [[.6, .6, .6], [.6, .6, .6], [.6, .6, .6]]]
    data[..., 3] = 2.
    weights = np.exp(posterior_log_event_weights(data, 2))
    assert weights.shape == (2, 8)
    np.testing.assert_allclose(weights.sum(1), .5)
    assert weights[0, 0] == pytest.approx(1/6)
    assert weights[0, 4] == pytest.approx(1/3)
    assert weights[1, 7] == pytest.approx(.5)


def test_refined_event_and_selection_changes_match_full_log_likelihood():
    protocol = json.loads((ROOT / 'configs/external_benchmark.json').read_text())
    rng = np.random.default_rng(76)
    event = np.log(rng.dirichlet(np.ones(8), size=5))
    fine_event = np.log(rng.dirichlet(np.ones(8), size=5))
    selection = rng.uniform(.01, .99, 8)/8
    fine_selection = selection*(1+rng.normal(0, .01, 8))
    model = InitializedGPPotential(2, event, selection, protocol['binned_gp'])
    reference = InitializedGPPotential(2, fine_event, fine_selection, protocol['binned_gp'])
    points = rng.normal(0, .3, (6, 12))
    points[:, 1:4] = np.log(.3)
    calls = []
    changes = integral_changes(model, points, fine_event, fine_selection, lambda: calls.append(1))
    assert len(calls) == 6
    expected = []
    for point in points:
        value = torch.tensor(point)
        expected.append(float(model(value)-reference(value)))
        field = model.field(value).numpy()
        index = len(expected)-1
        assert changes['event_log_likelihood_change'][index] == pytest.approx(
            (logsumexp(field+fine_event, axis=1)-logsumexp(field+event, axis=1)).sum(), abs=1e-13)
    np.testing.assert_allclose(changes['combined_log_likelihood_change'], expected, atol=1e-12)
    assert integral_report({name: np.zeros(8) for name in changes}, .2)['passed']
    assert not integral_report({'event': np.arange(8)}, .2)['passed']
    assert not integral_report({'event': []}, .2)['passed']


def test_sampling_amendment_preserves_priors_and_does_not_use_truth_or_test_data():
    config = json.loads(CONFIG.read_text())
    original = json.loads((ROOT / config['protocol']).read_text())
    protocol = effective_protocol(config, original)
    assert protocol['binned_gp'] == original['binned_gp']
    assert protocol['evaluation'] == original['evaluation']
    assert protocol['sampling']['retained_per_chain'] == 2400
    assert {k: v for k, v in protocol['sampling'].items() if k != 'retained_per_chain'} == {
        k: v for k, v in original['sampling'].items() if k != 'retained_per_chain'}
    assert config['bins_per_axis'] == [6, 12]
    assert config['verification_event_samples'] == original['evaluation']['verification_training_event_samples']
    sources = source_manifest(config)
    assert not any('test_observations' in name or '/truth.' in name for name in sources)
    assert config['training_catalog'] in sources
    budget = json.loads((ROOT / config['previous_cost']).read_text())
    assert config['maximum_seconds'] <= budget['remaining_study_seconds']
