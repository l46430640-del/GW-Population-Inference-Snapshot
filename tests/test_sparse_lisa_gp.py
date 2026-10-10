import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_external_gp_selected import InitializedGPPotential
from sample_lisa_transport import compile_potential, value_gradient
from sparse_lisa_gp import SparseEventGPPotential


@pytest.fixture(scope='module', autouse=True)
def precision():
    old = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(old[0])
    torch.set_num_threads(old[1])


@pytest.mark.parametrize('bins', [2, 6, 12])
def test_sparse_event_evaluation_preserves_dense_values_gradients_prior_and_selection(bins):
    config = json.loads((ROOT / 'configs/external_benchmark.json').read_text())['binned_gp']
    rng = np.random.default_rng(45)
    event = np.full((7, bins**3), -np.inf)
    for i in range(7):
        index = rng.choice(bins**3, i+1, replace=False)
        event[i, index] = np.log(rng.dirichlet(np.ones(i+1)))
    selection = rng.uniform(1e-15, 1., bins**3)/bins**3
    dense, sparse = [cls(bins, event, selection, config) for cls in (InitializedGPPotential, SparseEventGPPotential)]
    point = torch.tensor(np.r_[0., np.log([.3, .3, .3]), rng.normal(0, .2, bins**3)])
    expected, gradient = value_gradient(dense, point)
    value, derivative = value_gradient(sparse, point)
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(derivative, gradient, atol=1e-10, rtol=0)
    assert sparse.dimension == dense.dimension == bins**3+4
    torch.testing.assert_close(sparse.log_selection, dense.log_selection)
    torch.testing.assert_close(sparse.log_prior(point), dense.log_prior(point))
    assert sparse.event_indices.shape == (7, 7)
    for i in range(7):
        finite = torch.isfinite(sparse.sparse_log_event_weights[i])
        assert finite.sum() == i+1
        np.testing.assert_array_equal(sparse.event_indices[i, finite], np.flatnonzero(np.isfinite(event[i])))
    traced, _ = compile_potential(sparse, point, 1e-7)
    value, derivative = value_gradient(traced, point)
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(derivative, gradient, atol=1e-10, rtol=0)


def test_an_event_without_finite_support_is_rejected():
    config = json.loads((ROOT / 'configs/external_benchmark.json').read_text())['binned_gp']
    with pytest.raises(ValueError, match='finite nonzero support'):
        SparseEventGPPotential(2, np.full((1, 8), -np.inf), np.ones(8)/8, config)


def test_sparse_continuation_restores_original_length_without_relaxing_acceptance():
    from run_external_lisa_gp_sparse import CONFIG, settings, sources_and_inputs
    amendment = json.loads(CONFIG.read_text())
    config, protocol = settings(amendment)
    original = json.loads((ROOT / config['protocol']).read_text())
    assert protocol['sampling'] == original['sampling']
    assert protocol['binned_gp'] == original['binned_gp']
    assert protocol['evaluation'] == original['evaluation']
    assert config['retained_per_chain'] == 800
    sources, inputs, path = sources_and_inputs(amendment)
    assert inputs['log_event_weights'].shape == (135, 12**3)
    assert path == ROOT / 'results/analysis/external_lisa_gp/inputs_12.npz'
    assert not any('test_observations' in name or '/truth.' in name for name in sources)
