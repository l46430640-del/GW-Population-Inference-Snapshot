import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from initialize_population_guide import initialize_from_events
from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated


@pytest.mark.parametrize('family', ['kumaraswamy', 'normal'])
@pytest.mark.parametrize('components', [1, 2, 3])
def test_observed_initialization_matches_declared_latent_gaussian(family, components):
    torch.set_num_threads(1)
    torch.manual_seed(198)
    model = BayesianMixtureEstimator_NF_Correlated(
        3, components if family == 'kumaraswamy' else 0,
        components if family == 'normal' else 0, 0,
        num_flow_transforms=6, hidden_dim_flow=16, device='cpu',
        physical_bounds=([0, 0, 0], [1, 1, 1]))
    rng = np.random.default_rng(154)
    points = np.concatenate([rng.normal(center, .025, (30, 3))
                             for center in np.linspace(.15, .8, components)])
    data = torch.tensor(np.broadcast_to(points[:, None, :], (len(points), 8, 3)).copy(),
                        dtype=torch.float32)
    result = initialize_from_events(model, data, 180)
    location = torch.tensor(result['initial_latent_location'])
    with torch.no_grad():
        samples = model.flow_dist.sample((12000,))
        torch.testing.assert_close(samples.mean(0), location, atol=.022, rtol=0)
        torch.testing.assert_close(samples.std(0), torch.full_like(location, .5), atol=.015, rtol=0)
        expected = torch.distributions.Normal(location, .5).log_prob(samples[:30]).sum(-1)
        torch.testing.assert_close(model.flow_dist.log_prob(samples[:30]), expected,
                                   atol=3e-5, rtol=2e-5)
    assert sum(result['component_event_counts']) == len(points)
    assert np.std(result['initial_latent_location'][components:7*components]) > .1


def test_initializer_rejects_mixed_family_layout():
    model = BayesianMixtureEstimator_NF_Correlated(
        3, 1, 1, 0, num_flow_transforms=2, hidden_dim_flow=16, device='cpu',
        physical_bounds=([0, 0, 0], [1, 1, 1]))
    with pytest.raises(ValueError, match='homogeneous'):
        initialize_from_events(model, torch.zeros(12, 8, 3), 14)


def test_smooth_initializer_handles_odd_latent_dimension_and_declared_scale():
    model = BayesianMixtureEstimator_NF_Correlated(
        3, 0, 0, 0, num_corr_axis_spline_kumaraswamy=3,
        axis_spline_num_bins=16, m1_basis_type='smooth',
        num_flow_transforms=6, hidden_dim_flow=16, device='cpu',
        physical_bounds=([np.log10(3), 0, 0], [np.log10(150), 2.3, 1]))
    rng = np.random.default_rng(154)
    unit = np.concatenate([rng.normal(center, .025, (30, 3)) for center in (.2, .5, .8)])
    points = model.min_vals.numpy() + unit * (model.max_vals - model.min_vals).numpy()
    data = torch.tensor(np.broadcast_to(points[:, None, :], (len(points), 8, 3)).copy(), dtype=torch.float32)
    result = initialize_from_events(model, data, 180)
    location, scale = torch.tensor(result['initial_latent_location']), torch.tensor(result['initial_latent_scale'])
    assert model.latent_dim == 75
    with torch.no_grad():
        samples = model.flow_dist.sample((12000,))
        torch.testing.assert_close(samples.mean(0), location, atol=.022, rtol=0)
        torch.testing.assert_close(samples.std(0), scale, atol=.015, rtol=0)
        expected = torch.distributions.Normal(location, scale).log_prob(samples[:30]).sum(-1)
        torch.testing.assert_close(model.flow_dist.log_prob(samples[:30]), expected, atol=5e-5, rtol=2e-5)
