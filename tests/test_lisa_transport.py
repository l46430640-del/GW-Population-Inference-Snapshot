import json
import sys
from pathlib import Path

import numpy as np
import arviz as az
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correct_selected_posterior import sha256, swap_components
from diagnose_lisa_correction import target_value
from sample_lisa_transport import (
    TransportPotential, aligned_initial, check_precision, compile_potential, differentiable_log_target,
    lisa_features, run_chain, sampling_model, value_gradient,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', autouse=True)
def double_precision_one_thread():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def example():
    directory = ROOT / 'results/analysis/morphologies/lisa_bimodal_kumaraswamy_k2_s91001'
    config = json.loads((directory / 'summary.json').read_text())['configuration']
    model = sampling_model(config, checkpoint=directory / 'flow.pt')
    data = torch.tensor(np.load(directory / 'catalog.npz')['data'])
    product = np.load(ROOT / 'results/analysis/lisa_correction/repeat_732001.npz')
    initial, _ = aligned_initial(model, product)
    return model, data, initial


def test_differentiable_target_matches_retained_density_and_finite_difference():
    model, data, initial = example()
    value, gradient = value_gradient(lambda z: differentiable_log_target(model, z, data), initial)
    assert float(value) == pytest.approx(target_value(model, initial, data), abs=1e-10)
    direction = torch.linspace(-.3, .4, len(initial))
    step = 1e-6
    difference = (target_value(model, initial + step * direction, data)
                  - target_value(model, initial - step * direction, data)) / (2 * step)
    assert float(gradient @ direction) == pytest.approx(difference, abs=2e-5)


def test_sampling_precision_and_fixed_sobol_nodes():
    model, data, initial = example()
    count = model.lambda_det_mc_samples
    first = model._selection_mc_samples_norm(count, model.device)
    second = model._selection_mc_samples_norm(count, model.device)
    assert first.data_ptr() == second.data_ptr()
    record = check_precision(model, data, initial[None])
    assert record['maximum_absolute_log_target_difference'] < .001
    assert model.min_vals.dtype == torch.float64


def test_flow_transport_roundtrip_jacobian_and_potential():
    model, data, initial = example()
    potential = TransportPotential(model, data, 'flow')
    coordinate = potential.from_latent(initial).detach()
    latent = potential.to_latent(coordinate)
    torch.testing.assert_close(latent, initial, atol=1e-11, rtol=1e-11)
    jacobian = torch.autograd.functional.jacobian(potential.to_latent, coordinate)
    _, actual_logdet = torch.linalg.slogdet(jacobian)
    potential.to_latent(coordinate)
    analytic_logdet = model.flow_transforms.log_abs_det_jacobian(coordinate, initial)
    torch.testing.assert_close(actual_logdet, analytic_logdet, atol=1e-10, rtol=1e-10)
    assert float(potential(coordinate)) == pytest.approx(
        -target_value(model, initial, data) - float(actual_logdet), abs=1e-9)
    assert all(not p.requires_grad for p in model.flow_transforms.parameters())


@pytest.mark.parametrize('method', ['raw', 'flow'])
def test_jit_potential_preserves_values_and_gradients(method):
    model, data, initial = example()
    potential = TransportPotential(model, data, method)
    coordinate = potential.from_latent(initial).detach()
    _, checks = compile_potential(potential, coordinate, 1e-8)
    assert len(checks) == 3


def test_lisa_functionals_use_lisa_efficiency_and_are_label_invariant():
    model, _, initial = example()
    latent = torch.stack([initial, swap_components(model, initial)])
    names, values = lisa_features(model, latent)
    np.testing.assert_allclose(values[0], values[1], atol=1e-12)
    params = model._unflatten_latent_vector(initial, apply_numerical_limits=False)
    assert values[0, names.index('alpha')] == pytest.approx(float(model._selection_fraction_for_params(params)))
    assert abs(values[0, names.index('alpha')] - (.05 + .95 * (
        values[0, names.index('variance_mass')] + values[0, names.index('mean_mass')]**2))) > .001


class GaussianPotential:
    method = 'raw'

    def __call__(self, coordinate):
        return .5 * coordinate.square().sum()

    def to_latent(self, coordinate):
        return coordinate


@pytest.mark.parametrize('budget,expected', [(60, 'complete'), (0, 'time_limit')])
def test_chain_records_warmup_samples_and_budget_stop(budget, expected):
    config = {'warmup': 10, 'samples': 12, 'full_mass': True, 'target_acceptance': .8,
              'max_tree_depth': 3, 'maximum_seconds_per_chain': budget, 'jit_absolute_tolerance': 1e-10}
    report, arrays = run_chain(GaussianPotential(), torch.tensor([.2, -.4]), config, 732900)
    assert report['status'] == expected
    assert arrays['progress'].shape[1] == 5
    assert np.all(np.diff(arrays['progress'][:, 2]) >= 1)
    if expected == 'complete':
        assert report['warmup_completed'] == 10
        assert arrays['latent'].shape == (12, 2)
        assert (arrays['progress'][:, 0] == 0).sum() == 10
    else:
        assert report['warmup_completed'] == 1
        assert arrays['latent'].shape == (0, 2)
        assert report['divergences'] is None


@pytest.mark.parametrize('method', ['raw', 'flow'])
@pytest.mark.parametrize('seed', [733001, 733002])
def test_retained_pilot_provenance_coordinates_and_functionals(method, seed):
    output = ROOT / 'results/analysis/lisa_transport_pilot'
    path = output / f'{method}_{seed}.json'
    report = json.loads(path.read_text())
    for name, expected in report['source_sha256'].items():
        assert sha256(ROOT / name) == expected
    assert sha256(path.with_suffix('.npz')) == report['draw_product_sha256']
    assert sha256(output / 'precision.json') == report['precision_product_sha256']
    assert report['eligible_for_scientific_summaries'] is False
    product = np.load(path.with_suffix('.npz'))
    assert product['coordinate'].shape == product['latent'].shape == (report['samples_completed'], 20)
    assert product['warmup_coordinate'].shape == (report['warmup_completed'], 20)
    assert np.isfinite(product['progress']).all()
    assert np.all(np.diff(product['progress'][:, 2]) >= 1)
    model, data, _ = example()
    potential = TransportPotential(model, data, method)
    initial = potential.to_latent(torch.tensor(product['initial_coordinate']))
    np.testing.assert_allclose(initial.numpy(), product['initial_latent'], atol=1e-11)
    for index in np.unique(np.linspace(0, len(product['latent']) - 1, min(2, len(product['latent'])), dtype=int)):
        coordinate = torch.tensor(product['coordinate'][index])
        latent = potential.to_latent(coordinate)
        np.testing.assert_allclose(latent.numpy(), product['latent'][index], atol=1e-11)
        names, values = lisa_features(model, latent[None])
        assert names == product['feature_names'].tolist()
        np.testing.assert_allclose(values[0], product['features'][index], atol=1e-11)


def test_retained_pilots_share_initial_points_and_precision_check():
    output = ROOT / 'results/analysis/lisa_transport_pilot'
    for seed in (733001, 733002):
        raw = np.load(output / f'raw_{seed}.npz')
        flow = np.load(output / f'flow_{seed}.npz')
        np.testing.assert_array_equal(raw['initial_latent'], flow['initial_latent'])
    record = json.loads((output / 'precision.json').read_text())
    assert record['draws'] == 98
    assert max(abs(x) for x in record['log_target_differences']) == record['maximum_absolute_log_target_difference']
    assert record['maximum_absolute_log_target_difference'] < .001


def test_retained_pilot_summary_reconstructs_mixing_diagnostics():
    output = ROOT / 'results/analysis/lisa_transport_pilot'
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['eligible_for_scientific_summaries'] is False
    for name, expected in summary['chain_products'].items():
        assert sha256(output / name) == expected
    for method, result in summary['methods'].items():
        products = [np.load(output / f'{method}_{seed}.npz') for seed in summary['configuration']['seeds']]
        values = np.stack([p['features'] for p in products])
        names = products[0]['feature_names']
        dataset = az.from_dict(posterior={str(name): values[:, :, j] for j, name in enumerate(names)})
        diagnostic = az.summary(dataset, kind='diagnostics', round_to='none')
        assert diagnostic.to_dict(orient='index') == result['functionals']
        assert result['maximum_rhat'] == diagnostic.r_hat.max()
        assert result['minimum_bulk_ess'] == diagnostic.ess_bulk.min()
        assert result['minimum_tail_ess'] == diagnostic.ess_tail.min()
