import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correct_selected_posterior import sha256, swap_components
from run_lisa_production import acceptance, additional_functionals, initial_points, mixing_summary, numerical_check
from sample_lisa_transport import differentiable_log_target, lisa_features, sampling_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', autouse=True)
def one_thread_double_precision():
    previous = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_num_threads(previous[0])
    torch.set_default_dtype(previous[1])


def inputs():
    config = json.loads((ROOT / 'configs/lisa_production.json').read_text())
    original = json.loads((ROOT / 'results/analysis/morphologies/lisa_bimodal_kumaraswamy_k2_s91001/summary.json').read_text())['configuration']
    return config, sampling_model(original)


def test_dispersed_initialization_is_deterministic_and_uses_declared_labels():
    config, model = inputs()
    points, records = initial_points(config, model)
    again, repeated = initial_points(config, model)
    torch.testing.assert_close(points, again, rtol=0, atol=0)
    assert records == repeated
    assert len(points) == 4
    assert np.all(np.diff([r['alpha'] for r in records]) > 0)
    for point, row in zip(points, records):
        original = torch.tensor(np.load(ROOT / row['source'])['latent'][row['index']])
        if row['labels_swapped']:
            original = swap_components(model, original)
        torch.testing.assert_close(point, original, rtol=0, atol=0)


def test_ordered_coordinates_and_marginal_cdfs_are_label_invariant():
    config, model = inputs()
    points, _ = initial_points(config, model)
    names, first = additional_functionals(model, points, config['marginal_cdf_points'])
    second_names, second = additional_functionals(model, swap_components(model, points), config['marginal_cdf_points'])
    assert names == second_names
    assert first.shape == (4, 35)
    np.testing.assert_allclose(first, second, atol=1e-14)
    cdf = first[:, 20:].reshape(4, 5, 3)
    assert np.all(np.diff(cdf, axis=1) >= 0)
    assert cdf.min() >= 0 and cdf.max() <= 1


def test_mixing_summary_uses_chain_disagreement():
    rng = np.random.default_rng(734000)
    values = rng.normal(size=(4, 500, 2))
    converged = mixing_summary(['first', 'second'], values)
    assert converged['maximum_rhat'] < 1.01
    values[0, :, 0] += 4
    divergent = mixing_summary(['first', 'second'], values)
    assert divergent['maximum_rhat'] > 1.1


def test_acceptance_requires_every_declared_condition():
    config, _ = inputs()
    reports = [{'status': 'complete', 'warmup_completed': config['warmup'],
                'samples_completed': config['samples'], 'divergences': []} for _ in range(4)]
    mixing = {'maximum_rhat': 1.001, 'minimum_bulk_ess': 600, 'minimum_tail_ess': 600}
    numerical = {'groups': {k: {'total_change_sd': .1} for k in [*(f'chain_{i}' for i in range(4)), 'efficiency_extremes']},
                 'uniform_weight_ess_fraction': .98, 'uniform_mean_shift_in_sd': [.01],
                 'feature_change_in_sd': [.01], 'precision': {'maximum_absolute_log_target_difference': 1e-5}}
    assert acceptance(config, reports, mixing, numerical)['eligible_for_scientific_summaries']
    for key, value in [('maximum_rhat', 1.02), ('minimum_bulk_ess', 30), ('minimum_tail_ess', 30)]:
        assert not acceptance(config, reports, dict(mixing, **{key: value}), numerical)['eligible_for_scientific_summaries']
    for key, value in [('uniform_weight_ess_fraction', .5), ('uniform_mean_shift_in_sd', [.2]),
                        ('uniform_mean_shift_in_sd', [.01, np.nan]), ('feature_change_in_sd', [.01, np.nan]),
                        ('feature_change_in_sd', [.1]), ('groups', {'all': {'total_change_sd': .1}}),
                        ('precision', {'maximum_absolute_log_target_difference': .1})]:
        assert not acceptance(config, reports, mixing, dict(numerical, **{key: value}))['eligible_for_scientific_summaries']
    reports[0]['divergences'] = [4]
    assert not acceptance(config, reports, mixing, numerical)['eligible_for_scientific_summaries']
    assert not acceptance(config, reports[:3], mixing, numerical)['eligible_for_scientific_summaries']
    reports[0]['status'] = 'time_limit'
    assert not acceptance(config, reports, mixing, numerical)['eligible_for_scientific_summaries']


def test_refinement_retains_indices_and_restores_selection_settings():
    config, model = inputs()
    points, _ = initial_points(config, model)
    latent = points[:, None, :] + torch.arange(4)[None, :, None] * .001
    data = torch.tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]],
                         [[.6, .5, .7, 1.], [.7, .6, .8, 1.]]])
    names, values = lisa_features(model, latent.reshape(-1, 20))
    features = values.reshape(4, 4, -1)
    with torch.no_grad():
        log_target = np.asarray([float(differentiable_log_target(model, p, data))
                                for p in latent.reshape(-1, 20)]).reshape(4, 4)
    small = dict(config, refinement_draws_per_chain=2, refinement_extremes_per_chain=1,
                 fine_selection_points=512, fine_feature_transport_power=11)
    original = model.lambda_det_mc_samples, model.selection_mc_seed
    report, arrays = numerical_check(model, latent, features, names, log_target, data, data, small)
    assert (model.lambda_det_mc_samples, model.selection_mc_seed) == original
    np.testing.assert_allclose(arrays['event_change'], 0, atol=1e-10)
    for name, group in report['groups'].items():
        offsets = np.searchsorted(arrays['indices'], arrays['group_' + name])
        assert arrays['selection_change'][offsets].std() == pytest.approx(group['total_change_sd'])
    assert arrays['uniform_refinement_weights'].sum() == pytest.approx(1)
    assert 0 < report['uniform_weight_ess_fraction'] <= 1


def test_retained_production_target_and_initial_states():
    config, model = inputs()
    path = ROOT / config['output'] / 'target.json'
    target = json.loads(path.read_text())
    assert target['configuration'] == config
    for name, expected in target['source_sha256'].items():
        assert sha256(ROOT / name) == expected
    points, records = initial_points(config, model)
    assert target['initial_provenance'] == records
    np.testing.assert_array_equal(points.numpy(), target['initial_points'])


@pytest.mark.parametrize('seed', [734001, 734002, 734003, 734004])
def test_retained_production_chain_functionals_and_target(seed):
    config, model = inputs()
    output = ROOT / config['output']
    path = output / f'chain_{seed}.npz'
    report = json.loads(path.with_suffix('.json').read_text())
    assert sha256(path) == report['draw_product_sha256']
    assert sha256(output / 'target.json') == report['target_product_sha256']
    assert report['seed'] == seed and report['configuration'] == config
    product = np.load(path)
    assert product['latent'].shape == (report['samples_completed'], 20)
    assert product['warmup_coordinate'].shape == (report['warmup_completed'], 20)
    assert product['diagnostic_values'].shape == (report['samples_completed'], 48)
    assert np.isfinite(product['diagnostic_values']).all()
    np.testing.assert_array_equal(product['latent'], product['coordinate'])
    if report['status'] == 'complete':
        assert report['samples_completed'] == config['samples']
        assert report['warmup_completed'] == config['warmup']
    directory = ROOT / 'results/analysis/morphologies/lisa_bimodal_kumaraswamy_k2_s91001'
    data = torch.tensor(np.load(directory / 'catalog.npz')['data'], dtype=torch.float64)
    for index in [0, len(product['latent']) - 1]:
        point = torch.tensor(product['latent'][index])
        names, features = lisa_features(model, point[None], config['feature_transport_power'])
        assert names == product['feature_names'].tolist()
        np.testing.assert_allclose(features[0], product['features'][index], atol=1e-12)
        extra_names, extra = additional_functionals(model, point[None], config['marginal_cdf_points'])
        assert names + extra_names + ['log_target'] == product['diagnostic_names'].tolist()
        np.testing.assert_allclose(extra[0], product['diagnostic_values'][index, 12:-1], atol=1e-12)
        with torch.no_grad():
            value = float(differentiable_log_target(model, point, data))
        assert value == pytest.approx(product['log_target'][index], abs=1e-10)


def test_retained_production_decision_and_numerical_summary():
    config, _ = inputs()
    output = ROOT / config['output']
    summary = json.loads((output / 'summary.json').read_text())
    assert sha256(output / 'target.json') == summary['target_product_sha256']
    assert sha256(output / 'numerical.npz') == summary['numerical_product_sha256']
    for name, expected in summary['chain_reports'].items():
        assert sha256(output / name) == expected
    products = [dict(np.load(output / f'chain_{seed}.npz')) for seed in config['seeds']]
    reports = [json.loads((output / f'chain_{seed}.json').read_text()) for seed in config['seeds']]
    size = min(len(p['latent']) for p in products)
    mixing = mixing_summary(products[0]['diagnostic_names'].tolist(),
                            np.stack([p['diagnostic_values'][:size] for p in products]))
    assert mixing == summary['mixing']
    numerical = np.load(output / 'numerical.npz')
    total = numerical['event_change'] + numerical['selection_change']
    for name, row in summary['numerical']['groups'].items():
        offsets = np.searchsorted(numerical['indices'], numerical['group_' + name])
        assert total[offsets].std() == pytest.approx(row['total_change_sd'])
    features = np.stack([p['features'][:size] for p in products]).reshape(-1, 12)
    selected = features[numerical['uniform_indices']]
    weights = numerical['uniform_refinement_weights']
    assert weights.sum() == pytest.approx(1)
    assert 1 / (weights @ weights) / len(weights) == pytest.approx(summary['numerical']['uniform_weight_ess_fraction'])
    shift = abs(weights @ selected - selected.mean(0)) / features.std(0)
    np.testing.assert_allclose(shift, summary['numerical']['uniform_mean_shift_in_sd'])
    change = abs(numerical['fine_features'] - selected).max(0) / features.std(0)
    np.testing.assert_allclose(change, summary['numerical']['feature_change_in_sd'])
    assert acceptance(config, reports, mixing, summary['numerical']) == summary['decision']


def test_retained_refinement_recomputes_independent_integration():
    config, model = inputs()
    output = ROOT / config['output']
    check = np.load(output / 'numerical.npz')
    products = [dict(np.load(output / f'chain_{seed}.npz')) for seed in config['seeds']]
    count = min(len(p['latent']) for p in products)
    latent = np.stack([p['latent'][:count] for p in products]).reshape(-1, 20)
    log_target = np.stack([p['log_target'][:count] for p in products]).reshape(-1)
    original = json.loads((ROOT / config['source_config']).read_text())
    fine = torch.tensor(np.load(ROOT / original['fine_catalog'])['data'], dtype=torch.float64)
    old_count, old_seed = model.lambda_det_mc_samples, model.selection_mc_seed
    with torch.no_grad():
        for index in [0, len(check['indices']) - 1]:
            selected = check['indices'][index]
            point = torch.tensor(latent[selected])
            event = float(differentiable_log_target(model, point, fine))
            model.lambda_det_mc_samples = config['fine_selection_points']
            model.selection_mc_seed = old_seed + config['fine_selection_seed_offset']
            refined = float(differentiable_log_target(model, point, fine))
            model.lambda_det_mc_samples, model.selection_mc_seed = old_count, old_seed
            assert event - log_target[selected] == pytest.approx(check['event_change'][index], abs=1e-9)
            assert refined - event == pytest.approx(check['selection_change'][index], abs=1e-9)
