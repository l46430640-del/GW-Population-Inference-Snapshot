import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_external_lisa as comparison
from external_lisa_evaluation import (cell_integrals, gp_population, gp_projection,
    population_cells, population_projection, truth_features)
from external_selected_evaluation import coarsen, normal_points


@pytest.fixture(scope='module', autouse=True)
def precision():
    old = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(old[0])
    torch.set_num_threads(old[1])


def parameters():
    return {'log_k_a': np.log(np.full((2, 3), .9999)),
        'log_k_b': np.log(np.full((2, 3), .9999)), 'weights_logits': np.log([.4, .6]),
        'k_L_cholesky': np.broadcast_to(np.eye(3), (2, 3, 3)).copy()}


def test_declaration_preserves_experiment_precision_and_records_bounded_cost_extension():
    config = json.loads(comparison.CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    precedent = json.loads((ROOT / config['partition_precedent']).read_text())
    previous = json.loads((ROOT / config['posterior_registry']).read_text())
    assert config['experiment'] == protocol['experiments'][2]['name']
    assert config['test_event_samples'] == protocol['evaluation']['test_event_samples'] == 2048
    assert config['verification_test_samples'] == protocol['evaluation']['verification_test_event_samples'] == 8192
    for name in ('primary_cells_per_axis', 'verification_cells_per_axis'):
        assert config[name] == precedent[name]
    assert config['maximum_total_study_seconds'] == 45000
    assert config['maximum_total_study_seconds']-3600*protocol['resources']['maximum_total_cpu_hours'] == 1800
    assert config['maximum_seconds']+previous['cost']['total_study_charge_seconds'] < 45000
    assert config['resource_amendment']


def test_truth_features_are_exact_beta_mixture_moments_except_declared_selection():
    from validate_population_morphologies import BETA_A, BETA_B
    p = np.array([.55, .45])
    means = BETA_A/(BETA_A+BETA_B)
    result = truth_features(.32)
    np.testing.assert_allclose(result[:3], p @ means)
    assert result[6] == .32
    assert result[9] == pytest.approx(p @ (means[:, 0]*means[:, 1])-result[0]*result[1])
    assert result[9] < 0 < result[10]


def test_stratified_selection_retains_every_cell_and_normalized_truth():
    result = cell_integrals(lambda x: np.full(len(x), 8.), 6, 8, 51)
    np.testing.assert_allclose(result['selection'], .5/6**3, atol=1e-15)
    assert result['intrinsic'].sum() == pytest.approx(1.)
    assert result['detected'].sum() == pytest.approx(1.)
    assert result['features'][6] == pytest.approx(.5, abs=2e-4)


def test_population_transport_uses_lisa_selection_and_projection_matches_histograms():
    response = lambda x: 8+3*(x[:, 0]-.5)
    params = parameters()
    normal = normal_points(10, 55)
    product = population_cells(params, normal, 12, response)
    assert product['intrinsic'].sum() == pytest.approx(1.)
    assert product['detected'].sum() == pytest.approx(1.)
    assert product['features'][6] == pytest.approx(.5, abs=.001)
    rng = np.random.default_rng(80)
    gradients = {f'{kind}_{bins}': rng.normal(size=bins**3) for kind in ('intrinsic', 'detected') for bins in (6, 12)}
    actual = population_projection(params, normal, response, gradients)
    for label, gradient in gradients.items():
        kind, bins = label.split('_')
        assert actual[label] == pytest.approx(coarsen(product[kind], 12, int(bins)) @ gradient, abs=1e-14)


def test_gp_recovery_projection_matches_explicit_selected_subdivision():
    rng = np.random.default_rng(77)
    probability = rng.dirichlet(np.ones(8), size=3)
    selection = rng.uniform(1e-15, 1., 12**3)/12**3
    product = gp_population(probability, 2, 12, selection)
    for kind in ('intrinsic', 'detected'):
        assert product[kind].sum() == pytest.approx(1.)
        for partition in (6, 12):
            gradient = rng.normal(size=partition**3)
            projected = gp_projection(probability, 2, 12, selection, product['alpha'], f'{kind}_{partition}', gradient)
            for i in range(3):
                individual = gp_population(probability[i:i+1], 2, 12, selection)
                assert projected[i] == pytest.approx(coarsen(individual[kind], 12, partition) @ gradient, abs=1e-14)


def test_population_builder_preserves_all_draw_weights():
    p = parameters()
    samples = {name: np.repeat(value[None], 2, axis=0) for name, value in p.items()}
    samples['log_k_a'][1] += .3
    weights = np.array([.3, .7])
    result = comparison.population_product(samples, weights, 8, 22, 6, lambda: None)
    from build_lisa_analysis_assets import LISASNR
    rows = [population_cells({name: value[i] for name, value in samples.items()}, normal_points(8, 22), 6, LISASNR()) for i in range(2)]
    for kind in ('intrinsic', 'detected'):
        np.testing.assert_allclose(result[kind], sum(w*row[kind] for w, row in zip(weights, rows)), atol=1e-15)


def test_inputs_reuse_the_same_accepted_chains_and_weighted_copula_draws():
    config = json.loads(comparison.CONFIG.read_text())
    registry, accepted, posterior, weights, verification, model, samples, gp, sources = comparison.accepted_inputs(config)
    assert registry['gp_posteriors_ready_for_comparison']
    assert accepted['decision']['eligible_for_scientific_summaries']
    assert weights.shape == posterior['latent'].shape[:2] == (4, 500)
    assert weights.sum() == pytest.approx(1.)
    assert len(samples['weights_logits']) == 2000
    for key, (chains, inputs) in gp.items():
        assert len(chains) == 4
        assert all(len(row['coordinate']) == (2400 if key == '6' else 800) for row in chains)
        assert inputs['log_event_weights'].shape == (135, int(key)**3)
    assert len(verification['uniform_indices']) == 256
    assert config['training_verification'] in sources
    np.testing.assert_array_equal(model.min_vals.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(model.max_vals.numpy(), [1, 1, 1])
