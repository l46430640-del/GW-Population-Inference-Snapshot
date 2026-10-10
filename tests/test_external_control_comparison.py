import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_external_control as comparison
from binned_gp_benchmark import controlled_selection
from external_selected_evaluation import coarsen, copula_cells, normal_points


@pytest.fixture(scope='module', autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def test_settings_preserve_declared_case_samples_and_common_partition():
    config = json.loads(comparison.CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    precedent = json.loads((ROOT / config['partition_precedent']).read_text())
    registry = json.loads((ROOT / config['posterior_registry']).read_text())
    assert config['experiment'] == 'selected_gp_truth'
    assert 2**config['test_qmc_power'] == protocol['evaluation']['test_event_samples']
    assert config['verification_test_samples'] == protocol['evaluation']['verification_test_event_samples']
    for name in ('primary_cells_per_axis', 'verification_cells_per_axis'):
        assert config[name] == precedent[name]
    assert config['maximum_seconds'] < registry['cost']['remaining_study_seconds']
    assert config['verification_draws_per_chain'] == 128


def test_truth_subdivision_preserves_intrinsic_and_selected_mass_exactly():
    rng = np.random.default_rng(72)
    p = rng.dirichlet(np.ones(6**3))
    alpha = p @ (controlled_selection(6)*6**3)
    truth = {'probabilities': p, 'alpha': np.asarray(alpha)}
    result = comparison.truth_product(truth, np.arange(12), 6, 48)
    np.testing.assert_allclose(coarsen(result['intrinsic'], 48, 6), p, atol=1e-15)
    expected = p*controlled_selection(6)*6**3/alpha
    np.testing.assert_allclose(coarsen(result['detected'], 48, 6), expected, atol=1e-15)
    assert result['intrinsic'].sum() == pytest.approx(1.)
    assert result['detected'].sum() == pytest.approx(1.)


def test_population_builder_uses_every_weighted_draw_and_checks_budget():
    params = {'log_k_a': np.log(np.full((2, 3), .9999)),
        'log_k_b': np.log(np.full((2, 3), .9999)),
        'weights_logits': np.log([.4, .6]),
        'k_L_cholesky': np.broadcast_to(np.eye(3), (2, 3, 3)).copy()}
    samples = {name: np.repeat(value[None], 3, axis=0) for name, value in params.items()}
    samples['log_k_a'][1] += .2
    samples['log_k_b'][2] -= .1
    weights = np.array([.1, .3, .6])
    calls = []
    actual = comparison.population_product(samples, weights, 8, 9, 6, lambda: calls.append(1))
    products = [copula_cells({name: value[i] for name, value in samples.items()}, normal_points(8, 9), 6) for i in range(3)]
    for kind in ('intrinsic', 'detected'):
        np.testing.assert_allclose(actual[kind], sum(w*p[kind] for w, p in zip(weights, products)), atol=1e-15)
    assert len(calls) == 3
    assert actual['features'].shape == (3, 12)


def test_training_refinement_preserves_predictive_subset_order():
    selected = np.array([0, 3, 4, 7])
    saved = np.array([0, 1, 3, 4, 6, 7])
    primary = np.arange(48, dtype=float).reshape(8, 2, 3)/100
    fine = primary[saved, 1] + np.arange(6)[:, None]/100
    posterior = {'verification_indices': saved, 'verification_log_integrals': fine,
        'log_integrals': primary, 'weights': np.arange(1, 9, dtype=float).reshape(2, 4)/36}
    a, b = comparison.refined_subset_weights(posterior, selected)
    expected = np.array([1, 4, 5, 8], dtype=float)
    expected /= expected.sum()
    np.testing.assert_allclose(a, expected, atol=1e-15)
    expected *= np.exp(np.array([0, 2, 3, 5])*3/100)
    expected /= expected.sum()
    np.testing.assert_allclose(b, expected, atol=1e-15)
    with pytest.raises(ValueError, match='Every predictive verification draw'):
        comparison.refined_subset_weights(posterior, [2])


def test_only_accepted_ensembles_are_loaded_without_duplicate_prefixes():
    config = json.loads(comparison.CONFIG.read_text())
    registry, manifest, posterior, samples, shape, gp, sources = comparison.accepted_inputs(config)
    assert registry['posteriors_ready_for_comparison']
    assert manifest['experiment']['train_count'] == 96
    assert manifest['experiment']['test_count'] == 192
    assert shape == (4, 800)
    assert samples['weights_logits'].shape[0] == 3200
    assert posterior['weights'].sum() == pytest.approx(1.)
    assert set(gp) == {'6', '12'}
    for target, summary, chains, inputs in gp.values():
        assert summary['decision']['posterior_accepted']
        assert len(chains) == 4
        assert all(len(chain['coordinate']) == 2400 for chain in chains)
        assert len(inputs['log_event_weights']) == 96
    assert 'results/analysis/external_gp_control_catalog/test_catalog.npz' in sources
