import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_selected_mixture import selected_model
from run_external_control_production import (copula_functionals, effective_protocol,
    freeze_target, initial_copula_points, install_adapter, previous_cost)
from pyro.ops import stats


@pytest.fixture(scope='module', autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def settings():
    config = json.loads((ROOT / 'configs/external_control_production.json').read_text())
    original = json.loads((ROOT / config['protocol']).read_text())
    return config, original


def test_only_resource_cap_changes_and_total_study_budget_is_preserved():
    config, original = settings()
    expected = copy.deepcopy(original)
    expected['resources']['maximum_seconds_per_fit'] = 14400
    assert effective_protocol(config, original) == expected
    assert original['resources']['maximum_seconds_per_fit'] == 7200
    assert config['resource_amendment']['maximum_seconds_control_production'] == 18000
    config['resource_amendment']['maximum_seconds_control_production'] = 30000
    with pytest.raises(ValueError, match='study budget'):
        effective_protocol(config, original)


def test_four_distinct_training_only_starts_retain_the_original_first_point():
    config, _ = settings()
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    data = torch.tensor(np.load(ROOT / 'results/analysis/external_gp_control_catalog/train_catalog.npz')['data'])
    previous = np.load(ROOT / 'results/analysis/external_gp_control_copula_pilot/pilot.npz')['initial_coordinate']
    points, records = initial_copula_points(model, data, config, previous)
    assert points.shape == (4, 20) and np.isfinite(points.numpy()).all()
    assert len(np.unique(points.numpy(), axis=0)) == 4
    assert [r['seed'] for r in records] == config['copula_initialization_seeds']
    np.testing.assert_array_equal(points[0], previous)
    repeated, _ = initial_copula_points(model, data, config, previous)
    np.testing.assert_array_equal(points, repeated)


def test_shared_functionals_of_uniform_population_have_expected_order_and_values():
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    point = np.zeros((1, 20))
    point[:, 2:14] = np.log(.9999)
    names, values, _ = copula_functionals(model, point, 12, 847501)
    assert len(names) == 29 and len(set(names)) == 29
    np.testing.assert_allclose(values[0, :3], .5, atol=1e-12)
    np.testing.assert_allclose(values[0, 3:6], 1/12, atol=1e-12)
    assert values[0, 6] == pytest.approx(.05+.95/3, abs=1e-12)
    np.testing.assert_allclose(values[0, 12:21], np.tile([.25, .5, .75], 3), atol=1e-12)
    np.testing.assert_allclose(values[0, 21:], 1/8, atol=1e-12)
    assert names[12] == 'cdf_x1_0.25' and names[-1] == 'octant_111'


def test_adapter_is_the_checked_linear_storage_diagnostic():
    config, _ = settings()
    original = install_adapter(config)
    try:
        point = torch.tensor([4., 2., 3., -1.])
        np.testing.assert_array_equal(stats._cummin(point), [4., 2., 2., -1.])
    finally:
        stats._cummin = original


def test_target_cannot_change_after_freezing(tmp_path):
    first = freeze_target(tmp_path, {'source_sha256': {'example': '123'}})
    assert freeze_target(tmp_path, {'source_sha256': {'example': '123'}}) == first
    with pytest.raises(ValueError):
        freeze_target(tmp_path, {'source_sha256': {'example': '456'}})


def test_cost_does_not_double_count_nested_reports_and_charges_partial_work(tmp_path):
    folder = tmp_path / 'copula'
    folder.mkdir()
    (folder / 'chain_1.json').write_text(json.dumps({'process_elapsed_seconds': 100}))
    (folder / 'summary.json').write_text(json.dumps({'verification_seconds': 20}))
    (tmp_path / 'job_cost_000.json').write_text(json.dumps({'process_elapsed_seconds': 130}))
    assert previous_cost(tmp_path) == 130
    (folder / 'chain_2.json').write_text(json.dumps({'process_elapsed_seconds': 50}))
    assert previous_cost(tmp_path) == 170


def test_gp_extension_changes_only_sample_count_and_preserves_fit_settings():
    from extend_external_control_gp import extended_protocol
    config = json.loads((ROOT / 'configs/external_control_gp_extension.json').read_text())
    base, original = settings()
    for key in base:
        if key not in ('output', 'resource_amendment', 'decision'):
            assert config[key] == base[key]
    expected = effective_protocol(config, original)
    expected['sampling']['retained_per_chain'] = 2400
    assert extended_protocol(config, original) == expected
    assert config['sampling_extension']['original_retained_per_chain'] == 800
