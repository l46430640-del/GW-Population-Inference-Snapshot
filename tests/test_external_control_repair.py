from pathlib import Path
import json
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import repair_external_control_copula as repair


def test_cached_integrals_never_reuse_rejected_chain_states(monkeypatch):
    clouds = [np.ones((3, n, 4)) for n in (2, 4, 8)]
    primary = np.arange(24.).reshape(4, 2, 3)
    fine = np.arange(6.).reshape(2, 3)+100
    state = {'clouds': clouds, 'keep': np.array([True, False, True, True]),
        'fine_rows': np.array([0, 1, -1, -1]),
        'data': {'log_integrals': primary, 'verification_log_integrals': fine}, 'hits': 0, 'fresh': 0}
    monkeypatch.setattr(repair, 'STATE', state)
    calls = []
    def evaluate(sites, selected, indices, deadline, memory):
        calls.extend(indices)
        return np.full((len(indices), len(selected), 3), 999.)
    monkeypatch.setattr(repair, 'ORIGINAL_EVALUATE', evaluate)
    result = repair.cached_evaluations({}, clouds[:2], range(4), time.perf_counter()+30, 2)
    np.testing.assert_array_equal(result[[0, 2, 3]], primary[[0, 2, 3]])
    assert (result[1] == 999).all() and calls == [1]
    result = repair.cached_evaluations({}, clouds[2:], range(4), time.perf_counter()+30, 2)
    np.testing.assert_array_equal(result[0, 0], fine[0])
    assert (result[1:] == 999).all()
    assert calls == [1, 1, 2, 3]
    assert state['hits'] == 7 and state['fresh'] == 5
    with pytest.raises(ValueError, match='clouds differ'):
        repair.cached_evaluations({}, [clouds[2]+1], [0], time.perf_counter()+30, 2)


def test_repair_changes_the_sampler_but_no_population_information():
    config = json.loads((ROOT / 'configs/external_control_copula_repair.json').read_text())
    parent = json.loads((ROOT / 'configs/external_control_production.json').read_text())
    for key in parent:
        if key not in ('output', 'resource_amendment', 'decision', 'copula_fit_verification_reserve_seconds'):
            assert config[key] == parent[key]
    original = json.loads((ROOT / config['protocol']).read_text())
    expected = repair.ORIGINAL_PROTOCOL(config, original)
    expected['sampling']['target_acceptance'] = .97
    assert repair.repair_protocol(config, original) == expected
    assert config['sampling_repair']['replace_seed'] == 847203
    assert config['sampling_repair']['reuse_seeds'] == [847201, 847202, 847204]
    assert config['resource_amendment']['prior_study_budget_charge_seconds'] + 5604 <= 43200
