import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_external_lisa_gp_batched as batch


def test_independent_batches_preserve_count_rng_order_and_combined_precision(monkeypatch):
    calls = []
    def sampler(y, rho, response, count, rng):
        points = rng.uniform(size=(count, 3))
        calls.append((count, points.copy()))
        return points, points.mean(0), {'proposal_ess': 5.*count, 'proposal_points': 8*count}
    monkeypatch.setattr(batch, 'base_sampler', sampler)
    points, center, report = batch.batched_event_posterior(None, None, None, 8192, np.random.default_rng(8))
    np.testing.assert_array_equal(points, np.random.default_rng(8).uniform(size=(8192, 3)))
    np.testing.assert_allclose(center, points.mean(0), atol=1e-14)
    assert [n for n, _ in calls] == [2048]*4
    assert report['proposal_ess'] == pytest.approx(5*8192)
    assert report['proposal_points'] == 8*8192
    assert report['normalized_batch_weights'] == [.25]*4


def test_original_single_batch_requirement_cannot_be_relaxed(monkeypatch):
    def sampler(y, rho, response, count, rng):
        return np.zeros((count, 3)), np.zeros(3), {'proposal_ess': 3.9*count, 'proposal_points': 10000}
    monkeypatch.setattr(batch, 'base_sampler', sampler)
    with pytest.raises(ValueError, match='original event precision'):
        batch.batched_event_posterior(None, None, None, 8192, None)
