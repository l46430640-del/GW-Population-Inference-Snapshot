"""Check retained pilot evidence without repeating inference."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from binned_gp_benchmark import controlled_log_event_weights, controlled_selection
from correct_selected_posterior import sha256
from run_lisa_production import verify_product


@pytest.mark.parametrize('bins', [6, 12])
def test_gp_control_pilots_use_only_training_measurements_and_pass_checks(bins):
    path = ROOT / f'results/analysis/external_gp_control_gp_pilot/bins_{bins}.npz'
    report = verify_product(path)
    for name, expected in report['source_sha256'].items():
        assert sha256(ROOT / name) == expected
    assert report['configuration']['experiment'] == 'selected_gp_truth'
    assert report['ready_for_production_sampling']
    assert not report['eligible_for_scientific_comparisons']
    assert report['warmup_completed'] == report['samples_completed'] == 100
    assert report['divergences'] == []
    target = json.loads((path.parent / 'target.json').read_text())
    assert target['input_fields'] == ['observed', 'sigma']
    assert not any('test_catalog' in name or name.endswith('/truth.npz') for name in target['source_sha256'])
    catalog = np.load(ROOT / 'results/analysis/external_gp_control_catalog/train_catalog.npz')
    data = np.load(path)
    np.testing.assert_allclose(data['log_event_weights'], controlled_log_event_weights(catalog['observed'], catalog['sigma'], bins), atol=1e-12)
    np.testing.assert_array_equal(data['selection_coefficients'], controlled_selection(bins))
    error = float(abs(data['directional_derivative_automatic']-data['directional_derivative_finite_difference']).max())
    assert error == report['coordinate_verification']['maximum_directional_derivative_error']
    assert error < 1e-4
    assert data['covariance_jitter_log_target_difference'].std() < .2
    assert report['runtime']['peak_working_set_bytes'] < 2*1024**3
    assert report['process_elapsed_seconds'] <= 600


def test_copula_pilot_preserves_evidence_and_does_not_claim_a_posterior():
    path = ROOT / 'results/analysis/external_gp_control_copula_pilot/pilot.npz'
    report = verify_product(path)
    for name, expected in report['source_sha256'].items():
        assert sha256(ROOT / name) == expected
    assert not report['eligible_for_scientific_comparisons']
    assert report['input_fields'] == ['train_catalog:data', 'train_verification:data']
    assert not any('test_catalog' in name or name.endswith('/truth.npz') for name in report['source_sha256'])
    data = np.load(path)
    assert data['coordinate'].shape == (report['samples_completed'], 20)
    assert data['warmup_coordinate'].shape == (report['warmup_completed'], 20)
    assert np.isfinite(data['progress']).all()
    checked = report['verification']
    count = checked['draws_checked']
    if count:
        assert float(abs(data['derivative_automatic']-data['derivative_numeric']).max()) == checked['maximum_directional_derivative_error']
        assert float(abs(data['precision_log_target_difference']).max()) == checked['maximum_precision_log_target_difference']
    if count > 2:
        assert data['event_log_likelihood_difference'][1:].std(ddof=1) == checked['event_log_likelihood_difference_sd']
    expected = (report['status'] == 'complete' and report['divergences'] == [] and report['event_integration_passed']
                and report['resources_passed'] and checked['derivatives_passed'] and checked['precision_passed'])
    assert report['ready_for_production_sampling'] == expected
    if report['status'] == 'complete':
        assert report['samples_completed'] == report['warmup_completed'] == 100
    else:
        assert not report['ready_for_production_sampling']
