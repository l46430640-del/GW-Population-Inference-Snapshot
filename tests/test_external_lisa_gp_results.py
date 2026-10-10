"""Retained LISA GP preparation, accepted low resolution and bounded partial fit."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from binned_gp_benchmark import posterior_log_event_weights
from correct_selected_posterior import sha256
from lisa_gp_inputs import integral_report
from run_lisa_production import mixing_summary, verify_product

OUTPUT = ROOT / 'results/analysis/external_lisa_gp'


def test_all_preparation_and_chain_sources_match_their_products():
    for path in OUTPUT.rglob('*.json'):
        value = json.loads(path.read_text())
        for name, digest in value.get('source_sha256', {}).items():
            assert sha256(ROOT / name) == digest
        if 'draw_product_sha256' in value:
            assert sha256(path.with_suffix('.npz')) == value['draw_product_sha256']
        if 'target_product_sha256' in value:
            target = path.parent / 'target.json'
            assert sha256(target) == value['target_product_sha256']
        if 'summary_sha256' in value:
            assert sha256(path.parent / 'summary.json') == value['summary_sha256']


def test_independent_event_batches_keep_the_same_measurements_and_precision():
    config = json.loads((ROOT / 'configs/external_lisa_gp.json').read_text())
    primary = np.load(ROOT / config['training_catalog'])
    fine = np.load(OUTPUT / 'verification_catalog.npz')
    assert fine['data'].shape == (135, 8192, 4)
    for key in ('measurement', 'rho_observed', 'detected'):
        np.testing.assert_array_equal(primary[key], fine[key])
    np.testing.assert_array_equal(fine['data'][:, :, 3], 1.)
    report = verify_product(OUTPUT / 'verification_catalog.npz')
    assert len(report['event_diagnostics']) == 135
    for event in report['event_diagnostics']:
        assert event['batch_samples'] == [2048]*4
        assert event['normalized_batch_weights'] == [.25]*4
        ess = np.asarray([r['proposal_ess'] for r in event['batches']])
        assert min(ess) >= 8192
        assert event['proposal_ess'] == pytest.approx(16/np.sum(1/ess))
        assert event['proposal_ess'] >= 32768


@pytest.mark.parametrize('bins', [6, 12])
def test_gp_integrals_use_the_frozen_event_samples_and_keep_every_selection_bin(bins):
    config = json.loads((ROOT / 'configs/external_lisa_gp.json').read_text())
    primary = np.load(ROOT / config['training_catalog'])['data']
    fine = np.load(OUTPUT / 'verification_catalog.npz')['data']
    inputs = np.load(OUTPUT / f'inputs_{bins}.npz')
    np.testing.assert_array_equal(inputs['log_event_weights'], posterior_log_event_weights(primary, bins))
    np.testing.assert_array_equal(inputs['fine_log_event_weights'], posterior_log_event_weights(fine, bins))
    for name in ('selection_coefficients', 'fine_selection_coefficients'):
        assert inputs[name].shape == (bins**3,)
        assert np.all(inputs[name] > 0) and np.all(inputs[name] <= 1/bins**3)
    pilot = verify_product(OUTPUT / f'pilot_bins_{bins}.npz')
    assert pilot['ready_for_production_sampling']
    assert not pilot['eligible_for_scientific_comparisons']
    assert pilot['samples_completed'] == pilot['warmup_completed'] == 100


def test_accepted_six_bin_posterior_reconstructs_mixing_and_pooled_integrals():
    directory = OUTPUT / 'bins_6'
    summary = json.loads((directory / 'summary.json').read_text())
    final = json.loads((directory / 'acceptance.json').read_text())
    assert all(final['decision'].values())
    assert not final['scientific_scores_examined']
    products = []
    for seed in range(847201, 847205):
        report = verify_product(directory / f'chain_{seed}.npz')
        assert report['samples_completed'] == 2400 and report['warmup_completed'] == 500
        assert report['divergences'] == []
        products.append(dict(np.load(directory / f'chain_{seed}.npz')))
    checked = mixing_summary(products[0]['diagnostic_names'].tolist(), np.stack([p['diagnostic_values'] for p in products]))
    for key in ('maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess'):
        assert checked[key] == pytest.approx(summary['mixing'][key], rel=1e-12)
    numerical = {name: np.concatenate([p[name] for p in products]) for name in final['pooled_integral_verification']['standard_deviations']}
    assert integral_report(numerical, .2) == final['pooled_integral_verification']


def test_partial_twelve_bin_chain_is_not_accepted_or_scored_as_a_loss():
    directory = OUTPUT / 'bins_12'
    report = verify_product(directory / 'chain_847201.npz')
    final = json.loads((directory / 'acceptance.json').read_text())
    assert report['status'] == 'time_limit'
    assert report['warmup_completed'] == 500
    assert report['samples_completed'] == 1007
    assert report['divergences'] is None
    assert not final['decision']['posterior_accepted']
    assert not final['scientific_scores_examined']


def test_original_failure_and_all_runtime_costs_are_preserved():
    first, second = [json.loads((OUTPUT / f'job_cost_{i:03d}.json').read_text()) for i in (0, 1)]
    assert first['status'] == 'stopped' and 'importance ESS' in first['error']
    assert second['status'] == 'complete'
    cost = first['process_elapsed_seconds']+second['process_elapsed_seconds']
    assert second['all_lisa_gp_attempts_seconds'] == pytest.approx(cost)
    assert cost <= 4200
    assert second['remaining_study_seconds'] == pytest.approx(43200-second['total_study_charge_seconds'])
    assert all(row['runtime']['threads'] == 1 and row['runtime']['peak_working_set_bytes'] <= 2*1024**3 for row in (first, second))
