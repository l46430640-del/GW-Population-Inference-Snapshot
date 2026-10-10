"""Bounded external GP inference on the archived off-family LISA catalog."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import time
PROCESS_STARTED = time.perf_counter()

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pyro
from pyro.ops import stats
import torch

from benchmark_selected_mixture import runtime_metadata
from binned_gp_benchmark import posterior_log_event_weights
from build_lisa_analysis_assets import LISASNR
from correct_selected_posterior import sha256
from lisa_gp_inputs import integral_changes, integral_report, selection_coefficients
from lisa_noisy_catalog import posterior_from_measurements
from pilot_external_gp_control_copula import enforce_budget
from run_external_control_production import freeze_target, install_adapter
from run_external_gp_pilot import ROOT, run_pilot
from run_external_gp_selected import (InitializedGPPotential, checked_reports,
    density_functionals, summarize)
from run_lisa_production import verify_product, write_json
from run_symmetric_gp_pilot import check_coordinates
from symmetric_gp_benchmark import SymmetricBinnedGPPotential

CONFIG = ROOT / 'configs/external_lisa_gp.json'


def effective_protocol(config, protocol):
    result = copy.deepcopy(protocol)
    result['sampling']['retained_per_chain'] = config['retained_per_chain']
    return result


def source_manifest(config):
    path = ROOT / config['accepted_copula_summary']
    accepted = json.loads(path.read_text())
    if not accepted['decision']['eligible_for_scientific_summaries']:
        raise ValueError('The fixed copula posterior has not been accepted')
    for name, digest in accepted['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Accepted LISA source changed: {name}')
    catalog = ROOT / config['training_catalog']
    if accepted['configuration']['primary_event_catalog'] != config['training_catalog']:
        raise ValueError('External fit must use the same training event catalog')
    paths = [CONFIG, Path(__file__), path, catalog, ROOT / config['protocol'], ROOT / config['previous_cost'],
        ROOT / 'model_assets/lisa_analysis/phenom_d_grid.npz']
    scripts = ['lisa_gp_inputs', 'lisa_noisy_catalog', 'build_lisa_analysis_assets',
        'binned_gp_benchmark', 'symmetric_gp_benchmark', 'run_external_gp_pilot',
        'run_external_gp_selected', 'run_symmetric_gp_pilot', 'run_external_control_production',
        'run_external_gp_bounded_diagnostics', 'sample_lisa_transport', 'benchmark_selected_mixture',
        'run_lisa_production', 'correct_selected_posterior', 'pilot_external_gp_control_copula']
    paths.extend(ROOT / f'scripts/{name}.py' for name in scripts)
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def verification_catalog(config, output, catalog, sources, check):
    path = output / 'verification_catalog.npz'
    if path.with_suffix('.json').exists():
        report = verify_product(path)
        if report['source_sha256'] != sources:
            raise ValueError('Verification catalog source changed')
        return np.load(path)['data']
    if path.exists():
        raise FileExistsError('Unverified event samples are retained')
    started = time.perf_counter()
    rng, response = np.random.default_rng(config['verification_event_seed']), LISASNR()
    mask = catalog['detected']
    values, diagnostics = [], []
    for i, (y, rho) in enumerate(zip(catalog['measurement'][mask], catalog['rho_observed'][mask])):
        check()
        points, _, report = posterior_from_measurements(y, rho, response, config['verification_event_samples'], rng)
        values.append(points)
        diagnostics.append(report)
        if (i+1) % 25 == 0:
            print('LISA GP verification events', i+1, '/', mask.sum(), flush=True)
    points = np.asarray(values)
    data = np.concatenate([points, np.ones((*points.shape[:2], 1))], -1)
    np.savez_compressed(path, data=data, measurement=catalog['measurement'],
        rho_observed=catalog['rho_observed'], detected=mask)
    write_json(path.with_suffix('.json'), {'source_sha256': sources, 'draw_product_sha256': sha256(path),
        'seconds': time.perf_counter()-started, 'event_diagnostics': diagnostics, 'runtime': runtime_metadata()})
    return data


def prepare(config, protocol, output, sources, check):
    with np.load(ROOT / config['training_catalog']) as data:
        catalog = {name: data[name] for name in ('data', 'detected', 'measurement', 'rho_observed')}
    experiment = next(row for row in protocol['experiments'] if row['name'] == config['experiment'])
    if len(catalog['data']) != experiment['train_count'] or catalog['data'].shape[1] != protocol['evaluation']['training_event_samples']:
        raise ValueError('Training count or event sample count differs from the declaration')
    if not np.all(catalog['rho_observed'][catalog['detected']] > 8):
        raise ValueError('Training catalog contains undetected events')
    fine = verification_catalog(config, output, catalog, sources, check)
    sources = dict(sources)
    for suffix in ('npz', 'json'):
        path = output / ('verification_catalog.'+suffix)
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    response, result = LISASNR(), {}
    for bins in config['bins_per_axis']:
        path = output / f'inputs_{bins}.npz'
        if path.with_suffix('.json').exists():
            report = verify_product(path)
            if report['source_sha256'] != sources:
                raise ValueError('LISA GP input provenance changed')
            result[bins] = dict(np.load(path))
            continue
        if path.exists():
            raise FileExistsError('Unverified bin integrals are retained')
        started = time.perf_counter()
        arrays = {'log_event_weights': posterior_log_event_weights(catalog['data'], bins),
            'fine_log_event_weights': posterior_log_event_weights(fine, bins),
            'selection_coefficients': selection_coefficients(response, bins, config['selection_power'],
                config['selection_seed'], config['selection_cell_block'], check),
            'fine_selection_coefficients': selection_coefficients(response, bins, config['verification_selection_power'],
                config['verification_selection_seed'], config['selection_cell_block'], check)}
        np.savez_compressed(path, **arrays)
        write_json(path.with_suffix('.json'), {'source_sha256': sources, 'draw_product_sha256': sha256(path),
            'seconds': time.perf_counter()-started, 'runtime': runtime_metadata(),
            'empty_event_bins_primary': int(np.isneginf(arrays['log_event_weights']).sum()),
            'empty_event_bins_verification': int(np.isneginf(arrays['fine_log_event_weights']).sum()),
            'minimum_selection_coefficient': float(arrays['selection_coefficients'].min()),
            'maximum_selection_coefficient_change': float(abs(arrays['selection_coefficients']-arrays['fine_selection_coefficients']).max())})
        print('Prepared LISA GP bins', bins, round(time.perf_counter()-started, 2), flush=True)
        result[bins] = arrays
    for bins in config['bins_per_axis']:
        for suffix in ('npz', 'json'):
            path = output / f'inputs_{bins}.{suffix}'
            sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result, sources


def run_chain(config, protocol, inputs, bins, chain, pilot, deadline, check):
    sampling = protocol['sampling']
    seed = sampling['chain_seeds'][chain]
    settings = dict(config, seed=seed, warmup=config['pilot_warmup'] if pilot else sampling['warmup'],
        retained=config['pilot_retained'] if pilot else sampling['retained_per_chain'],
        target_acceptance=sampling['target_acceptance'], maximum_tree_depth=sampling['maximum_tree_depth'],
        maximum_seconds=min(config['pilot_maximum_seconds'] if pilot else config['maximum_seconds'],
            deadline-time.perf_counter()-config['verification_reserve_seconds']))
    if settings['maximum_seconds'] <= 0:
        raise RuntimeError('Insufficient time remains for a chain and numerical checks')
    potential = InitializedGPPotential(bins, inputs['log_event_weights'], inputs['selection_coefficients'],
        protocol['binned_gp'], config['covariance_axis_jitter'])
    potential.set_initial_hyperparameters(config['initial_amplitude_and_lengths'][chain])
    reference = SymmetricBinnedGPPotential(bins, inputs['log_event_weights'], inputs['selection_coefficients'],
        protocol['binned_gp'], config['verification_covariance_axis_jitter'])
    pyro.clear_param_store()
    started = time.perf_counter()
    report, arrays = run_pilot(potential, settings)
    check()
    verification, coordinate_checks = check_coordinates(potential, reference, arrays['coordinate'], config)
    arrays.update(coordinate_checks)
    numerical = integral_changes(potential, arrays['coordinate'], inputs['fine_log_event_weights'],
        inputs['fine_selection_coefficients'], check)
    numerical_report = integral_report(numerical, protocol['evaluation']['maximum_log_likelihood_difference_sd'])
    arrays.update(numerical)
    extra_names = density_functionals(torch.zeros(bins**3), bins, config['marginal_cdf_points'])[0]
    extra = []
    with torch.no_grad():
        for coordinate in arrays['coordinate']:
            check()
            extra.append(density_functionals(potential.field(torch.tensor(coordinate)), bins, config['marginal_cdf_points'])[1])
    arrays.update(diagnostic_names=np.asarray(list(arrays['feature_names'])+extra_names),
        diagnostic_values=np.column_stack([arrays['features'], np.asarray(extra).reshape(-1, len(extra_names))]))
    numerical_ok = (numerical_report['passed'] and verification['derivatives_passed']
        and verification['jitter_difference_sd'] is not None
        and verification['jitter_difference_sd'] <= protocol['evaluation']['maximum_log_likelihood_difference_sd'])
    report.update(seed=seed, coordinate_verification=verification, integral_verification=numerical_report,
        covariance_jitter_log_target_difference_sd=verification['jitter_difference_sd'],
        event_quadrature_passed=numerical_report['passed'],
        ready_for_production_sampling=bool(report['status'] == 'complete' and report['divergences'] == [] and numerical_ok),
        process_elapsed_seconds=time.perf_counter()-started, runtime=runtime_metadata())
    return report, arrays


def save_chain(path, report, arrays, target_hash):
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Completed or partial chain is retained')
    np.savez_compressed(path, **arrays)
    report.update(draw_product_sha256=sha256(path), target_product_sha256=target_hash)
    write_json(path.with_suffix('.json'), report)
    print('Saved', path.name, report['status'], report['integral_verification'], flush=True)


def run(config, protocol, output, stage, deadline, check):
    inputs, sources = prepare(config, protocol, output, source_manifest(config), check)
    signature = {'configuration': config, 'effective_protocol': protocol, 'source_sha256': sources,
        'experiment': config['experiment'], 'input_fields': ['training event posterior samples',
            'training measurement', 'training observed SNR', 'training detected mask', 'fixed SNR response grid']}
    target_hash = freeze_target(output, signature)
    if stage == 'prepare':
        return
    for bins in config['bins_per_axis']:
        path = output / f'pilot_bins_{bins}.npz'
        if path.with_suffix('.json').exists():
            report = verify_product(path)
            if report['target_product_sha256'] != target_hash:
                raise ValueError('Pilot target differs from current numerical inputs')
        else:
            report, arrays = run_chain(config, protocol, inputs[bins], bins, 0, True, deadline, check)
            save_chain(path, report, arrays, target_hash)
        if not report['ready_for_production_sampling']:
            raise RuntimeError(f'Pilot for {bins} bins needs numerical or sampling repair')
    if stage == 'pilot':
        return
    for bins in config['bins_per_axis']:
        directory = output / f'bins_{bins}'
        if (directory / 'summary.json').exists():
            continue
        signature_here = copy.deepcopy(signature)
        signature_here.update(bins_per_axis=bins, inputs_product_sha256=sha256(output / f'inputs_{bins}.npz'))
        path = output / f'pilot_bins_{bins}.json'
        signature_here['source_sha256'][path.relative_to(ROOT).as_posix()] = sha256(path)
        digest = freeze_target(directory, signature_here)
        previous = checked_reports(directory, digest, protocol)
        for chain, seed in enumerate(protocol['sampling']['chain_seeds']):
            if seed in {row['seed'] for row in previous}:
                continue
            check()
            report, arrays = run_chain(config, protocol, inputs[bins], bins, chain, False, deadline, check)
            save_chain(directory / f'chain_{seed}.npz', report, arrays, digest)
            if report['status'] != 'complete':
                break
        summary = summarize(directory, protocol, digest)
        reports = checked_reports(directory, digest, protocol)
        combined = {name: [] for name in ('event_log_likelihood_change', 'selection_log_likelihood_change', 'combined_log_likelihood_change')}
        for row in reports:
            with np.load(directory / f"chain_{row['seed']}.npz") as arrays:
                for name in combined:
                    combined[name].extend(arrays[name])
        numerical = integral_report(combined, protocol['evaluation']['maximum_log_likelihood_difference_sd'])
        decision = dict(summary['decision'])
        decision['pooled_integrals_passed'] = numerical['passed']
        decision['posterior_accepted'] = all(value for name, value in decision.items() if name != 'posterior_accepted')
        write_json(directory / 'acceptance.json', {'decision': decision, 'pooled_integral_verification': numerical,
            'summary_sha256': sha256(directory / 'summary.json'), 'target_product_sha256': digest,
            'scientific_scores_examined': False})
        print('LISA GP acceptance', bins, decision, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'pilot', 'all'], default='all')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    protocol = effective_protocol(config, json.loads((ROOT / config['protocol']).read_text()))
    cost = json.loads((ROOT / config['previous_cost']).read_text())
    if config['maximum_seconds'] > cost['remaining_study_seconds']:
        raise ValueError('LISA GP allocation exceeds the remaining study allowance')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    old = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('job_cost_*.json'))
    deadline = PROCESS_STARTED+config['maximum_seconds']-old
    check = lambda: enforce_budget(deadline, config['maximum_resident_gib'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    original = install_adapter(config)
    status, error = 'complete', None
    try:
        run(config, protocol, output, args.stage, deadline, check)
        check()
    except BaseException as exc:
        status, error = 'stopped', str(exc)
        raise
    finally:
        stats._cummin = original
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(output / f'job_cost_{len(list(output.glob("job_cost_*.json"))):03d}.json',
            {'status': status, 'error': error, 'stage': args.stage, 'process_elapsed_seconds': seconds,
             'all_lisa_gp_attempts_seconds': old+seconds,
             'total_study_charge_seconds': cost['total_study_charge_seconds']+old+seconds,
             'remaining_study_seconds': cost['remaining_study_seconds']-old-seconds,
             'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})


if __name__ == '__main__':
    main()
