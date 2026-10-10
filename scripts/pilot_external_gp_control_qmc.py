"""Prepare and test reduced event integration on the fixed GP control."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import time
PROCESS_STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
from scipy.special import logsumexp
import torch

from benchmark_selected_mixture import feature_draws, runtime_metadata, selected_model
from build_external_gp_control import ROOT, verify_manifest
from check_selected_mixture_integrals import event_integrals
from correct_selected_posterior import latent_sites, sha256
from external_selected_evaluation import event_log_integrals
from pilot_external_gp_control_copula import BudgetExceeded, enforce_budget, run_bounded, verify_target
from run_lisa_production import verify_product, write_json
from sample_lisa_transport import TransportPotential, compile_potential, value_gradient
from selected_event_qmc import selected_event_cloud

CONFIG = ROOT / 'configs/external_gp_control_qmc_pilot.json'


def verified_sources(report):
    for name, expected in report['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')


def parameters(model, points):
    with torch.no_grad():
        return {name: value.numpy() for name, value in latent_sites(model, torch.tensor(points)).items()}


def integrate_states(model, points, clouds, config, deadline):
    sites = parameters(model, points)
    values = []
    status = 'complete'
    try:
        for i in range(len(points)):
            row = []
            for cloud in clouds:
                enforce_budget(deadline, config['maximum_resident_gib'])
                row.append(event_log_integrals({name: value[i] for name, value in sites.items()}, cloud))
            values.append(row)
    except BudgetExceeded as exc:
        status = str(exc)
    return np.asarray(values).reshape(-1, len(clouds), len(clouds[0])), status


def integration_summary(values, protocol):
    """Short-chain correction diagnostics; never a convergence assessment."""
    reports, weights = [], []
    for target in range(1, values.shape[1]):
        difference = (values[:, target]-values[:, 0]).sum(-1)
        raw = np.exp(difference-logsumexp(difference))
        sd = float(difference.std(ddof=1)) if len(difference) > 1 else None
        reff, k = None, None
        if len(difference) >= 20:
            reff = float(az.ess(difference[None], method='mean', relative=True))
            reff = float(np.clip(reff, 1/len(difference), 1)) if np.isfinite(reff) else 1/len(difference)
            _, tail = az.psislw(difference, reff=reff)
            k = float(tail) if np.isfinite(tail) else None
        ess = float(1/np.sum(raw**2)/len(raw))
        reports.append({'log_likelihood_difference_sd': sd, 'weight_ess_fraction': ess,
            'relative_efficiency': reff, 'pareto_k': k,
            'passed': bool(sd is not None and sd <= protocol['evaluation']['maximum_log_likelihood_difference_sd']
                and ess >= protocol['sampling']['minimum_reuse_weight_ess_fraction']
                and k is not None and k < protocol['sampling']['maximum_pareto_k_for_reuse'])})
        weights.append(raw)
    return reports, np.asarray(weights)


def prepare(config, directory, output, model, sources):
    if any(output.iterdir()):
        raise FileExistsError('Preparation products are retained')
    deadline = PROCESS_STARTED+config['preparation_maximum_seconds']
    catalog = np.load(directory / 'train_catalog.npz')
    paths, clouds = [], []
    for label in ('primary', 'verification'):
        product = selected_event_cloud(catalog['observed'], catalog['sigma'],
            config[f'{label}_power'], config[f'{label}_scramble_seed'])
        path = output / f'{label}_events.npz'
        np.savez_compressed(path, **product)
        paths.append(path)
        clouds.append(product['data'])
    old = np.load(ROOT / config['previous_pilot'])
    points = np.vstack([old['initial_coordinate'], old['warmup_coordinate']])
    clouds.extend([np.load(directory / 'train_verification.npz')['data'], catalog['data']])
    values, status = integrate_states(model, points, clouds, config, deadline)
    difference = (values[:, 1:]-values[:, :1]).sum(-1)
    # Alternate the timed order so the same target is not always measured first.
    indices = np.unique(np.linspace(0, len(points)-1, 5, dtype=int))
    elapsed = np.full((len(indices), 2), np.nan)
    kernels, checks = [], []
    for cloud in (clouds[0], catalog['data']):
        enforce_budget(deadline, config['maximum_resident_gib'])
        traced, checked = compile_potential(TransportPotential(model, torch.tensor(cloud), 'raw'),
            torch.tensor(points[0]), config['jit_absolute_tolerance'])
        kernels.append(traced)
        checks.append(checked)
    for row, index in enumerate(indices):
        for target in ((0, 1) if row % 2 == 0 else (1, 0)):
            enforce_budget(deadline, config['maximum_resident_gib'])
            start = time.perf_counter()
            value_gradient(kernels[target], torch.tensor(points[index]))
            elapsed[row, target] = time.perf_counter()-start
    sites = parameters(model, points[indices])
    core, _ = event_integrals(model, sites, torch.tensor(clouds[0]))
    kernel_error = float(abs(core-values[indices, 0]).sum(-1).max())
    path = output / 'preparation.npz'
    np.savez_compressed(path, coordinate=points[:len(values)], log_integrals=values,
        cloud_names=np.asarray(['qmc_512', 'qmc_2048', 'iid_8192', 'iid_2048']),
        timing_indices=indices, gradient_seconds=elapsed, core_log_integrals=core)
    paths.append(path)
    resources = runtime_metadata()
    process_seconds = time.perf_counter()-PROCESS_STARTED
    passed = (status == 'complete' and len(values) == len(points)
        and kernel_error <= config['maximum_kernel_log_likelihood_difference']
        and process_seconds <= config['preparation_maximum_seconds']
        and resources['peak_working_set_bytes'] <= config['maximum_resident_gib']*1024**3)
    report = {'configuration': config, 'status': status, 'states_checked': len(values),
        'scope': 'Original initialization and stopped warmup; not posterior draws. No recovery or test prediction.',
        'input_fields': ['train_catalog:observed,sigma,data', 'train_verification:data', 'previous_pilot:initial_coordinate,warmup_coordinate'],
        'source_sha256': sources, 'product_sha256': {p.name: sha256(p) for p in paths},
        'draw_product_sha256': sha256(path), 'jit_checks': checks,
        'maximum_numpy_core_event_log_likelihood_difference': kernel_error,
        'warmup_log_likelihood_difference_sd': difference.std(0, ddof=1).tolist(),
        'warmup_maximum_absolute_log_likelihood_difference': np.max(abs(difference), axis=0).tolist(),
        'median_gradient_seconds': np.median(elapsed, axis=0).tolist(),
        'median_paired_gradient_speed_ratio': float(np.median(elapsed[:, 1]/elapsed[:, 0])),
        'process_elapsed_seconds': process_seconds, 'runtime': resources,
        'ready_for_bounded_pilot': bool(passed), 'eligible_for_scientific_comparisons': False}
    write_json(output / 'preparation.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'states_checked',
        'warmup_log_likelihood_difference_sd', 'median_paired_gradient_speed_ratio', 'ready_for_bounded_pilot')}), flush=True)


def pilot(config, protocol, directory, output, model, sources):
    if (output / 'pilot.npz').exists() or (output / 'pilot.json').exists():
        raise FileExistsError('Completed or partial pilot is retained')
    prepared = verify_product(output / 'preparation.npz')
    verified_sources(prepared)
    for name, expected in prepared['product_sha256'].items():
        if sha256(output / name) != expected:
            raise ValueError(f'Prepared product changed: {name}')
    if not prepared['ready_for_bounded_pilot']:
        raise ValueError('Preparation did not pass')
    paths = [output / 'preparation.json', *[output / name for name in prepared['product_sha256']]]
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    clouds = [np.load(output / f'{name}_events.npz')['data'] for name in ('primary', 'verification')]
    clouds.append(np.load(directory / 'train_verification.npz')['data'])
    data, fine = torch.tensor(clouds[0]), torch.tensor(clouds[1])
    initial = torch.tensor(np.load(ROOT / config['previous_pilot'])['initial_coordinate'])
    potential = TransportPotential(model, data, 'raw')
    deadline = PROCESS_STARTED+config['maximum_seconds']
    report, arrays = run_bounded(potential, initial, config, deadline-config['verification_reserve_seconds'])
    indices = np.unique(np.linspace(0, len(arrays['coordinate'])-1,
        min(config['derivative_check_draws'], len(arrays['coordinate'])), dtype=int))
    points = np.vstack([initial.numpy(), arrays['coordinate'][indices]])
    verification, checked = verify_target(model, potential, data, fine, points, config, deadline)
    arrays.update(checked)
    values, status = integrate_states(model, arrays['coordinate'], clouds, config, deadline)
    if len(values):
        summaries, weights = integration_summary(values, protocol)
    else:
        summaries, weights = [], np.empty((2, 0))
    arrays.update(all_log_integrals=values, correction_weights=weights,
                  cloud_names=np.asarray(['qmc_512', 'qmc_2048', 'iid_8192']))
    integration_ok = (status == 'complete' and len(values) == config['retained']
                      and len(summaries) == 2 and all(item['passed'] for item in summaries))
    if len(arrays['coordinate']):
        names, features = feature_draws(parameters(model, arrays['coordinate']), power=config['feature_transport_power'])
        arrays.update(features=features, feature_names=np.asarray(names))
    path = output / 'pilot.npz'
    np.savez_compressed(path, **arrays)
    elapsed, runtime = time.perf_counter()-PROCESS_STARTED, runtime_metadata()
    resources_ok = (elapsed <= config['maximum_seconds']
                    and runtime['peak_working_set_bytes'] <= config['maximum_resident_gib']*1024**3)
    ready = (report['status'] == 'complete' and report['divergences'] == [] and integration_ok
             and resources_ok and verification['derivatives_passed'] and verification['precision_passed'])
    report.update(configuration=config, source_sha256=sources, draw_product_sha256=sha256(path),
        input_fields=['primary_events:data', 'verification_events:data', 'train_verification:data', 'previous_pilot:initial_coordinate'],
        coordinate_basis='unchanged raw latent variables; float64 bounds', verification=verification,
        integration_status=status, integration_draws_checked=len(values),
        integration_targets=['qmc_2048', 'iid_8192'], integration_summaries=summaries,
        correction_scope='All retained single-chain pilot states; reff estimated from each log ratio. Preliminary only.',
        event_integration_passed=bool(integration_ok), resources_passed=bool(resources_ok),
        runtime=runtime, process_elapsed_seconds=elapsed,
        ready_for_production_sampling=bool(ready), eligible_for_scientific_comparisons=False)
    write_json(output / 'pilot.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'warmup_completed', 'samples_completed',
        'divergences', 'integration_summaries', 'process_elapsed_seconds', 'ready_for_production_sampling')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'pilot'], required=True)
    stage = parser.parse_args().stage
    config = json.loads(CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    catalog_config = json.loads((ROOT / config['catalog_config']).read_text())
    directory, output = ROOT / catalog_config['output'], ROOT / config['output']
    verify_manifest(directory)
    previous = verify_product(ROOT / config['previous_pilot'])
    verified_sources(previous)
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    sources = dict(previous['source_sha256'])
    paths = [CONFIG, Path(__file__), ROOT / config['previous_pilot'],
        (ROOT / config['previous_pilot']).with_suffix('.json'), ROOT / 'scripts/selected_event_qmc.py',
        ROOT / 'scripts/external_selected_evaluation.py', ROOT / 'scripts/run_lisa_production.py']
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    output.mkdir(parents=True, exist_ok=True)
    if stage == 'prepare':
        prepare(config, directory, output, model, sources)
    else:
        pilot(config, protocol, directory, output, model, sources)


if __name__ == '__main__':
    main()
