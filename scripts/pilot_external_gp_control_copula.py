"""Bounded training-only copula pilot for the fixed GP-family catalog."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import time
PROCESS_STARTED = time.perf_counter()

import json
from pathlib import Path

import numpy as np
import psutil
import pyro
from pyro.infer import MCMC, NUTS
import torch

from benchmark_selected_mixture import feature_draws, runtime_metadata, selected_model
from binned_gp_benchmark import FEATURE_NAMES
from build_external_gp_control import ROOT, verify_manifest
from check_selected_mixture_integrals import event_integrals
from correct_selected_posterior import latent_sites, sha256
from initialize_population_guide import initialize_from_events
from run_lisa_production import write_json
from sample_lisa_transport import (TransportPotential, check_precision, compile_potential,
    value_gradient)

CONFIG = ROOT / 'configs/external_gp_control_copula_pilot.json'


class BudgetExceeded(RuntimeError):
    pass


def enforce_budget(deadline, maximum_resident_gib):
    if time.perf_counter() >= deadline:
        raise BudgetExceeded('time_limit')
    memory = psutil.Process().memory_info()
    if getattr(memory, 'peak_wset', memory.rss) > maximum_resident_gib*1024**3:
        raise BudgetExceeded('memory_limit')


def run_bounded(potential, initial, config, deadline):
    started = time.perf_counter()
    calls, warmup, samples, progress = 0, [], [], []
    status, error, checks, kernel, sampler = 'complete', None, [], None, None
    try:
        enforce_budget(deadline, config['maximum_resident_gib'])
        traced, checks = compile_potential(potential, initial, config['jit_absolute_tolerance'])
        def counted(values):
            nonlocal calls
            calls += 1
            enforce_budget(deadline, config['maximum_resident_gib'])
            return traced(values['coordinate'])
        def record(kernel, state, stage, iteration):
            (warmup if stage == 'Warmup' else samples).append(state['coordinate'].detach().clone())
            progress.append([int(stage != 'Warmup'), iteration, calls, time.perf_counter()-started, float(kernel.step_size)])
            if (iteration+1) % 20 == 0:
                print('Copula', stage, iteration+1, 'seconds', round(progress[-1][3], 1), flush=True)
        pyro.set_rng_seed(config['seed'])
        kernel = NUTS(potential_fn=counted, full_mass=config['full_mass'],
                      target_accept_prob=config['target_acceptance'], max_tree_depth=config['maximum_tree_depth'])
        sampler = MCMC(kernel, initial_params={'coordinate': initial}, num_chains=1,
                       warmup_steps=config['warmup'], num_samples=config['retained'],
                       disable_progbar=True, hook_fn=record)
        sampler.run()
    except BudgetExceeded as exc:
        status = str(exc)
    except (RuntimeError, ValueError) as exc:
        status, error = 'numerical_failure', str(exc)
    diagnostics = sampler.diagnostics() if status == 'complete' else None
    report = {'status': status, 'error': error, 'warmup_completed': len(warmup), 'samples_completed': len(samples),
        'sampling_and_compilation_seconds': time.perf_counter()-started, 'potential_evaluations': calls,
        'jit_checks': checks, 'divergences': diagnostics['divergences']['chain 0'] if diagnostics else None,
        'acceptance_rate': diagnostics['acceptance rate']['chain 0'] if diagnostics else None,
        'step_size': float(kernel.step_size) if kernel else None}
    arrays = {'initial_coordinate': initial.numpy(),
        'coordinate': torch.stack(samples).numpy() if samples else np.empty((0, len(initial))),
        'warmup_coordinate': torch.stack(warmup).numpy() if warmup else np.empty((0, len(initial))),
        'progress': np.asarray(progress).reshape(-1, 5)}
    return report, arrays


def derivative_directions(dimension):
    if dimension != 20:
        raise ValueError('This check is for the retained two-component copula layout')
    result = np.zeros((5, dimension))
    result[0, :2] = [1., -1.]
    result[1, 2:14], result[2, 2:14] = np.sin(np.arange(12)+1), np.cos(np.arange(12)+1)
    result[3, 14:], result[4, 14:] = np.sin(np.arange(6)+1), np.cos(np.arange(6)+1)
    return torch.tensor(result/np.linalg.norm(result, axis=1, keepdims=True))


def verify_target(model, potential, data, fine, points, config, deadline):
    directions = derivative_directions(points.shape[1])
    step = config['derivative_check_step']
    automatic, numeric, differences, precision = [], [], [], []
    status = 'complete'
    try:
        for point in torch.tensor(points):
            enforce_budget(deadline, config['maximum_resident_gib'])
            _, gradient = value_gradient(potential, point)
            automatic.append((directions @ gradient).numpy())
            row = []
            with torch.no_grad():
                for direction in directions:
                    enforce_budget(deadline, config['maximum_resident_gib'])
                    full = (potential(point+step*direction)-potential(point-step*direction))/(2*step)
                    half = (potential(point+step/2*direction)-potential(point-step/2*direction))/step
                    row.append(float((4*half-full)/3))
            numeric.append(row)
            checked = check_precision(model, data, point[None])
            precision.extend(checked['log_target_differences'])
            with torch.no_grad():
                sites = {name: value.numpy() for name, value in latent_sites(model, point[None]).items()}
                primary, _ = event_integrals(model, sites, data)
                refined, _ = event_integrals(model, sites, fine)
            differences.append(float((refined-primary).sum()))
    except BudgetExceeded as exc:
        status = str(exc)
    completed = len(differences)
    arrays = {'checked_coordinate': points[:completed], 'event_log_likelihood_difference': np.asarray(differences),
        'precision_log_target_difference': np.asarray(precision[:completed]),
        'derivative_automatic': np.asarray(automatic[:completed]).reshape(-1, 5),
        'derivative_numeric': np.asarray(numeric[:completed]).reshape(-1, 5)}
    derivative_error = float(abs(arrays['derivative_automatic']-arrays['derivative_numeric']).max()) if completed else None
    precision_error = float(np.max(abs(arrays['precision_log_target_difference']))) if completed else None
    sd = float(np.std(differences[1:], ddof=1)) if completed > 2 else None
    report = {'status': status, 'draws_checked': completed, 'draws_requested': len(points),
        'maximum_directional_derivative_error': derivative_error,
        'maximum_precision_log_target_difference': precision_error,
        'event_log_likelihood_difference_sd': sd,
        'event_sd_scope': 'Retained pilot draws only; the first checked coordinate is initialization.',
        'derivatives_passed': completed == len(points) and derivative_error < config['maximum_directional_derivative_error'],
        'precision_passed': completed == len(points) and precision_error < config['maximum_precision_log_target_difference']}
    return report, arrays


def main():
    config = json.loads(CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    catalog_config = json.loads((ROOT / config['catalog_config']).read_text())
    directory, output = ROOT / catalog_config['output'], ROOT / config['output']
    manifest = verify_manifest(directory)
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Completed or partial pilot is retained')
    data = torch.tensor(np.load(directory / 'train_catalog.npz')['data'])
    fine = torch.tensor(np.load(directory / 'train_verification.npz')['data'])
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    initialization = initialize_from_events(model, data, config['initialization_seed'])
    generator = torch.Generator().manual_seed(config['initialization_seed']+1)
    initial = torch.tensor(initialization['initial_latent_location']) + config['initial_jitter']*torch.randn(model.latent_dim, generator=generator)
    potential = TransportPotential(model, data, 'raw')
    deadline = PROCESS_STARTED+config['maximum_seconds']
    report, arrays = run_bounded(potential, initial, config, deadline-config['verification_reserve_seconds'])
    indices = np.unique(np.linspace(0, len(arrays['coordinate'])-1,
        min(config['derivative_check_draws'], len(arrays['coordinate'])), dtype=int))
    points = np.vstack([arrays['initial_coordinate'], arrays['coordinate'][indices]])
    verification, checked = verify_target(model, potential, data, fine, points, config, deadline)
    arrays.update(checked)
    if len(arrays['coordinate']):
        with torch.no_grad():
            sites = {name: value.numpy() for name, value in latent_sites(model, torch.tensor(arrays['coordinate'])).items()}
        names, features = feature_draws(sites, power=config['feature_transport_power'])
    else:
        names, features = FEATURE_NAMES, np.empty((0, len(FEATURE_NAMES)))
    arrays.update(features=features, feature_names=np.asarray(names))
    path = output / 'pilot.npz'
    np.savez_compressed(path, **arrays)
    sources = [CONFIG, ROOT / config['protocol'], ROOT / config['catalog_config'], directory / 'manifest.json',
        directory / 'train_catalog.npz', directory / 'train_verification.npz', Path(__file__),
        ROOT / 'scripts/build_external_gp_control.py', ROOT / 'scripts/benchmark_selected_mixture.py',
        ROOT / 'scripts/validate_population_morphologies.py', ROOT / 'scripts/initialize_population_guide.py',
        ROOT / 'scripts/sample_lisa_transport.py', ROOT / 'scripts/correct_selected_posterior.py',
        ROOT / 'scripts/check_selected_mixture_integrals.py', ROOT / 'src/gwpopulation/core.py']
    elapsed = time.perf_counter()-PROCESS_STARTED
    memory = runtime_metadata()
    complete = report['status'] == 'complete' and report['divergences'] == []
    integration_ok = (verification['event_log_likelihood_difference_sd'] is not None
        and verification['event_log_likelihood_difference_sd'] <= protocol['evaluation']['maximum_log_likelihood_difference_sd'])
    resources_ok = elapsed <= config['maximum_seconds'] and memory['peak_working_set_bytes'] <= config['maximum_resident_gib']*1024**3
    report.update(configuration=config, initialization=initialization, coordinate_basis='unchanged raw latent variables; float64 bounds',
        input_fields=['train_catalog:data', 'train_verification:data'], verification=verification,
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}, draw_product_sha256=sha256(path),
        process_elapsed_seconds=elapsed, runtime=memory, event_integration_passed=integration_ok, resources_passed=resources_ok,
        ready_for_production_sampling=bool(complete and integration_ok and resources_ok and verification['derivatives_passed'] and verification['precision_passed']),
        eligible_for_scientific_comparisons=False)
    write_json(path.with_suffix('.json'), report)
    print(json.dumps({key: report[key] for key in ('status', 'warmup_completed', 'samples_completed',
        'divergences', 'verification', 'process_elapsed_seconds', 'ready_for_production_sampling')}), flush=True)


if __name__ == '__main__':
    main()
