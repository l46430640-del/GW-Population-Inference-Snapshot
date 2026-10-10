"""Bounded production posteriors for the fixed external-family control."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import time
PROCESS_STARTED = time.perf_counter()

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path

import arviz as az
import numpy as np
import pyro
from pyro.ops import stats
import torch

from benchmark_selected_mixture import runtime_metadata, selected_model
from binned_gp_benchmark import FEATURE_NAMES, controlled_log_event_weights, controlled_selection
from build_external_gp_control import ROOT, verify_manifest
from correct_selected_posterior import latent_sites, sha256, swap_components
from external_selected_evaluation import copula_cells, event_log_integrals, normal_points
from initialize_population_guide import initialize_from_events
from pilot_external_gp_control_copula import enforce_budget, run_bounded, verify_target
from pilot_external_gp_control_qmc import verified_sources
from refine_lisa_posterior import corrected_statistics, ratio_weights
from run_external_gp_bounded_diagnostics import cumulative_minimum
from run_external_gp_pilot import run_pilot
from run_external_gp_selected import (InitializedGPPotential, checked_reports, density_functionals,
    production_decision, summarize as summarize_gp)
from run_lisa_production import mixing_summary, verify_product, write_json
from run_symmetric_gp_pilot import check_coordinates
from sample_lisa_transport import TransportPotential
from symmetric_gp_benchmark import SymmetricBinnedGPPotential

CONFIG = ROOT / 'configs/external_control_production.json'


def effective_protocol(config, original):
    protocol = copy.deepcopy(original)
    amendment = config['resource_amendment']
    if (amendment['prior_study_budget_charge_seconds'] + amendment['maximum_seconds_control_production']
            > original['resources']['maximum_total_cpu_hours']*3600):
        raise ValueError('New allocation exceeds the study budget')
    protocol['resources']['maximum_seconds_per_fit'] = amendment['maximum_seconds_per_fit']
    return protocol


def install_adapter(config):
    adapter = config['diagnostic_adapter']
    original = stats._cummin
    if (sha256(ROOT / adapter['source']) != adapter['sha256']
            or pyro.__version__ != adapter['pyro_version']
            or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != adapter['original_cummin_sha256']):
        raise ValueError('Pyro diagnostic adapter differs from the validated version')
    stats._cummin = cumulative_minimum
    return original


def freeze_target(output, signature):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'target.json'
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError('Existing target differs from requested sources')
    else:
        write_json(path, signature)
    return sha256(path)


def previous_cost(output):
    total = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('*/chain_*.json'))
    summary = output / 'copula/summary.json'
    if summary.exists():
        total += json.loads(summary.read_text())['verification_seconds']
    jobs = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('job_cost_*.json'))
    return max(total, jobs)


def initial_copula_points(model, catalog, config, previous):
    points, records = [], []
    for chain, seed in enumerate(config['copula_initialization_seeds']):
        initialization = initialize_from_events(model, catalog, seed)
        generator = torch.Generator().manual_seed(seed+1)
        point = torch.tensor(initialization['initial_latent_location']) + config['copula_initial_jitter']*torch.randn(model.latent_dim, generator=generator)
        if config['swap_initial_labels'][chain]:
            point = swap_components(model, point)
        points.append(point)
        records.append(initialization)
    result = torch.stack(points)
    if not np.array_equal(result[0].numpy(), previous):
        raise ValueError('First start differs from the validated original initialization')
    if len(np.unique(result.numpy(), axis=0)) != 4:
        raise ValueError('Four distinct initial coordinates required')
    return result, records


def copula_functionals(model, points, power, seed):
    with torch.no_grad():
        sites = {name: value.numpy() for name, value in latent_sites(model, torch.tensor(points)).items()}
    normal = normal_points(power, seed)
    values = []
    for i in range(len(points)):
        result = copula_cells({name: value[i] for name, value in sites.items()}, normal, 2)
        values.append(np.r_[result['features'], result['additional']])
    extra_names = density_functionals(torch.zeros(8), 2, [.25, .5, .75])[0]
    return FEATURE_NAMES+extra_names, np.asarray(values).reshape(len(points), 29), sites


def evaluate_clouds(sites, clouds, indices, deadline, memory):
    values = []
    for j, i in enumerate(indices):
        row = []
        for cloud in clouds:
            enforce_budget(deadline, memory)
            row.append(event_log_integrals({name: value[i] for name, value in sites.items()}, cloud))
        values.append(row)
        if (j+1) % 400 == 0:
            print('Control likelihood verification', j+1, '/', len(indices), flush=True)
    return np.asarray(values)


def copula_summary(config, protocol, output, model, clouds, target_hash, deadline):
    started = time.perf_counter()
    if (output / 'posterior.npz').exists() or (output / 'summary.json').exists():
        raise FileExistsError('Completed or partial numerical summary is retained')
    reports = checked_reports(output, target_hash, protocol)
    sampling = protocol['sampling']
    complete = (len(reports) == 4 and all(r['status'] == 'complete' and r['samples_completed'] == sampling['retained_per_chain']
        and r['warmup_completed'] == sampling['warmup'] for r in reports))
    decision = {'complete': complete, 'divergences_passed': complete and all(r['divergences'] == [] for r in reports),
        'mixing_passed': False, 'numerical_passed': False, 'correction_precision_passed': False, 'resources_passed': False}
    summary = {'configuration': config, 'target_product_sha256': target_hash,
        'chain_reports_sha256': {f"chain_{r['seed']}.json": sha256(output / f"chain_{r['seed']}.json") for r in reports},
        'decision': decision, 'comparison_results_available': False, 'mixing': None}
    if complete:
        products = [dict(np.load(output / f"chain_{r['seed']}.npz")) for r in reports]
        names = products[0]['diagnostic_names'].tolist()
        if any(p['diagnostic_names'].tolist() != names for p in products):
            raise ValueError('Diagnostic functionals differ across chains')
        values = np.stack([p['diagnostic_values'] for p in products])
        mixing = mixing_summary(names, values)
        decision['mixing_passed'] = (mixing['maximum_rhat'] <= sampling['maximum_functional_rhat']
            and mixing['minimum_bulk_ess'] >= sampling['minimum_functional_bulk_ess']
            and mixing['minimum_tail_ess'] >= sampling['minimum_functional_tail_ess'])
        points = np.concatenate([p['coordinate'] for p in products])
        with torch.no_grad():
            sites = {name: value.numpy() for name, value in latent_sites(model, torch.tensor(points)).items()}
        log_event = evaluate_clouds(sites, clouds[:2], range(len(points)), deadline, protocol['resources']['maximum_resident_gib'])
        difference = (log_event[:, 1]-log_event[:, 0]).sum(-1).reshape(4, -1)
        reff = float(np.clip(az.ess(difference, method='mean', relative=True), 1/difference.size, 1))
        weights, correction = ratio_weights(difference, reff)
        count = sampling['retained_per_chain']
        groups = {f'chain_{c}': c*count+np.linspace(0, count-1, config['verification_draws_per_chain'], dtype=int) for c in range(4)}
        extremes = config['verification_extreme_draws']
        alpha = values[:, :, 6].ravel()
        groups['efficiency_extremes'] = np.r_[np.argsort(alpha)[:extremes], np.argsort(alpha)[-extremes:]]
        groups['largest_weights'] = np.argsort(weights.ravel())[-extremes:]
        indices = np.unique(np.concatenate(list(groups.values())))
        fine = evaluate_clouds(sites, [clouds[2]], indices, deadline, protocol['resources']['maximum_resident_gib'])[:, 0]
        fine_difference = (fine-log_event[indices, 1]).sum(-1)
        verification = {name: float(fine_difference[np.searchsorted(indices, group)].std(ddof=1)) for name, group in groups.items()}
        chain_difference = difference.std(1, ddof=1)
        correction_ok = (correction['pareto_k'] < sampling['maximum_pareto_k_for_reuse']
            and correction['weight_ess_fraction'] >= sampling['minimum_reuse_weight_ess_fraction'])
        numerical_ok = (max(verification.values()) <= protocol['evaluation']['maximum_log_likelihood_difference_sd']
            and max(chain_difference) <= protocol['evaluation']['maximum_log_likelihood_difference_sd']
            and all(r['verification']['derivatives_passed'] and r['verification']['precision_passed'] for r in reports))
        uniform = np.concatenate([groups[f'chain_{c}'] for c in range(4)])
        _, feature_fine, _ = copula_functionals(model, points[uniform], config['fine_feature_power'], config['feature_seed'])
        feature_change = np.max(abs(feature_fine-values[:, :, :29].reshape(-1, 29)[uniform]), axis=0)/values[:, :, :29].reshape(-1, 29).std(0)
        feature_ok = np.max(feature_change) <= config['maximum_feature_change_in_posterior_sd']
        statistics = corrected_statistics(values[:, :, :29], names[:29], weights, 29, config)
        precise = (statistics['maximum_influence_rhat'] <= sampling['maximum_functional_rhat']
            and min(statistics['minimum_mean_precision_equivalent_draws'], statistics['minimum_cdf_precision_equivalent_draws'])
                >= config['minimum_corrected_precision_equivalent_draws'])
        path = output / 'posterior.npz'
        np.savez_compressed(path, coordinate=points.reshape(4, count, -1), weights=weights, log_ratio=difference,
            feature_names=np.asarray(names[:29]), features=values[:, :, :29], log_integrals=log_event,
            verification_indices=indices, verification_log_integrals=fine,
            fine_feature_indices=uniform, fine_features=feature_fine,
            **{f'verification_group_{name}': group for name, group in groups.items()})
        decision.update(numerical_passed=bool(numerical_ok and correction_ok and feature_ok), correction_precision_passed=bool(precise))
        summary.update(mixing=mixing, correction=correction, corrected_statistics=statistics,
            coarse_to_fine_log_likelihood_difference_sd_by_chain=chain_difference.tolist(),
            independent_event_log_likelihood_difference_sd=verification,
            feature_change_in_posterior_sd=feature_change.tolist(), posterior_product_sha256=sha256(path))
    seconds = time.perf_counter()-started
    total = sum(r['process_elapsed_seconds'] for r in reports)+seconds
    peak = runtime_metadata()
    decision['resources_passed'] = (total <= protocol['resources']['maximum_seconds_per_fit']
        and peak['peak_working_set_bytes'] <= protocol['resources']['maximum_resident_gib']*1024**3)
    decision['posterior_accepted'] = all(decision.values())
    summary.update(verification_seconds=seconds, recorded_process_elapsed_seconds=total, runtime=peak,
        sampling_and_compilation_seconds=sum(r['sampling_and_compilation_seconds'] for r in reports))
    write_json(output / 'summary.json', summary)
    print(json.dumps({'model': 'copula', 'decision': decision, 'mixing': None if summary['mixing'] is None else
        {k: summary['mixing'][k] for k in ('maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess')}}), flush=True)


def run_model(label, config, protocol, directory, root, sources, global_deadline):
    fit_started = time.perf_counter()
    output = root / label
    if (output / 'summary.json').exists():
        return
    copula = label == 'copula'
    pilot_config = json.loads((ROOT / config['copula_pilot_config' if copula else 'gp_pilot_config']).read_text())
    pilot_dir = ROOT / pilot_config['output']
    verified = verify_product(pilot_dir / ('pilot.npz' if copula else f"bins_{label.split('_')[1]}.npz"))
    verified_sources(verified)
    if not verified['ready_for_production_sampling']:
        raise ValueError('Pilot is not ready for production')
    source = dict(sources)
    for path in [pilot_dir / ('pilot.json' if copula else f"bins_{label.split('_')[1]}.json")]:
        source[path.relative_to(ROOT).as_posix()] = sha256(path)
    catalog = np.load(directory / 'train_catalog.npz')
    signature = {'configuration': config, 'effective_protocol': protocol, 'source_sha256': source,
        'experiment': 'selected_gp_truth', 'model': label, 'input_fields': ['training observations and uncertainty only']}
    if copula:
        model = selected_model()
        model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
        old = np.load(ROOT / pilot_config['previous_pilot'])
        initial, records = initial_copula_points(model, torch.tensor(catalog['data']), config, old['initial_coordinate'])
        clouds = [np.load(pilot_dir / f'{name}_events.npz')['data'] for name in ('primary', 'verification')]
        clouds.append(np.load(directory / 'train_verification.npz')['data'])
        prepared = verify_product(pilot_dir / 'preparation.npz')
        verified_sources(prepared)
        for name, digest in prepared['product_sha256'].items():
            if sha256(pilot_dir / name) != digest:
                raise ValueError('Prepared integration cloud changed')
            source[(pilot_dir / name).relative_to(ROOT).as_posix()] = digest
        signature.update(initial_coordinates=initial.numpy().tolist(), initial_provenance=records,
            cloud_sizes=[cloud.shape[1] for cloud in clouds])
    else:
        bins = int(label.split('_')[1])
        weights = controlled_log_event_weights(catalog['observed'], catalog['sigma'], bins, pilot_config['quadrature_order'])
        fine = controlled_log_event_weights(catalog['observed'], catalog['sigma'], bins, pilot_config['verification_quadrature_order'])
        error = float(np.max(abs(weights-fine)))
        if error > pilot_config['maximum_quadrature_log_weight_difference']:
            raise ValueError('GP event quadrature needs refinement')
        selection = controlled_selection(bins)
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'inputs.npz'
        if path.exists():
            saved = np.load(path)
            np.testing.assert_array_equal(saved['log_event_weights'], weights)
            np.testing.assert_array_equal(saved['selection_coefficients'], selection)
        else:
            np.savez_compressed(path, log_event_weights=weights, selection_coefficients=selection)
        signature.update(inputs_product_sha256=sha256(path), quadrature_maximum_log_weight_change=error,
            initial_amplitude_and_lengths=config['initial_amplitude_and_lengths'])
    target_hash = freeze_target(output, signature)
    previous = checked_reports(output, target_hash, protocol)
    old_cost = sum(r['process_elapsed_seconds'] for r in previous)
    deadline = min(global_deadline, fit_started+protocol['resources']['maximum_seconds_per_fit']-old_cost)
    reserve = config['copula_fit_verification_reserve_seconds'] if copula else config['chain_verification_reserve_seconds']
    accounted = fit_started
    for chain, seed in enumerate(protocol['sampling']['chain_seeds']):
        if seed in {r['seed'] for r in previous}:
            continue
        if time.perf_counter() >= deadline-reserve:
            break
        chain_config = dict(pilot_config, seed=seed, warmup=protocol['sampling']['warmup'],
            retained=protocol['sampling']['retained_per_chain'], target_acceptance=protocol['sampling']['target_acceptance'],
            maximum_tree_depth=protocol['sampling']['maximum_tree_depth'], maximum_seconds=deadline-reserve-time.perf_counter())
        pyro.clear_param_store()
        if copula:
            potential = TransportPotential(model, torch.tensor(clouds[0]), 'raw')
            report, arrays = run_bounded(potential, initial[chain], chain_config, deadline-reserve)
            points = arrays['coordinate']
            names, features, _ = copula_functionals(model, points, config['feature_power'], config['feature_seed'])
            indices = np.unique(np.linspace(0, len(points)-1, min(3, len(points)), dtype=int))
            checked_points = np.vstack([initial[chain].numpy(), points[indices]])
            verification, checks = verify_target(model, potential, torch.tensor(clouds[0]), torch.tensor(clouds[1]), checked_points, chain_config, deadline)
            with torch.no_grad():
                log_target = np.asarray([-float(potential(torch.tensor(point))) for point in points])
            arrays.update(checks)
            arrays.update(features=features, feature_names=np.asarray(names), diagnostic_names=np.asarray(names+['log_posterior']),
                diagnostic_values=np.column_stack([features, log_target]))
            report.update(verification=verification)
        else:
            potential = InitializedGPPotential(bins, weights, selection, protocol['binned_gp'], pilot_config['covariance_axis_jitter'])
            potential.set_initial_hyperparameters(config['initial_amplitude_and_lengths'][chain])
            reference = SymmetricBinnedGPPotential(bins, weights, selection, protocol['binned_gp'], pilot_config['verification_covariance_axis_jitter'])
            report, arrays = run_pilot(potential, chain_config)
            verification, refined = check_coordinates(potential, reference, arrays['coordinate'], pilot_config)
            arrays.update(refined)
            extra_names, extra_values = density_functionals(torch.zeros(bins**3), bins, config['marginal_cdf_points'])[0], []
            with torch.no_grad():
                for point in arrays['coordinate']:
                    extra_values.append(density_functionals(potential.field(torch.tensor(point)), bins, config['marginal_cdf_points'])[1])
            arrays.update(diagnostic_names=np.asarray(list(arrays['feature_names'])+extra_names),
                diagnostic_values=np.column_stack([arrays['features'], np.asarray(extra_values).reshape(-1, len(extra_names))]))
            report.update(coordinate_verification=verification, event_quadrature_passed=True,
                covariance_jitter_log_target_difference_sd=verification['jitter_difference_sd'])
        path = output / f'chain_{seed}.npz'
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError('Completed or partial chain is retained')
        np.savez_compressed(path, **arrays)
        report.update(seed=seed, target_product_sha256=target_hash, draw_product_sha256=sha256(path),
            process_elapsed_seconds=time.perf_counter()-accounted, runtime=runtime_metadata())
        write_json(path.with_suffix('.json'), report)
        print('Saved', label, seed, report['status'], round(report['process_elapsed_seconds'], 2), flush=True)
        accounted = time.perf_counter()
        del arrays
        if report['status'] != 'complete':
            break
    if copula:
        copula_summary(config, protocol, output, model, clouds, target_hash, deadline)
    else:
        summarize_gp(output, protocol, target_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['all', 'gp_6', 'gp_12', 'copula'], default='all')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    protocol = effective_protocol(config, json.loads((ROOT / config['protocol']).read_text()))
    directory = ROOT / json.loads((ROOT / config['catalog_config']).read_text())['output']
    verify_manifest(directory)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    root = ROOT / config['output']
    root.mkdir(parents=True, exist_ok=True)
    paths = [CONFIG, Path(__file__), ROOT / config['protocol'], ROOT / config['catalog_config'],
        ROOT / config['gp_pilot_config'], ROOT / config['copula_pilot_config'],
        directory / 'manifest.json', directory / 'train_catalog.npz', directory / 'train_verification.npz']
    scripts = ['run_external_gp_selected', 'run_external_gp_pilot', 'run_symmetric_gp_pilot', 'binned_gp_benchmark',
        'symmetric_gp_benchmark', 'run_external_gp_bounded_diagnostics', 'pilot_external_gp_control_copula',
        'pilot_external_gp_control_qmc', 'external_selected_evaluation', 'refine_lisa_posterior',
        'initialize_population_guide', 'sample_lisa_transport', 'benchmark_selected_mixture', 'run_lisa_production',
        'correct_selected_posterior', 'check_selected_mixture_integrals']
    paths += [ROOT / f'scripts/{name}.py' for name in scripts]
    paths.append(ROOT / 'src/gwpopulation/core.py')
    sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    deadline = PROCESS_STARTED+config['resource_amendment']['maximum_seconds_control_production']-previous_cost(root)
    original = install_adapter(config)
    outcome, error = 'complete', None
    try:
        for label in (['gp_6', 'gp_12', 'copula'] if args.model == 'all' else [args.model]):
            enforce_budget(deadline, protocol['resources']['maximum_resident_gib'])
            run_model(label, config, protocol, directory, root, sources, deadline)
    except BaseException as exc:
        outcome, error = 'stopped', str(exc)
        raise
    finally:
        stats._cummin = original
        path = root / f'job_cost_{len(list(root.glob("job_cost_*.json"))):03d}.json'
        write_json(path, {'status': outcome, 'error': error, 'requested_models': args.model,
            'process_elapsed_seconds': time.perf_counter()-PROCESS_STARTED,
            'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})


if __name__ == '__main__':
    main()
