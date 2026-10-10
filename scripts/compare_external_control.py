"""Recovery and held-out prediction for the fixed GP-family population."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import time
PROCESS_STARTED = time.perf_counter()

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from benchmark_selected_mixture import runtime_metadata, selected_model
from build_external_gp_control import ROOT, event_cloud, verify_manifest
from check_external_copula_selected import params_at, selected_indices
from compare_external_selected import (difference_precision, gp_product,
    posterior_functionals, prediction_diagnostics, recovery_gradients)
from compare_lisa_posterior import cached, moment_check
from correct_selected_posterior import latent_sites, sha256
from external_selected_evaluation import (coarsen, copula_cells, event_log_integrals,
    expand_gp_cells, normal_points)
from pilot_external_gp_control_copula import enforce_budget
from run_lisa_production import write_json
from selected_event_qmc import selected_event_cloud
from summarize_selected_mixture import paired_scores

CONFIG = ROOT / 'configs/external_control_comparison.json'


def accepted_inputs(config):
    registry_path = ROOT / config['posterior_registry']
    registry = json.loads(registry_path.read_text())
    if not registry['posteriors_ready_for_comparison'] or registry['scientific_scores_examined']:
        raise ValueError('The checked, unevaluated control ensembles are required')
    sources = dict(registry['source_sha256'])
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Posterior source changed: {name}')
    catalog = ROOT / config['catalog']
    manifest = verify_manifest(catalog)
    if manifest['configuration']['experiment'] != config['experiment']:
        raise ValueError('Catalog and comparison experiments differ')
    paths = [CONFIG, Path(__file__), registry_path, ROOT / config['protocol'],
        ROOT / config['partition_precedent'], catalog / 'manifest.json']
    paths.extend(catalog / name for name in manifest['product_sha256'])
    scripts = ['compare_external_selected', 'compare_lisa_posterior',
        'check_external_copula_selected', 'summarize_selected_mixture',
        'predict_corrected_posterior', 'selected_event_qmc', 'build_external_gp_control']
    paths.extend(ROOT / f'scripts/{name}.py' for name in scripts)
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    directory = ROOT / registry['models']['copula']['directory']
    posterior = dict(np.load(directory / 'posterior.npz'))
    shape = posterior['coordinate'].shape[:2]
    model = selected_model()
    model.min_vals, model.max_vals = model.min_vals.double(), model.max_vals.double()
    with torch.no_grad():
        samples = {name: value.numpy() for name, value in latent_sites(model,
            torch.tensor(posterior['coordinate'].reshape(-1, model.latent_dim))).items()}
    gp = {}
    for key in ('6', '12'):
        directory = ROOT / registry['models']['gp_'+key]['directory']
        target = json.loads((directory / 'target.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        chains = []
        for seed in target['effective_protocol']['sampling']['chain_seeds']:
            with np.load(directory / f'chain_{seed}.npz') as data:
                chains.append({name: data[name] for name in ('coordinate', 'features')})
        gp[key] = (target, summary, chains, dict(np.load(directory / 'inputs.npz')))
    return registry, manifest, posterior, samples, shape, gp, sources


def truth_product(truth, features, bins, fine):
    probability = truth['probabilities']
    return {'intrinsic': expand_gp_cells(probability, bins, fine),
        'detected': expand_gp_cells(probability/float(truth['alpha']), bins, fine, True),
        'features': np.asarray(features), 'alpha': truth['alpha']}


def population_product(samples, weights, power, seed, bins, check):
    normal = normal_points(power, seed)
    means = {kind: np.zeros(bins**3) for kind in ('intrinsic', 'detected')}
    features, additional, alpha = [], [], []
    for i, weight in enumerate(weights):
        check()
        product = copula_cells(params_at(samples, i), normal, bins)
        for kind in means:
            means[kind] += weight*product[kind]
        features.append(product['features'])
        additional.append(product['additional'])
        alpha.append(product['transport_alpha'])
        if (i+1) % 400 == 0:
            print('Control population', power, i+1, '/', len(weights), flush=True)
    return dict(means, features=np.asarray(features), additional=np.asarray(additional),
        transport_alpha=np.asarray(alpha))


def population_influences(samples, config, bins, partitions, gradients, check):
    normal = normal_points(config['verification_population_transport_power'], config['verification_population_seed'])
    result = {label: [] for label in gradients}
    for i in range(len(samples['weights_logits'])):
        check()
        product = copula_cells(params_at(samples, i), normal, bins)
        for partition in partitions:
            for kind in ('intrinsic', 'detected'):
                label = f'{kind}_{partition}'
                result[label].append(coarsen(product[kind], bins, partition) @ gradients[label])
        if (i+1) % 400 == 0:
            print('Control distance influences', i+1, flush=True)
    return {label: np.asarray(values) for label, values in result.items()}


def evaluate_events(samples, data, indices, check):
    values = []
    for i, index in enumerate(indices):
        check()
        values.append(event_log_integrals(params_at(samples, index), data))
        if (i+1) % 400 == 0:
            print('Control predictive integration', i+1, '/', len(indices), flush=True)
    return {'indices': np.asarray(indices), 'log_integrals': np.asarray(values)}


def refined_subset_weights(posterior, indices):
    saved = posterior['verification_indices']
    positions = np.searchsorted(saved, indices)
    if np.any(positions >= len(saved)) or not np.array_equal(saved[positions], indices):
        raise ValueError('Every predictive verification draw needs its finer training integral')
    difference = (posterior['verification_log_integrals'][positions]
        - posterior['log_integrals'][indices, 1]).sum(-1)
    primary = posterior['weights'].ravel()[indices]
    primary = primary/primary.sum()
    refined = np.exp(np.log(primary)+difference-logsumexp(np.log(primary)+difference))
    return primary, refined


def comparison(config, protocol, output, check):
    registry, manifest, posterior, samples, shape, gp, sources = accepted_inputs(config)
    weights = posterior['weights'].ravel()
    partitions = [config['primary_cells_per_axis'], config['verification_cells_per_axis']]
    fine = max(partitions)
    def obtain(label, builder):
        check()
        product, _ = cached(output, label, sources, builder)
        check()
        return product
    catalog = ROOT / config['catalog']
    truth = obtain('truth', lambda: truth_product(dict(np.load(catalog / 'truth.npz')),
        manifest['truth_functionals'], manifest['experiment']['truth_grid_per_axis'], fine))
    coarse = obtain('copula_recovery_coarse', lambda: population_product(samples, weights,
        config['population_transport_power'], config['population_seed'], fine, check))
    population = obtain('copula_recovery', lambda: population_product(samples, weights,
        config['verification_population_transport_power'], config['verification_population_seed'], fine, check))
    distances, gradients = recovery_gradients(population, truth, fine, partitions)
    influences = obtain('copula_influences', lambda: population_influences(samples,
        config, fine, partitions, gradients, check))
    with np.load(catalog / 'test_catalog.npz') as data:
        test = {name: data[name] for name in ('observed', 'sigma')}
    products = {key: obtain('gp_'+key, lambda key=key: gp_product(config, protocol, gp,
        key, truth, partitions, test)) for key in gp}
    events = obtain('test_events', lambda: selected_event_cloud(test['observed'], test['sigma'],
        config['test_qmc_power'], config['test_qmc_seed']))
    primary = obtain('copula_prediction', lambda: evaluate_events(samples, events['data'],
        np.arange(len(weights)), check))
    del events
    indices = selected_indices(*shape, config['verification_draws_per_chain'])
    events_fine = obtain('test_events_verification', lambda: {'data': event_cloud(test['observed'],
        test['sigma'], config['verification_test_samples'], config['verification_test_seed'])})
    refined = obtain('copula_prediction_verification', lambda: evaluate_events(samples,
        events_fine['data'], indices, check))
    del events_fine
    alpha = population['features'][:, 6]
    prediction, precision = prediction_diagnostics(primary['log_integrals']-np.log(alpha[:, None]),
        weights, shape, config['batch_lengths'])
    subset_weight, fine_weight = refined_subset_weights(posterior, indices)
    subset_shape = (shape[0], len(indices)//shape[0])
    first = primary['log_integrals'][indices]-np.log(alpha[indices, None])
    second = refined['log_integrals']-np.log(alpha[indices, None])
    numerical_prediction = {
        'event_integration': difference_precision(first, second, subset_weight, subset_weight, subset_shape),
        'training_target_and_event_integration': difference_precision(first, second, subset_weight, fine_weight, subset_shape),
        'subset_minus_all_draws_total_score': float((logsumexp(first+np.log(subset_weight[:, None]), axis=0)-prediction).sum()),
        'verification_draws': len(indices), 'gp_quadrature_total_score_changes': {},
        'paired_total_score_changes': {}}
    feature_result = posterior_functionals(population['features'], weights, shape, truth['features'], config)
    joint = {label: {'distance': value, **{k: v for k, v in moment_check(influences[label],
        weights, shape, config['batch_lengths']).items() if k != 'mean'}} for label, value in distances.items()}
    coarse_distances, _ = recovery_gradients(coarse, truth, fine, partitions)
    numerical_recovery = {'posterior_transport_js_changes': {label: distances[label]-value for label, value in coarse_distances.items()},
        'partition_js_changes': {}, 'maximum_transport_relative_alpha_error': float(np.max(abs(population['transport_alpha']/alpha-1))),
        'maximum_covariance_refinement_change': float(np.max(abs(population['features'][:, 9:]-coarse['features'][:, 9:])))}
    result = {'copula': {'joint': joint, 'prediction_precision': precision, 'functionals': feature_result}, 'gp': {}}
    comparisons = {}
    score_change = numerical_prediction['training_target_and_event_integration']['total_score_change']
    for key, product in products.items():
        gp_shape = product['features'].shape[:2]
        uniform = np.full(np.prod(gp_shape), 1/np.prod(gp_shape))
        score, score_precision = prediction_diagnostics(product['log_integrals'], uniform, gp_shape, config['batch_lengths'])
        comparisons[key] = dict(paired_scores(prediction, score),
            posterior_mcse=float(np.hypot(precision['mcse'], score_precision['mcse'])))
        ds, _ = recovery_gradients(product, truth, fine, partitions)
        result['gp'][key] = {'joint': {label: {'distance': value, **{k: v for k, v in moment_check(
            product['influence_'+label], uniform, gp_shape, config['batch_lengths']).items() if k != 'mean'}} for label, value in ds.items()},
            'prediction_precision': score_precision,
            'functionals': posterior_functionals(product['features'], uniform, gp_shape, truth['features'], config)}
        change = float((product['verification_log_prediction']-score).sum())
        numerical_prediction['gp_quadrature_total_score_changes'][key] = change
        numerical_prediction['paired_total_score_changes'][key] = score_change-change
    rhats = []
    for key, row in [('copula', result['copula']), *result['gp'].items()]:
        numerical_recovery['partition_js_changes'][key] = {kind: row['joint'][f'{kind}_{fine}']['distance']
            - row['joint'][f'{kind}_{min(partitions)}']['distance'] for kind in ('intrinsic', 'detected')}
        rhats.extend(v['influence_rhat'] for v in row['joint'].values())
        rhats.extend([row['prediction_precision']['influence_rhat'], row['functionals']['maximum_influence_rhat']])
    changes = list(numerical_recovery['posterior_transport_js_changes'].values())
    changes.extend(value for group in numerical_recovery['partition_js_changes'].values() for value in group.values())
    evaluation = protocol['evaluation']
    decision = {'posteriors_accepted': registry['posteriors_ready_for_comparison'],
        'recovery_precision_passed': bool(max(map(abs, changes)) < evaluation['maximum_joint_distance_numerical_change']),
        'prediction_precision_passed': bool(max(map(abs, numerical_prediction['paired_total_score_changes'].values())) < evaluation['maximum_paired_score_numerical_change']),
        'derived_mixing_passed': bool(max(rhats) <= protocol['sampling']['maximum_functional_rhat'])}
    check()
    decision['resources_passed'] = True
    decision['eligible_for_scientific_comparison'] = all(decision.values())
    result.update(configuration=config, source_sha256=sources, experiment=config['experiment'],
        comparator_label=protocol['binned_gp']['label'], decision=decision,
        copula_minus_gp_test_scores=comparisons, numerical_recovery=numerical_recovery,
        numerical_prediction=numerical_prediction, maximum_derived_influence_rhat=max(rhats),
        posterior_costs=registry['cost'],
        evaluation_costs={p.stem: {'seconds': json.loads(p.read_text())['seconds'],
            'peak_bytes': json.loads(p.read_text())['runtime']['peak_working_set_bytes']}
            for p in output.glob('*.json') if 'seconds' in json.loads(p.read_text())},
        product_sha256={p.name: sha256(p) for p in output.glob('*.npz')},
        interpretation='The first fixed population drawn from the GP family, with 96 detected training and 192 independent test events. Report both resolutions and all outcomes. Predictive densities are averaged before taking logs; only common event-evidence constants cancel. Single-catalog intervals do not establish frequentist coverage. The LISA external comparison remains required.')
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': decision, 'scores': comparisons, 'joint': {key: row['joint']
        for key, row in [('copula', result['copula']), *result['gp'].items()]}}), flush=True)


def main():
    config = json.loads(CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    registry = json.loads((ROOT / config['posterior_registry']).read_text())
    if config['maximum_seconds'] > registry['cost']['remaining_study_seconds']:
        raise ValueError('Evaluation allocation exceeds the study budget')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed comparison is retained')
    old_cost = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('job_cost_*.json'))
    deadline = PROCESS_STARTED+config['maximum_seconds']-old_cost
    def check():
        enforce_budget(deadline, protocol['resources']['maximum_resident_gib'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    status, error = 'complete', None
    try:
        comparison(config, protocol, output, check)
        check()
    except BaseException as exc:
        status, error = 'stopped', str(exc)
        raise
    finally:
        seconds = time.perf_counter()-PROCESS_STARTED
        write_json(output / f'job_cost_{len(list(output.glob("job_cost_*.json"))):03d}.json',
            {'status': status, 'error': error, 'process_elapsed_seconds': seconds,
             'all_evaluation_attempts_seconds': old_cost+seconds,
             'total_study_charge_seconds': registry['cost']['total_study_charge_seconds']+old_cost+seconds,
             'remaining_study_seconds': registry['cost']['remaining_study_seconds']-old_cost-seconds,
             'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})


if __name__ == '__main__':
    main()
