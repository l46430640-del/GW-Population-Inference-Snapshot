"""Complete the fixed LISA comparison using common likelihood and rate prior."""
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

from benchmark_selected_mixture import runtime_metadata
from binned_gp_benchmark import posterior_log_event_weights
from build_lisa_analysis_assets import LISASNR
from check_external_copula_selected import params_at, selected_indices
from check_selected_mixture_integrals import event_integrals
from compare_external_selected import (difference_precision, posterior_functionals,
    prediction_diagnostics, recovery_gradients)
from compare_lisa_posterior import cached, moment_check
from correct_selected_posterior import latent_sites, sha256
from external_lisa_evaluation import (cell_integrals, gp_population, gp_projection,
    population_cells, population_projection)
from external_selected_evaluation import event_log_integrals, normal_points
from pilot_external_gp_control_copula import enforce_budget
from predict_corrected_posterior import weighted_prediction
from run_external_lisa_gp_batched import batched_event_posterior
from run_lisa_production import verify_product, write_json
from sample_lisa_transport import sampling_model
from symmetric_gp_benchmark import SymmetricBinnedGPPotential

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/external_lisa_comparison.json'


def accepted_inputs(config):
    registry_path = ROOT / config['posterior_registry']
    registry = json.loads(registry_path.read_text())
    if not registry['gp_posteriors_ready_for_comparison']:
        raise ValueError('Both accepted GP resolutions are required')
    source = ROOT / config['copula_source']
    accepted = json.loads((source / 'summary.json').read_text())
    if not accepted['decision']['eligible_for_scientific_summaries']:
        raise ValueError('The accepted copula posterior is required')
    old = ROOT / config['existing_comparison']
    old_summary = json.loads((old / 'summary.json').read_text())
    sources = {**registry['source_sha256'], **accepted['source_sha256'], **old_summary['source_sha256']}
    for directory, report in ((source, accepted), (old, old_summary)):
        sources.update({(directory / name).relative_to(ROOT).as_posix(): digest for name, digest in report['product_sha256'].items()})
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    paths = [CONFIG, Path(__file__), registry_path, source / 'summary.json', old / 'summary.json',
        ROOT / config['existing_configuration'], ROOT / config['partition_precedent']]
    paths.extend(ROOT / f'scripts/{name}.py' for name in ('external_lisa_evaluation', 'compare_external_selected',
        'external_selected_evaluation', 'compare_lisa_posterior', 'check_external_copula_selected',
        'check_selected_mixture_integrals', 'refine_lisa_posterior', 'summarize_selected_mixture',
        'validate_population_morphologies', 'predict_corrected_posterior'))
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    posterior = dict(np.load(source / 'posterior.npz'))
    weights = np.load(source / 'weights.npz')['weights']
    verification = dict(np.load(source / 'verification.npz'))
    old_config = json.loads((ROOT / config['existing_configuration']).read_text())
    original = json.loads((ROOT / old_config['baselines'][1] / 'summary.json').read_text())['configuration']
    model = sampling_model(original)
    with torch.no_grad():
        samples = {name: value.numpy() for name, value in latent_sites(model, torch.tensor(
            posterior['latent'].reshape(-1, posterior['latent'].shape[-1]))).items()}
    gp = {}
    for key, row in registry['models'].items():
        directory = ROOT / row['directory']
        target = json.loads((directory / 'target.json').read_text())
        chains = []
        for seed in target['effective_protocol']['sampling']['chain_seeds']:
            path = directory / f'chain_{seed}.npz'
            verify_product(path)
            with np.load(path) as data:
                chains.append({name: data[name] for name in ('coordinate', 'features', 'combined_log_likelihood_change')})
        gp[key] = (chains, dict(np.load(ROOT / row['inputs'])))
    return registry, accepted, posterior, weights, verification, model, samples, gp, sources


def evaluate_events(samples, data, indices, check):
    values = []
    for i, index in enumerate(indices):
        check()
        values.append(event_log_integrals(params_at(samples, index), data))
        if (i+1) % 200 == 0:
            print('LISA event integrals', i+1, '/', len(indices), flush=True)
    return {'indices': np.asarray(indices), 'log_integrals': np.asarray(values)}


def event_cloud(observations, count, seed, check):
    response, rng = LISASNR(), np.random.default_rng(seed)
    samples, ess, batch_ess = [], [], []
    for y, rho in zip(observations['measurement'], observations['rho_observed']):
        check()
        points, _, report = batched_event_posterior(y, rho, response, count, rng)
        samples.append(points)
        ess.append(report['proposal_ess'])
        batch_ess.append([r['proposal_ess'] for r in report['batches']])
    samples = np.asarray(samples)
    return {'data': np.concatenate([samples, np.ones((*samples.shape[:2], 1))], -1),
        'proposal_ess': np.asarray(ess), 'batch_proposal_ess': np.asarray(batch_ess)}


def population_product(samples, weights, power, seed, bins, check):
    normal, response = normal_points(power, seed), LISASNR()
    means = {name: np.zeros(bins**3) for name in ('intrinsic', 'detected')}
    features = []
    for i, weight in enumerate(weights):
        check()
        product = population_cells(params_at(samples, i), normal, bins, response)
        for name in means:
            means[name] += weight*product[name]
        features.append(product['features'])
        if (i+1) % 400 == 0:
            print('LISA recovery', power, i+1, flush=True)
    return dict(means, features=np.asarray(features))


def influence_product(samples, power, seed, gradients, check):
    normal, response = normal_points(power, seed), LISASNR()
    result = {name: [] for name in gradients}
    for i in range(len(samples['weights_logits'])):
        check()
        values = population_projection(params_at(samples, i), normal, response, gradients)
        for name in result:
            result[name].append(values[name])
    return {name: np.asarray(value) for name, value in result.items()}


@torch.no_grad()
def gp_product(chains, inputs, bins, protocol, truth, partitions, events, fine_events, check):
    fine = max(partitions)
    model = SymmetricBinnedGPPotential(bins, inputs['log_event_weights'], inputs['selection_coefficients'], protocol['binned_gp'])
    coordinates = np.concatenate([row['coordinate'] for row in chains])
    probability = np.empty((len(coordinates), bins**3))
    for i, point in enumerate(coordinates):
        check()
        probability[i] = torch.softmax(model.field(torch.tensor(point)), -1).numpy()
    population = gp_population(probability, bins, fine, truth['selection'])
    alpha = probability @ inputs['selection_coefficients']*bins**3
    fine_alpha = probability @ inputs['fine_selection_coefficients']*bins**3
    result = dict(population, features=np.stack([row['features'][:, :12] for row in chains]), probability=probability)
    _, gradients = recovery_gradients(result, truth, fine, partitions)
    for label, gradient in gradients.items():
        result['influence_'+label] = gp_projection(probability, bins, fine, truth['selection'], population['alpha'], label, gradient)
    def predict(data, efficiency):
        logw = posterior_log_event_weights(data, bins)
        integral = probability @ np.exp(logw).T
        if np.any(integral <= 0):
            raise ValueError('A GP predictive integral underflows')
        return np.log(integral)+3*np.log(bins)-np.log(efficiency[:, None])
    result['log_integrals'] = predict(events, alpha)
    result['fine_log_integrals'] = predict(fine_events, fine_alpha)
    difference = np.concatenate([row['combined_log_likelihood_change'] for row in chains])
    result['fine_weights'] = np.exp(difference-logsumexp(difference))
    result['maximum_grid_relative_alpha_error'] = np.asarray(np.max(abs(population['alpha']/alpha-1)))
    return result


def comparison(config, protocol, output, check):
    registry, accepted, posterior, weights, verification, model, samples, gp, sources = accepted_inputs(config)
    shape, weight = weights.shape, weights.ravel()
    alpha = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
    old = ROOT / config['existing_comparison']
    fine = config['verification_cells_per_axis']
    partitions = [config['primary_cells_per_axis'], fine]
    def obtain(name, builder):
        check()
        product, _ = cached(output, name, sources, builder)
        check()
        return product
    train = np.load(ROOT / accepted['configuration']['primary_event_catalog'])['data']
    train_fine = np.load(ROOT / config['training_verification'])['data']
    indices = verification['indices']
    uniform = verification['uniform_indices']
    np.testing.assert_array_equal(uniform, selected_indices(*shape, config['verification_draws_per_chain']))
    primary_train = obtain('training_primary', lambda: evaluate_events(samples, train, indices, check))
    fine_train = obtain('training_verification', lambda: evaluate_events(samples, train_fine, indices, check))
    event_change = (fine_train['log_integrals']-primary_train['log_integrals']).sum(-1)
    selection_change = -len(train)*np.log(verification['alpha_verification']/alpha[indices])
    change = event_change+selection_change
    groups = {name.removeprefix('group_'): value for name, value in verification.items() if name.startswith('group_')}
    target_checks = {name: {kind: float(values[np.searchsorted(indices, group)].std())
        for kind, values in [('event', event_change), ('selection', selection_change), ('combined', change)]}
        for name, group in dict(groups, pooled=indices).items()}
    kernel_indices = selected_indices(*shape, config['kernel_check_draws_per_chain'])
    def kernel_check():
        chosen = {name: value[kernel_indices] for name, value in samples.items()}
        expected, _ = event_integrals(model, chosen, torch.tensor(train))
        actual = evaluate_events(samples, train, kernel_indices, check)['log_integrals']
        return {'indices': kernel_indices, 'core': expected, 'numpy': actual}
    kernel = obtain('kernel_verification', kernel_check)
    kernel_error = float(abs((kernel['core']-kernel['numpy']).sum(-1)).max())
    target_decision = {'original_accepted': accepted['decision']['eligible_for_scientific_summaries'],
        'weights_passed': accepted['weight_diagnostics']['pareto_k'] < protocol['sampling']['maximum_pareto_k_for_reuse']
            and accepted['weight_diagnostics']['weight_ess_fraction'] >= protocol['sampling']['minimum_reuse_weight_ess_fraction'],
        'integrals_passed': max(v for row in target_checks.values() for v in row.values()) <= protocol['evaluation']['maximum_log_likelihood_difference_sd'],
        'kernel_passed': kernel_error <= config['maximum_kernel_total_log_likelihood_difference']}
    obtain('training_changes', lambda: {'indices': indices, 'uniform_indices': uniform,
        'event_change': event_change, 'selection_change': selection_change, 'combined_change': change})
    target_report = {'decision': target_decision, 'groups': target_checks,
        'maximum_kernel_total_log_likelihood_difference': kernel_error, 'scientific_scores_examined': False,
        'source_sha256': sources}
    target_path = output / 'copula_target_summary.json'
    if target_path.exists():
        if json.loads(target_path.read_text()) != target_report:
            raise ValueError('Cached target verification changed')
    else:
        write_json(target_path, target_report)
    if not all(target_decision.values()):
        raise ValueError('Copula target does not meet the external numerical protocol')
    del train, train_fine
    observations = dict(np.load(old / 'test_observations.npz'))
    events = np.load(old / 'events_fine.npz')['data']
    if events.shape != (192, config['test_event_samples'], 4):
        raise ValueError('Archived primary test integration has a different size')
    fine_events = obtain('test_events_verification', lambda: event_cloud(observations,
        config['verification_test_samples'], config['verification_test_seed'], check))
    positions = np.searchsorted(indices, uniform)
    subset_weight = weight[uniform]/weight[uniform].sum()
    fine_weight = np.exp(np.log(subset_weight)+change[positions]-logsumexp(np.log(subset_weight)+change[positions]))
    def primary_prediction():
        old_indices = np.load(old / 'derived.npz')['nuts_verification_indices']
        np.testing.assert_array_equal(old_indices, uniform)
        old_alpha = np.load(old / 'nuts_population_fine.npz')['alpha']
        np.testing.assert_allclose(old_alpha, verification['alpha_verification'][positions], rtol=0, atol=1e-12)
        values = np.empty((len(weight), len(events)))
        values[uniform] = np.load(old / 'nuts_prediction_fine.npz')['log_integrals']+np.log(old_alpha[:, None])
        missing = np.setdiff1d(np.arange(len(weight)), uniform)
        values[missing] = evaluate_events(samples, events, missing, check)['log_integrals']
        return {'log_integrals': values-np.log(alpha[:, None]), 'reused_indices': uniform}
    prediction = obtain('copula_prediction', primary_prediction)
    prediction_fine = obtain('copula_prediction_verification', lambda: evaluate_events(samples, fine_events['data'], uniform, check))
    primary_log = prediction['log_integrals']
    refined_log = prediction_fine['log_integrals']-np.log(verification['alpha_verification'][positions, None])
    copula_score, copula_precision = prediction_diagnostics(primary_log, weight, shape, config['batch_lengths'])
    numerical_prediction = {'event_selection_and_training_refinement': difference_precision(primary_log[uniform],
        refined_log, subset_weight, fine_weight, (shape[0], config['verification_draws_per_chain'])),
        'subset_minus_all_draws_total_score': float((weighted_prediction(primary_log[uniform], subset_weight)-copula_score).sum()),
        'verification_draws': len(uniform), 'gp_total_score_changes': {}, 'paired_total_score_changes': {}}
    truth_coarse = obtain('truth_coarse', lambda: cell_integrals(LISASNR(), fine,
        config['cell_integration_power'], config['cell_seed'], check))
    truth = obtain('truth', lambda: cell_integrals(LISASNR(), fine,
        config['verification_cell_integration_power'], config['verification_cell_seed'], check))
    coarse = obtain('copula_recovery_coarse', lambda: population_product(samples, weight,
        config['population_transport_power'], config['population_seed'], fine, check))
    population = obtain('copula_recovery', lambda: population_product(samples, weight,
        config['verification_population_transport_power'], config['verification_population_seed'], fine, check))
    distances, gradients = recovery_gradients(population, truth, fine, partitions)
    influences = obtain('copula_influences', lambda: influence_product(samples,
        config['verification_population_transport_power'], config['verification_population_seed'], gradients, check))
    # Selection fractions in functional summaries use the accepted target integral.
    transport_alpha = population['features'][:, 6].copy()
    features = population['features'].copy()
    features[:, 6] = alpha
    rows = {'copula': {'joint': {label: {'distance': value, **{k: v for k, v in moment_check(
        influences[label], weight, shape, config['batch_lengths']).items() if k != 'mean'}} for label, value in distances.items()},
        'prediction_precision': copula_precision,
        'functionals': posterior_functionals(features, weight, shape, truth['features'], config)}}
    coarse_distance, _ = recovery_gradients(coarse, truth_coarse, fine, partitions)
    numerical_recovery = {'copula_transport_and_truth_changes': {name: distances[name]-value for name, value in coarse_distance.items()},
        'maximum_transport_relative_alpha_error': float(np.max(abs(transport_alpha/alpha-1))),
        'maximum_covariance_refinement_change': float(abs(population['features'][:, 9:]-coarse['features'][:, 9:]).max()),
        'gp_cell_integration_changes': {}, 'partition_changes': {}, 'gp_grid_relative_alpha_error': {}}
    comparisons, gp_rows = {}, {}
    from summarize_selected_mixture import paired_scores
    for key, (chains, inputs) in gp.items():
        product = obtain('gp_'+key, lambda: gp_product(chains, inputs, int(key), protocol, truth, partitions, events, fine_events['data'], check))
        gp_shape = product['features'].shape[:2]
        gp_weight = np.full(np.prod(gp_shape), 1/np.prod(gp_shape))
        score, precision = prediction_diagnostics(product['log_integrals'], gp_weight, gp_shape, config['batch_lengths'])
        score_change = float((weighted_prediction(product['fine_log_integrals'], product['fine_weights'])-score).sum())
        numerical_prediction['gp_total_score_changes'][key] = score_change
        numerical_prediction['paired_total_score_changes'][key] = numerical_prediction['event_selection_and_training_refinement']['total_score_change']-score_change
        comparisons[key] = dict(paired_scores(copula_score, score), posterior_mcse=float(np.hypot(copula_precision['mcse'], precision['mcse'])))
        ds, _ = recovery_gradients(product, truth, fine, partitions)
        gp_rows[key] = {'joint': {label: {'distance': value, **{k: v for k, v in moment_check(product['influence_'+label],
            gp_weight, gp_shape, config['batch_lengths']).items() if k != 'mean'}} for label, value in ds.items()},
            'prediction_precision': precision, 'functionals': posterior_functionals(product['features'], gp_weight, gp_shape, truth['features'], config)}
        coarse_gp = gp_population(product['probability'], int(key), fine, truth_coarse['selection'])
        cds, _ = recovery_gradients(coarse_gp, truth_coarse, fine, partitions)
        numerical_recovery['gp_cell_integration_changes'][key] = {label: ds[label]-value for label, value in cds.items()}
        numerical_recovery['gp_grid_relative_alpha_error'][key] = float(product['maximum_grid_relative_alpha_error'])
    rhats = []
    for key, row in [('copula', rows['copula']), *gp_rows.items()]:
        numerical_recovery['partition_changes'][key] = {kind: row['joint'][f'{kind}_{fine}']['distance']-
            row['joint'][f'{kind}_{min(partitions)}']['distance'] for kind in ('intrinsic', 'detected')}
        rhats.extend(value['influence_rhat'] for value in row['joint'].values())
        rhats.extend([row['prediction_precision']['influence_rhat'], row['functionals']['maximum_influence_rhat']])
    recovery_changes = list(numerical_recovery['copula_transport_and_truth_changes'].values())
    recovery_changes.extend(v for group in numerical_recovery['gp_cell_integration_changes'].values() for v in group.values())
    recovery_changes.extend(v for group in numerical_recovery['partition_changes'].values() for v in group.values())
    decision = {'posteriors_accepted': True,
        'recovery_precision_passed': max(map(abs, recovery_changes)) < protocol['evaluation']['maximum_joint_distance_numerical_change'],
        'prediction_precision_passed': max(map(abs, numerical_prediction['paired_total_score_changes'].values())) < protocol['evaluation']['maximum_paired_score_numerical_change'],
        'grid_selection_precision_passed': max(numerical_recovery['gp_grid_relative_alpha_error'].values()) < config['maximum_grid_relative_alpha_change'],
        'derived_mixing_passed': max(rhats) <= protocol['sampling']['maximum_functional_rhat']}
    check()
    decision['resources_passed'] = True
    decision['eligible_for_scientific_comparison'] = all(decision.values())
    result = dict(rows, gp=gp_rows, configuration=config, experiment=config['experiment'],
        source_sha256=sources, comparator_label=protocol['binned_gp']['label'], decision=decision,
        copula_target_checks=target_checks, copula_minus_gp_test_scores=comparisons,
        numerical_recovery=numerical_recovery, numerical_prediction=numerical_prediction,
        maximum_derived_influence_rhat=max(rhats), truth_functionals=truth['features'].tolist(),
        product_sha256={p.name: sha256(p) for p in output.glob('*.npz')},
        scope='One fixed off-family LISA population, 135 detected training events and 192 independent test events. Both resolutions and all outcomes are retained. Predictive densities are averaged before taking logs; common event-evidence constants cancel. The SNR measurement is included once in each event likelihood. Functional intervals do not establish repeated-catalog coverage.')
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': decision, 'scores': comparisons, 'recovery': {key: row['joint']
        for key, row in [('copula', rows['copula']), *gp_rows.items()]}}), flush=True)


def main():
    config = json.loads(CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    registry = json.loads((ROOT / config['posterior_registry']).read_text())
    prior = registry['cost']['total_study_charge_seconds']
    if config['maximum_seconds']+prior > config['maximum_total_study_seconds']:
        raise ValueError('Evaluation allocation exceeds the amended study budget')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed comparison is retained')
    old = sum(json.loads(p.read_text())['process_elapsed_seconds'] for p in output.glob('job_cost_*.json'))
    deadline = PROCESS_STARTED+config['maximum_seconds']-old
    check = lambda: enforce_budget(deadline, protocol['resources']['maximum_resident_gib'])
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
             'all_evaluation_attempts_seconds': old+seconds, 'total_study_charge_seconds': prior+old+seconds,
             'remaining_study_seconds': config['maximum_total_study_seconds']-prior-old-seconds,
             'study_ceiling_seconds': config['maximum_total_study_seconds'],
             'configuration_sha256': sha256(CONFIG), 'runtime': runtime_metadata()})


if __name__ == '__main__':
    main()
