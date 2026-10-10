"""Recovery and paired prediction for the first declared external experiment."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from binned_gp_benchmark import FEATURE_NAMES, controlled_log_event_weights, controlled_selection
from check_external_copula_selected import (ROOT, CONFIG, evaluate_events, make_events,
    params_at, read_copula, selected_indices, settings)
from compare_lisa_posterior import cached, moment_check
from correct_selected_posterior import sha256
from external_selected_evaluation import (coarsen, copula_cells, expand_gp_cells,
    gp_gradient, interval_score, js_gradient, normal_points)
from predict_corrected_posterior import weighted_prediction
from refine_lisa_posterior import corrected_statistics, influence_diagnostics
from run_lisa_production import mixing_summary, verify_product, write_json
from summarize_selected_mixture import paired_scores, truth_samples
from symmetric_gp_benchmark import SymmetricBinnedGPPotential


def accepted_inputs(config):
    samples, shape, sources = read_copula(config)
    output = ROOT / config['output']
    target_path = output / 'copula_target_summary.json'
    target = json.loads(target_path.read_text())
    if not target['decision']['target_reusable'] or target['source_sha256'] != sources:
        raise ValueError('A verified copula target is required')
    for name, expected in target['product_sha256'].items():
        if sha256(output / name) != expected:
            raise ValueError(f'Target product changed: {name}')
    paths = [target_path, Path(__file__), ROOT / 'scripts/summarize_selected_mixture.py',
             ROOT / 'scripts/predict_corrected_posterior.py']
    original_comparison = ROOT / config['copula_source'] / 'comparison.json'
    for name, expected in json.loads(original_comparison.read_text())['sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Archived copula product changed: {name}')
    paths.append(original_comparison)
    paths.extend(output / name for name in target['product_sha256'])
    gp = {}
    for key, directory in config['gp_sources'].items():
        directory = ROOT / directory
        summary_path, gp_target_path = directory / 'summary.json', directory / 'target.json'
        summary = json.loads(summary_path.read_text())
        gp_target = json.loads(gp_target_path.read_text())
        if not summary['decision']['posterior_accepted'] or sha256(gp_target_path) != summary['target_product_sha256']:
            raise ValueError('An accepted GP posterior is required')
        for name, expected in gp_target['source_sha256'].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f'GP source changed: {name}')
            sources[name] = expected
        if sha256(directory / 'inputs.npz') != gp_target['inputs_product_sha256']:
            raise ValueError('GP target arrays changed')
        paths.extend([summary_path, gp_target_path, directory / 'inputs.npz'])
        chains = []
        for name, expected in summary['chain_reports_sha256'].items():
            report_path = directory / name
            report = verify_product(report_path.with_suffix('.npz'))
            if sha256(report_path) != expected or report['target_product_sha256'] != summary['target_product_sha256']:
                raise ValueError('GP chain provenance changed')
            with np.load(report_path.with_suffix('.npz')) as draws:
                chains.append({name: draws[name] for name in ('coordinate', 'features')})
            paths.extend([report_path, report_path.with_suffix('.npz')])
        gp[key] = (gp_target, summary, chains, dict(np.load(directory / 'inputs.npz')))
        original = gp_target['configuration']
        if 'sampling_extension' in original:
            paths.append(ROOT / original['sampling_extension']['previous_summary'])
        if 'amendment' in original:
            paths.append(ROOT / original['amendment']['prior_attempt_summary'])
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    return samples, shape, target, gp, sources


def truth_product(config, bins, fine=False):
    prefix = 'verification_' if fine else ''
    normal = normal_points(config[prefix+'truth_transport_power'], config[prefix+'truth_population_seed'])
    total = len(normal)
    result = None
    for start in range(0, total, 2**18):
        block = normal[start:start+2**18]
        product = copula_cells(params_at(truth_samples(), 0), block, bins)
        product['detected'] *= product['transport_alpha']
        if result is None:
            result = {name: np.zeros_like(value) for name, value in product.items()}
        for name, value in product.items():
            result[name] += len(block)/total*value
    result['detected'] /= result['transport_alpha']
    return result


def population_product(samples, weights, config, bins, fine=False):
    prefix = 'verification_' if fine else ''
    normal = normal_points(config[prefix+'population_transport_power'], config[prefix+'population_seed'])
    means = {kind: np.zeros(bins**3) for kind in ('intrinsic', 'detected')}
    features, additional, alpha = [], [], []
    for i, weight in enumerate(weights):
        product = copula_cells(params_at(samples, i), normal, bins)
        for kind in means:
            means[kind] += weight*product[kind]
        features.append(product['features'])
        additional.append(product['additional'])
        alpha.append(product['transport_alpha'])
        if (i+1) % 400 == 0:
            print('Population integration', i+1, '/', len(weights), flush=True)
    return dict(means, features=np.asarray(features), additional=np.asarray(additional),
                transport_alpha=np.asarray(alpha))


def recovery_gradients(population, truth, fine, partitions):
    gradients, distances = {}, {}
    for bins in partitions:
        for kind in ('intrinsic', 'detected'):
            label = f'{kind}_{bins}'
            distance, gradient = js_gradient(coarsen(population[kind], fine, bins), coarsen(truth[kind], fine, bins))
            distances[label] = distance
            gradients[label] = gradient
    return distances, gradients


def population_influences(samples, config, bins, partitions, gradients):
    normal = normal_points(config['population_transport_power'], config['population_seed'])
    result = {label: [] for label in gradients}
    for i in range(len(samples['weights_logits'])):
        product = copula_cells(params_at(samples, i), normal, bins)
        for partition in partitions:
            for kind in ('intrinsic', 'detected'):
                label = f'{kind}_{partition}'
                result[label].append(coarsen(product[kind], bins, partition) @ gradients[label])
    return {label: np.asarray(values) for label, values in result.items()}


@torch.no_grad()
def gp_product(config, protocol, gp, key, truth, partitions, test):
    target, summary, chains, inputs = gp[key]
    bins, fine = int(key), max(partitions)
    model = SymmetricBinnedGPPotential(bins, inputs['log_event_weights'], inputs['selection_coefficients'], protocol['binned_gp'])
    coordinates = np.concatenate([c['coordinate'] for c in chains])
    probability = np.empty((len(coordinates), bins**3))
    for i, coordinate in enumerate(coordinates):
        probability[i] = torch.softmax(model.field(torch.tensor(coordinate)), dim=-1).numpy()
    alpha = probability @ (controlled_selection(bins)*bins**3)
    means = {'intrinsic': expand_gp_cells(probability.mean(0), bins, fine),
             'detected': expand_gp_cells((probability/alpha[:, None]).mean(0), bins, fine, True)}
    result = dict(means, features=np.stack([c['features'][:, :12] for c in chains]))
    _, gradients = recovery_gradients(means, truth, fine, partitions)
    for label, gradient in gradients.items():
        kind, partition = label.split('_')
        projection = probability @ gp_gradient(gradient, bins, int(partition), kind == 'detected')
        result['influence_'+label] = projection/alpha if kind == 'detected' else projection
    log_weights = controlled_log_event_weights(test['observed'], test['sigma'], bins, 32)
    fine_log_weights = controlled_log_event_weights(test['observed'], test['sigma'], bins, 64)
    def predict(log_w):
        integral = probability @ np.exp(log_w).T
        if np.any(integral <= 0):
            raise ValueError('Predictive integral underflow')
        return np.log(integral) + 3*np.log(bins) - np.log(alpha[:, None])
    result['log_integrals'] = predict(log_weights)
    result['verification_log_prediction'] = weighted_prediction(predict(fine_log_weights))
    result['maximum_event_log_weight_difference'] = np.asarray(np.max(abs(log_weights-fine_log_weights)))
    return result


def prediction_diagnostics(log_integrals, weights, shape, batch_lengths):
    weights = np.asarray(weights)/np.sum(weights)
    log_prediction = weighted_prediction(log_integrals, weights)
    derivative = np.exp(log_integrals-log_prediction).sum(-1)
    precision = moment_check(derivative, weights, shape, batch_lengths)
    return log_prediction, precision


def difference_precision(primary, refined, weights, refined_weights, shape):
    first, second = weighted_prediction(primary, weights), weighted_prediction(refined, refined_weights)
    n = len(weights)
    a = np.exp(primary-first).sum(-1)
    b = np.exp(refined-second).sum(-1)
    influence = n*(refined_weights*(b-refined_weights @ b)-weights*(a-weights @ a))
    check = influence_diagnostics(influence.reshape(*shape, 1), (8, 16, 32))
    return {'total_score_change': float((second-first).sum()),
            'mcse_of_change': float(check['mcse'][0]), 'influence_rhat': float(check['rhat'][0])}


def posterior_functionals(values, weights, shape, truth, config):
    result = corrected_statistics(values.reshape(*shape, -1), FEATURE_NAMES,
        weights.reshape(shape), len(FEATURE_NAMES),
        {'batch_lengths': config['batch_lengths'], 'cdf_probabilities': [.05, .5, .95]})
    for i, name in enumerate(FEATURE_NAMES):
        row = result['functionals'][name]
        lower, median, upper = row['quantiles']
        row.update(truth=float(truth[i]), error_of_mean=float(row['mean']-truth[i]),
                   interval_score_90=float(interval_score(lower, upper, truth[i])),
                   truth_in_interval=bool(lower <= truth[i] <= upper))
    return result


def sampling_costs(config, gp):
    copula = [json.loads((ROOT / config['copula_source'] / f'nuts_chain_{i}.json').read_text()) for i in range(4)]
    attempts = {config['gp_sources'][key]: entry[1] for key, entry in gp.items()}
    original = gp['12'][0]['configuration']
    for path in (original['sampling_extension']['previous_summary'], original['amendment']['prior_attempt_summary']):
        attempts[str(Path(path).parent).replace('\\', '/')] = json.loads((ROOT / path).read_text())
    attempt_rows = {name: {'accepted': report['decision']['posterior_accepted'],
        'recorded_process_elapsed_seconds': report['recorded_process_elapsed_seconds'],
        'maximum_peak_working_set_bytes': report['maximum_peak_working_set_bytes']} for name, report in attempts.items()}
    return {'copula_original_sampling_seconds': sum(r['seconds'] for r in copula),
            'copula_original_peak_bytes': max(r['runtime']['peak_working_set_bytes'] for r in copula),
            'accepted_gp': {key: {k: entry[1][k] for k in ('sampling_and_compilation_seconds',
                'recorded_process_elapsed_seconds', 'maximum_peak_working_set_bytes')} for key, entry in gp.items()},
            'all_gp_selected_attempts': attempt_rows,
            'all_gp_selected_attempts_recorded_seconds': sum(r['recorded_process_elapsed_seconds'] for r in attempts.values()),
            'scope': 'Original copula sampling and every GP selected-catalog production attempt, including failed and replayed sampling. Earlier pilots are in their own reports. Evaluation costs are separate; these are not end-to-end software speed comparisons.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['recovery', 'prediction', 'summary', 'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config, protocol = settings()
    evaluation = protocol['evaluation']
    samples, shape, target, gp, sources = accepted_inputs(config)
    output = ROOT / config['output']
    partitions = [evaluation['joint_cells_per_axis'], evaluation['verification_joint_cells_per_axis']]
    fine = max(partitions)
    weights = np.load(output / 'copula_weights.npz')['weights'].ravel()
    test = dict(np.load(ROOT / config['copula_source'] / 'test_catalog.npz'))
    def obtain(label, builder):
        product, metadata = cached(output, label, sources, builder)
        if metadata['runtime']['peak_working_set_bytes'] > protocol['resources']['maximum_resident_gib']*1024**3:
            raise RuntimeError('Comparison exceeded the declared memory limit')
        return product
    truth = obtain('recovery_truth', lambda: truth_product(config, fine))
    truth_fine = obtain('recovery_truth_verification', lambda: truth_product(config, fine, True))
    population = obtain('copula_recovery', lambda: population_product(samples, weights, config, fine))
    population_fine = obtain('copula_recovery_verification', lambda: population_product(samples, weights, config, fine, True))
    distances, gradients = recovery_gradients(population, truth, fine, partitions)
    influences = obtain('copula_recovery_influences', lambda: population_influences(samples, config, fine, partitions, gradients))
    gp_products = {key: obtain('gp_'+key+'_comparison', lambda key=key: gp_product(config, protocol, gp, key, truth, partitions, test))
                   for key in gp}
    if args.stage == 'recovery':
        return
    events = obtain('test_events', lambda: make_events(test['observed'], evaluation['test_event_samples'], evaluation['evaluation_seed']))
    prediction = obtain('copula_test_primary', lambda: evaluate_events(samples, events['data'], np.arange(len(weights))))
    indices = selected_indices(*shape, config['verification_draws_per_chain'])
    fine_events = obtain('test_events_verification', lambda: make_events(test['observed'], evaluation['verification_test_event_samples'], evaluation['verification_seed']))
    prediction_fine = obtain('copula_test_verification', lambda: evaluate_events(samples, fine_events['data'], indices))
    if args.stage == 'prediction':
        return
    alpha = population['features'][:, 6]
    log_prediction, precision = prediction_diagnostics(prediction['log_integrals']-np.log(alpha[:, None]), weights, shape, config['batch_lengths'])
    train_fine = np.load(output / 'train_verification.npz')
    train_primary = np.load(output / 'train_primary.npz')
    train_difference = (train_fine['log_integrals'][np.searchsorted(train_fine['indices'], indices)]
                        - train_primary['log_integrals'][indices]).sum(-1)
    subset_weight = weights[indices]/weights[indices].sum()
    fine_weight = np.exp(np.log(subset_weight)+train_difference-logsumexp(np.log(subset_weight)+train_difference))
    primary_integrals = prediction['log_integrals'][indices]-np.log(alpha[indices, None])
    fine_integrals = prediction_fine['log_integrals']-np.log(alpha[indices, None])
    subset_shape = (shape[0], len(indices)//shape[0])
    numerical_prediction = {
        'event_integration': difference_precision(primary_integrals, fine_integrals, subset_weight, subset_weight, subset_shape),
        'training_target_and_event_integration': difference_precision(primary_integrals, fine_integrals, subset_weight, fine_weight, subset_shape),
        'subset_minus_all_draws_total_score': float((weighted_prediction(primary_integrals, subset_weight)-log_prediction).sum()),
        'verification_draws': len(indices),
        'scope': 'Numerical differences pair the same posterior draws; the subset versus all-draw score is a separate posterior Monte Carlo check.'}
    feature_result = posterior_functionals(population['features'], weights, shape, truth_fine['features'], config)
    names = FEATURE_NAMES + [f'cdf_x{a}_{x}' for a in (1, 2, 3) for x in (.25, .5, .75)] + [f'octant_{i}' for i in range(8)]
    mixing = mixing_summary(names, np.column_stack([population_fine['features'], population_fine['additional']]).reshape(*shape, -1))
    joint = {label: {'distance': distance, **{k: v for k, v in moment_check(influences[label], weights, shape, config['batch_lengths']).items() if k != 'mean'}}
             for label, distance in distances.items()}
    refined_distances, _ = recovery_gradients(population_fine, truth_fine, fine, partitions)
    numerical_recovery = {'copula_transport_and_truth_changes': {k: refined_distances[k]-v for k, v in distances.items()},
        'copula_partition_changes': {kind: distances[f'{kind}_{fine}']-distances[f'{kind}_{min(partitions)}'] for kind in ('intrinsic', 'detected')},
        'maximum_copula_transport_relative_alpha_error': float(np.max(abs(population['transport_alpha']/alpha-1))),
        'maximum_covariance_refinement_change': float(np.max(abs(population['features'][:, 9:]-population_fine['features'][:, 9:]))),
        'truth_covariance_refinement_change': (truth_fine['features'][9:]-truth['features'][9:]).tolist(),
        'gp_truth_changes': {}, 'gp_partition_changes': {}}
    comparisons, gp_result, gp_score_changes = {}, {}, {}
    for key, product in gp_products.items():
        gp_shape = product['features'].shape[:2]
        uniform = np.full(np.prod(gp_shape), 1/np.prod(gp_shape))
        gp_score, gp_precision = prediction_diagnostics(product['log_integrals'], uniform, gp_shape, config['batch_lengths'])
        comparison = paired_scores(log_prediction, gp_score)
        comparison['posterior_mcse'] = float(np.hypot(precision['mcse'], gp_precision['mcse']))
        comparisons[key] = comparison
        gp_distances, _ = recovery_gradients(product, truth, fine, partitions)
        gp_fine_distances, _ = recovery_gradients(product, truth_fine, fine, partitions)
        gp_joint = {label: {'distance': distance, **{k: v for k, v in moment_check(product['influence_'+label], uniform, gp_shape, config['batch_lengths']).items() if k != 'mean'}}
                    for label, distance in gp_distances.items()}
        gp_result[key] = {'joint': gp_joint, 'prediction_precision': gp_precision,
                         'functionals': posterior_functionals(product['features'], uniform, gp_shape, truth_fine['features'], config)}
        numerical_recovery['gp_truth_changes'][key] = {label: gp_fine_distances[label]-v for label, v in gp_distances.items()}
        numerical_recovery['gp_partition_changes'][key] = {kind: gp_distances[f'{kind}_{fine}']-gp_distances[f'{kind}_{min(partitions)}'] for kind in ('intrinsic', 'detected')}
        gp_score_changes[key] = float((product['verification_log_prediction']-gp_score).sum())
    numerical_prediction['gp_quadrature_total_score_changes'] = gp_score_changes
    recovery_changes = list(numerical_recovery['copula_transport_and_truth_changes'].values()) + list(numerical_recovery['copula_partition_changes'].values())
    for name in ('gp_truth_changes', 'gp_partition_changes'):
        for group in numerical_recovery[name].values():
            recovery_changes.extend(group.values())
    score_change = numerical_prediction['training_target_and_event_integration']['total_score_change']
    numerical_prediction['paired_total_score_changes'] = {key: score_change-v for key, v in gp_score_changes.items()}
    derived_rhats = [v['influence_rhat'] for v in joint.values()] + [precision['influence_rhat'], feature_result['maximum_influence_rhat']]
    for row in gp_result.values():
        derived_rhats.extend(v['influence_rhat'] for v in row['joint'].values())
        derived_rhats.extend([row['prediction_precision']['influence_rhat'], row['functionals']['maximum_influence_rhat']])
    decision = {
        'copula_target_reusable': target['decision']['target_reusable'],
        'copula_density_mixing_passed': mixing['maximum_rhat'] <= protocol['sampling']['maximum_functional_rhat'] and min(mixing['minimum_bulk_ess'], mixing['minimum_tail_ess']) >= protocol['sampling']['minimum_functional_bulk_ess'],
        'recovery_precision_passed': max(map(abs, recovery_changes)) < evaluation['maximum_joint_distance_numerical_change'],
        'prediction_precision_passed': max(map(abs, numerical_prediction['paired_total_score_changes'].values())) < evaluation['maximum_paired_score_numerical_change'],
        'derived_mixing_passed': max(derived_rhats) <= protocol['sampling']['maximum_functional_rhat']}
    decision['eligible_for_scientific_comparison'] = all(decision.values())
    summary = {'configuration': config, 'source_sha256': sources, 'decision': decision,
        'experiment': config['experiment'], 'comparator_label': protocol['binned_gp']['label'],
        'copula': {'joint': joint, 'prediction_precision': precision, 'functionals': feature_result, 'density_mixing': mixing},
        'gp': gp_result, 'copula_minus_gp_test_scores': comparisons,
        'numerical_recovery': numerical_recovery, 'numerical_prediction': numerical_prediction,
        'maximum_derived_influence_rhat': max(derived_rhats), 'sampling_costs': sampling_costs(config, gp),
        'evaluation_costs': {p.stem: {'seconds': json.loads(p.read_text())['seconds'], 'peak_bytes': json.loads(p.read_text())['runtime']['peak_working_set_bytes']}
                             for p in output.glob('*.json') if 'seconds' in json.loads(p.read_text())},
        'product_sha256': {p.name: sha256(p) for p in output.glob('*.npz')},
        'interpretation': 'One compact population in the copula family. Predictive differences omit only common event-evidence constants. Other declared populations remain required; single-catalog intervals do not measure frequentist coverage.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps({'decision': decision, 'scores': comparisons, 'copula_joint': joint,
                      'gp_joint': {k: v['joint'] for k, v in gp_result.items()}}), flush=True)


if __name__ == '__main__':
    main()
