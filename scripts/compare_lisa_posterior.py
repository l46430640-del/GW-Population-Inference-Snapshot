"""Recovery, rate, and independent event prediction from a frozen LISA posterior."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, ndtr, ndtri
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta, gamma, qmc
import torch

from benchmark_selected_mixture import runtime_metadata
from build_lisa_analysis_assets import LISASNR
from check_selected_mixture_integrals import event_integrals
from correct_selected_posterior import latent_sites, sha256, weighted_quantiles
from lisa_noisy_catalog import SIGMA, posterior_from_measurements
from predict_corrected_posterior import array_hash, normalized_weights, weighted_prediction
from refine_lisa_posterior import influence_diagnostics
from run_lisa_production import write_json
from sample_lisa_transport import sampling_model
from summarize_analysis_populations import components, probability_cells
from summarize_selected_mixture import paired_scores
from validate_population_morphologies import BETA_A, BETA_B, ROOT, truth_sample


def cached(output, label, sources, builder):
    path = output / (label + '.npz')
    if path.with_suffix('.json').exists():
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['source_sha256'] != sources or sha256(path) != metadata['product_sha256']:
            raise ValueError(f'Cached comparison has changed inputs: {label}')
        return dict(np.load(path)), metadata
    if path.exists():
        raise FileExistsError(f'Unverified product retained: {path}')
    started = time.perf_counter()
    product = builder()
    if any(not np.isfinite(v).all() for v in product.values() if np.asarray(v).dtype.kind in 'fc'):
        raise ValueError(f'Nonfinite comparison product: {label}')
    np.savez_compressed(path, **product)
    metadata = {'source_sha256': sources, 'product_sha256': sha256(path),
                'seconds': time.perf_counter()-started, 'runtime': runtime_metadata()}
    write_json(path.with_suffix('.json'), metadata)
    print(label, f"{metadata['seconds']:.2f} seconds", flush=True)
    return product, metadata


def verified_sources(config, path):
    source = ROOT / config['source']
    accepted = json.loads((source / 'summary.json').read_text())
    if not accepted['decision']['eligible_for_scientific_summaries']:
        raise ValueError('A numerically accepted posterior is required')
    sources = dict(accepted['source_sha256'])
    sources.update({(source / name).relative_to(ROOT).as_posix(): value
                    for name, value in accepted['product_sha256'].items()})
    for name, value in sources.items():
        if sha256(ROOT / name) != value:
            raise ValueError(f'Accepted source changed: {name}')
    paths = [path, source / 'summary.json', ROOT / 'scripts/compare_lisa_posterior.py',
             ROOT / 'scripts/summarize_analysis_populations.py',
             ROOT / 'scripts/check_selected_mixture_integrals.py', ROOT / 'scripts/summarize_selected_mixture.py']
    for relative in config['baselines']:
        paths.extend(ROOT / relative / name for name in ('summary.json', 'flow.pt', 'catalog.npz'))
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    return accepted, sources


def moment_check(values, weights, chain_shape=None, batch_lengths=(25, 50)):
    values = np.asarray(values, dtype=float)
    weights = normalized_weights(weights, len(values))
    mean = weights @ values
    influence = len(values) * weights * (values - mean)
    if chain_shape is None:
        mcse = influence.std(ddof=1) / np.sqrt(len(values))
        rhat = None
    else:
        checked = influence_diagnostics(influence.reshape(*chain_shape, 1), batch_lengths)
        mcse, rhat = float(checked['mcse'][0]), float(checked['rhat'][0])
    return {'mean': float(mean), 'mcse': float(mcse), 'influence_rhat': rhat}


def distance_check(cells, truth, weights, chain_shape=None):
    weights = normalized_weights(weights, len(cells))
    mean = weights @ cells
    distance = float(jensenshannon(mean, truth, base=2))
    gradient = np.zeros_like(mean)
    positive = mean > 0
    if distance > 1e-12:
        gradient[positive] = np.log2(2 * mean[positive] / (mean[positive]+truth[positive])) / (4 * distance)
    diagnostic = moment_check(cells @ gradient, weights, chain_shape)
    return {'distance': distance, 'mcse': diagnostic['mcse'], 'influence_rhat': diagnostic['influence_rhat']}


def rate_summary(alpha, weights, count, truth_rate, chain_shape=None):
    alpha = np.asarray(alpha, dtype=float)
    weights = normalized_weights(weights, len(alpha))
    if count <= 0 or np.any((alpha <= 0) | (alpha > 1)) or not np.isfinite(alpha).all():
        raise ValueError('Positive count and efficiencies in (0,1] are required')
    def cdf(rate):
        return float(weights @ gammainc(count, alpha * rate))
    probabilities = [.05, .5, .95]
    upper = gamma.ppf(.9999, count) / alpha.min()
    quantiles = [brentq(lambda r: cdf(r)-p, 0, upper) for p in probabilities]
    mean = weights @ (count / alpha)
    sd = np.sqrt(weights @ (count * (count+1) / alpha**2) - mean**2)
    return {'mean': float(mean), 'sd': float(sd), 'quantiles_05_50_95': quantiles,
            'mean_precision': moment_check(count / alpha, weights, chain_shape),
            'cdf_at_truth': moment_check(gammainc(count, alpha * truth_rate), weights, chain_shape),
            'cdf_precision_at_quantiles': [moment_check(gammainc(count, alpha * r), weights, chain_shape)
                                           for r in quantiles]}


def exact_truth_probabilities(bins):
    edges = np.linspace(0, 1, bins+1)
    mass = np.diff(beta.cdf(edges[:, None, None], BETA_A, BETA_B), axis=0)
    joint = sum(w * np.einsum('i,j,k->ijk', mass[:, c, 0], mass[:, c, 1], mass[:, c, 2]).ravel()
                for c, w in enumerate([.55, .45]))
    return joint, np.einsum('bcd,c->db', mass, [.55, .45])


def truth_product(config, fine=False):
    prefix = 'fine_' if fine else ''
    uniform = qmc.Sobol(3, scramble=True, seed=config[prefix+'truth_seed']).random_base2(config[prefix+'truth_power'])
    response = LISASNR()
    cells, alpha, hidden = np.zeros(config['cells_per_axis']**3), 0., 0.
    for c, weight in enumerate([.55, .45]):
        points = beta.ppf(uniform, BETA_A[c], BETA_B[c])
        detection = ndtr(response(points)-8)
        cells += weight * detection.mean() * probability_cells(points, config['cells_per_axis'], detection)
        alpha += weight * detection.mean()
        hidden += weight * np.mean(detection < .1)
    intrinsic, _ = exact_truth_probabilities(config['cells_per_axis'])
    _, marginals = exact_truth_probabilities(config['marginal_bins'])
    return {'intrinsic': intrinsic, 'detected': cells/alpha, 'marginals': marginals,
            'alpha': np.asarray(alpha), 'hidden': np.asarray(hidden)}


def test_observations(config):
    rng, response = np.random.default_rng(config['test_catalog_seed']), LISASNR()
    truths, measurements, statistics = [], [], []
    while sum(map(len, truths)) < config['test_events']:
        points = truth_sample('bimodal', 256, rng)
        r = response(points) + rng.normal(size=len(points))
        y = points + rng.normal(size=points.shape) * SIGMA
        selected = r > 8
        truths.append(points[selected])
        measurements.append(y[selected])
        statistics.append(r[selected])
    count = config['test_events']
    return {'truth': np.concatenate(truths)[:count], 'measurement': np.concatenate(measurements)[:count],
            'rho_observed': np.concatenate(statistics)[:count]}


def event_samples(config, observations, fine=False):
    prefix = 'fine_' if fine else ''
    rng, response = np.random.default_rng(config[prefix+'event_seed']), LISASNR()
    samples, ess = [], []
    for y, r in zip(observations['measurement'], observations['rho_observed']):
        points, _, diagnostic = posterior_from_measurements(y, r, response, config[prefix+'event_samples'], rng)
        samples.append(points)
        ess.append(diagnostic['proposal_ess'])
    samples = np.asarray(samples)
    return {'data': np.concatenate([samples, np.ones((*samples.shape[:2], 1))], -1), 'proposal_ess': np.asarray(ess)}


@torch.no_grad()
def baseline_draws(config, index):
    directory = ROOT / config['baselines'][index]
    original = json.loads((directory / 'summary.json').read_text())['configuration']
    model = sampling_model(original, checkpoint=directory / 'flow.pt')
    torch.manual_seed(config['baseline_seed'] + index)
    return {'latent': model.flow_dist.sample((config['baseline_draws'],)).numpy()}


@torch.no_grad()
def selection_fractions(model, latent, count, seed):
    previous = model.lambda_det_mc_samples, model.selection_mc_seed
    try:
        model.lambda_det_mc_samples, model.selection_mc_seed = count, seed
        return np.asarray([float(model._selection_fraction_for_params(
            model._unflatten_latent_vector(torch.tensor(point), apply_numerical_limits=False))) for point in latent])
    finally:
        model.lambda_det_mc_samples, model.selection_mc_seed = previous


@torch.no_grad()
def population_product(model, latent, config, fine=False):
    prefix = 'fine_' if fine else ''
    normal = ndtri(qmc.Sobol(3, scramble=True, seed=config[prefix+'transport_seed']).random_base2(config[prefix+'transport_power']))
    response = LISASNR()
    edges = torch.linspace(0, 1, config['marginal_bins']+1)
    result = {key: [] for key in ('intrinsic', 'detected', 'marginals', 'hidden', 'transport_alpha', 'alpha')}
    for i, point in enumerate(latent):
        weights, marginal, chol = components(model, point)
        uniform = torch.tensor(ndtr(np.einsum('nj,kij->nki', normal, chol.numpy()))).clamp(1e-15, 1-1e-15)
        points = np.clip(marginal.icdf(uniform).numpy(), 0, 1)
        detect = ndtr(response(points.reshape(-1, 3))-8).reshape(points.shape[:2])
        alpha = weights @ detect.mean(0)
        result['intrinsic'].append(sum(w * probability_cells(points[:, c], config['cells_per_axis'])
                                        for c, w in enumerate(weights)))
        result['detected'].append(sum(w * detect[:, c].mean() * probability_cells(points[:, c], config['cells_per_axis'], detect[:, c])
                                       for c, w in enumerate(weights)) / alpha)
        cdf = marginal.cdf(edges[:, None, None].expand(-1, len(weights), 3)).numpy()
        result['marginals'].append(np.einsum('bcd,c->db', np.diff(cdf, axis=0), weights))
        result['hidden'].append(weights @ (detect < .1).mean(0))
        result['transport_alpha'].append(alpha)
        if (i+1) % 500 == 0:
            print('Population draws', i+1, '/', len(latent), flush=True)
    result['alpha'] = selection_fractions(model, latent,
        config['fine_selection_points'] if fine else model.lambda_det_mc_samples,
        config['fine_selection_seed'] if fine else model.selection_mc_seed)
    return {key: np.asarray(value) for key, value in result.items()}


@torch.no_grad()
def prediction_product(model, latent, alpha, data):
    values, ess = [], []
    data = torch.tensor(data, dtype=torch.float64)
    for start in range(0, len(latent), 64):
        sites = {name: value.numpy() for name, value in latent_sites(model, torch.tensor(latent[start:start+64])).items()}
        log_integrals, effective = event_integrals(model, sites, data)
        values.append(log_integrals - np.log(alpha[start:start+64, None]))
        ess.append(effective)
    return {'log_integrals': np.concatenate(values), 'minimum_event_ess': np.concatenate(ess)}


def scientific_summary(population, prediction, weights, truth, chain_shape, count, injected_rate):
    weights = normalized_weights(weights, len(population['alpha']))
    score = weighted_prediction(prediction['log_integrals'], weights)
    marginal_mean = np.einsum('n,ndb->db', weights, population['marginals'])
    bands = np.array([[weighted_quantiles(population['marginals'][:, d, b], weights)
                       for b in range(marginal_mean.shape[1])] for d in range(3)])
    summary = {'joint': {name: distance_check(population[name], truth[name], weights, chain_shape)
                         for name in ('intrinsic', 'detected')},
               'marginal_distances': jensenshannon(marginal_mean, truth['marginals'], axis=1, base=2).tolist(),
               'hidden': moment_check(population['hidden'], weights, chain_shape),
               'hidden_quantiles': weighted_quantiles(population['hidden'], weights).tolist(),
               'alpha_quantiles': weighted_quantiles(population['alpha'], weights).tolist(),
               'rate': rate_summary(population['alpha'], weights, count, injected_rate, chain_shape),
               'prediction_precision': moment_check(np.exp(prediction['log_integrals']-score).sum(1), weights, chain_shape)}
    return summary, {'log_prediction': score, 'marginal_mean': marginal_mean, 'marginal_quantiles': bands,
                     'intrinsic_mean_cells': weights @ population['intrinsic'],
                     'detected_mean_cells': weights @ population['detected']}


def numerical_summary(primary, fine, prediction, fine_prediction, weights, indices, truth, fine_truth):
    selected_weights = normalized_weights(weights[indices], len(indices))
    row = {}
    for name in ('intrinsic', 'detected'):
        first = float(jensenshannon(selected_weights @ primary[name][indices], truth[name], base=2))
        second = float(jensenshannon(selected_weights @ fine[name], fine_truth[name], base=2))
        row[name+'_distance_change'] = second-first
    row['hidden_change'] = float(selected_weights @ (fine['hidden']-primary['hidden'][indices]))
    row['maximum_relative_alpha_change'] = float(np.max(abs(fine['alpha']/primary['alpha'][indices]-1)))
    first = weighted_prediction(prediction['log_integrals'][indices], selected_weights)
    second = weighted_prediction(fine_prediction['log_integrals'], selected_weights)
    full = weighted_prediction(prediction['log_integrals'], weights)
    row['prediction_rmse_change'] = float(np.sqrt(np.mean((second-first)**2)))
    row['subset_vs_full_prediction_rmse'] = float(np.sqrt(np.mean((first-full)**2)))
    return row, first, second


def comparison_decision(config, numerical, comparisons, max_rhat):
    baselines = {'vi_' + relative.rsplit('_s', 1)[1] for relative in config['baselines']}
    result = {'all_comparisons_present': set(numerical) == baselines | {'nuts'} and set(comparisons) == baselines,
              'population_integration_passed': all(abs(r[k]) < config['maximum_joint_distance_change']
                 for r in numerical.values() for k in ('intrinsic_distance_change', 'detected_distance_change')),
              'hidden_mass_integration_passed': all(abs(r['hidden_change']) < config['maximum_hidden_mass_change'] for r in numerical.values()),
              'prediction_integration_passed': all(r['prediction_rmse_change'] < config['maximum_prediction_rmse_change'] for r in numerical.values()),
              'paired_score_integration_passed': all(abs(r['fine_paired_score_change']) < config['maximum_paired_score_change'] for r in comparisons.values()),
              'derived_mixing_passed': bool(max_rhat < config['maximum_derived_rhat'])}
    result['eligible_for_scientific_comparison'] = all(result.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lisa_scientific_comparison.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    path = ROOT / args.config
    config = json.loads(path.read_text())
    accepted, sources = verified_sources(config, path)
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed scientific comparison is retained')
    costs = {}
    def obtain(label, builder, extra=None):
        product, metadata = cached(output, label, dict(sources, **(extra or {})), builder)
        costs[label] = metadata['seconds']
        return product
    truth = obtain('truth', lambda: truth_product(config))
    fine_truth = obtain('truth_fine', lambda: truth_product(config, True))
    observations = obtain('test_observations', lambda: test_observations(config))
    observation_hash = {'test_observations': sha256(output / 'test_observations.npz')}
    events = obtain('events', lambda: event_samples(config, observations), observation_hash)
    fine_events = obtain('events_fine', lambda: event_samples(config, observations, True), observation_hash)
    original = json.loads((ROOT / config['baselines'][1] / 'summary.json').read_text())['configuration']
    model = sampling_model(original)
    model.lambda_det_mc_samples = accepted['configuration']['primary_selection_points']
    model.selection_mc_seed = accepted['configuration']['primary_selection_seed']
    posterior = np.load(ROOT / config['source'] / 'posterior.npz')
    weights = np.load(ROOT / config['source'] / 'weights.npz')['weights']
    samples = {'nuts': posterior['latent'].reshape(-1, 20)}
    sample_weights = {'nuts': weights.ravel()}
    for index, relative in enumerate(config['baselines']):
        label = 'vi_' + relative.rsplit('_s', 1)[1]
        samples[label] = obtain(label+'_draws', lambda i=index: baseline_draws(config, i))['latent']
        sample_weights[label] = np.full(config['baseline_draws'], 1/config['baseline_draws'])
    if config['primary_baseline'] not in samples:
        raise ValueError('Primary baseline must be declared among the frozen fits')
    count = len(np.load(ROOT / config['baselines'][1] / 'catalog.npz')['data'])
    injected_rate = float(np.load(ROOT / config['baselines'][1] / 'catalog.npz')['mean_intrinsic'])
    rows, arrays, numerical, coarse_subset, fine_subset = {}, {}, {}, {}, {}
    for label, latent in samples.items():
        weight = sample_weights[label]
        chain_shape = weights.shape if label == 'nuts' else None
        population = obtain(label+'_population', lambda: population_product(model, latent, config), {'latent': array_hash(latent)})
        if label == 'nuts':
            expected = posterior['features'][:, :, posterior['feature_names'].tolist().index('alpha')].ravel()
            np.testing.assert_allclose(population['alpha'], expected, rtol=0, atol=1e-12)
        prediction = obtain(label+'_prediction', lambda: prediction_product(model, latent, population['alpha'], events['data']),
                            {'population': sha256(output / (label+'_population.npz')), 'events': sha256(output / 'events.npz')})
        rows[label], derived = scientific_summary(population, prediction, weight, truth, chain_shape, count, injected_rate)
        arrays.update({label+'_'+name: value for name, value in derived.items()})
        if chain_shape:
            indices = np.concatenate([c * weights.shape[1] + np.linspace(0, weights.shape[1]-1, config['verification_draws_per_chain'], dtype=int)
                                      for c in range(weights.shape[0])])
        else:
            indices = np.linspace(0, len(latent)-1, config['verification_baseline_draws'], dtype=int)
        arrays[label+'_verification_indices'] = indices
        fine_population = obtain(label+'_population_fine', lambda: population_product(model, latent[indices], config, True),
                                 {'latent': array_hash(latent[indices])})
        fine_prediction = obtain(label+'_prediction_fine', lambda: prediction_product(model, latent[indices], fine_population['alpha'], fine_events['data']),
                                 {'latent': array_hash(latent[indices]), 'population': sha256(output / (label+'_population_fine.npz')),
                                  'events': sha256(output / 'events_fine.npz')})
        numerical[label], coarse_subset[label], fine_subset[label] = numerical_summary(
            population, fine_population, prediction, fine_prediction, weight, indices, truth, fine_truth)
        arrays[label+'_coarse_subset_prediction'] = coarse_subset[label]
        arrays[label+'_fine_subset_prediction'] = fine_subset[label]
    comparisons = {}
    for label in samples:
        if label == 'nuts':
            continue
        row = paired_scores(arrays['nuts_log_prediction'], arrays[label+'_log_prediction'])
        row['posterior_integration_mcse'] = float(np.hypot(rows['nuts']['prediction_precision']['mcse'], rows[label]['prediction_precision']['mcse']))
        row['fine_paired_score_change'] = float((fine_subset['nuts']-fine_subset[label]).sum() - (coarse_subset['nuts']-coarse_subset[label]).sum())
        row['coarse_subset_total_gain'] = float((coarse_subset['nuts']-coarse_subset[label]).sum())
        row['fine_subset_total_gain'] = float((fine_subset['nuts']-fine_subset[label]).sum())
        comparisons[label] = row
    rhat_rows = [rows['nuts']['joint'][k] for k in ('intrinsic', 'detected')]
    rhat_rows += [rows['nuts']['hidden'], rows['nuts']['prediction_precision'], rows['nuts']['rate']['mean_precision'], rows['nuts']['rate']['cdf_at_truth']]
    rhat_rows += rows['nuts']['rate']['cdf_precision_at_quantiles']
    max_rhat = max(r['influence_rhat'] for r in rhat_rows)
    decision = comparison_decision(config, numerical, comparisons, max_rhat)
    np.savez_compressed(output / 'derived.npz', **arrays)
    result = {'configuration': config, 'source_sha256': sources, 'results': rows, 'comparisons': comparisons,
              'numerical': numerical, 'decision': decision, 'maximum_derived_rhat': max_rhat,
              'truth': {'alpha': float(truth['alpha']), 'hidden': float(truth['hidden']), 'intrinsic_rate': injected_rate,
                        'detected_truth_refinement_distance': float(jensenshannon(truth['detected'], fine_truth['detected'], base=2))},
              'seconds': costs, 'total_seconds': sum(costs.values()), 'runtime': runtime_metadata(),
              'product_sha256': {p.name: sha256(p) for p in output.glob('*.npz')},
              'scope': 'One fixed training catalog, three frozen variational fits, and independently generated detected test events. Scores omit common event evidence. Test-event SE and posterior-integration MCSE are reported separately. Rate is intrinsic expected count in one unit exposure. No repeated-catalog coverage claim.'}
    write_json(output / 'summary.json', result)
    print(json.dumps({'decision': decision, 'comparisons': comparisons}, indent=2), flush=True)


if __name__ == '__main__':
    main()
