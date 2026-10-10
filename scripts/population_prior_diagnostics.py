"""Compare population priors and fitted posteriors on scientific functionals."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Beta
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from build_lisa_analysis_assets import LISASNR
from lvk_injections import covered_mass_pairs
from analyze_lvk_population import ROOT, estimator, load_inputs
from analysis_results import configuration, lisa_baselines
from summarize_lvk_marginals import marginal_distributions, piecewise_mass_icdf
from summarize_analysis_populations import components
from validate_population_morphologies import build_model


def draw_prior_latents(model, count, seed):
    """Draw the model prior, rather than the standard Gaussian flow base."""
    allowed = {'weights', 'k_marginals', 'k_cholesky', 'tnormal_marginals',
               'tnormal_cholesky', 'axis_spline_kumaraswamy_marginals',
               'axis_spline_kumaraswamy_cholesky'}
    unsupported = {key for key, size in model.dims.items() if size and key not in allowed}
    if unsupported or model.m1_roughness_lambda != 0:
        raise ValueError('Prior sampler does not support this family or a tilted roughness prior')
    if str(model.device) != 'cpu':
        raise ValueError('This reproducible diagnostic uses CPU sampling')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        values = torch.randn(count, model.latent_dim, dtype=torch.float64)
        offset = 0
        for name, size in model.dims.items():
            block = values[:, offset:offset + size]
            offset += size
            if not size:
                continue
            if name.endswith('_cholesky'):
                n_components = size // model.cholesky_params_dim
                # Independent partial correlations give the exact LKJ measure.
                # The installed Torch 2.5.1 onion sampler fails exchangeability.
                columns = torch.tril_indices(model.data_dim, model.data_dim, -1)[1]
                shape = 1.5 + .5 * (model.data_dim - columns.to(torch.float64) - 2)
                partial = 2 * Beta(shape, shape).sample((count, n_components)) - 1
                block.copy_(partial.atanh().reshape(count, size))
            elif name == 'tnormal_marginals':
                shaped = block.reshape(count, -1, model.data_dim, 2)
                uniform = torch.rand(shaped[..., 0].shape, dtype=torch.float64)
                shaped[..., 0] = torch.logit(uniform.clamp_min(torch.finfo(torch.float64).tiny))
    return values.numpy()


@torch.no_grad()
def transported_population(model, latent, normal, kind):
    if kind == 'lisa':
        weights, marginal, cholesky = components(model, latent)
        probability = ndtr(np.einsum('nj,kij->nki', normal, cholesky.numpy()))
        points = marginal.icdf(torch.tensor(np.clip(probability, 1e-15, 1 - 1e-15))).numpy()
        return weights, points
    params = model._unflatten_latent_vector(torch.tensor(latent, dtype=torch.float64))
    weights = torch.softmax(params['weights_logits'], -1).numpy()
    distributions, cholesky = marginal_distributions(model, params)
    probability = np.clip(ndtr(np.einsum('nj,kij->nki', normal, cholesky.numpy())), 1e-15, 1 - 1e-15)
    points = np.stack([
        piecewise_mass_icdf(distributions[0], probability[:, :, 0]),
        distributions[1].icdf(torch.tensor(probability[:, :, 1])).numpy(),
        distributions[2].icdf(torch.tensor(probability[:, :, 2])).numpy(),
    ], axis=-1)
    return weights, points


def weighted_means(values, component_weights, count):
    return np.mean(values[:count], axis=0) @ component_weights


@torch.no_grad()
def scientific_features(model, latent, normal, kind, snr=None):
    weights, points = transported_population(model, latent, normal, kind)
    if kind == 'lisa':
        pdet = ndtr(snr(points.reshape(-1, 3)).reshape(points.shape[:2]) - 8.)
        indicators = {'alpha': pdet, 'weak_detection_fraction': pdet < .1,
                      'mass_above_1e8': points[:, :, 0] > .75}
        support = np.ones(points.shape[:2], dtype=bool)
    else:
        physical = model.min_vals.numpy() + points * (model.max_vals - model.min_vals).numpy()
        mass = 10**physical[:, :, 0]
        secondary = 3 + (mass - 3) * physical[:, :, 2]
        support = covered_mass_pairs(mass, secondary)
        indicators = {'mass_below_5': mass < 5, 'mass_below_10': mass < 10,
                      'mass_above_50': mass > 50, 'z_above_1': physical[:, :, 1] > 1,
                      'q_below_half': secondary / mass < .5}
    fine, changes = {}, {}
    coarse_count = len(normal) // 4
    coarse_a = weighted_means(support, weights, coarse_count)
    fine_a = weighted_means(support, weights, len(normal))
    if min(coarse_a, fine_a) <= 0:
        raise ValueError('Transported population has no samples in the covered domain')
    for name, values in indicators.items():
        coarse = weighted_means(values * support, weights, coarse_count) / coarse_a
        fine[name] = float(weighted_means(values * support, weights, len(normal)) / fine_a)
        changes[name] = float(abs(fine[name] - coarse))
    fine['domain_probability'] = float(fine_a)
    changes['domain_probability'] = float(abs(fine_a - coarse_a))
    if kind == 'lvk':
        params = model._unflatten_latent_vector(torch.tensor(latent, dtype=torch.float32))
        raw_alpha = float(model._selection_fraction_for_params(params))
        log_weights = model._last_selection_log_weights.detach().double()
        log_weights = log_weights - log_weights.max()
        injection_weights = log_weights.exp()
        fine['alpha'] = raw_alpha / fine_a
        fine['injection_ess'] = float(injection_weights.sum().square() / injection_weights.square().sum())
        changes['alpha_domain_normalization_only'] = float(abs(raw_alpha / fine_a - raw_alpha / coarse_a))
    return fine, changes


def feature_summary(values):
    return {name: {'quantiles_05_50_95': np.quantile(array, [.05, .5, .95]).tolist(),
                   'mean': float(np.mean(array)), 'sd': float(np.std(array))}
            for name, array in values.items()}


def transpose_records(records):
    return {name: np.asarray([record[name] for record in records]) for name in records[0]}


def analyze(directory, output, prior_draws, power, seed):
    started = time.perf_counter()
    source = json.loads((directory / 'summary.json').read_text())
    config = source['configuration']
    kind = 'lisa' if config.get('detector') == 'lisa' else 'lvk'
    if kind == 'lisa':
        model = build_model(config['family'], config['components'], 4096, seed)
        snr = LISASNR()
    else:
        args = SimpleNamespace(**config)
        arrays = load_inputs(args.spin, args.posterior_samples, args.injections, args.mmax)
        args.full_injections = True
        model = estimator(args, arrays)
        snr = None
    prior = draw_prior_latents(model, prior_draws, seed)
    posterior = np.load(directory / 'posterior.npz')['latent']
    normal = ndtri(qmc.Sobol(3, scramble=True, seed=seed + 1).random_base2(power))
    all_values, all_changes = {}, {}
    for label, latents in (('prior', prior), ('posterior', posterior)):
        records, changes = [], []
        for latent in latents:
            values, change = scientific_features(model, latent, normal, kind, snr)
            records.append(values)
            changes.append(change)
        all_values[label] = transpose_records(records)
        all_changes[label] = transpose_records(changes)
        print(directory.name, label, len(latents), 'draws complete', flush=True)
    summary = {label: feature_summary(values) for label, values in all_values.items()}
    numerical = {label: {name: np.quantile(values, [.5, .9, .99, 1]).tolist()
                         for name, values in changes.items()}
                 for label, changes in all_changes.items()}
    widths = {}
    for name in all_values['prior']:
        prior_width = np.diff(np.quantile(all_values['prior'][name], [.05, .95]))[0]
        post_width = np.diff(np.quantile(all_values['posterior'][name], [.05, .95]))[0]
        widths[name] = float(post_width / prior_width) if prior_width > 0 else None
    hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (directory / 'summary.json', directory / 'posterior.npz',
                           ROOT / 'src/gwpopulation/core.py', ROOT / 'scripts/population_prior_diagnostics.py')}
    result = {'source': directory.relative_to(ROOT).as_posix(), 'kind': kind,
              'prior_draws': len(prior), 'posterior_draws': len(posterior),
              'points_per_component': len(normal), 'seed': seed,
              'summary': summary, 'posterior_to_prior_90_interval_width': widths,
              'coarse_to_fine_absolute_change_50_90_99_max': numerical,
              'selection_note': ('LISA sensitivity averaged over transported population points' if kind == 'lisa'
                                 else 'Full recovered injection campaign, divided by the joint-domain probability; ESS is reported without removing low-ESS draws'),
              'interpretation': 'Intervals describe prior and conditional posterior distributions, not repeated-catalog coverage.',
              'sha256': hashes, 'seconds': time.perf_counter() - started}
    target = output / directory.name
    target.mkdir(parents=True, exist_ok=True)
    (target / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    products = {'prior_latent': prior, 'posterior_latent': posterior}
    for label, values in all_values.items():
        products.update({f'{label}_{name}': value for name, value in values.items()})
    np.savez_compressed(target / 'functionals.npz', **products)
    print(json.dumps({'source': result['source'], 'seconds': result['seconds'], 'summary': summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='*')
    parser.add_argument('--prior-draws', type=int, default=512)
    parser.add_argument('--power', type=int, default=14)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--output', default='results/analysis/prior_diagnostics')
    args = parser.parse_args()
    if args.prior_draws < 2 or args.power < 4:
        raise ValueError('At least two prior draws and 16 population points are required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = configuration()
    directories = args.directories or sorted(set(lisa_baselines().values())
        | set(config['lisa_sensitivity']) | {entry['directory'] for entry in config['lvk_sensitivity']})
    for directory in directories:
        analyze(ROOT / directory, ROOT / args.output, args.prior_draws, args.power, args.seed)


if __name__ == '__main__':
    main()
