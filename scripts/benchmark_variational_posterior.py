"""Compare the same three-dimensional copula hyperposterior using VI and NUTS.

The complete-detection example isolates posterior approximation from numerical
selection integration. Event centers are independent Gaussian noise draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import psutil
import pyro
import torch
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from scipy.special import ndtr
from scipy.stats import truncnorm, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated


def build_model():
    model = BayesianMixtureEstimator_NF_Correlated(
        data_dim=3, num_corr_kumaraswamy=1, num_corr_tnormal=0,
        num_corr_tpareto=0, hidden_dim_flow=32, num_flow_transforms=6,
        physical_bounds=([0., 0., 0.], [1., 1., 1.]), device='cpu')
    model.obs_neff_threshold = 0
    model.selection_neff_multiplier = 0
    model.q_monotonicity_lambda = 0
    model.kumaraswamy_log_param_clip = None
    return model


def mock_catalog(seed, n_events=80, n_samples=128):
    rng = np.random.default_rng(seed)
    a, b = np.array([2., 3., 2.]), np.array([4., 2., 1.5])
    correlation = np.array([[1., .55, -.2], [.55, 1., .15], [-.2, .15, 1.]])
    u = ndtr(rng.multivariate_normal(np.zeros(3), correlation, size=n_events))
    truth = (1 - (1 - u)**(1 / b))**(1 / a)
    sigma = np.array([.035, .045, .035])
    centers = truth + rng.normal(size=truth.shape) * sigma
    posterior = truncnorm.rvs(
        -centers[:, None, :] / sigma, (1 - centers[:, None, :]) / sigma,
        loc=centers[:, None, :], scale=sigma,
        size=(n_events, n_samples, 3), random_state=rng)
    data = torch.tensor(np.concatenate([posterior, np.ones((n_events, n_samples, 1))], axis=-1),
                        dtype=torch.float32)
    return data, {'truth': truth, 'centers': centers, 'sigma': sigma,
                  'a': a, 'b': b, 'correlation': correlation}


def parameters_from_vi(model, n_draws):
    with torch.no_grad():
        latent = model.flow_dist.sample((n_draws,))
        params = model._unflatten_latent_vector(latent)
        chol = model._build_cholesky_from_unconstrained(params['k_cholesky_params'], 3, 'cpu')
    return {'log_k_a': params['log_k_a'].numpy(), 'log_k_b': params['log_k_b'].numpy(),
            'k_L_cholesky': chol.numpy(), 'weights_logits': params['weights_logits'].numpy()}


def scientific_parameters(samples):
    a = np.exp(samples['log_k_a'][..., 0, :]) + .0001
    b = np.exp(samples['log_k_b'][..., 0, :]) + .0001
    chol = samples['k_L_cholesky'][..., 0, :, :]
    correlation = chol @ np.swapaxes(chol, -1, -2)
    return np.concatenate([a, b, correlation[..., [1, 2, 2], [0, 0, 1]]], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/analysis/vi_benchmark')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--particles', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--reuse-nuts', action='store_true')
    parser.add_argument('--nuts-source', help='Reuse independently sampled chains for the identical catalog and target')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    data, truth = mock_catalog(args.seed)
    np.savez_compressed(output / 'catalog.npz', posterior=data.numpy(), **truth)
    target_hash = hashlib.sha256((ROOT / 'src/gwpopulation/core.py').read_bytes()).hexdigest()
    metadata = dict(vars(args), python=platform.python_version(), torch=torch.__version__,
                    pyro=pyro.__version__, processor=platform.processor(), threads=1,
                    n_events=len(data), samples_per_event=data.shape[1],
                    selection='complete detection', hyperparameters=10,
                    core_sha256=target_hash)
    if args.nuts_source:
        source = ROOT / args.nuts_source
        previous = json.loads((source / 'configuration.json').read_text())
        for key in ('seed', 'warmup', 'samples', 'chains'):
            if previous[key] != metadata[key]:
                raise ValueError(f'MCMC source configuration differs: {key}')
        if not np.array_equal(np.load(source / 'catalog.npz')['posterior'], data.numpy()):
            raise ValueError('MCMC source uses a different event catalog')
        for chain in range(args.chains):
            for extension in ('npz', 'json'):
                shutil.copy2(source / f'nuts_chain_{chain}.{extension}', output)
        metadata['mcmc_source_core_sha256'] = previous['core_sha256']
        metadata['mcmc_source'] = str(source.relative_to(ROOT))
    if args.reuse_nuts:
        previous = json.loads((output / 'configuration.json').read_text())
        for key in ('seed', 'warmup', 'samples', 'chains', 'core_sha256'):
            if previous[key] != metadata[key]:
                raise ValueError(f'Cached MCMC configuration differs: {key}')
    (output / 'configuration.json').write_text(json.dumps(metadata, indent=2) + '\n')
    chains, nuts_times, divergences = [], [], []
    for chain in range(args.chains):
        cache = output / f'nuts_chain_{chain}.npz'
        diagnostic_cache = output / f'nuts_chain_{chain}.json'
        if args.reuse_nuts or args.nuts_source:
            chains.append(dict(np.load(cache)))
            diagnostic = json.loads(diagnostic_cache.read_text())
            nuts_times.append(diagnostic['seconds'])
            divergences.append(diagnostic['divergences'])
            continue
        pyro.set_rng_seed(args.seed + 100 + chain)
        pyro.clear_param_store()
        model = build_model()
        kernel = NUTS(model._pyro_model, target_accept_prob=.85, max_tree_depth=7)
        sampler = MCMC(kernel, num_samples=args.samples, warmup_steps=args.warmup,
                       num_chains=1, disable_progbar=True)
        start = time.perf_counter()
        sampler.run(data)
        nuts_times.append(time.perf_counter() - start)
        draws = {k: v.numpy() for k, v in sampler.get_samples().items()}
        np.savez_compressed(output / f'nuts_chain_{chain}.npz', **draws)
        chains.append(draws)
        divergences.append(sampler.diagnostics()['divergences']['chain 0'])
        diagnostic_cache.write_text(json.dumps({'seconds': nuts_times[-1],
                                               'divergences': divergences[-1]}, indent=2) + '\n')
        print(f'NUTS chain {chain}: {nuts_times[-1]:.1f} s, {len(divergences[-1])} divergences', flush=True)
    stacked = {key: np.stack([chain[key] for chain in chains]) for key in chains[0]}
    nuts_features = scientific_parameters(stacked)
    inference_data = az.from_dict(posterior={'parameters': nuts_features})
    names = ['a_mass', 'a_redshift', 'a_ratio', 'b_mass', 'b_redshift', 'b_ratio',
             'rho_mass_redshift', 'rho_mass_ratio', 'rho_redshift_ratio']
    diagnostics = {
        'rhat': az.rhat(inference_data, method='rank')['parameters'].values.tolist() if args.chains > 1 else None,
        'ess_bulk': az.ess(inference_data, method='bulk')['parameters'].values.tolist(),
        'ess_tail': az.ess(inference_data, method='tail')['parameters'].values.tolist(),
        'divergences_per_chain': [len(x) for x in divergences],
        'seconds_per_chain': nuts_times,
    }
    summaries = []
    for run in range(args.starts):
        pyro.set_rng_seed(args.seed + 200 + run)
        pyro.clear_param_store()
        model = build_model()
        optimizer = pyro.optim.ClippedAdam({'lr': .001, 'clip_norm': 5., 'lrd': .9997})
        svi = SVI(model._pyro_model, model._pyro_guide, optimizer,
                  loss=Trace_ELBO(num_particles=args.particles))
        losses = []
        start = time.perf_counter()
        for step in range(args.steps):
            loss = svi.step(data)
            if not np.isfinite(loss):
                raise RuntimeError(f'Nonfinite VI objective in start {run} step {step}')
            losses.append(loss)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        vi_draws = parameters_from_vi(model, 10000)
        draw_seconds = time.perf_counter() - start
        vi_features = scientific_parameters(vi_draws)
        np.savez_compressed(output / f'vi_start_{run}.npz', **vi_draws, losses=np.asarray(losses))
        torch.save({'state': model.flow_transforms.state_dict(), 'seed': args.seed + 200 + run},
                   output / f'vi_start_{run}.pt')
        summary = {'start': run, 'train_seconds': train_seconds, 'draw_seconds_10000': draw_seconds,
                   'last_500_mean_negative_elbo': float(np.mean(losses[-500:])), 'parameters': []}
        for i, name in enumerate(names):
            reference, approx = nuts_features[..., i].ravel(), vi_features[:, i]
            lo, hi = np.quantile(reference, [.05, .95])
            summary['parameters'].append({
                'name': name, 'nuts_quantiles': np.quantile(reference, [.05, .5, .95]).tolist(),
                'vi_quantiles': np.quantile(approx, [.05, .5, .95]).tolist(),
                'sd_ratio': float(approx.std() / reference.std()),
                'mean_offset_in_nuts_sd': float((approx.mean() - reference.mean()) / reference.std()),
                'wasserstein_in_nuts_sd': float(wasserstein_distance(reference, approx) / reference.std()),
                'vi_probability_below_nuts_05': float(np.mean(approx < lo)),
                'vi_probability_above_nuts_95': float(np.mean(approx > hi)),
            })
        summaries.append(summary)
        print(f'VI start {run}: {train_seconds:.1f} s; SD ratios '
              f'{[round(p["sd_ratio"], 2) for p in summary["parameters"]]}', flush=True)
    memory = psutil.Process().memory_info()
    result = {'configuration': metadata, 'names': names, 'nuts': diagnostics, 'vi': summaries,
              'peak_working_set_bytes': getattr(memory, 'peak_wset', memory.rss)}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
