"""Independent inference and held-out prediction for a selected noisy mixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from types import MethodType

import arviz as az
import numpy as np
import psutil
import pyro
import torch
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide.initialization import init_to_value
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import qmc, truncnorm

from benchmark_variational_posterior import parameters_from_vi
from initialize_population_guide import initialize_from_events
from analyze_lvk_population import ROOT
from validate_population_morphologies import build_model
from gwpopulation.core import SafeKumaraswamy


def protocol():
    return json.loads((ROOT / 'configs/model_comparisons.json').read_text())['selected_mixture_benchmark']


def runtime_metadata():
    memory = psutil.Process().memory_info()
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'pyro': pyro.__version__, 'processor': platform.processor(),
            'threads': torch.get_num_threads(),
            'peak_working_set_bytes': getattr(memory, 'peak_wset', memory.rss)}


def detection_probability(points):
    return .05 + .95 * points[..., 0]**2


def kumaraswamy_moment(a, b, order):
    return (b.log() + torch.lgamma(1 + order / a) + torch.lgamma(b)
            - torch.lgamma(1 + order / a + b)).exp()


def exact_selection(params):
    a = params['log_k_a'].exp() + .0001
    b = params['log_k_b'].exp() + .0001
    second = kumaraswamy_moment(a[..., 0], b[..., 0], 2)
    return (torch.softmax(params['weights_logits'], -1) * (.05 + .95 * second)).sum(-1)


def independent_log_probability(self, marginal_dist, cholesky, data_norm):
    return marginal_dist.log_prob(data_norm.unsqueeze(2)).sum(-1)


def selected_model(independent=False):
    model = build_model('kumaraswamy', 2, 64, 17)
    model.selection_function = lambda points: detection_probability(points)
    model._selection_fraction_for_params = exact_selection
    if independent:
        # Correlation sites remain independent prior auxiliaries and integrate
        # to one; the physical component densities have identity copulas.
        model._copula_log_prob = MethodType(independent_log_probability, model)
    return model


def sample_selected_truth(count, seed):
    config = protocol()
    rng = np.random.default_rng(seed)
    a, b = np.asarray(config['kumaraswamy_a']), np.asarray(config['kumaraswamy_b'])
    chol = np.linalg.cholesky(config['correlations'])
    accepted = []
    while sum(map(len, accepted)) < count:
        indices = rng.choice(2, size=512, p=config['component_weights'])
        gaussian = np.einsum('nij,nj->ni', chol[indices], rng.normal(size=(512, 3)))
        uniform = ndtr(gaussian)
        points = (1 - (1 - uniform)**(1 / b[indices]))**(1 / a[indices])
        keep = rng.uniform(size=len(points)) < detection_probability(points)
        accepted.append(points[keep])
    return np.concatenate(accepted)[:count]


def event_posterior(measurement, sample_count, rng):
    sigma = np.asarray(protocol()['measurement_standard_deviations'])
    accepted = []
    while sum(map(len, accepted)) < sample_count:
        proposal = truncnorm.rvs(-measurement / sigma, (1 - measurement) / sigma,
                                loc=measurement, scale=sigma,
                                size=(max(256, 2 * sample_count), 3), random_state=rng)
        keep = rng.uniform(size=len(proposal)) < detection_probability(proposal)
        accepted.append(proposal[keep])
    return np.concatenate(accepted)[:sample_count]


def mock_catalog(count, seed, sample_count):
    truth = sample_selected_truth(count, seed)
    rng = np.random.default_rng(seed + 1000)
    sigma = np.asarray(protocol()['measurement_standard_deviations'])
    observed = truth + rng.normal(size=truth.shape) * sigma
    posterior = np.asarray([event_posterior(y, sample_count, rng) for y in observed])
    data = np.concatenate([posterior, np.ones((*posterior.shape[:2], 1))], axis=-1)
    return {'data': data, 'truth': truth, 'observed': observed, 'sigma': sigma}


def initial_sites(model, data, seed, jitter=.15):
    initialization = initialize_from_events(model, data, seed)
    raw = torch.tensor(initialization['initial_latent_location'], dtype=data.dtype)
    generator = torch.Generator().manual_seed(seed + 10000)
    raw = raw + jitter * torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
    params = model._unflatten_latent_vector(raw, apply_numerical_limits=False)
    return {'weights_logits': params['weights_logits'], 'log_k_a': params['log_k_a'],
            'log_k_b': params['log_k_b'],
            'k_L_cholesky': model._build_cholesky_from_unconstrained(params['k_cholesky_params'], 3, 'cpu')}


@torch.no_grad()
def feature_draws(samples, independent=False, power=10):
    leading = np.shape(samples['log_k_a'])[:-2]
    a = torch.tensor(np.exp(samples['log_k_a']).reshape(-1, 2, 3) + .0001)
    b = torch.tensor(np.exp(samples['log_k_b']).reshape(-1, 2, 3) + .0001)
    weights = torch.softmax(torch.tensor(samples['weights_logits']).reshape(-1, 2), -1)
    chol = np.asarray(samples['k_L_cholesky']).reshape(-1, 2, 3, 3)
    mean = (weights[..., None] * kumaraswamy_moment(a, b, 1)).sum(1)
    second = (weights[..., None] * kumaraswamy_moment(a, b, 2)).sum(1)
    variance = second - mean.square()
    marginal = SafeKumaraswamy(a, b)
    low_mass = (weights * marginal.cdf(torch.tensor(.25))[..., 0]).sum(-1)
    high_mass = (weights * (1 - marginal.cdf(torch.tensor(.8))[..., 0])).sum(-1)
    alpha = .05 + .95 * second[:, 0]
    normal = ndtri(qmc.Sobol(3, scramble=True, seed=54310).random_base2(power))
    covariances = []
    for index in range(len(a)):
        matrix = np.broadcast_to(np.eye(3), (2, 3, 3)) if independent else chol[index]
        probability = np.clip(ndtr(np.einsum('nj,kij->nki', normal, matrix)), 1e-14, 1-1e-14)
        points = SafeKumaraswamy(a[index], b[index]).icdf(torch.tensor(probability)).numpy()
        products = np.stack([points[..., 0] * points[..., 1], points[..., 0] * points[..., 2],
                             points[..., 1] * points[..., 2]], axis=-1).mean(0)
        cross = weights[index].numpy() @ products
        m = mean[index].numpy()
        covariances.append(cross - np.array([m[0]*m[1], m[0]*m[2], m[1]*m[2]]))
    values = np.column_stack([mean.numpy(), variance.numpy(), alpha.numpy(), low_mass.numpy(),
                              high_mass.numpy(), covariances])
    names = ['mean_mass', 'mean_redshift', 'mean_ratio', 'variance_mass', 'variance_redshift',
             'variance_ratio', 'alpha', 'mass_below_quarter', 'mass_above_0p8',
             'cov_mass_redshift', 'cov_mass_ratio', 'cov_redshift_ratio']
    return names, values.reshape(*leading, len(names))


@torch.no_grad()
def log_predictive(model, samples, data, draw_limit=512):
    total = len(samples['weights_logits'])
    indices = np.linspace(0, total - 1, min(total, draw_limit), dtype=int)
    values = []
    for index in indices:
        params = {key: torch.tensor(np.asarray(value)[index], dtype=data.dtype) for key, value in samples.items()}
        log_density = model._log_pdf_phys_from_params(params, data[..., :3].reshape(-1, 3)).reshape(*data.shape[:2])
        event_log_integrals = torch.logsumexp(log_density - data[..., 3].log(), dim=-1) - np.log(data.shape[1])
        values.append((event_log_integrals - exact_selection(params).log()).numpy())
    return logsumexp(values, axis=0) - np.log(len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/analysis/selected_mixture')
    parser.add_argument('--mode', choices=('nuts', 'vi', 'independent', 'summarize'), default='nuts')
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--samples', type=int, default=800)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--chain-start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--starts', type=int, default=2)
    parser.add_argument('--max-tree-depth', type=int, default=8)
    parser.add_argument('--particles', type=int, default=4)
    parser.add_argument('--jit', action='store_true', help='Trace the same NUTS potential after equivalence checks')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = protocol()
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        'src/gwpopulation/core.py', 'scripts/benchmark_selected_mixture.py',
        'scripts/initialize_population_guide.py', 'configs/model_comparisons.json')}
    signature = {'protocol': config, 'source_sha256': source_hashes}
    signature_path = output / 'target.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Existing output belongs to a different target or implementation')
    signature_path.write_text(json.dumps(signature, indent=2) + '\n')
    catalogs = {}
    for label, count, seed in (('train', config['selected_training_events'], config['catalog_seed']),
                                ('test', config['selected_test_events'], config['test_seed'])):
        path = output / f'{label}_catalog.npz'
        if not path.exists():
            np.savez_compressed(path, **mock_catalog(count, seed, config['posterior_samples_per_event']))
        catalogs[label] = torch.tensor(np.load(path)['data'])
    data, test = catalogs['train'], catalogs['test']
    if args.mode == 'nuts':
        for chain in range(args.chain_start, args.chain_start + args.chains):
            path = output / f'nuts_chain_{chain}.npz'
            if path.exists():
                raise FileExistsError(f'Completed chain is retained: {path}')
            pyro.clear_param_store()
            pyro.set_rng_seed(671200 + chain)
            model = selected_model()
            initial_jitter = (.15, .35, .65, 1.)[chain % 4]
            values = initial_sites(model, data, 671300 + chain, initial_jitter)
            kernel = NUTS(model._pyro_model, init_strategy=init_to_value(values=values),
                          target_accept_prob=.9, max_tree_depth=args.max_tree_depth, full_mass=True,
                          jit_compile=args.jit, ignore_jit_warnings=True)
            def progress(kernel, samples, stage, iteration):
                if (iteration + 1) % 100 == 0:
                    print('nuts', chain, stage, iteration + 1, flush=True)
            sampler = MCMC(kernel, warmup_steps=args.warmup, num_samples=args.samples,
                           num_chains=1, disable_progbar=True, hook_fn=progress)
            start = time.perf_counter()
            sampler.run(data)
            seconds = time.perf_counter() - start
            draws = {key: value.numpy() for key, value in sampler.get_samples().items()}
            np.savez_compressed(path, **draws)
            sampler_diagnostics = sampler.diagnostics()
            diagnostic = {'chain': chain, 'seconds': seconds, 'configuration': vars(args),
                          'divergences': sampler_diagnostics['divergences']['chain 0'],
                          'acceptance_rate': sampler_diagnostics['acceptance rate']['chain 0'],
                          'step_size': float(kernel.step_size),
                          'initial_jitter_standard_deviation': initial_jitter,
                          'runtime': runtime_metadata()}
            path.with_suffix('.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
            print(json.dumps(diagnostic), flush=True)
    elif args.mode in ('vi', 'independent'):
        independent = args.mode == 'independent'
        for run in range(args.starts):
            path = output / f'{args.mode}_start_{run}.npz'
            if path.exists():
                raise FileExistsError(f'Completed optimization is retained: {path}')
            pyro.clear_param_store()
            pyro.set_rng_seed(671400 + run)
            model = selected_model(independent)
            initialization = initialize_from_events(model, data, 671500 + run)
            svi = SVI(model._pyro_model, model._pyro_guide,
                      pyro.optim.ClippedAdam({'lr': .001, 'clip_norm': 5., 'lrd': .9997}),
                      Trace_ELBO(num_particles=args.particles))
            start, losses = time.perf_counter(), []
            for step in range(args.steps):
                loss = svi.step(data)
                if not np.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite objective at {step}')
                losses.append(loss)
                if (step + 1) % 1000 == 0:
                    print(args.mode, run, step + 1, float(np.mean(losses[-500:])), flush=True)
            seconds = time.perf_counter() - start
            draws = parameters_from_vi(model, 4000)
            np.savez_compressed(path, **draws)
            torch.save({'state': model.flow_transforms.state_dict()}, path.with_suffix('.pt'))
            np.save(output / f'{args.mode}_losses_{run}.npy', losses)
            names, features = feature_draws(draws, independent)
            prediction = log_predictive(model, draws, test)
            np.savez_compressed(output / f'{args.mode}_features_{run}.npz',
                                features=features, names=names, test_log_predictive=prediction)
            result = {'seconds': seconds, 'configuration': vars(args), 'initialization': initialization,
                      'runtime': runtime_metadata(),
                      'last_1000_mean_negative_elbo': float(np.mean(losses[-1000:])),
                      'test_score_sum_up_to_shared_event_constants': float(prediction.sum())}
            path.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    else:
        paths = sorted(output.glob('nuts_chain_*.npz'))
        if len(paths) < 2:
            raise ValueError('At least two independent completed chains are required')
        chains = [dict(np.load(path)) for path in paths]
        samples = {key: np.stack([chain[key] for chain in chains]) for key in chains[0]}
        names, features = feature_draws(samples)
        inference_data = az.from_dict(posterior={'functionals': features})
        result = {'names': names, 'chains': len(chains), 'retained_per_chain': features.shape[1],
                  'rhat': az.rhat(inference_data)['functionals'].values.tolist(),
                  'ess_bulk': az.ess(inference_data, method='bulk')['functionals'].values.tolist(),
                  'ess_tail': az.ess(inference_data, method='tail')['functionals'].values.tolist(),
                  'quantiles_05_50_95': np.quantile(features.reshape(-1, len(names)), [.05, .5, .95], axis=0).T.tolist()}
        np.savez_compressed(output / 'nuts_features.npz', features=features, names=names)
        (output / 'nuts_summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
