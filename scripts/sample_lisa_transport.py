"""Compare ordinary and flow-coordinate NUTS for the unchanged LISA target."""
from __future__ import annotations

import argparse
import json
import time
import warnings

import arviz as az
import numpy as np
import pyro
from pyro.infer import MCMC, NUTS
import torch

from benchmark_selected_mixture import feature_draws, runtime_metadata
from correct_selected_posterior import latent_sites, sha256, swap_components
from diagnose_lisa_correction import default_dtype, lisa_model, verified_inputs
from validate_population_morphologies import ROOT


def sampling_model(configuration, checkpoint=None):
    model = lisa_model(configuration, checkpoint=checkpoint)
    # The copula Gaussianization casts to the bounds dtype in the shared core.
    model.min_vals = model.min_vals.double()
    model.max_vals = model.max_vals.double()
    original_points = model._selection_mc_samples_norm
    count = configuration['quadrature']
    fixed_points = original_points(count, model.device).clone()
    # Cache before tracing, so Sobol scrambling never enters the traced graph.
    model._selection_mc_samples_norm = lambda n, device: (
        fixed_points.to(device) if n == count else original_points(n, device))
    return model


def clear_transport_cache(model):
    model.flow_dist.clear_cache()
    for transform in model.flow_transforms.parts:
        transform.clear_cache()


@torch.no_grad()
def check_precision(model, data, points):
    low, high = model.min_vals, model.max_vals
    differences = []
    try:
        for point in points:
            model.min_vals, model.max_vals = low, high
            precise = differentiable_log_target(model, point, data)
            model.min_vals, model.max_vals = low.float(), high.float()
            previous = differentiable_log_target(model, point, data)
            differences.append(float(precise - previous))
    finally:
        model.min_vals, model.max_vals = low, high
    return {'draws': len(points), 'log_target_differences': differences,
            'maximum_absolute_log_target_difference': max(abs(x) for x in differences)}


def differentiable_log_target(model, latent, data):
    with default_dtype(latent.dtype):
        trace = pyro.poutine.trace(pyro.poutine.condition(
            model._pyro_model, data=latent_sites(model, latent))).get_trace(data)
        return trace.log_prob_sum() + model._latent_log_abs_det_jacobian(latent)


class TransportPotential:
    def __init__(self, model, data, method):
        if method not in ('raw', 'flow'):
            raise ValueError('Unknown sampling coordinates')
        self.model, self.data, self.method = model, data, method
        for parameter in model.flow_transforms.parameters():
            parameter.requires_grad_(False)

    def to_latent(self, coordinate):
        if self.method == 'raw':
            return coordinate
        clear_transport_cache(self.model)
        return self.model.flow_transforms(coordinate)

    def from_latent(self, latent):
        if self.method == 'raw':
            return latent
        clear_transport_cache(self.model)
        return self.model.flow_transforms.inv(latent)

    def __call__(self, coordinate):
        latent = self.to_latent(coordinate)
        value = -differentiable_log_target(self.model, latent, self.data)
        if self.method == 'flow':
            value = value - self.model.flow_transforms.log_abs_det_jacobian(coordinate, latent)
        return value


def value_gradient(function, point):
    variable = point.detach().clone().requires_grad_(True)
    value = function(variable)
    gradient, = torch.autograd.grad(value, variable)
    return value.detach(), gradient.detach()


def compile_potential(potential, initial, tolerance):
    # Explicitly compare new inputs: transform caches must not freeze a trajectory.
    potential(initial)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.jit.TracerWarning)
        traced = torch.jit.trace(lambda point: potential(point), (initial,), check_trace=False)
    generator = torch.Generator().manual_seed(733100)
    checks = []
    for scale in (0., .03, .1):
        point = initial + scale * torch.randn(initial.shape, dtype=initial.dtype, generator=generator)
        eager_value, eager_gradient = value_gradient(potential, point)
        jit_value, jit_gradient = value_gradient(traced, point)
        checks.append({'scale': scale, 'absolute_value_difference': float(abs(eager_value - jit_value)),
                       'maximum_absolute_gradient_difference': float(abs(eager_gradient - jit_gradient).max())})
    if any(not np.isfinite(list(row.values())).all()
           or max(row['absolute_value_difference'], row['maximum_absolute_gradient_difference']) > tolerance
           for row in checks):
        raise ValueError(f'Traced target differs from the original: {checks}')
    return traced, checks


@torch.no_grad()
def aligned_initial(model, product):
    index = int(np.argmax(product['raw_weights']))
    point = torch.tensor(product['latent'][index], dtype=torch.float64)
    candidates = torch.stack([point, swap_components(model, point)])
    clear_transport_cache(model)
    label = int(model.flow_dist.log_prob(candidates).argmax())
    return candidates[label], {'index': index, 'labels_swapped': bool(label),
                               'saved_log_target': float(product['log_target'][index]),
                               'saved_alpha': float(product['alpha'][index])}


@torch.no_grad()
def lisa_features(model, latent, power=10):
    if not len(latent):
        return [], np.empty((0, 0))
    samples = {name: value.numpy() for name, value in latent_sites(model, latent).items()}
    names, features = feature_draws(samples, power=power)
    for index, point in enumerate(latent):
        params = model._unflatten_latent_vector(point, apply_numerical_limits=False)
        features[index, names.index('alpha')] = float(model._selection_fraction_for_params(params))
    return names, features


class SamplingBudgetExceeded(RuntimeError):
    pass


def run_chain(potential, initial, config, seed):
    started = time.perf_counter()
    traced, checks = compile_potential(potential, initial, config['jit_absolute_tolerance'])
    compilation_seconds = time.perf_counter() - started
    calls = 0

    def counted(values):
        nonlocal calls
        calls += 1
        return traced(values['coordinate'])

    warmup, posterior, records = [], [], []
    sample_started = time.perf_counter()

    def progress(kernel, samples, stage, iteration):
        coordinate = samples['coordinate'].detach().clone()
        (warmup if stage == 'Warmup' else posterior).append(coordinate)
        records.append([int(stage != 'Warmup'), iteration, calls,
                        time.perf_counter() - sample_started, float(kernel.step_size)])
        if (iteration + 1) % 20 == 0:
            print(potential.method, seed, stage, iteration + 1,
                  'seconds', round(records[-1][3], 1), 'evaluations', calls, flush=True)
        if records[-1][3] > config['maximum_seconds_per_chain']:
            raise SamplingBudgetExceeded('Per-chain pilot time limit reached')

    pyro.set_rng_seed(seed)
    kernel = NUTS(potential_fn=counted, full_mass=config['full_mass'],
                  target_accept_prob=config['target_acceptance'], max_tree_depth=config['max_tree_depth'])
    sampler = MCMC(kernel, initial_params={'coordinate': initial.detach().clone()},
                   warmup_steps=config['warmup'], num_samples=config['samples'],
                   num_chains=1, disable_progbar=True, hook_fn=progress)
    status = 'complete'
    with pyro.validation_enabled(False):
        try:
            sampler.run()
        except SamplingBudgetExceeded:
            status = 'time_limit'
    sampling_seconds = time.perf_counter() - sample_started
    coordinate = torch.stack(posterior) if posterior else torch.empty((0, len(initial)), dtype=initial.dtype)
    with torch.no_grad():
        latent = potential.to_latent(coordinate) if len(coordinate) else coordinate.clone()
    diagnostic = sampler.diagnostics() if status == 'complete' else None
    report = {'status': status, 'seed': seed, 'method': potential.method,
              'warmup_completed': len(warmup), 'samples_completed': len(posterior),
              'compilation_seconds': compilation_seconds, 'sampling_seconds': sampling_seconds,
              'potential_evaluations': calls, 'jit_checks': checks,
              'step_size': float(kernel.step_size),
              'divergences': diagnostic['divergences']['chain 0'] if diagnostic else None,
              'acceptance_rate': diagnostic['acceptance rate']['chain 0'] if diagnostic else None}
    arrays = {'latent': latent.numpy(), 'coordinate': coordinate.numpy(),
              'initial_coordinate': initial.detach().numpy(),
              'warmup_coordinate': torch.stack(warmup).numpy() if warmup else np.empty((0, len(initial))),
              'progress': np.asarray(records)}
    return report, arrays


def summarize(config, output):
    results = {}
    for method in config['methods']:
        reports = [json.loads((output / f'{method}_{seed}.json').read_text()) for seed in config['seeds']]
        complete = all(row['status'] == 'complete' for row in reports)
        results[method] = {'complete': complete,
                           'sampling_seconds': sum(row['sampling_seconds'] for row in reports),
                           'potential_evaluations': sum(row['potential_evaluations'] for row in reports)}
        if not complete:
            continue
        products = [np.load(output / f'{method}_{seed}.npz') for seed in config['seeds']]
        names = products[0]['feature_names'].tolist()
        if any(p['feature_names'].tolist() != names for p in products):
            raise ValueError('Chains have different functionals')
        features = np.stack([p['features'] for p in products])
        dataset = az.from_dict(posterior={name: features[:, :, j] for j, name in enumerate(names)})
        diagnostic = az.summary(dataset, kind='diagnostics', round_to='none')
        results[method]['functionals'] = diagnostic.to_dict(orient='index')
        results[method]['maximum_rhat'] = float(diagnostic.r_hat.max())
        results[method]['minimum_bulk_ess'] = float(diagnostic.ess_bulk.min())
        results[method]['minimum_tail_ess'] = float(diagnostic.ess_tail.min())
        results[method]['minimum_bulk_ess_per_second'] = float(diagnostic.ess_bulk.min()) / results[method]['sampling_seconds']
        results[method]['divergences'] = sum(len(row['divergences']) for row in reports)
    result = {'configuration': config, 'methods': results, 'eligible_for_scientific_summaries': False,
              'interpretation': 'Short cost and mixing pilot; no converged LISA posterior is asserted.',
              'chain_products': {p.name: sha256(p) for p in sorted(output.glob('*_73*.json'))}}
    path = output / 'summary.json'
    if path.exists():
        raise FileExistsError(f'Completed summary is retained: {path}')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lisa_transport.json')
    parser.add_argument('--method', choices=('raw', 'flow'))
    parser.add_argument('--chain', type=int, choices=(0, 1))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    if not config['pilot_only']:
        raise ValueError('This declared experiment is a cost pilot')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(config, output)
        return
    source_config_path = ROOT / config['source_config']
    source_config = json.loads(source_config_path.read_text())
    configurations, data, _, sources = verified_inputs(source_config, source_config_path)
    paths = [config_path, ROOT / 'scripts/sample_lisa_transport.py']
    paths += [ROOT / name for name in config['initial_draws']]
    for name in config['initial_draws']:
        path = ROOT / name
        metadata = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != metadata['draw_product_sha256']:
            raise ValueError('Initial proposal draws differ from their archived product')
        paths.append(path.with_suffix('.json'))
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    source_index = config['flow_source_index']
    precision_path = output / 'precision.json'
    if precision_path.exists():
        precision = json.loads(precision_path.read_text())
        if precision['source_sha256'] != sources:
            raise ValueError('Precision check belongs to different sources')
    else:
        model = sampling_model(configurations[source_index])
        points = []
        for name in source_config['sources']:
            values = np.load(ROOT / name / 'posterior.npz')['latent']
            indices = np.linspace(0, len(values) - 1, config['precision_draws_per_start'], dtype=int)
            points.extend(values[indices])
        for name in config['initial_draws']:
            product = np.load(ROOT / name)
            points.append(product['latent'][int(product['raw_weights'].argmax())])
        precision = check_precision(model, torch.tensor(data), torch.tensor(np.asarray(points)))
        precision['source_sha256'] = sources
        precision_path.write_text(json.dumps(precision, indent=2, allow_nan=False) + '\n')
    if precision['maximum_absolute_log_target_difference'] >= config['precision_log_density_tolerance']:
        raise ValueError('Full float64 evaluation changes the target beyond the declared tolerance')
    for method in ([args.method] if args.method else config['methods']):
        for chain in ([args.chain] if args.chain is not None else range(len(config['seeds']))):
            seed = config['seeds'][chain]
            path = output / f'{method}_{seed}.json'
            if path.exists() or path.with_suffix('.npz').exists():
                raise FileExistsError(f'Completed pilot is retained: {path}')
            pyro.clear_param_store()
            model = sampling_model(configurations[source_index], checkpoint=ROOT / source_config['sources'][source_index] / 'flow.pt')
            initial, provenance = aligned_initial(model, np.load(ROOT / config['initial_draws'][chain]))
            potential = TransportPotential(model, torch.tensor(data), method)
            coordinate = potential.from_latent(initial).detach()
            report, arrays = run_chain(potential, coordinate, config, seed)
            names, features = lisa_features(model, torch.tensor(arrays['latent']), config['feature_transport_power'])
            arrays.update(initial_latent=initial.numpy(), feature_names=np.asarray(names), features=features)
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            report.update(configuration=config, source_sha256=sources, initial=provenance,
                          runtime=runtime_metadata(), draw_product_sha256=sha256(path.with_suffix('.npz')),
                          precision_product_sha256=sha256(precision_path),
                          eligible_for_scientific_summaries=False)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
