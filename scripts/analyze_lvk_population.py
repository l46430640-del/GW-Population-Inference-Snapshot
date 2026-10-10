"""GWTC-3 shape inference with explicit coordinates and fixed spin assumptions."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from scipy.special import betainc, expit
from scipy.stats import beta, norm, qmc

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA256 = hashlib.sha256((ROOT / 'src/gwpopulation/core.py').read_bytes()).hexdigest()
MODEL_SOURCE_SHA256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
    'src/gwpopulation/core.py', 'src/gwpopulation/smooth_mass.py',
    'scripts/analyze_lvk_population.py', 'scripts/initialize_population_guide.py')}
sys.path.insert(0, str(ROOT / 'src'))
from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated
from lvk_injections import MIXTURE_NAME, covered_mass_pairs, load_recovered

EXCLUDED = {'GW170817', 'S190425z', 'S190426c', 'S190814bv', 'S190917u', 'S200105ae', 'S200115j'}


def coordinates(m1, m2, z, mmin=3.):
    width = np.maximum(1 - mmin / m1, 1e-8)
    u = (m2 / m1 - mmin / m1) / width
    points = np.column_stack([np.log10(m1), z, u])
    jacobian = np.log(10) * m1**2 * width
    return points, jacobian


def spin_density(a1, a2, mode):
    valid = (a1 >= 0) & (a1 <= .99) & (a2 >= 0) & (a2 <= .99)
    if mode == 'uniform':
        density = np.full_like(a1, 1 / (.99**2 * 4))
    else:
        density = beta.pdf(a1, 2, 5) * beta.pdf(a2, 2, 5) / (4 * betainc(2, 5, .99)**2)
    return np.where(valid, density, 0.)


def reference_density(points, mmax):
    """A fixed importance proposal; its density cancels in the likelihood."""
    m1 = 10**points[:, 0]
    z, u = points[:, 1], points[:, 2]
    mass = (.85 * m1**-2.5 / ((5.**-1.5 - mmax**-1.5) / 1.5)
            + .15 * norm.pdf(m1, 35., 4.)) * expit((m1 - 5.) / .8)
    density = mass * m1 * np.log(10) * beta.pdf(z / 2.3, 2., 4.) / 2.3 * beta.pdf(u, 2., 1.3)
    valid = (m1 > 3) & (m1 < mmax) & (z > 0) & (z < 2.3) & (u > 0) & (u < 1)
    return np.where(valid, density, 0.)


def load_inputs(mode, sample_count, injection_count, mmax, seed=372100):
    directory = ROOT / 'external_data/lvk/analysis_inputs'
    cache = directory / f'importance_mixture_v2_{mode}_{sample_count}_{injection_count}_{mmax:g}_{seed}.npz'
    if cache.exists():
        arrays = np.load(cache)
        return ({key: arrays[key] for key in arrays.files})
    with (directory / 'sampleDict_FAR_1_in_1_yr.pickle').open('rb') as handle:
        events = pickle.load(handle)
    injection, injection_metadata = load_recovered(directory / MIXTURE_NAME)
    if mmax > injection_metadata['all_primary_mass_range'][1]:
        raise ValueError('Population support exceeds the injection campaign mass range')
    rng = np.random.default_rng(seed)
    data, names, reference_event_ess = [], [], []
    for name, event in events.items():
        if name in EXCLUDED:
            continue
        m1, m2, z, a1, a2, prior = [np.asarray(event[key])
                                  for key in ('m1', 'm2', 'z', 'a1', 'a2', 'z_prior')]
        points, jacobian = coordinates(m1, m2, z)
        # Spin priors in PE are constant in (a1,a2,cos(t1),cos(t2)).
        effective_prior = prior * jacobian / np.maximum(spin_density(a1, a2, mode), 1e-30)
        proposal = reference_density(points, mmax)
        weights = proposal / effective_prior
        reference_event_ess.append(float(weights.sum()**2 / np.sum(weights**2)))
        index = rng.choice(len(m1), sample_count, replace=True, p=weights / weights.sum())
        data.append(np.column_stack([points[index], proposal[index]]))
        names.append(name)
    all_found = len(injection['m1'])
    keep = np.arange(all_found)
    m1, m2, z, a1, a2 = [np.asarray(injection[key])[keep] for key in ('m1', 'm2', 'z', 'a1', 'a2')]
    points, jacobian = coordinates(m1, m2, z)
    spin = spin_density(a1, a2, mode)
    valid = ((m1 > 3) & (m1 < mmax) & (m2 >= 3) & (z > 0) & (z < 2.3)
             & (points[:, 2] >= 0) & (points[:, 2] <= 1) & (spin > 0))
    pdraw = np.asarray(injection['joint_draw'])[keep]/np.asarray(injection['mixture_weight'])[keep]
    effective_draw = pdraw[valid] * jacobian[valid] / spin[valid]
    proposal = reference_density(points[valid], mmax)
    weights = proposal / effective_draw
    reference_alpha = weights.sum() / float(injection['nTrials'])
    index = rng.choice(len(weights), injection_count, replace=True, p=weights / weights.sum())
    arrays = {'data': np.asarray(data, dtype=np.float32), 'event_names': np.array(names),
              'injection_points': points[valid][index].astype(np.float32),
              'injection_log_p_draw': np.log(proposal[index] / reference_alpha).astype(np.float32),
              'n_draw': float(injection_count),
              'injection_retained': int(valid.sum()), 'found_injections_total': all_found,
              'reference_event_ess': reference_event_ess,
              'reference_injection_ess': float(weights.sum()**2 / np.sum(weights**2)),
              'full_injection_points': points[valid].astype(np.float32),
              'full_injection_log_p_draw': np.log(effective_draw).astype(np.float32),
              'full_n_draw': float(injection['nTrials'])}
    np.savez_compressed(cache, **arrays)
    return arrays


def estimator(args, arrays):
    spline = args.family in ('spline', 'smooth')
    model = BayesianMixtureEstimator_NF_Correlated(
        3, num_corr_kumaraswamy=0 if spline else args.components, num_corr_tnormal=0,
        num_corr_tpareto=0, num_corr_axis_spline_kumaraswamy=args.components if spline else 0,
        hidden_dim_flow=64, num_flow_transforms=6,
        m1_basis_type='smooth' if args.family == 'smooth' else 'two_resolution',
        axis_spline_num_bins=16, m1_smooth_length=getattr(args, 'mass_length', .35),
        m1_two_resolution_global_bins=5, m1_two_resolution_local_bins=16,
        m1_two_resolution_local_min_solar_mass=5., m1_two_resolution_local_max_solar_mass=50.,
        physical_bounds=([np.log10(3), 0., 0.], [np.log10(args.mmax), 2.3, 1.]), device='cpu')
    full = getattr(args, 'full_injections', False)
    points = arrays['full_injection_points' if full else 'injection_points']
    model.injection_data = {key: torch.tensor(points[:, axis]) for axis, key in enumerate(('log_m1', 'z', 'q'))}
    model.injection_data.update(log_p_draw=torch.tensor(arrays['full_injection_log_p_draw' if full else 'injection_log_p_draw']),
                                Ndraw=float(arrays['full_n_draw' if full else 'n_draw']))
    model.correct_selection_effects = True
    model.obs_neff_threshold = 0
    model.selection_neff_multiplier = 0
    model.q_monotonicity_lambda = 0
    model.kumaraswamy_log_param_clip = None
    # The normalization of the population conditioned on this domain cancels
    # between N event factors and alpha**N in the rate-marginalized likelihood.
    model.event_support_function = lambda points: covered_mass_pairs(
        10**points[..., 0], 3+(10**points[..., 0]-3)*points[..., 2])
    if not np.all(covered_mass_pairs(10**points[:,0],3+(10**points[:,0]-3)*points[:,2])):
        raise ValueError('Selection injections outside the declared joint population domain')
    return model


@torch.no_grad()
def summarize(model, arrays, n_draws=192):
    volume = float(torch.prod(model.max_vals - model.min_vals))
    points_norm = torch.tensor(qmc.Sobol(3, scramble=True, seed=993002).random_base2(15), dtype=torch.float32)
    physical = (points_norm * (model.max_vals - model.min_vals) + model.min_vals).numpy()
    m1 = 10**physical[:, 0]
    real_q = 3 / m1 + (1 - 3 / m1) * physical[:, 2]
    displayed = np.column_stack([physical[:, 0], physical[:, 1], real_q])
    edges = [np.linspace(np.log10(3), float(model.max_vals[0]), 81),
             np.linspace(0, 2.3, 81), np.linspace(0, 1, 81)]
    bins = [np.clip(np.searchsorted(edge, displayed[:, i]) - 1, 0, 79) for i, edge in enumerate(edges)]
    latents = model.flow_dist.sample((n_draws,))
    marginal_draws, alpha, neff_sel, neff_event, norm_integrals = [], [], [], [], []
    data = torch.tensor(arrays['data'])
    normalized_pe = (data[..., :3].reshape(-1, 3) - model.min_vals) / (model.max_vals - model.min_vals)
    for latent in latents:
        params = model._unflatten_latent_vector(latent)
        density = model._log_pdf_phys_from_params(params, points_norm).exp().numpy()
        norm_integrals.append(float(density.mean() * volume))
        weights = density / density.sum()
        marginal_draws.append([np.bincount(index, weights=weights, minlength=80) / np.diff(edge)
                               for index, edge in zip(bins, edges)])
        alpha.append(float(model._selection_fraction_for_params(params)))
        inj_weights = model._last_selection_log_weights.exp()
        neff_sel.append(float(inj_weights.sum()**2 / inj_weights.square().sum()))
        log_pdf = model._log_pdf_phys_from_params(params, normalized_pe).reshape(*data.shape[:2])
        log_weights = log_pdf - data[..., 3].log()
        log_weights = log_weights.masked_fill(~model.event_support_function(data[..., :3]), -torch.inf)
        pe_weights = (log_weights - log_weights.max(dim=1, keepdim=True).values).exp()
        neff_event.append(float((pe_weights.sum(dim=1)**2 / pe_weights.square().sum(dim=1)).min()))
    result = {'event_count': len(data), 'posterior_samples_per_event': data.shape[1],
              'minimum_event_ess_quantiles': np.quantile(neff_event, [.05, .5, .95]).tolist(),
              'selection_ess_quantiles': np.quantile(neff_sel, [.05, .5, .95]).tolist(),
              'alpha_quantiles': np.quantile(alpha, [.05, .5, .95]).tolist(),
              'density_integral_quantiles': np.quantile(norm_integrals, [.05, .5, .95]).tolist(),
              'injection_retained': int(arrays['injection_retained']), 'event_names': arrays['event_names'].tolist()}
    products = {'latent': latents.numpy(), 'marginal_draws': np.asarray(marginal_draws),
                'marginal_x': np.asarray([(edge[:-1] + edge[1:]) / 2 for edge in edges]),
                'alpha': alpha, 'neff_selection': neff_sel, 'neff_event': neff_event}
    return result, products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/analysis/lvk_mixture')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--family', choices=('spline', 'kumaraswamy', 'smooth'), default='spline')
    parser.add_argument('--mass-length', type=float, default=.35)
    parser.add_argument('--components', type=int, default=3)
    parser.add_argument('--spin', choices=('uniform', 'low'), default='uniform')
    parser.add_argument('--mmax', type=float, default=150.)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--particles', type=int, default=2)
    parser.add_argument('--posterior-samples', type=int, default=256)
    parser.add_argument('--injections', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=812301)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--initial-state')
    parser.add_argument('--full-injections', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--initialize-events', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tag = f'{args.family}_k{args.components}_{args.spin}_m{args.mmax:g}_s{args.seed}'
    if args.family == 'smooth':
        tag += f'_l{args.mass_length:g}'
    output = ROOT / args.output / tag
    output.mkdir(parents=True, exist_ok=True)
    arrays = load_inputs(args.spin, args.posterior_samples, args.injections, args.mmax)
    pyro.clear_param_store()
    pyro.set_rng_seed(args.seed)
    model = estimator(args, arrays)
    initialization = None
    if args.initialize_events:
        from initialize_population_guide import initialize_from_events
        initialization = initialize_from_events(model,torch.tensor(arrays['data']),args.seed)
    if args.resume:
        state = torch.load(output / 'flow.pt', weights_only=False)
        model.flow_transforms.load_state_dict(state['state'])
    elif args.initial_state:
        state = torch.load(ROOT / args.initial_state, weights_only=False)
        model.flow_transforms.load_state_dict(state['state'])
    data = torch.tensor(arrays['data'])
    svi = SVI(model._pyro_model, model._pyro_guide,
              pyro.optim.ClippedAdam({'lr': args.learning_rate, 'clip_norm': 5., 'lrd': .9998}),
              loss=Trace_ELBO(num_particles=args.particles))
    losses, start = [], time.perf_counter()
    for step in range(args.steps):
        loss = svi.step(data)
        if not np.isfinite(loss):
            raise RuntimeError(f'Nonfinite objective at step {step}')
        losses.append(loss)
        if (step + 1) % 500 == 0:
            print(tag, step + 1, float(np.mean(losses[-250:])), flush=True)
    elapsed = time.perf_counter() - start
    if args.smoke_only:
        print(json.dumps({'seconds': elapsed, 'steps': args.steps,
                          'diagnostics': model._latest_model_diagnostics}), flush=True)
        return
    torch.save({'state': model.flow_transforms.state_dict(), 'configuration': vars(args)}, output / 'flow.pt')
    model.is_trained = True
    summary, products = summarize(model, arrays)
    summary.update(configuration=vars(args), train_seconds=elapsed,
                   core_sha256=CORE_SHA256,
                   model_source_sha256=MODEL_SOURCE_SHA256,
                   injection_campaign=MIXTURE_NAME,
                   negative_elbo_last_1000=float(np.mean(losses[-1000:])))
    if initialization is not None:
        summary['initialization'] = initialization
    np.savez_compressed(output / 'posterior.npz', losses=losses, **products)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
