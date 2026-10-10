"""Controlled noisy catalogs with a coherent Gaussian detection experiment.

The detection statistic is observed, so its likelihood is included in each
event posterior. These experiments isolate morphology and selection effects;
they are not LISA waveform or search forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from scipy.special import ndtr, ndtri
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta, multivariate_normal, norm, qmc, truncnorm

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA256 = hashlib.sha256((ROOT / 'src/gwpopulation/core.py').read_bytes()).hexdigest()
sys.path.insert(0, str(ROOT / 'src'))
from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated
from gwpopulation.rate import conditional_rate_draws

MORPHOLOGIES = ('correlated', 'bimodal', 'curved')
SLOPE = np.array([2., -8., 2.])
OFFSET = 13.
THRESHOLD = 8.
SIGMA = np.array([.04, .055, .04])
CORR = np.array([[1., .6, -.25], [.6, 1., .1], [-.25, .1, 1.]])
A, B = np.array([2., 3., 2.]), np.array([4., 2., 1.5])
BETA_A = np.array([[5., 9., 3.], [12., 4., 9.]])
BETA_B = np.array([[12., 5., 9.], [4., 10., 3.]])


def snr(x):
    return OFFSET + x @ SLOPE


def selection(x):
    return torch.special.ndtr(OFFSET + x @ torch.tensor(SLOPE, dtype=x.dtype) - THRESHOLD)


def truth_sample(name, n, rng):
    if name == 'correlated':
        u = ndtr(rng.multivariate_normal(np.zeros(3), CORR, size=n))
        return (1 - (1 - u)**(1 / B))**(1 / A)
    if name == 'bimodal':
        component = (rng.uniform(size=n) > .55).astype(int)
        return rng.beta(BETA_A[component], BETA_B[component])
    x = rng.beta(2., 2., size=n)
    loc_y, loc_z = .13 + .72 * x**2, .85 - .65 * x
    y = truncnorm.rvs(-loc_y / .085, (1 - loc_y) / .085, loc=loc_y,
                      scale=.085, random_state=rng)
    z = truncnorm.rvs(-loc_z / .11, (1 - loc_z) / .11, loc=loc_z,
                      scale=.11, random_state=rng)
    return np.column_stack([x, y, z])


def truth_pdf(name, x):
    if name == 'correlated':
        f = A * B * x**(A - 1) * (1 - x**A)**(B - 1)
        eta = ndtri(np.clip(1 - (1 - x**A)**B, 1e-12, 1 - 1e-12))
        copula = np.exp(multivariate_normal.logpdf(eta, cov=CORR) - norm.logpdf(eta).sum(axis=-1))
        return f.prod(axis=-1) * copula
    if name == 'bimodal':
        first = beta.pdf(x, BETA_A[0], BETA_B[0]).prod(axis=-1)
        second = beta.pdf(x, BETA_A[1], BETA_B[1]).prod(axis=-1)
        return .55 * first + .45 * second
    loc_y, loc_z = .13 + .72 * x[:, 0]**2, .85 - .65 * x[:, 0]
    return (beta.pdf(x[:, 0], 2, 2)
            * truncnorm.pdf(x[:, 1], -loc_y / .085, (1 - loc_y) / .085, loc_y, .085)
            * truncnorm.pdf(x[:, 2], -loc_z / .11, (1 - loc_z) / .11, loc_z, .11))


def event_posterior(y, rho, n_samples, rng):
    precision = np.diag(SIGMA**-2) + np.outer(SLOPE, SLOPE)
    covariance = np.linalg.inv(precision)
    mean = covariance @ (y / SIGMA**2 + SLOPE * (rho - OFFSET))
    accepted = []
    attempts = 0
    while sum(len(chunk) for chunk in accepted) < n_samples:
        proposal = rng.multivariate_normal(mean, covariance, size=max(256, 2 * n_samples))
        accepted.append(proposal[((proposal > 0) & (proposal < 1)).all(axis=1)])
        attempts += 1
        if attempts > 10000:
            raise RuntimeError('Truncated Gaussian posterior rejection stalled')
    return np.concatenate(accepted)[:n_samples], mean, covariance


def catalog(name, seed, mean_intrinsic=160., n_samples=256):
    rng = np.random.default_rng(seed)
    truth = truth_sample(name, rng.poisson(mean_intrinsic), rng)
    rho = snr(truth) + rng.normal(size=len(truth))
    y = truth + rng.normal(size=truth.shape) * SIGMA
    detected = rho > THRESHOLD
    posteriors, centers = [], []
    for yi, rhoi in zip(y[detected], rho[detected]):
        samples, center, covariance = event_posterior(yi, rhoi, n_samples, rng)
        posteriors.append(samples)
        centers.append(center)
    posterior = np.asarray(posteriors)
    data = np.concatenate([posterior, np.ones((*posterior.shape[:2], 1))], axis=-1)
    return torch.tensor(data, dtype=torch.float32), {
        'truth_intrinsic': truth, 'detected': detected, 'rho_observed': rho,
        'measurement': y, 'posterior_centers': centers, 'covariance': covariance,
        'mean_intrinsic': mean_intrinsic,
    }


def build_model(family, k, quadrature, seed):
    model = BayesianMixtureEstimator_NF_Correlated(
        data_dim=3, num_corr_kumaraswamy=k if family == 'kumaraswamy' else 0,
        num_corr_tnormal=k if family == 'normal' else 0, num_corr_tpareto=0,
        hidden_dim_flow=48, num_flow_transforms=6, physical_bounds=([0.] * 3, [1.] * 3),
        device='cpu')
    model.obs_neff_threshold = 0
    model.selection_neff_multiplier = 0
    model.q_monotonicity_lambda = 0
    model.kumaraswamy_log_param_clip = None
    model.correct_selection_effects = True
    model.selection_function = selection
    model.selection_mc_mode = 'fixed_sobol'
    model.lambda_det_mc_samples = quadrature
    model.selection_mc_seed = seed
    model.observation_years = 1.
    return model


@torch.no_grad()
def summarize(model, name, data, truth, seed, n_draws=192):
    rng = np.random.default_rng(seed)
    points = qmc.Sobol(3, scramble=True, seed=seed).random_base2(15)
    target = truth_pdf(name, points)
    pdet = ndtr(snr(points) - THRESHOLD)
    latent = model.flow_dist.sample((n_draws,))
    densities, alphas, training_alphas, neff, norm_integrals = [], [], [], [], []
    point_tensor = torch.tensor(points, dtype=torch.float32)
    for draw in latent:
        params = model._unflatten_latent_vector(draw)
        density = model._log_pdf_phys_from_params(params, point_tensor).exp().numpy()
        norm_integrals.append(float(density.mean()))
        densities.append(density / density.sum())
        training_alphas.append(float(model._selection_fraction_for_params(params)))
        if model.selection_integration == 'population_qmc':
            original_count = model.lambda_det_mc_samples
            model.lambda_det_mc_samples = 32768
            alphas.append(float(model._selection_fraction_for_params(params)))
            model.lambda_det_mc_samples = original_count
        else:
            alphas.append(float(np.sum(density * pdet) / np.sum(density)))
        event_pdf = model._log_pdf_phys_from_params(params, data[..., :3].reshape(-1, 3)).exp()
        event_pdf = event_pdf.reshape(*data.shape[:2]).numpy()
        neff.append(float(np.min(event_pdf.sum(axis=1)**2 / (event_pdf**2).sum(axis=1))))
    densities = np.asarray(densities)
    predictive = densities.mean(axis=0)
    joint_distance = float(jensenshannon(predictive, target, base=2))
    truth_distances = jensenshannon(densities, np.broadcast_to(target, densities.shape), base=2, axis=1)
    within = jensenshannon(densities[::2], densities[1::2], base=2, axis=1)
    alpha_true = float(np.mean(target * pdet))
    rates = conditional_rate_draws(len(data), 1., torch.tensor(alphas).repeat_interleave(32))
    intrinsic = rates['intrinsic_mean'].numpy()
    detected_mean = rates['detected_mean'].numpy()
    replicated = rates['replicated_detected_count'].numpy()
    independent_count = int(rng.poisson(truth['mean_intrinsic'] * alpha_true))
    # Marginals integrate the same dense 3D quadrature, with identical bins for all draws.
    edges = np.linspace(0, 1, 65)
    marginal_draws, marginal_truth = [], []
    for axis in range(3):
        bins = np.minimum(np.searchsorted(edges, points[:, axis], side='right') - 1, 63)
        marginal_draws.append(np.array([np.bincount(bins, weights=d, minlength=64) * 64 for d in densities]))
        marginal_truth.append(np.bincount(bins, weights=target / target.sum(), minlength=64) * 64)
    summary = {
        'n_intrinsic_realized': len(truth['truth_intrinsic']), 'n_detected': len(data),
        'injected_intrinsic_mean': float(truth['mean_intrinsic']), 'alpha_true': alpha_true,
        'truth_integral': float(target.mean()), 'quadrature_points': len(points),
        'posterior_density_integral_quantiles': np.quantile(norm_integrals, [.05, .5, .95]).tolist(),
        'alpha_quantiles': np.quantile(alphas, [.05, .5, .95]).tolist(),
        'alpha_training_to_fine_absolute_difference_quantiles': np.quantile(
            abs(np.asarray(alphas) - training_alphas), [.5, .9, .99, 1]).tolist(),
        'intrinsic_mean_quantiles': np.quantile(intrinsic, [.05, .5, .95]).tolist(),
        'detected_mean_quantiles': np.quantile(detected_mean, [.05, .5, .95]).tolist(),
        'independent_detected_count': independent_count,
        'predictive_detected_count_quantiles': np.quantile(replicated, [.05, .5, .95]).tolist(),
        'joint_js_distance_predictive_to_truth': joint_distance,
        'joint_js_distance_draw_to_truth_quantiles': np.quantile(truth_distances, [.05, .5, .95]).tolist(),
        'joint_js_distance_within_posterior_quantiles': np.quantile(within, [.05, .5, .95]).tolist(),
        'minimum_event_ess_quantiles': np.quantile(neff, [.05, .5, .95]).tolist(),
    }
    arrays = dict(latent=latent.numpy(), marginal_x=(edges[:-1] + edges[1:]) / 2,
                  marginal_draws=np.array(marginal_draws), marginal_truth=np.array(marginal_truth),
                  alpha=np.array(alphas), intrinsic_mean=intrinsic)
    return summary, arrays


def main():
    global snr, selection
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/analysis/morphologies')
    parser.add_argument('--morphology', choices=MORPHOLOGIES, default='correlated')
    parser.add_argument('--family', choices=('kumaraswamy', 'normal'), default='kumaraswamy')
    parser.add_argument('--components', type=int, default=2)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--particles', type=int, default=4)
    parser.add_argument('--quadrature', type=int, default=2048)
    parser.add_argument('--posterior-samples', type=int, default=256)
    parser.add_argument('--catalog-seed', type=int, default=823010)
    parser.add_argument('--training-seed', type=int, default=91000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--initial-state')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--initialize-events', action='store_true')
    parser.add_argument('--detector', choices=('linear', 'lisa'), default='linear')
    parser.add_argument('--integration', choices=('uniform', 'population_qmc'), default='population_qmc')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tag = f'{args.morphology}_{args.family}_k{args.components}_s{args.training_seed}'
    if args.detector == 'lisa':
        tag = 'lisa_' + tag
    output = ROOT / args.output / tag
    output.mkdir(parents=True, exist_ok=True)
    seed = args.catalog_seed + MORPHOLOGIES.index(args.morphology)
    generator = catalog
    if args.detector == 'lisa':
        from build_lisa_analysis_assets import LISASNR
        from lisa_noisy_catalog import catalog as lisa_catalog
        snr = LISASNR()
        selection = lambda points: torch.special.ndtr(snr.torch(points) - THRESHOLD)
        generator = lisa_catalog
    catalog_cache = ROOT / args.output / f'{args.detector}_{args.morphology}_catalog_s{seed}_n{args.posterior_samples}.npz'
    if catalog_cache.exists():
        cached = np.load(catalog_cache)
        data = torch.tensor(cached['data'])
        truth = {key: cached[key] for key in cached.files if key != 'data'}
    else:
        data, truth = generator(args.morphology, seed, n_samples=args.posterior_samples)
        np.savez_compressed(catalog_cache, data=data.numpy(), **truth)
    np.savez_compressed(output / 'catalog.npz', data=data.numpy(), **truth)
    pyro.clear_param_store()
    pyro.set_rng_seed(args.training_seed)
    model = build_model(args.family, args.components, args.quadrature, seed)
    model.selection_integration = args.integration
    initialization = None
    if args.initialize_events:
        from initialize_population_guide import initialize_from_events
        initialization = initialize_from_events(model,data,args.training_seed)
    if args.resume:
        state = torch.load(output / 'flow.pt', weights_only=False)
        model.flow_transforms.load_state_dict(state['state'])
    elif args.initial_state:
        state = torch.load(ROOT / args.initial_state, weights_only=False)
        model.flow_transforms.load_state_dict(state['state'])
    optimizer = pyro.optim.ClippedAdam({'lr': args.learning_rate, 'clip_norm': 5., 'lrd': .9998})
    svi = SVI(model._pyro_model, model._pyro_guide, optimizer,
              loss=Trace_ELBO(num_particles=args.particles))
    losses, start = [], time.perf_counter()
    for step in range(args.steps):
        loss = svi.step(data)
        if not np.isfinite(loss):
            raise RuntimeError(f'Nonfinite objective at step {step}')
        losses.append(loss)
        if (step + 1) % 1000 == 0:
            print(tag, step + 1, float(np.mean(losses[-250:])), flush=True)
    elapsed = time.perf_counter() - start
    model.is_trained = True
    torch.save({'state': model.flow_transforms.state_dict(), 'configuration': vars(args)}, output / 'flow.pt')
    summary, arrays = summarize(model, args.morphology, data, truth, seed + 500)
    np.savez_compressed(output / 'posterior.npz', losses=losses, **arrays)
    summary.update(configuration=vars(args), train_seconds=elapsed,
                   negative_elbo_last_1000=float(np.mean(losses[-1000:])),
                   core_sha256=CORE_SHA256)
    if initialization is not None:
        summary['initialization'] = initialization
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
