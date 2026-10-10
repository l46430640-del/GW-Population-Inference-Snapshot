"""Bounded paired evaluation of historical and corrected saved posteriors."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import arviz as az
import numpy as np
import psutil
import pyro
from scipy.special import logsumexp, ndtr, ndtri, rel_entr
from scipy.stats import qmc
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from benchmark_selected_mixture import exact_selection, selected_model
from build_lisa_analysis_assets import LISASNR
from correct_selected_posterior import latent_sites, weighted_quantiles
from diagnose_lisa_correction import default_dtype
from external_selected_evaluation import coarsen
from lvk_injections import covered_mass_pairs
from analyze_lvk_population import estimator, load_inputs
from sample_lisa_transport import sampling_model
from summarize_lvk_marginals import marginal_distributions, piecewise_mass_icdf
from validate_population_morphologies import build_model, MORPHOLOGIES
from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated, SafeKumaraswamy, TruncatedNormal
from gwpopulation.stable_model import CorrectedPopulationEstimator


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='ascii', newline='\n')


class Sources:
    def __init__(self):
        self.manifest = json.loads((ROOT/'results/analysis/result_manifest.json').read_text())['sha256']
        self.hashes = {}

    def add(self, name, expected=None):
        path = ROOT/name
        name = path.relative_to(ROOT).as_posix()
        actual = self.hashes.get(name)
        if actual is None:
            actual = digest(path)
            self.hashes[name] = actual
        expected = expected or self.manifest.get(name)
        if expected is not None and actual != expected:
            raise ValueError('Historical source or product changed: ' + name)
        return path

    def json(self, name):
        value = json.loads(self.add(name).read_text())
        for source, expected in value.get('source_sha256', {}).items():
            self.add(source, expected)
        return value

    def arrays(self, name, keys=None):
        with np.load(self.add(name)) as data:
            return {key: data[key] for key in (data.files if keys is None else keys)}


def indices_for_shape(shape, count):
    chains, draws = shape
    count = min(draws, count)
    return (np.arange(chains)[:, None]*draws + np.linspace(0, draws-1, count, dtype=int)).ravel()


def corrected_model(historical, independent=False):
    # Only likelihood coordinates are used here; no new guide is fitted.
    signature = inspect.signature(BayesianMixtureEstimator_NF_Correlated.__init__)
    args = {name: getattr(historical, name) for name in signature.parameters
            if name != 'self' and hasattr(historical, name)}
    args.update(hidden_dim_flow=8, num_flow_transforms=1, physical_bounds=historical.provided_bounds,
                independent_components=independent)
    with contextlib.redirect_stdout(io.StringIO()):
        model = CorrectedPopulationEstimator(**args)
    model.correct_selection_effects = historical.correct_selection_effects
    model.selection_function = historical.selection_function
    model.injection_data = getattr(historical, 'injection_data', None)
    model.observation_years = historical.observation_years
    model.event_support_function = getattr(historical, 'event_support_function', None)
    return model


def old_event_integrals(model, params, data, block=8):
    results = []
    for start in range(0, len(data), block):
        row = data[start:start+block]
        unit = (row[..., :3]-model.min_vals)/(model.max_vals-model.min_vals)
        pdf = model._log_pdf_phys_from_params(params, unit.reshape(-1, 3)).reshape(row.shape[:2])
        support = getattr(model, 'event_support_function', None)
        if support is not None:
            pdf = torch.where(support(row[..., :3]), pdf, -torch.inf)
        weights = row[..., 4].log() if row.shape[-1] == 5 else torch.zeros_like(row[..., 3])
        results.append(torch.logsumexp(pdf-row[..., 3].log()+weights, -1)-torch.logsumexp(weights, -1))
    return torch.cat(results)


def event_integrals(model, params, data, block):
    if not isinstance(model, CorrectedPopulationEstimator):
        return old_event_integrals(model, params, data, block)
    return torch.cat([model.event_log_likelihoods(params, data[i:i+block]) for i in range(0, len(data), block)])


def log_alpha(model, params):
    if not isinstance(model, CorrectedPopulationEstimator):
        if getattr(model, 'injection_data', None) is not None:
            points = torch.stack([model.injection_data[k] for k in ('log_m1', 'z', 'q')], -1)
            unit = (points-model.min_vals)/(model.max_vals-model.min_vals)
            return model._log_selection_fraction_from_params(params, unit)
    return model._selection_fraction_for_params(params).log()


def legacy_transport(model, params, normal, independent=False, lvk=False):
    if lvk:
        distributions, chol = marginal_distributions(model, params)
        probability = ndtr(np.einsum('nj,kij->nki', normal, chol.numpy())).clip(1e-15, 1-1e-15)
        return np.stack([piecewise_mass_icdf(distributions[0], probability[..., 0]),
                         distributions[1].icdf(torch.tensor(probability[..., 1])).numpy(),
                         distributions[2].icdf(torch.tensor(probability[..., 2])).numpy()], -1)
    if model.num_corr_kumaraswamy:
        marginal = SafeKumaraswamy(params['log_k_a'].exp()+.0001, params['log_k_b'].exp()+.0001)
        prefix = 'k'
    else:
        marginal = TruncatedNormal(params['tn_locs'], params['log_tn_scales'].exp()+.0001, low=.0001, high=.9999)
        prefix = 'tnormal'
    chol = params[prefix+'_L_cholesky']
    if independent:
        chol = torch.eye(3).expand_as(chol)
    probability = ndtr(np.einsum('nj,kij->nki', normal, chol.numpy())).clip(1e-14, 1-1e-14)
    return marginal.icdf(torch.tensor(probability)).numpy()


def population_quantities(model, params, unit, bins, alpha, count, lvk):
    weight = params['weights_logits'].double().clamp(-20, 20).softmax(-1).numpy()
    physical = model.min_vals.numpy()+(model.max_vals-model.min_vals).numpy()*unit
    support = np.ones(unit.shape[:2], dtype=bool)
    values = unit.copy()
    if lvk:
        mass = 10**physical[..., 0]
        secondary = 3+(mass-3)*unit[..., 2]
        support = covered_mass_pairs(mass, secondary)
        values = np.stack([mass, physical[..., 1], secondary/mass], -1)
        population_axes = unit.copy()
        population_axes[..., 2] = secondary/mass
    else:
        population_axes = unit
    norm = np.mean(support, 0) @ weight
    point_weight = support*weight[None]/(len(unit)*norm)
    means = (values*point_weight[..., None]).sum((0, 1))
    tail = ((unit[..., 0] < .25)*point_weight).sum(), ((unit[..., 0] > .8)*point_weight).sum()
    years = model.observation_years or 1.
    features = np.r_[means, tail, alpha, count*norm/(alpha*years), norm]
    index = np.minimum((population_axes*bins).astype(int), bins-1)
    flat = (index[..., 0]*bins+index[..., 1])*bins+index[..., 2]
    cells = np.bincount(flat.ravel(), weights=point_weight.ravel(), minlength=bins**3)
    if lvk:
        return features, cells, cells.copy()
    detection = model.selection_function(torch.tensor(physical.reshape(-1, 3))).numpy().reshape(unit.shape[:2])
    detected = np.bincount(flat.ravel(), weights=(point_weight*detection).ravel(), minlength=bins**3)
    return features, cells, detected/detected.sum()


def weight_diagnostics(delta, old_weights):
    delta, old_weights = np.asarray(delta), np.asarray(old_weights)
    weights = old_weights/old_weights.sum()
    new = np.exp(np.log(weights)+delta-logsumexp(np.log(weights)+delta))
    relative = np.exp(delta-logsumexp(delta))
    k = 0. if np.ptp(delta) < 1e-10 else float(az.psislw(delta, reff=1.)[1])
    return new, {'delta_mean': float(delta.mean()), 'delta_sd': float(delta.std(ddof=1)),
                 'delta_min': float(delta.min()), 'delta_max': float(delta.max()),
                 'ratio_ess_fraction': float(1/(relative@relative)/len(relative)),
                 'combined_weight_ess': float(1/(new@new)), 'maximum_combined_weight': float(new.max()),
                 'pareto_k': k if np.isfinite(k) else None,
                 'pareto_k_finite': bool(np.isfinite(k)), 'psis_reff': 1.,
                 'interpretation': 'Subset weight concentration only; neither MCMC ESS nor correction of VI error.'}


def prediction_summary(old, new, old_weights, new_weights):
    first = logsumexp(old+np.log(old_weights/old_weights.sum())[:, None], 0)
    fixed = logsumexp(new+np.log(old_weights/old_weights.sum())[:, None], 0)
    second = logsumexp(new+np.log(new_weights)[:, None], 0)
    difference = second-first
    return {'old_log_predictive': first, 'new_fixed_draw_log_predictive': fixed, 'new_log_predictive': second,
            'change': difference}, {'events': len(first), 'total_log_score_change': float(difference.sum()),
                                   'fixed_draw_total_log_score_change': float((fixed-first).sum()),
                                   'event_change_sd': float(difference.std(ddof=1)),
                                   'largest_absolute_event_change': float(abs(difference).max())}


def js_distance(first, second):
    first, second = first/first.sum(), second/second.sum()
    middle = (first+second)/2
    divergence = float((rel_entr(first, middle).sum()+rel_entr(second, middle).sum())/(2*math.log(2)))
    if divergence < -1e-12:
        raise ValueError('Negative Jensen-Shannon divergence exceeds roundoff.')
    return math.sqrt(max(0., divergence))


def cases(sources, config):
    analysis = sources.json('configs/analysis_runs.json')
    selected = 'results/analysis/selected_mixture'
    sources.json(selected+'/target.json')
    chains = [sources.arrays(f'{selected}/nuts_chain_{i}.npz') for i in range(4)]
    samples = {key: np.concatenate([c[key] for c in chains]) for key in chains[0]}
    shape = (4, len(chains[0]['weights_logits']))
    for label, train, weight in (
        ('controlled_nuts', selected+'/train_catalog.npz', None),
        ('external_compact', 'results/analysis/external_selected_comparison/train_events.npz',
         'results/analysis/external_selected_comparison/copula_weights.npz')):
        legacy = selected_model()
        if weight is not None:
            # The refined external target used the independent float64 NumPy
            # evaluator. The original controlled fit retains float32 Gaussianization.
            legacy.min_vals, legacy.max_vals = legacy.min_vals.double(), legacy.max_vals.double()
        yield dict(label=label, category='mcmc', model=legacy, samples=samples, shape=shape,
                   data=sources.arrays(train, ['data'])['data'], exact=True, lvk=False,
                   old_weights=np.ones(np.prod(shape)) if weight is None else sources.arrays(weight, ['weights'])['weights'].ravel(),
                   test=sources.arrays(selected+'/test_catalog.npz' if weight is None else
                                      'results/analysis/external_selected_comparison/test_events.npz', ['data'])['data'],
                   comparison='results/analysis/external_selected_comparison' if weight else None,
                   cached_events=None if weight is None else sources.arrays(
                       'results/analysis/external_selected_comparison/train_primary.npz', ['log_integrals'])['log_integrals'])
    for family in ('vi', 'independent'):
        for start in (0, 1):
            samples_vi = sources.arrays(f'{selected}/{family}_start_{start}.npz')
            n = len(samples_vi['weights_logits'])
            yield dict(label=f'controlled_{family}_{start}', category='vi', model=selected_model(family == 'independent'),
                       samples=samples_vi, shape=(1, n), data=sources.arrays(selected+'/train_catalog.npz', ['data'])['data'],
                       test=sources.arrays(selected+'/test_catalog.npz', ['data'])['data'],
                       exact=True, lvk=False, independent=family == 'independent', old_weights=np.ones(n))
    directory = 'results/analysis/external_control_copula_repaired/copula'
    sources.json(directory+'/target.json')
    control = sources.arrays(directory+'/posterior.npz', ['coordinate', 'weights', 'log_integrals'])
    control_model = selected_model()
    control_model.min_vals, control_model.max_vals = control_model.min_vals.double(), control_model.max_vals.double()
    yield dict(label='external_gp_truth', category='mcmc', model=control_model, latent=control['coordinate'].reshape(-1, 20),
               shape=control['coordinate'].shape[:2], old_weights=control['weights'].ravel(), exact=True, lvk=False,
               data=sources.arrays('results/analysis/external_gp_control_qmc_pilot/verification_events.npz', ['data'])['data'],
               test=sources.arrays('results/analysis/external_control_comparison/test_events.npz', ['data'])['data'],
               comparison='results/analysis/external_control_comparison', cached_events=control['log_integrals'][:, 1])
    lisa_directory = analysis['lisa_baseline_candidates']['bimodal'][0]
    model_config = sources.json(lisa_directory+'/summary.json')['configuration']
    refinement = sources.json('configs/lisa_refinement.json')
    sources.json('results/analysis/lisa_refinement/summary.json')
    lisa = sources.arrays('results/analysis/lisa_refinement/posterior.npz', ['latent'])
    weights = sources.arrays('results/analysis/lisa_refinement/weights.npz', ['weights'])['weights']
    lisa_model = sampling_model(model_config)
    lisa_model.lambda_det_mc_samples = refinement['primary_selection_points']
    lisa_model.selection_mc_seed = refinement['primary_selection_seed']
    yield dict(label='lisa_nuts_external', category='mcmc', model=lisa_model, latent=lisa['latent'].reshape(-1, 20),
               shape=lisa['latent'].shape[:2], old_weights=weights.ravel(), exact=False, lvk=False,
               data=sources.arrays(refinement['primary_event_catalog'], ['data'])['data'],
               test=sources.arrays('results/analysis/lisa_scientific_comparison/events_fine.npz', ['data'])['data'],
               comparison='results/analysis/external_lisa_comparison_double')
    lisa_dirs = set(sum(analysis['lisa_baseline_candidates'].values(), [])+analysis['lisa_sensitivity'])
    snr = LISASNR()
    for directory in sorted(lisa_dirs):
        c = sources.json(directory+'/summary.json')['configuration']
        with default_dtype(torch.float32):
            m = build_model(c['family'], c['components'], c['quadrature'], c['catalog_seed']+MORPHOLOGIES.index(c['morphology']))
        m.selection_integration = c['integration']
        m.selection_function = lambda points: torch.special.ndtr(snr.torch(points)-8.)
        latent = sources.arrays(directory+'/posterior.npz', ['latent'])['latent']
        yield dict(label=Path(directory).name, category='vi', model=m, latent=latent, shape=(1, len(latent)),
                   data=sources.arrays(directory+'/catalog.npz', ['data'])['data'], exact=False, lvk=False,
                   old_weights=np.ones(len(latent)))
    for entry in analysis['lvk_sensitivity']:
        directory = entry['directory']
        c = sources.json(directory+'/summary.json')['configuration']
        cache = f"external_data/lvk/analysis_inputs/importance_mixture_v2_{c['spin']}_{c['posterior_samples']}_{c['injections']}_{c['mmax']:g}_372100.npz"
        sources.add(cache)  # Require the old cache, never generate a replacement.
        arrays = load_inputs(c['spin'], c['posterior_samples'], c['injections'], c['mmax'])
        with default_dtype(torch.float32):
            m = estimator(SimpleNamespace(**c), arrays)
        latent = sources.arrays(directory+'/posterior.npz', ['latent'])['latent']
        yield dict(label='lvk_'+Path(directory).name, category='vi', model=m, latent=latent, shape=(1, len(latent)),
                   data=arrays['data'], exact=False, lvk=True, old_weights=np.ones(len(latent)),
                   event_names=arrays['event_names'].tolist())


@torch.no_grad()
def evaluate_case(case, config, check):
    old = case['model']
    chosen = indices_for_shape(case['shape'], config['mcmc_draws_per_chain'] if case['category'] == 'mcmc' else config['vi_draws'])
    new = corrected_model(old, case.get('independent', False))
    native_bounds = new.min_vals.clone(), new.max_vals.clone()
    new.min_vals, new.max_vals = old.min_vals.double(), old.max_vals.double()
    precise = copy.copy(old)
    precise.min_vals, precise.max_vals = old.min_vals.double(), old.max_vals.double()
    if getattr(old, 'injection_data', None) is not None:
        precise.injection_data = {k: v.double() if torch.is_tensor(v) else v for k, v in old.injection_data.items()}
        new.injection_data = precise.injection_data
    if case['exact']:
        new._selection_fraction_for_params = exact_selection
    elif not case['lvk']:
        fixed = old._selection_mc_samples_norm(old.lambda_det_mc_samples, old.device).double()
        precise._selection_mc_samples_norm = lambda n, device: fixed
        new._selection_mc_samples_norm = lambda n, device: fixed
        new.lambda_det_mc_samples = len(fixed)
    data = torch.tensor(case['data'])
    normal = ndtri(qmc.Sobol(3, scramble=True, seed=config['population_transport_seed']).random_base2(config['population_transport_power']))
    rows, populations, old_test, new_test = [], [], [], []
    bounds_indices = set(np.linspace(0, len(chosen)-1, config['native_bounds_checks_per_case'], dtype=int))
    native_deltas, reconstructed_errors, cached_errors = [], [], []
    for position, index in enumerate(chosen):
        check()
        if 'latent' in case:
            raw = torch.tensor(case['latent'][index])
            # LVK mass families need all their named parameter sites.
            if case['lvk'] or old.num_corr_tnormal:
                p_old = old._unflatten_latent_vector(raw)
                p = precise._unflatten_latent_vector(raw.double())
                for family, prefix, *_ in CorrectedPopulationEstimator._families:
                    if getattr(old, 'num_corr_'+family):
                        for m, param in ((old, p_old), (precise, p)):
                            param[prefix+'_L_cholesky'] = m._build_cholesky_from_unconstrained(param[prefix+'_cholesky_params'], 3, 'cpu')
            else:
                p_old = latent_sites(old, raw)
                p = latent_sites(precise, raw.double())
        else:
            p_old = {key: torch.tensor(value[index]) for key, value in case['samples'].items()}
            p = {key: value.double() for key, value in p_old.items()}
        # Older LVK summary adapters expect unconstrained correlation entries.
        p_summary = precise._unflatten_latent_vector(torch.tensor(case['latent'][index]).double()) if case['lvk'] else p
        old_e = event_integrals(old, p_old, data, config['event_block'])
        if case.get('cached_events') is not None:
            error = float(np.max(abs(old_e.numpy()-case['cached_events'][index])))
            cached_errors.append(error)
            if error > 1e-7:
                raise ValueError(f'Historical event cache is not reproduced: {error}')
        precise_e = event_integrals(precise, p, data.double(), config['event_block'])
        new_e = event_integrals(new, p, data.double(), config['event_block'])
        a_old, a_precise, a_new = [log_alpha(m, param) for m, param in ((old, p_old), (precise, p), (new, p))]
        targets = [e.double().sum()-len(data)*a.double() for e, a in ((old_e, a_old), (precise_e, a_precise), (new_e, a_new))]
        if position == 0:
            trace = pyro.poutine.trace(pyro.poutine.condition(old._pyro_model, data=p_old)).get_trace(data)
            actual = trace.nodes['obs_log_prob']['fn'].log_factor.sum().double()
            actual -= len(data)*(old.max_vals-old.min_vals).log().sum().double()
            error = float(actual-targets[0])
            reconstructed_errors.append(error)
            if abs(error) > (1e-3 if data.dtype == torch.float32 else 1e-7):
                raise ValueError(f'Historical factor reconstruction differs: {error}')
        if position in bounds_indices:
            matched = new.min_vals, new.max_vals
            new.min_vals, new.max_vals = native_bounds
            native = event_integrals(new, p, data.double(), config['event_block']).sum()-len(data)*log_alpha(new, p)
            native_deltas.append([int(index), float(native-targets[2])])
            new.min_vals, new.max_vals = matched
        old_points = legacy_transport(precise, p_summary, normal, case.get('independent', False), case['lvk'])
        new_points = new.component_transport(p, torch.tensor(normal)).numpy()
        first = population_quantities(precise, p, old_points, config['population_bins'], float(a_precise.exp()), len(data), case['lvk'])
        second = population_quantities(new, p, new_points, config['population_bins'], float(a_new.exp()), len(data), case['lvk'])
        rows.append(np.r_[[float(v) for v in targets], [float(v) for v in (a_old, a_precise, a_new)], first[0], second[0]])
        populations.append(np.stack([first[1], second[1], first[2], second[2]]))
        if 'test' in case:
            test = torch.tensor(case['test'], dtype=torch.float64)
            old_test.append((event_integrals(precise, p, test, config['event_block'])-a_precise).numpy())
            new_test.append((event_integrals(new, p, test, config['event_block'])-a_new).numpy())
        if (position+1) % 32 == 0:
            print(case['label'], position+1, '/', len(chosen), flush=True)
    rows, populations = np.array(rows), np.array(populations)
    if not np.isfinite(rows).all() or not np.isfinite(populations).all():
        raise ValueError('Nonfinite paired evaluation; retain the failure rather than sanitize it.')
    old_weights = case['old_weights'][chosen]
    old_weights /= old_weights.sum()
    delta = rows[:, 2]-rows[:, 0]
    weights, diagnostics = weight_diagnostics(delta, old_weights)
    density_weights, density_diagnostics = weight_diagnostics(rows[:, 2]-rows[:, 1], old_weights)
    feature_names = ['mean_mass' if case['lvk'] else 'mean_unit_mass', 'mean_redshift' if case['lvk'] else 'mean_unit_redshift',
                     'mean_mass_ratio' if case['lvk'] else 'mean_unit_ratio', 'mass_below_quarter_support',
                     'mass_above_0p8_support', 'alpha', 'conditional_rate_mean', 'domain_probability']
    old_features, new_features = rows[:, 6:14], rows[:, 14:22]
    old_mean, new_mean = old_weights@old_features, weights@new_features
    sd = np.sqrt(old_weights@((old_features-old_mean)**2))
    feature_rows = {}
    for j, name in enumerate(feature_names):
        feature_rows[name] = {'old_mean': old_mean[j], 'new_mean': new_mean[j],
            'old_sd': sd[j], 'shift_in_old_sd': float((new_mean[j]-old_mean[j])/sd[j]) if sd[j] > 1e-12 else None,
            'fixed_draw_mean_change': float(old_weights@(new_features[:, j]-old_features[:, j])),
            'old_quantiles_05_50_95': weighted_quantiles(old_features[:, j], old_weights, [.05, .5, .95]).tolist(),
            'new_quantiles_05_50_95': weighted_quantiles(new_features[:, j], weights, [.05, .5, .95]).tolist()}
    cells, cell_rows = {}, {}
    for label, a, b in (('intrinsic', 0, 1), ('detected', 2, 3)):
        before, fixed, after = old_weights@populations[:, a], old_weights@populations[:, b], weights@populations[:, b]
        cells.update({label+'_old': before, label+'_fixed': fixed, label+'_new': after})
        cell_rows[label] = {'fixed_draw_total_variation': float(.5*abs(fixed-before).sum()),
                            'total_variation': float(.5*abs(after-before).sum()),
                            'js_distance': js_distance(before, after)}
    summary = dict(label=case['label'], category=case['category'], draws=len(chosen), chain_shape=list(case['shape']),
                   historical_event_reconstruction_error=reconstructed_errors,
                   historical_event_cache_max_abs_error=max(cached_errors, default=None),
                   data_shape=list(data.shape), data_dtype=str(data.dtype),
                   weight_diagnostics=diagnostics, density_only_weight_diagnostics=density_diagnostics,
                   precision_delta_sd=float((rows[:, 1]-rows[:, 0]).std(ddof=1)),
                   native_bounds_target_change=native_deltas, features=feature_rows, cells=cell_rows,
                   lvk_detected_cells_available=not case['lvk'],
                   posterior_certified=False, historical_results_superseded=False)
    shifts = [abs(v['shift_in_old_sd']) for v in feature_rows.values() if v['shift_in_old_sd'] is not None]
    summary['screen_flagged'] = bool(diagnostics['delta_sd'] > config['screen_log_ratio_sd']
        or diagnostics['ratio_ess_fraction'] < config['screen_weight_ess_fraction']
        or max(shifts, default=0.) > config['screen_maximum_feature_shift_sd']
        or cell_rows['intrinsic']['total_variation'] > config['screen_maximum_cell_total_variation'])
    arrays = dict(indices=chosen, rows=rows, old_weights=old_weights, new_weights=weights,
                  density_only_weights=density_weights, feature_names=np.array(feature_names), **cells)
    if old_test:
        product, summary['prediction'] = prediction_summary(np.array(old_test), np.array(new_test), old_weights, weights)
        arrays.update(product)
    return summary, arrays


def frozen_gp_comparison(case, summary, arrays, sources, bins):
    directory = case.get('comparison')
    if not directory:
        return
    compact = case['label'] == 'external_compact'
    truth = sources.arrays(directory+('/recovery_truth.npz' if compact else '/truth.npz'), ['intrinsic', 'detected'])
    records = {}
    for resolution in (6, 12):
        product = sources.arrays(f'{directory}/gp_{resolution}'+('_comparison' if compact else '')+'.npz',
                                 ['log_integrals', 'intrinsic', 'detected'])
        score = logsumexp(product['log_integrals'], 0)-math.log(len(product['log_integrals']))
        row = {'old_subset_minus_gp_log_score': float((arrays['old_log_predictive']-score).sum()),
               'reweighted_subset_minus_gp_log_score': float((arrays['new_log_predictive']-score).sum()),
               'gp_draws': len(product['log_integrals']), 'recovery': {}}
        for kind in ('intrinsic', 'detected'):
            fine = round(len(truth[kind])**(1/3))
            target = coarsen(truth[kind], fine, bins)
            gp_fine = round(len(product[kind])**(1/3))
            gp = coarsen(product[kind], gp_fine, bins)
            row['recovery'][kind] = {label: js_distance(values, target) for label, values in (
                ('old_subset', arrays[kind+'_old']), ('reweighted_subset', arrays[kind+'_new']), ('gp', gp))}
        records[str(resolution)] = row
    summary['frozen_gp_comparison'] = records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/density_impact.json')
    parser.add_argument('--case', help='Run one named case, charging the same shared allowance.')
    args = parser.parse_args()
    start = PROCESS_STARTED
    config = json.loads((ROOT/args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    previous = list(output.glob('attempt_*/cost.json'))
    charged = sum(json.loads(p.read_text())['seconds'] for p in previous)
    directory = output/f'attempt_{len(previous):03d}'
    directory.mkdir(exist_ok=False)
    process = psutil.Process()
    peak, completed, status = 0, [], 'failed'
    sources = Sources()
    for path in (ROOT/'src/gwpopulation').glob('*.py'):
        sources.add(path)
    for name in ('configs/density_impact.json',
                 'docs/analysis/stable_population_validation.json', 'scripts/check_density_impact.py'):
        sources.add(name)
    # Record the local adapters actually imported by this runner, including
    # their transitive numerical dependencies, without relabeling old records.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().is_relative_to(ROOT/'scripts'):
            sources.add(Path(filename).resolve())
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    def check():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        elapsed = time.perf_counter()-start
        if elapsed+charged >= config['maximum_seconds'] or peak > config['maximum_resident_gib']*2**30:
            raise RuntimeError('Declared shared time or resident-memory allowance exhausted.')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > config['maximum_output_mib']*2**20:
            raise RuntimeError('Declared output allowance exhausted.')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            iterator = cases(sources, config)
        while True:
            check()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    case = next(iterator)
            except StopIteration:
                break
            if args.case and args.case != case['label']:
                continue
            print('Starting', case['label'], flush=True)
            began = time.perf_counter()
            summary, arrays = evaluate_case(case, config, check)
            frozen_gp_comparison(case, summary, arrays, sources, config['population_bins'])
            summary['seconds'] = time.perf_counter()-began
            np.savez_compressed(directory/(case['label']+'.npz'), **arrays)
            summary['product_sha256'] = digest(directory/(case['label']+'.npz'))
            write_json(directory/(case['label']+'.json'), summary)
            completed.append(summary)
            print('Finished', case['label'], 'flagged=', summary['screen_flagged'], 'seconds=', round(summary['seconds'], 2), flush=True)
        if not completed:
            raise ValueError('No matching case was evaluated.')
        status = 'complete'
    finally:
        elapsed = time.perf_counter()-start
        write_json(directory/'cost.json', {'status': status, 'seconds': elapsed, 'previous_seconds': charged,
                   'total_charge_seconds': charged+elapsed, 'maximum_seconds': config['maximum_seconds'],
                   'peak_resident_bytes': peak, 'case_filter': args.case, 'completed_cases': [r['label'] for r in completed]})
        write_json(directory/'sources.json', {'source_sha256': sources.hashes, 'configuration': config})
    write_json(directory/'summary.json', {'status': status, 'cases': completed,
               'posterior_certified': False, 'historical_results_superseded': False})
    print(json.dumps({'status': status, 'cases': len(completed), 'seconds': elapsed, 'output': str(directory)}))


if __name__ == '__main__':
    main()
