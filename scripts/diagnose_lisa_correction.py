"""Test whether saved LISA flows support reliable posterior reweighting."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import time

import numpy as np
import torch

from benchmark_selected_mixture import runtime_metadata
from build_lisa_analysis_assets import LISASNR
from correct_selected_posterior import (
    draw_proposal, importance_weights, latent_log_target, proposal_log_prob,
    sha256, weighted_quantiles,
)
from predict_corrected_posterior import systematic_indices
from validate_population_morphologies import ROOT, MORPHOLOGIES, build_model


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def lisa_model(configuration, dtype=torch.float64, checkpoint=None):
    if (configuration['detector'], configuration['family'], configuration['components']) != ('lisa', 'kumaraswamy', 2):
        raise ValueError('This experiment requires a two-component Kumaraswamy LISA fit')
    seed = configuration['catalog_seed'] + MORPHOLOGIES.index(configuration['morphology'])
    with default_dtype(dtype):
        model = build_model('kumaraswamy', 2, configuration['quadrature'], seed)
    model.selection_integration = configuration['integration']
    snr = LISASNR()
    model.selection_function = lambda points: torch.special.ndtr(snr.torch(points) - 8.)
    # Preserve the original float32 Sobol coordinates; only evaluation dtype changes.
    original_points = model._selection_mc_samples_norm
    model._selection_mc_samples_norm = lambda count, device: original_points(count, device).to(dtype=dtype)
    if checkpoint is not None:
        saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
        model.flow_transforms.load_state_dict(saved['state'])
    return model


@torch.no_grad()
def target_value(model, latent, data):
    with default_dtype(latent.dtype):
        return float(latent_log_target(model, latent, data))


def verified_inputs(config, config_path):
    paths = [config_path, ROOT / 'scripts/diagnose_lisa_correction.py',
             ROOT / 'scripts/correct_selected_posterior.py', ROOT / 'scripts/predict_corrected_posterior.py',
             ROOT / 'scripts/benchmark_selected_mixture.py', ROOT / 'scripts/benchmark_variational_posterior.py',
             ROOT / 'scripts/validate_population_morphologies.py', ROOT / 'scripts/build_lisa_analysis_assets.py',
             ROOT / 'src/gwpopulation/core.py', ROOT / 'model_assets/lisa_analysis/phenom_d_grid.npz',
             ROOT / config['fine_catalog']]
    configurations, catalogs = [], []
    for relative in config['sources']:
        directory = ROOT / relative
        configurations.append(json.loads((directory / 'summary.json').read_text())['configuration'])
        catalogs.append(dict(np.load(directory / 'catalog.npz')))
        paths.extend(directory / name for name in ('summary.json', 'flow.pt', 'posterior.npz', 'catalog.npz'))
    settings = ('detector', 'morphology', 'family', 'components', 'quadrature', 'integration', 'catalog_seed', 'posterior_samples')
    if len({tuple(c[key] for key in settings) for c in configurations}) != 1:
        raise ValueError('Proposal starts have different population targets')
    for catalog in catalogs[1:]:
        for key in catalogs[0]:
            np.testing.assert_array_equal(catalog[key], catalogs[0][key])
    fine = dict(np.load(ROOT / config['fine_catalog']))
    for key in ('truth_intrinsic', 'measurement', 'rho_observed', 'detected'):
        np.testing.assert_array_equal(fine[key], catalogs[0][key])
    return configurations, catalogs[0]['data'], fine['data'], {
        path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def precision_check(config, configurations, models, data, output, sources):
    path = output / 'precision.json'
    if path.exists():
        report = json.loads(path.read_text())
        if report['source_sha256'] != sources:
            raise ValueError('Precision check belongs to different inputs')
        return report
    started = time.perf_counter()
    records = []
    for relative, configuration, model in zip(config['sources'], configurations, models):
        directory = ROOT / relative
        legacy = lisa_model(configuration, torch.float32, directory / 'flow.pt')
        latents = np.load(directory / 'posterior.npz')['latent']
        indices = np.linspace(0, len(latents) - 1, config['precision_draws_per_start'], dtype=int)
        differences, density_differences = [], []
        for value in latents[indices]:
            raw = torch.tensor(value, dtype=torch.float32)
            precise = raw.double()
            differences.append(target_value(model, precise, data.double()) - target_value(legacy, raw, data.float()))
            with torch.no_grad():
                density_differences.append(float(model.flow_dist.log_prob(precise) - legacy.flow_dist.log_prob(raw)))
        records.append({'source': relative, 'draws': len(indices),
                        'target_difference': differences, 'proposal_difference': density_differences})
    maximum_target = max(abs(value) for row in records for value in row['target_difference'])
    maximum_proposal = max(abs(value) for row in records for value in row['proposal_difference'])
    report = {'source_sha256': sources, 'seconds': time.perf_counter() - started,
              'maximum_absolute_target_difference': maximum_target,
              'maximum_absolute_proposal_difference': maximum_proposal,
              'passed': bool(max(maximum_target, maximum_proposal) < config['precision_log_density_tolerance']),
              'records': records}
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Precision check', maximum_target, maximum_proposal, flush=True)
    return report


@torch.no_grad()
def refinement_check(model, latent, data, fine_data, log_target, weights, config, seed):
    groups = {
        'uniform': systematic_indices(np.ones(len(latent)), min(len(latent), config['check_uniform_draws']), seed),
        'psis': systematic_indices(weights['psis'], config['check_weighted_draws'], seed + 1),
        'largest_raw_weights': np.argsort(weights['raw'])[-config['check_largest_weights']:],
    }
    indices = np.unique(np.concatenate(list(groups.values())))
    original_count = model.lambda_det_mc_samples
    event_change, selection_change = [], []
    try:
        for index in indices:
            point = latent[index]
            event_target = target_value(model, point, fine_data)
            model.lambda_det_mc_samples = config['fine_selection_points']
            refined_target = target_value(model, point, fine_data)
            model.lambda_det_mc_samples = original_count
            event_change.append(event_target - log_target[index])
            selection_change.append(refined_target - event_target)
    finally:
        model.lambda_det_mc_samples = original_count
    event_change, selection_change = np.array(event_change), np.array(selection_change)
    result = {}
    for name, selected in groups.items():
        offsets = np.searchsorted(indices, selected)
        total = event_change[offsets] + selection_change[offsets]
        result[name] = {'draws': len(selected), 'unique_draws': len(np.unique(selected)),
                        'total_change_sd': float(total.std()),
                        'total_change_05_50_95': np.quantile(total, [.05, .5, .95]).tolist(),
                        'maximum_absolute_total_change': float(abs(total).max()),
                        'event_change_sd': float(event_change[offsets].std()),
                        'selection_change_sd': float(selection_change[offsets].std())}
    return result, {'refinement_indices': indices, 'refinement_event_change': event_change,
                    'refinement_selection_change': selection_change,
                    **{f'refinement_{key}_indices': value for key, value in groups.items()}}


def eligibility(diagnostic, numerical, precision_passed, config):
    stable = (np.isfinite(diagnostic['pareto_k'])
              and diagnostic['pareto_k'] < min(config['maximum_pareto_k'], diagnostic['pareto_k_limit'])
              and all(diagnostic[key]['ess'] >= config['minimum_effective_samples'] for key in ('raw', 'psis')))
    numerical_passed = all(value['total_change_sd'] < config['maximum_refined_log_target_sd']
                           for value in numerical.values())
    return {'stable_weights': bool(stable), 'precision_passed': bool(precision_passed),
            'integration_passed': bool(numerical_passed),
            'eligible_for_corrected_summaries': bool(stable and precision_passed and numerical_passed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lisa_correction.json')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    configurations, coarse_array, fine_array, sources = verified_inputs(config, config_path)
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    data, fine = torch.tensor(coarse_array, dtype=torch.float64), torch.tensor(fine_array, dtype=torch.float64)
    models = [lisa_model(c, checkpoint=ROOT / relative / 'flow.pt')
              for c, relative in zip(configurations, config['sources'])]
    precision = precision_check(config, configurations, models, data, output, sources)
    if not precision['passed']:
        raise ValueError('Numerical precision change exceeds the declared tolerance; report retained')
    seeds = [config['pilot_seed']] if args.pilot else config['repeat_seeds']
    count = config['pilot_draws'] if args.pilot else config['draws_per_repeat']
    for seed in seeds:
        label = f'pilot_{seed}' if args.pilot else f'repeat_{seed}'
        path = output / f'{label}.json'
        if path.exists() or path.with_suffix('.npz').exists():
            raise FileExistsError(f'Completed draws are retained: {path}')
        started = time.perf_counter()
        latent, source_index, permuted = draw_proposal(models, count, seed)
        log_proposal = proposal_log_prob(models, latent).numpy()
        log_target, alpha = np.empty(count), np.empty(count)
        for index, point in enumerate(latent):
            log_target[index] = target_value(models[0], point, data)
            with torch.no_grad():
                params = models[0]._unflatten_latent_vector(point, apply_numerical_limits=False)
                alpha[index] = float(models[0]._selection_fraction_for_params(params))
            if (index + 1) % 1024 == 0:
                print(label, index + 1, 'target evaluations', flush=True)
        evaluation_seconds = time.perf_counter() - started
        weights, diagnostic = importance_weights(log_target - log_proposal)
        numerical, arrays = refinement_check(models[0], latent, data, fine, log_target, weights, config,
                                              config['check_seed'] + seed)
        decision = eligibility(diagnostic, numerical, precision['passed'], config)
        if args.pilot:
            decision['eligible_for_corrected_summaries'] = False
        alpha_summary = {}
        for name, w in {'unweighted': np.ones(count) / count, **weights}.items():
            alpha_summary[name] = {'mean': float(np.dot(w, alpha)),
                                   'quantiles_05_50_95': weighted_quantiles(alpha, w).tolist()}
        np.savez_compressed(path.with_suffix('.npz'), latent=latent.numpy(), log_target=log_target,
                            log_proposal=log_proposal, source_index=source_index.numpy(), permuted=permuted.numpy(),
                            raw_weights=weights['raw'], psis_weights=weights['psis'], alpha=alpha, **arrays)
        result = {'configuration': config, 'seed': seed, 'pilot': args.pilot, 'source_sha256': sources,
                  'events': len(data), 'event_samples': data.shape[1],
                  'target_evaluation_seconds': evaluation_seconds,
                  'seconds': time.perf_counter() - started, 'runtime': runtime_metadata(),
                  'weight_diagnostics': diagnostic, 'refinement': numerical, 'decision': decision,
                  'diagnostic_alpha_summaries': alpha_summary,
                  'draw_product_sha256': sha256(path.with_suffix('.npz')),
                  'scope': 'Applicability diagnosis for one LISA catalog. Weighted functionals are diagnostic only unless both repeats pass. No MCMC reference or repeated-catalog coverage is implied.'}
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: result[key] for key in ('seed', 'seconds', 'weight_diagnostics', 'decision')}), flush=True)


if __name__ == '__main__':
    main()
