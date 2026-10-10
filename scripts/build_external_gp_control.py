"""Construct the predeclared GP-family positive control without fit selection."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch

from benchmark_selected_mixture import event_posterior, protocol as copula_protocol, runtime_metadata
from binned_gp_benchmark import controlled_selection, grid_geometry, population_functionals
from correct_selected_posterior import sha256
from run_lisa_production import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/external_gp_control_catalog.json'


def settings():
    config = json.loads(CONFIG.read_text())
    protocol = json.loads((ROOT / config['protocol']).read_text())
    experiment = next(row for row in protocol['experiments'] if row['name'] == config['experiment'])
    return config, protocol, experiment


def gp_truth(experiment):
    bins = experiment['truth_grid_per_axis']
    lower, upper = grid_geometry(bins)
    centers = (lower+upper)/2
    delta = (centers[:, None]-centers[None, :])/np.asarray(experiment['truth_length_scales'])
    covariance = experiment['truth_sigma']**2*np.exp(-.5*np.sum(delta**2, axis=-1))
    covariance += experiment['truth_covariance_jitter']*np.eye(len(centers))
    cholesky = np.linalg.cholesky(covariance)
    white = np.random.default_rng(experiment['truth_seed']).standard_normal(len(centers))
    field = cholesky @ white
    probabilities = np.exp(field-logsumexp(field))
    selection = controlled_selection(bins)*bins**3
    return {'centers': centers, 'covariance': covariance, 'cholesky': cholesky,
            'white': white, 'field': field, 'probabilities': probabilities,
            'detected_probabilities': probabilities*selection/(probabilities @ selection),
            'alpha': np.asarray(probabilities @ selection)}


def selected_catalog(truth, bins, count, seed, sigma, batch_size):
    rng = np.random.default_rng(seed)
    lower, _ = grid_geometry(bins)
    points, indices = [], []
    proposals = 0
    while sum(map(len, points)) < count:
        index = rng.choice(bins**3, size=batch_size, p=truth['probabilities'])
        proposal = lower[index] + rng.random((batch_size, 3))/bins
        keep = rng.random(batch_size) < (.05+.95*proposal[:, 0]**2)
        points.append(proposal[keep])
        indices.append(index[keep])
        proposals += batch_size
    latent = np.concatenate(points)[:count]
    index = np.concatenate(indices)[:count]
    observed = latent + rng.normal(size=(count, 3))*sigma
    return {'truth': latent, 'truth_bin': index, 'observed': observed,
            'sigma': np.asarray(sigma), 'proposals_generated': np.asarray(proposals)}


def event_cloud(observed, sigma, count, seed):
    # The retained rejection sampler uses this measurement law, including the
    # detection indicator in the uniform-prior event posterior exactly once.
    if not np.array_equal(sigma, copula_protocol()['measurement_standard_deviations']):
        raise ValueError('The common controlled measurement law has changed')
    rng = np.random.default_rng(seed)
    points = np.asarray([event_posterior(y, count, rng) for y in observed])
    return np.concatenate([points, np.ones((*points.shape[:2], 1))], -1)


def verify_manifest(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Catalog source changed: {name}')
    for name, expected in manifest['product_sha256'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Catalog product changed: {name}')
    return manifest


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config, protocol, experiment = settings()
    output = ROOT / config['output']
    if (output / 'manifest.json').exists():
        verify_manifest(output)
        print('Existing GP control verified; no population or catalog redrawn.', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Unverified partial catalog is retained')
    started = time.perf_counter()
    truth = gp_truth(experiment)
    bins = experiment['truth_grid_per_axis']
    sigma = np.asarray(experiment['measurement_standard_deviations'])
    train = selected_catalog(truth, bins, experiment['train_count'], experiment['catalog_seed'], sigma, config['proposal_batch_size'])
    test = selected_catalog(truth, bins, experiment['test_count'], experiment['test_seed'], sigma, config['proposal_batch_size'])
    train['data'] = event_cloud(train['observed'], sigma, config['training_event_samples'], config['training_event_seed'])
    verification = {'data': event_cloud(train['observed'], sigma, config['verification_training_event_samples'], config['verification_training_event_seed']),
                    'observed': train['observed'], 'sigma': sigma}
    for name, product in [('truth', truth), ('train_catalog', train), ('train_verification', verification), ('test_catalog', test)]:
        if any(not np.isfinite(value).all() for value in product.values()):
            raise ValueError('Nonfinite catalog product')
        np.savez_compressed(output / (name+'.npz'), **product)
    functional = population_functionals(torch.tensor(truth['field']), controlled_selection(bins), bins).numpy()
    paths = [CONFIG, ROOT / config['protocol'], Path(__file__), ROOT / 'scripts/benchmark_selected_mixture.py',
             ROOT / 'scripts/binned_gp_benchmark.py', ROOT / 'configs/model_comparisons.json',
             ROOT / 'scripts/correct_selected_posterior.py', ROOT / 'scripts/run_lisa_production.py']
    manifest = {'configuration': config, 'experiment': experiment,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
        'product_sha256': {p.name: sha256(p) for p in output.glob('*.npz')},
        'truth_covariance_reconstruction_error': float(abs(truth['cholesky'] @ truth['cholesky'].T-truth['covariance']).max()),
        'truth_probability_sum': float(truth['probabilities'].sum()), 'truth_alpha': float(truth['alpha']),
        'truth_functionals': functional.tolist(),
        'fit_inputs': {'copula': ['train_catalog:data'], 'gp': ['train_catalog:observed', 'train_catalog:sigma']},
        'test_access': 'Not used for initialization, model choice or pilot fitting.',
        'generation_seconds': time.perf_counter()-started, 'runtime': runtime_metadata(),
        'numpy_version': np.__version__, 'eligible_for_scientific_comparisons': False}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('truth_covariance_reconstruction_error', 'truth_probability_sum',
        'truth_alpha', 'generation_seconds')}), flush=True)


if __name__ == '__main__':
    main()
