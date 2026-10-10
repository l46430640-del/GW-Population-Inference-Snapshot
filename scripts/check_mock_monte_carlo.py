"""Compare event integrals using independent, larger posterior samples."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from validate_population_morphologies import ROOT, MORPHOLOGIES, build_model, catalog


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--draws', type=int, default=96)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = ROOT / args.directory
    summary = json.loads((directory/'summary.json').read_text())
    config = summary['configuration']
    seed = config['catalog_seed'] + MORPHOLOGIES.index(config['morphology'])
    generator = catalog
    if config['detector'] == 'lisa':
        from lisa_noisy_catalog import catalog as generator
    cache = directory.parent/f"{config['detector']}_{config['morphology']}_catalog_s{seed}_n{args.samples}.npz"
    if cache.exists():
        fine_data = np.load(cache)['data']
    else:
        fine, truth = generator(config['morphology'], seed, n_samples=args.samples)
        original = np.load(directory/'catalog.npz')
        np.testing.assert_array_equal(truth['truth_intrinsic'], original['truth_intrinsic'])
        np.testing.assert_array_equal(truth['measurement'], original['measurement'])
        np.testing.assert_array_equal(truth['rho_observed'], original['rho_observed'])
        np.savez_compressed(cache, data=fine.numpy(), **truth)
        fine_data = fine.numpy()
    coarse_data = np.load(directory/'catalog.npz')['data']
    latent = np.load(directory/'posterior.npz')['latent'][:args.draws]
    model = build_model(config['family'], config['components'], config['quadrature'], seed)
    log_means, minima = [], []
    for array in (coarse_data, fine_data):
        points = torch.tensor(array[..., :3].reshape(-1, 3))
        log_values, effective = [], []
        for value in latent:
            params = model._unflatten_latent_vector(torch.tensor(value))
            logp = model._log_pdf_phys_from_params(params, points).reshape(array.shape[:2])
            log_values.append((torch.logsumexp(logp, 1)-np.log(array.shape[1])).numpy())
            w = (logp-logp.max(1, keepdim=True).values).exp()
            effective.append(float((w.sum(1).square()/w.square().sum(1)).min()))
        log_means.append(np.array(log_values))
        minima.append(effective)
    error = log_means[0]-log_means[1]
    result = {'coarse_samples': len(coarse_data[0]), 'fine_samples': args.samples,
              'posterior_draws': len(latent),
              'event_log_integral_difference_quantiles': np.quantile(error, [.05,.5,.95]).tolist(),
              'summed_log_integral_difference_quantiles': np.quantile(error.sum(1), [.05,.5,.95]).tolist(),
              'summed_log_integral_difference_std': float(error.sum(1).std()),
              'minimum_fine_event_ess_quantiles': np.quantile(minima[1], [.05,.5,.95]).tolist()}
    (directory/'event_integral_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
