"""Conserved joint cell probabilities and exact marginal CDFs for mock fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
import torch

from validate_population_morphologies import ROOT, build_model, truth_sample
from gwpopulation.core import SafeKumaraswamy, TruncatedNormal


def components(model, latent):
    params = model._unflatten_latent_vector(torch.tensor(latent, dtype=torch.float64))
    weights = torch.softmax(params['weights_logits'], -1).numpy()
    if model.num_corr_kumaraswamy:
        marginal = SafeKumaraswamy(params['log_k_a'].exp()+.0001, params['log_k_b'].exp()+.0001)
        cholesky = model._build_cholesky_from_unconstrained(params['k_cholesky_params'], 3, 'cpu')
    else:
        marginal = TruncatedNormal(params['tn_locs'], params['log_tn_scales'].exp()+.0001, low=.0001, high=.9999)
        cholesky = model._build_cholesky_from_unconstrained(params['tnormal_cholesky_params'], 3, 'cpu')
    return weights, marginal, cholesky


def probability_cells(points, bins, weights=None):
    index = np.minimum((np.clip(points, 0, 1)*bins).astype(int), bins-1)
    flat = (index[:, 0]*bins+index[:, 1])*bins+index[:, 2]
    result = np.bincount(flat, weights=weights, minlength=bins**3)
    return result/result.sum()


@torch.no_grad()
def summarize(directory, power=19):
    summary = json.loads((directory/'summary.json').read_text())
    config = summary['configuration']
    model = build_model(config['family'], config['components'], 4096, 33)
    latents = np.load(directory/'posterior.npz')['latent']
    base = qmc.Sobol(3, scramble=True, seed=918225).random_base2(power)
    normal = ndtri(base)
    truth = truth_sample(config['morphology'], 2**20, np.random.default_rng(918226))
    bins_list = (12, 20, 28)
    targets = {bins: probability_cells(truth, bins) for bins in bins_list}
    cells = {bins: [] for bins in bins_list}
    coarse_cells = []
    edges = np.linspace(0, 1, 97)
    marginal_draws = []
    for latent in latents:
        weights, marginal, cholesky = components(model, latent)
        u = torch.tensor(ndtr(np.einsum('nj,kij->nki', normal, cholesky.numpy()))).clamp(1e-15, 1-1e-15)
        points = marginal.icdf(u).numpy()
        for bins in bins_list:
            cells[bins].append(sum(weight*probability_cells(points[:, k], bins)
                                   for k, weight in enumerate(weights)))
        coarse_cells.append(sum(weight*probability_cells(points[:2**(power-2), k], 20)
                                for k, weight in enumerate(weights)))
        cdf = marginal.cdf(torch.tensor(edges)[:, None, None].expand(-1, len(weights), 3))
        marginal_draws.append((np.diff(cdf.numpy(), axis=0)*weights[None, :, None]).sum(axis=1).T*96)
    distances = {}
    for bins in bins_list:
        p = np.array(cells[bins])
        distances[str(bins)] = {
            'predictive_to_truth': float(jensenshannon(p.mean(axis=0), targets[bins], base=2)),
            'draw_to_truth_quantiles': np.quantile(jensenshannon(p, np.broadcast_to(targets[bins], p.shape), axis=1, base=2), [.05,.5,.95]).tolist(),
            'within_posterior_quantiles': np.quantile(jensenshannon(p[::2], p[1::2], axis=1, base=2), [.05,.5,.95]).tolist(),
        }
    fine = np.array(cells[20])
    coarse = np.array(coarse_cells)
    marginal_truth = np.array([np.histogram(truth[:, i], bins=edges, density=True)[0] for i in range(3)])
    result = {'method': 'joint cell probabilities from transported Sobol points',
              'points_per_component': len(base), 'truth_sample_count': len(truth),
              'bins_per_axis': bins_list, 'distances': distances,
              'coarse_to_fine_distance_quantiles': np.quantile(jensenshannon(coarse, fine, axis=1, base=2), [.5,.9,.99,1]).tolist(),
              'coarse_to_fine_within_distance_change_quantiles': np.quantile(abs(
                  jensenshannon(coarse[::2], coarse[1::2], axis=1, base=2)
                  - jensenshannon(fine[::2], fine[1::2], axis=1, base=2)), [.5,.9,.99,1]).tolist(),
              'coarse_predictive_to_truth': float(jensenshannon(coarse.mean(axis=0), targets[20], base=2)),
              'maximum_cell_normalization_error': float(abs(fine.sum(axis=1)-1).max()),
              'core_sha256': hashlib.sha256((ROOT/'src/gwpopulation/core.py').read_bytes()).hexdigest()}
    (directory/'joint_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(directory/'density_summary.npz', marginal_x=(edges[:-1]+edges[1:])/2,
                        marginal_draws=np.array(marginal_draws).transpose(1,0,2), marginal_truth=marginal_truth,
                        joint_draws=fine, joint_truth=targets[20])
    print(directory.name, json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--power', type=int, default=19)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for directory in args.directories:
        summarize(ROOT / directory, args.power)


if __name__ == '__main__':
    main()
