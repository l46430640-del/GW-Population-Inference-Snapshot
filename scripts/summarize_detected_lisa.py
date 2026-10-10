"""Distinguish measured structure from selection-hidden population probability."""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.spatial.distance import jensenshannon
from scipy.stats import qmc
import torch

from build_lisa_analysis_assets import LISASNR
from summarize_analysis_populations import components, probability_cells
from validate_population_morphologies import ROOT, build_model, truth_sample


@torch.no_grad()
def summarize(directory, power=17):
    config = json.loads((directory/'summary.json').read_text())['configuration']
    model = build_model(config['family'], config['components'], 4096, 33)
    response = LISASNR()
    normal = ndtri(qmc.Sobol(3, scramble=True, seed=872300).random_base2(power))
    truth = truth_sample(config['morphology'], 2**20, np.random.default_rng(872301))
    detected_truth = probability_cells(truth, 12, ndtr(response(truth)-8))
    latents = np.load(directory/'posterior.npz')['latent']
    cells, efficiencies, hidden = [], [], []
    for latent in latents:
        weights, marginal, cholesky = components(model, latent)
        uniform = torch.tensor(ndtr(np.einsum('nj,kij->nki', normal, cholesky.numpy())))
        points = marginal.icdf(uniform.clamp(1e-15, 1-1e-15)).numpy()
        detection = ndtr(response(np.clip(points.reshape(-1,3),0,1))-8).reshape(points.shape[:2])
        efficiency = detection.mean(0)
        selected_weights = weights*efficiency
        selected_weights /= selected_weights.sum()
        cells.append(sum(weight*probability_cells(points[:,k],12,detection[:,k])
                         for k,weight in enumerate(selected_weights)))
        efficiencies.append(float(weights@efficiency))
        hidden.append(float(weights@np.mean(detection<.1,axis=0)))
    cells = np.array(cells)
    within = jensenshannon(cells[::2],cells[1::2],base=2,axis=1)
    result = {'definition':'p_det(theta|Lambda) = P_det(theta) p(theta|Lambda) / alpha(Lambda)',
              'cells_per_axis':12,'samples_per_component':len(normal),
              'predictive_to_truth':float(jensenshannon(cells.mean(0),detected_truth,base=2)),
              'within_posterior_quantiles':np.quantile(within,[.05,.5,.95]).tolist(),
              'intrinsic_probability_with_detection_below_0p1':np.quantile(hidden,[.05,.5,.95]).tolist(),
              'truth_probability_with_detection_below_0p1':float(np.mean(ndtr(response(truth)-8)<.1)),
              'alpha_quantiles':np.quantile(efficiencies,[.05,.5,.95]).tolist()}
    (directory/'detected_joint_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(directory/'detected_joint_summary.npz',joint_draws=cells,joint_truth=detected_truth)
    print(directory.name,json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories',nargs='+')
    parser.add_argument('--power',type=int,default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for directory in args.directories:
        summarize(ROOT/directory,args.power)


if __name__ == '__main__':
    main()
