"""Common-support marginal comparisons and detected joint probability summaries."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import jensenshannon
import torch

from analyze_lvk_population import ROOT, estimator, load_inputs
from analysis_results import configuration
from summarize_analysis_populations import probability_cells


def common_marginals(data):
    # Extend compact-support densities by zero rather than conditioning marginals
    # separately on a mass cut, which would not condition the joint population.
    limits = [(np.log10(3),np.log10(300)),(0,2.3),(0,1)]
    result, retained = [], []
    for axis,(low,high) in enumerate(limits):
        x = data['marginal_x'][axis]
        dx = x[1]-x[0]
        original_edges = np.r_[x-dx/2,x[-1]+dx/2]
        cdfs = np.c_[np.zeros(len(data['marginal_draws'][axis])),
                     np.cumsum(data['marginal_draws'][axis]*dx,axis=1)]
        edges = np.linspace(low,high,97)
        bins = np.array([np.diff(np.interp(edges,original_edges,cdf)) for cdf in cdfs])
        retained.append(bins.sum(1))
        result.append(bins/bins.sum(1,keepdims=True))
    return np.array(result), retained


@torch.no_grad()
def summarize(directory):
    source = json.loads((directory/'summary.json').read_text())
    config = SimpleNamespace(**source['configuration'])
    data = np.load(directory/'density_summary.npz')
    reference = np.load(ROOT/'results/analysis/lvk_reference/density_summary.npz')
    marginals, retained = common_marginals(data)
    comparison, _ = common_marginals(reference)
    distances = jensenshannon(marginals.mean(1),comparison.mean(1),base=2,axis=1)
    within = jensenshannon(marginals[:,::2],marginals[:,1::2],base=2,axis=2)
    reference_within = jensenshannon(comparison[:,::2],comparison[:,1::2],base=2,axis=2)
    probabilities = {}
    for name,axis,threshold in (('m1_below_5',0,np.log10(5)),('m1_below_10',0,1.),
                               ('m1_above_50',0,np.log10(50)),('m1_above_150',0,np.log10(150)),('z_above_1',1,1.),
                               ('q_below_0p5',2,.5)):
        x = data['marginal_x'][axis]
        dx = x[1]-x[0]
        cdfs = np.c_[np.zeros(len(data['marginal_draws'][axis])),
                     np.cumsum(data['marginal_draws'][axis]*dx,axis=1)]
        edges = np.r_[x-dx/2,x[-1]+dx/2]
        values = np.array([np.interp(threshold,edges,cdf) for cdf in cdfs])
        if 'above' in name:
            values = 1-values
        values = np.clip(values,0,1)
        probabilities[name] = np.quantile(values,[.05,.5,.95]).tolist()
    arrays = load_inputs(config.spin,config.posterior_samples,config.injections,config.mmax)
    model = estimator(config,arrays)
    physical = arrays['full_injection_points']
    unit = (torch.tensor(physical)-model.min_vals)/(model.max_vals-model.min_vals)
    mass = 10**physical[:,0]
    q = 3/mass+(1-3/mass)*physical[:,2]
    common = np.column_stack([(physical[:,0]-np.log10(3))/2,physical[:,1]/2.3,q])
    log_draw = torch.tensor(arrays['full_injection_log_p_draw'])
    detected = []
    for latent in np.load(directory/'posterior.npz')['latent']:
        params = model._unflatten_latent_vector(torch.tensor(latent))
        log_weight = model._log_pdf_phys_from_params(params,unit)-log_draw
        weight = (log_weight-log_weight.max()).exp().numpy()
        detected.append(probability_cells(common,12,weight))
    detected = np.array(detected)
    result = {'common_integration_domain':{'m1':[3,300],'z':[0,2.3],'q':[0,1]},
              'support_convention':'normalized original marginal densities extended by zero outside their original support',
              'marginal_order':['log10_m1','z','q'],'distance':'square-root Jensen-Shannon, base 2',
              'marginal_mean_distance_to_reference':distances.tolist(),
              'within_model_distance_quantiles':np.quantile(within,[.05,.5,.95],axis=1).T.tolist(),
              'within_reference_distance_quantiles':np.quantile(reference_within,[.05,.5,.95],axis=1).T.tolist(),
              'intrinsic_probabilities':probabilities,
              'maximum_probability_normalization_error':float(max(abs(p-1).max() for p in retained)),
              'detected_joint_grid':{'m1':[3,300],'z':[0,2.3],'q':[0,1],'cells_per_axis':12},
              'within_detected_joint_distance_quantiles':np.quantile(jensenshannon(
                  detected[::2],detected[1::2],base=2,axis=1),[.05,.5,.95]).tolist()}
    np.savez_compressed(directory/'comparison_summary.npz',marginal_probabilities=marginals,
                        detected_joint_probabilities=detected)
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(directory.name,json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories',nargs='*')
    args = parser.parse_args()
    torch.set_num_threads(1)
    directories = args.directories or [entry['directory'] for entry in configuration()['lvk_sensitivity']]
    for directory in directories:
        summarize(ROOT/directory)


if __name__ == '__main__':
    main()
