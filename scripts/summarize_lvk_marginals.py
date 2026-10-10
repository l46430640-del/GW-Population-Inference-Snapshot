"""Integrate the joint copula population conditioned on injection coverage."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
import torch

from analyze_lvk_population import ROOT, estimator, load_inputs
from gwpopulation.core import SafeKumaraswamy
from lvk_injections import covered_mass_pairs


def mass_quadrature(model, order):
    low, high = np.log10(3), float(model.max_vals[0])
    knots = np.unique(np.r_[np.linspace(0,1,6),
                           (np.linspace(np.log10(5),np.log10(50),17)-low)/(high-low)])
    nodes, weights = leggauss(order)
    widths = np.diff(knots)
    points = (knots[:-1,None] + (nodes[None]+1)*widths[:,None]/2).ravel()
    weights = (weights[None]*widths[:,None]/2).ravel()
    return points, weights


def marginal_distributions(model, params):
    if model.num_corr_axis_spline_kumaraswamy:
        full = model._axis_spline_kumaraswamy_marginal_dist(
            params['axis_sk_m1_logits'], params['axis_sk_log_k_a'], params['axis_sk_log_k_b'])
        chol = model._build_cholesky_from_unconstrained(params['axis_sk_cholesky_params'], 3, 'cpu')
        return full.marginal_dists, chol
    a, b = params['log_k_a'].exp()+.0001, params['log_k_b'].exp()+.0001
    chol = model._build_cholesky_from_unconstrained(params['k_cholesky_params'], 3, 'cpu')
    return [SafeKumaraswamy(a[:,i], b[:,i]) for i in range(3)], chol


def piecewise_mass_icdf(distribution, probabilities):
    if not hasattr(distribution, 'global_dist'):
        return distribution.icdf(torch.tensor(probabilities)).numpy()
    knots = np.unique(np.r_[np.linspace(0,1,distribution.global_dist.num_knots),
                           np.linspace(float(distribution.local_low),float(distribution.local_high),
                                       distribution.local_dist.num_knots)])
    widths = np.diff(knots)
    first = distribution.log_prob(torch.tensor(knots[:-1]+.25*widths)[:,None]).exp().numpy()
    second = distribution.log_prob(torch.tensor(knots[:-1]+.75*widths)[:,None]).exp().numpy()
    slope = 2*(second-first)
    height = first-.25*slope
    areas = widths[:,None]*(height+.5*slope)
    cumulative = np.vstack([np.zeros(areas.shape[1]),areas.cumsum(axis=0)])
    result = np.zeros_like(probabilities)
    for k in range(probabilities.shape[1]):
        index = np.clip(np.searchsorted(cumulative[:,k],probabilities[:,k],side='right')-1,0,len(widths)-1)
        target = (probabilities[:,k]-cumulative[index,k])/widths[index]
        root = np.sqrt(np.maximum(height[index,k]**2+2*slope[index,k]*target,0))
        fraction = 2*target/(height[index,k]+root)
        result[:,k] = knots[index]+widths[index]*np.clip(fraction,0,1)
    return result


@torch.no_grad()
def marginal_bins_transport(model, latent, normal, n_bins=96):
    params = model._unflatten_latent_vector(torch.tensor(latent,dtype=torch.float64))
    weights = torch.softmax(params['weights_logits'],-1).numpy()
    distributions, cholesky = marginal_distributions(model,params)
    edges = np.linspace(0,1,n_bins+1)
    cdfs = [(dist.cdf(torch.tensor(edges)[:,None]).numpy()*weights).sum(axis=1)
            for dist in distributions[:2]]
    matrix = (cholesky@cholesky.transpose(-1,-2)).numpy()
    correlation = matrix[:,0,2]
    probability_mass = np.broadcast_to(ndtr(normal[:,0,None]),(len(normal),len(weights)))
    mass_unit = piecewise_mass_icdf(distributions[0],probability_mass.copy())
    probability_u = ndtr(normal[:,0,None]*correlation+normal[:,1,None]*np.sqrt(1-correlation**2))
    u = distributions[2].icdf(torch.tensor(probability_u)).numpy()
    mass = 10**(float(model.min_vals[0])+mass_unit*float(model.max_vals[0]-model.min_vals[0]))
    ratio = 3/mass+(1-3/mass)*u
    q_probabilities = sum(weight*np.histogram(ratio[:,k],bins=edges)[0]/len(ratio)
                          for k,weight in enumerate(weights))
    q_coarse = sum(weight*np.histogram(ratio[:len(ratio)//4,k],bins=edges)[0]/(len(ratio)//4)
                   for k,weight in enumerate(weights))
    widths = np.array([float(model.max_vals[0]-model.min_vals[0]),2.3,1.])
    bins = np.vstack([np.diff(cdfs,axis=1),q_probabilities])*n_bins/widths[:,None]
    return bins, .5*np.sum(abs(q_coarse-q_probabilities))


@torch.no_grad()
def marginal_bins(model, latent, order=32, n_bins=96):
    params = model._unflatten_latent_vector(torch.tensor(latent, dtype=torch.float64))
    weights = torch.softmax(params['weights_logits'], -1).numpy()
    distributions, cholesky = marginal_distributions(model, params)
    edges = np.linspace(0,1,n_bins+1)
    cdfs = []
    for dist in distributions[:2]:
        cdfs.append((dist.cdf(torch.tensor(edges)[:,None]).numpy()*weights).sum(axis=1))
    points, quadrature_weights = mass_quadrature(model, order)
    mass = 10**(np.log10(3)+points*(float(model.max_vals[0])-np.log10(3)))
    f_mass = distributions[0].log_prob(torch.tensor(points)[:,None]).exp().numpy()
    eta = ndtri(np.clip(distributions[0].cdf(torch.tensor(points)[:,None]).numpy(), 1e-15,1-1e-15))
    matrix = (cholesky @ cholesky.transpose(-1,-2)).numpy()
    correlation = matrix[:,0,2]
    u = (edges[:,None]-3/mass[None])/(1-3/mass[None])
    fu = distributions[2].cdf(torch.tensor(np.clip(u,0,1))[:,:,None]).numpy()
    probability = ndtr((ndtri(fu)-eta[None]*correlation)/np.sqrt(1-correlation**2))
    q_cdf = np.sum(probability * f_mass[None] * quadrature_weights[None,:,None] * weights, axis=(1,2))
    q_cdf[0], q_cdf[-1] = 0., 1.
    cdfs.append(q_cdf)
    widths = np.array([float(model.max_vals[0])-np.log10(3),2.3,1.])
    return np.diff(cdfs,axis=1)*n_bins/widths[:,None]


@torch.no_grad()
def conditioned_marginal_bins(model, latent, normal, n_bins=96):
    params = model._unflatten_latent_vector(torch.tensor(latent,dtype=torch.float64))
    weights = torch.softmax(params['weights_logits'],-1).numpy()
    distributions, cholesky = marginal_distributions(model,params)
    probability = np.clip(ndtr(np.einsum('nj,kij->nki',normal,cholesky.numpy())),1e-15,1-1e-15)
    unit_mass = piecewise_mass_icdf(distributions[0],probability[:,:,0])
    unit_z = distributions[1].icdf(torch.tensor(probability[:,:,1])).numpy()
    u = distributions[2].icdf(torch.tensor(probability[:,:,2])).numpy()
    mass = 10**(float(model.min_vals[0])+unit_mass*float(model.max_vals[0]-model.min_vals[0]))
    secondary = 3+(mass-3)*u
    points = np.stack([unit_mass,unit_z,secondary/mass],axis=-1)
    keep = covered_mass_pairs(mass,secondary)
    results, normalizations = [], []
    edges = np.linspace(0,1,n_bins+1)
    for count in (len(normal)//4,len(normal)):
        normalization = sum(weight*np.mean(keep[:count,k]) for k,weight in enumerate(weights))
        bins = np.zeros((3,n_bins))
        for k,weight in enumerate(weights):
            accepted = points[:count,k][keep[:count,k]]
            for axis in range(3):
                bins[axis] += weight*np.histogram(accepted[:,axis],bins=edges)[0]/count
        results.append(bins/normalization)
        normalizations.append(normalization)
    widths = np.array([float(model.max_vals[0]-model.min_vals[0]),2.3,1.])
    change = .5*np.sum(abs(results[1]-results[0]),axis=1)
    return results[1]*n_bins/widths[:,None], change, normalizations[1], abs(np.diff(normalizations)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = ROOT/args.directory
    config = SimpleNamespace(**json.loads((directory/'summary.json').read_text())['configuration'])
    model = estimator(config, load_inputs(config.spin, config.posterior_samples, config.injections, config.mmax))
    latents = np.load(directory/'posterior.npz')['latent']
    normal = ndtri(qmc.Sobol(3,scramble=True,seed=90118).random_base2(19))
    evaluated = [conditioned_marginal_bins(model,latent,normal) for latent in latents]
    fine = np.array([item[0] for item in evaluated])
    x = (np.arange(96)+.5)/96
    physical_x = np.array([np.log10(3)+x*(np.log10(config.mmax)-np.log10(3)),2.3*x,x])
    dx = np.array([(np.log10(config.mmax)-np.log10(3))/96,2.3/96,1/96])
    normalization = np.array([item[2] for item in evaluated])
    raw_alpha = np.load(directory/'posterior.npz')['alpha']
    arrays = load_inputs(config.spin, config.posterior_samples, config.injections, config.mmax)
    data = arrays['data']
    mass = 10**data[...,0]
    outside = ~covered_mass_pairs(mass,3+(mass-3)*data[...,2])
    result = {'method': 'three-dimensional transported Sobol population points conditioned on the covered joint mass domain',
              'joint_domain': 'm1 <= 100 Msun or m2 >= 10 Msun, inside the declared box with m2 >= 3 Msun',
              'coarse_samples_per_component': len(normal)//4, 'fine_samples_per_component': len(normal),
              'marginal_total_variation_change_quantiles': np.quantile([item[1] for item in evaluated],[.5,.9,.99,1],axis=0).T.tolist(),
              'base_probability_in_domain_quantiles':np.quantile(normalization,[.05,.5,.95]).tolist(),
              'maximum_domain_probability_resolution_change':float(max(item[3] for item in evaluated)),
              'conditional_selection_fraction_quantiles':np.quantile(raw_alpha/normalization,[.05,.5,.95]).tolist(),
              'compressed_event_points_outside_domain':int(outside.sum()),
              'shape_likelihood_normalizer_cancels':True,
              'maximum_probability_normalization_error': float(abs(fine.sum(-1)*dx-1).max()),
              'minimum_bin_density': float(fine.min())}
    np.savez_compressed(directory/'density_summary.npz', marginal_x=physical_x,
                        marginal_draws=fine.transpose(1,0,2),domain_probability=normalization,
                        conditional_alpha=raw_alpha/normalization)
    (directory/'marginal_integration_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
