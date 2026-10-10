"""Reconstruct official PL+Peak marginal draws before taking any quantiles.

Equations and conditional normalization follow the published helper
figures/read_O3_LVK_results.py in Callister & Farr's PRX companion repository.
"""
from __future__ import annotations

import hashlib
import json

from astropy.cosmology import Planck15
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit
from scipy.stats import norm

from analyze_lvk_population import ROOT
from lvk_injections import covered_mass_pairs


def smoothing(mass, minimum, delta):
    x = (mass-minimum)/delta
    interior = (x>0)&(x<1)
    safe = np.clip(x,1e-12,1-1e-12)
    result = expit(-1/safe-1/(safe-1))
    return np.where(interior,result,(x>=1).astype(float))


def main():
    path = ROOT/'external_data/lvk/analysis_inputs/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json'
    source = json.loads(path.read_text())['posterior']['content']
    rng = np.random.default_rng(241200)
    indices = rng.choice(len(source['alpha']),512,replace=False)
    mass, ratio, redshift = np.geomspace(3,150,2001), np.linspace(0,1,801), np.linspace(0,2.3,2001)
    mass_edges, q_edges, z_edges = np.geomspace(3,150,97), np.linspace(0,1,97), np.linspace(0,2.3,97)
    dv = Planck15.differential_comoving_volume(redshift).value
    draws, domain_probabilities = [], []
    for i in indices:
        p = {key: np.asarray(source[key])[i] for key in ('alpha','mmin','mmax','delta_m','lam','mpp','sigpp','beta','lamb')}
        power = 1-p['alpha']
        mass_pl = power*mass**(-p['alpha'])/(p['mmax']**power-p['mmin']**power)
        mass_pl[(mass<p['mmin'])|(mass>p['mmax'])] = 0
        pm = ((1-p['lam'])*mass_pl+p['lam']*norm.pdf(mass,p['mpp'],p['sigpp']))*smoothing(mass,p['mmin'],p['delta_m'])
        pm /= np.trapz(pm,mass)
        pq_given_m = np.maximum(ratio[:,None],1e-12)**p['beta']*smoothing(ratio[:,None]*mass[None],p['mmin'],p['delta_m'])
        normalization = np.trapz(pq_given_m,ratio,axis=0)
        pq_given_m /= np.where(normalization>0,normalization,1.)
        supported = pq_given_m*covered_mass_pairs(mass[None],ratio[:,None]*mass[None])
        retained = np.trapz(supported,ratio,axis=0)
        domain_probability = np.trapz(pm*retained,mass)
        pq = np.trapz(supported*pm[None],mass,axis=1)/domain_probability
        pm = pm*retained/domain_probability
        domain_probabilities.append(domain_probability)
        pz = (1+redshift)**(p['lamb']-1)*dv
        pz /= np.trapz(pz,redshift)
        cm = cumulative_trapezoid(pm,mass,initial=0)
        cq = cumulative_trapezoid(pq,ratio,initial=0)
        cz = cumulative_trapezoid(pz,redshift,initial=0)
        draws.append([np.diff(np.interp(mass_edges,mass,cm))/np.diff(np.log10(mass_edges)),
                      np.diff(np.interp(z_edges,redshift,cz))/np.diff(z_edges),
                      np.diff(np.interp(q_edges,ratio,cq))/np.diff(q_edges)])
    output = ROOT/'results/analysis/lvk_reference'
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'density_summary.npz', marginal_draws=np.array(draws).transpose(1,0,2),
                        marginal_x=np.array([(np.log10(mass_edges[:-1])+np.log10(mass_edges[1:]))/2,
                                             (z_edges[:-1]+z_edges[1:])/2,(q_edges[:-1]+q_edges[1:])/2]),
                        posterior_indices=indices)
    result = {'model': 'LVK PowerLaw+Peak with Default spin; mass and redshift separable',
              'draws':len(indices), 'seed':241200, 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'mass_support':[3,150], 'redshift_support':[0,2.3],
              'density_measure':['log10(m1/Msun)','z','q'],
              'redshift_density':'(1+z)^(lambda-1) dVc/dz, Planck15 cosmology',
              'normalization':'normalize each hyperposterior draw before computing pointwise quantiles',
              'joint_domain':'condition on m1 <= 100 Msun or m2 >= 10 Msun within the stated box',
              'base_probability_in_domain_quantiles':np.quantile(domain_probabilities,[.05,.5,.95]).tolist(),
              'source':'https://doi.org/10.5281/zenodo.8087858'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
