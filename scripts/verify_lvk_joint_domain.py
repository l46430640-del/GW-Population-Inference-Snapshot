"""Document joint injection coverage and the effect of the support mask on PE."""
from __future__ import annotations

import json
import pickle

import h5py
import numpy as np

from lvk_injections import MIXTURE_NAME, covered_mass_pairs
from analyze_lvk_population import EXCLUDED, ROOT


def main():
    directory = ROOT/'external_data/lvk/analysis_inputs'
    with h5py.File(directory/MIXTURE_NAME) as handle:
        group = handle['injections']
        mass,secondary = group['mass1_source'][:],group['mass2_source'][:]
    if np.any(~covered_mass_pairs(mass,secondary)):
        raise AssertionError('Recheck the declared mixture campaign domain')
    intervals = []
    for low,high in ((90,100),(100,120),(120,150),(150,200),(200,300)):
        mask = (mass>low)&(mass<=high)
        intervals.append({'primary_mass_interval':[low,high],'injections':int(mask.sum()),
                          'smallest_secondary_mass':float(secondary[mask].min())})
    with (directory/'sampleDict_FAR_1_in_1_yr.pickle').open('rb') as handle:
        events = pickle.load(handle)
    excluded = {name:int(np.sum(~covered_mass_pairs(event['m1'],event['m2'])))
                for name,event in events.items() if name not in EXCLUDED}
    compressed = {}
    for path in directory.glob('importance_mixture_v2_*_1024_12000_*.npz'):
        data = np.load(path)['data']
        primary = 10**data[...,0]
        pe_secondary = 3+(primary-3)*data[...,2]
        compressed[path.name] = int(np.sum(~covered_mass_pairs(primary,pe_secondary)))
    report = {'injection_source':MIXTURE_NAME,
              'domain':'within the declared box: m1 <= 100 Msun or m2 >= 10 Msun',
              'all_injections_outside_domain':int(np.sum(~covered_mass_pairs(mass,secondary))),
              'mass_intervals':intervals,'full_PE_points_outside_domain':excluded,
              'compressed_PE_points_outside_domain':compressed,
              'normalization':'p_C = p_0 * 1_C / A; Z_i,C = Z_i,0,C/A and alpha_C = alpha_0,C/A; A cancels from product(Z_i,C)/alpha_C**N',
              'reuse_scope':'Fits launched before the event-support callback have identical finite-sample shape likelihoods when their compressed PE count outside this domain is zero. Full-source PE checks apply the joint support mask and retain the original denominator.'}
    (ROOT/'docs/analysis/lvk_joint_domain_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='full_PE_points_outside_domain'},indent=2))


if __name__ == '__main__':
    main()
