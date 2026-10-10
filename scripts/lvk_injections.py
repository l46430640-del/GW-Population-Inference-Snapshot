"""Read search injections in the same mass, redshift, and spin measure as PE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIXTURE_NAME = 'o1+o2+o3_mixture_real+semianalytic-LIGO-T2100377-v2.hdf5'
SEARCHES = ('far_gstlal','far_pycbc_bbh','far_pycbc_hyperbank')


def covered_mass_pairs(m1, m2):
    """The BBH/IMBH mixture excludes m1 > 100 with m2 < 10 solar masses."""
    return (m1 <= 100.) | (m2 >= 10.)


def spin_measure_jacobian(a1,a2):
    # Integrate uniform azimuths after changing each Cartesian spin to (a,cos tilt,phi).
    return 4*np.pi**2*np.asarray(a1)**2*np.asarray(a2)**2


def load_recovered(path):
    with h5py.File(path) as source:
        group = source['injections']
        run = group['name'][:]
        far = np.min([group[key][:] for key in SEARCHES],axis=0)
        keep = np.where(run==b'o3',far<1,group['optimal_snr_net'][:]>=10)
        m1,m2,z = [group[key][:][keep] for key in ('mass1_source','mass2_source','redshift')]
        spins = [np.sqrt(sum(group[f'spin{index}{axis}'][:][keep]**2 for axis in 'xyz')) for index in (1,2)]
        raw_draw = group['sampling_pdf'][:][keep]
        mixture_weight = group['mixture_weight'][:][keep]
        joint_draw = raw_draw*spin_measure_jacobian(*spins)
        result = {'m1':m1,'m2':m2,'z':z,'a1':spins[0],'a2':spins[1],
                  'joint_draw':joint_draw,'mixture_weight':mixture_weight,
                  'nTrials':float(source.attrs['total_generated'])}
        metadata = {'source':Path(path).name,'total_generated':result['nTrials'],
                    'analysis_time_seconds':float(source.attrs['analysis_time_s']),
                    'recovered':int(keep.sum()),'searches':SEARCHES,
                    'selection':'O3: minimum FAR < 1/year in GstLAL and the two PyCBC searches; O1/O2: network SNR >= 10',
                    'all_primary_mass_range':[float(group['mass1_source'][:].min()),float(group['mass1_source'][:].max())],
                    'found_primary_mass_range':[float(m1.min()),float(m1.max())],
                    'found_redshift_range':[float(z.min()),float(z.max())],
                    'found_above_100':int(np.sum(m1>100)),
                    'mixture_weight_range':[float(mixture_weight.min()),float(mixture_weight.max())],
                    'spin_coordinate_conversion':'p_draw(a1,a2,cost1,cost2) = p_draw(cartesian spins) * (2*pi*a1^2)*(2*pi*a2^2)'}
    if not np.all(np.isfinite(joint_draw)&(joint_draw>0)&(mixture_weight>0)):
        raise ValueError('Invalid recovered-injection drawing density')
    return result,metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-bbh',type=Path)
    args = parser.parse_args()
    directory = ROOT/'external_data/lvk/analysis_inputs'
    result,metadata = load_recovered(directory/MIXTURE_NAME)
    if args.validate_bbh:
        baseline,baseline_metadata = load_recovered(args.validate_bbh)
        with (directory/'injectionDict_FAR_1_in_1.pickle').open('rb') as handle:
            published = pickle.load(handle)
        index = np.lexsort((baseline['z'],baseline['m1']))
        other = np.lexsort((published['z'],published['m1']))
        np.testing.assert_array_equal(baseline['m1'][index],published['m1'][other])
        np.testing.assert_array_equal(baseline['z'][index],published['z'][other])
        ratio = baseline['joint_draw'][index]/(published['p_draw_m1m2z'][other]*published['p_draw_a1a2cost1cost2'][other])
        np.testing.assert_allclose(ratio,1,rtol=1e-12)
        metadata['bbh_measure_crosscheck'] = {'matched_recovered_injections':len(index),
            'maximum_relative_draw_difference':float(abs(ratio-1).max()),
            'all_primary_mass_range':baseline_metadata['all_primary_mass_range'],
            'consequence':'BBH-only injections cannot estimate selection for population support above 100 Msun.'}
    (ROOT/'docs/analysis/lvk_injection_measure_check.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
