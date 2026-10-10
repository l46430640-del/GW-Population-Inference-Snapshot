"""Run selected analysis fits sequentially with a bounded CPU footprint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', choices=('lisa-main','lisa-sensitivity','lvk-baseline','lvk-family','lvk-support'))
    args = parser.parse_args()
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    jobs = []
    if args.batch == 'lisa-main':
        for morphology in ('correlated', 'bimodal', 'curved'):
            jobs.append((f'lisa_{morphology}_main', [
                'scripts/validate_population_morphologies.py', '--detector', 'lisa',
                '--morphology', morphology, '--steps', '6000', '--particles', '2',
                '--quadrature', '4096', '--posterior-samples', '256']))
    elif args.batch == 'lisa-sensitivity':
        for family, components, seed in (('kumaraswamy',2,91001),('kumaraswamy',2,91002),
                                         ('kumaraswamy',1,91001),('kumaraswamy',3,91001),
                                         ('normal',2,91001)):
            jobs.append((f'lisa_bimodal_{family}_k{components}_s{seed}',[
                'scripts/validate_population_morphologies.py','--detector','lisa',
                '--morphology','bimodal','--family',family,'--components',str(components),
                '--training-seed',str(seed),'--initialize-events','--steps','5000',
                '--particles','2','--quadrature','4096','--posterior-samples','256']))
    elif args.batch.startswith('lvk-'):
        base = ['scripts/analyze_lvk_population.py','--output','results/analysis/lvk_mixture',
                '--particles','2','--posterior-samples','1024','--full-injections']
        pilot = 'results/analysis/lvk_refined/spline_k3_uniform_m150_s812301/flow.pt'
        if args.batch == 'lvk-baseline':
            jobs.append(('lvk_mixture_spline_baseline',base+['--family','spline','--components','3',
                         '--initial-state',pilot,'--learning-rate','.0003','--steps','3000']))
            jobs.append(('lvk_mixture_spline_spin_low',base+['--family','spline','--components','3',
                         '--initial-state','results/analysis/lvk_mixture/spline_k3_uniform_m150_s812301/flow.pt',
                         '--learning-rate','.0003','--steps','3000','--spin','low']))
        elif args.batch == 'lvk-family':
            for components,steps in ((3,6000),(2,5000)):
                jobs.append((f'lvk_mixture_kumaraswamy_k{components}',base+[
                    '--family','kumaraswamy','--components',str(components),'--steps',str(steps),'--initialize-events']))
        else:
            jobs.append(('lvk_mixture_spline_support_300',base+['--family','spline','--components','3',
                         '--initial-state',pilot,'--learning-rate','.0003','--steps','3000','--mmax','300']))
    for name, command in jobs:
        print('Starting', name, flush=True)
        with (ROOT / 'results' / f'analysis_{name}.log').open('w') as log:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed', name, flush=True)


if __name__ == '__main__':
    main()
