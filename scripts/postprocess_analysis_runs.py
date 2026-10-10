"""Run missing or stale checks for completed analysis fits, one process at a time."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from analysis_results import ROOT, configuration


def products_current(directory, outputs, dependencies):
    targets = [directory/name for name in outputs]
    inputs = [directory/name for name in ('summary.json', 'posterior.npz', 'flow.pt')]
    inputs += dependencies
    return (all(path.exists() for path in targets)
            and min(path.stat().st_mtime_ns for path in targets)
            >= max(path.stat().st_mtime_ns for path in inputs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-only',action='store_true')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--lvk-directories', nargs='+', help='Check only these ground-based fit directories')
    args = parser.parse_args()
    config = configuration()
    lisa = set(config['lisa_sensitivity'])
    for candidates in config['lisa_baseline_candidates'].values():
        lisa.update(candidates)
    lvk = {entry['directory'] for entry in config['lvk_sensitivity']}
    if args.lvk_directories:
        lisa, lvk = set(), set(args.lvk_directories)
    environment = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    tasks = {
        'lisa': [(('joint_diagnostics.json','density_summary.npz'),'summarize_analysis_populations.py'),
                 (('detected_joint_diagnostics.json','detected_joint_summary.npz'),'summarize_detected_lisa.py'),
                 (('event_integral_check.json',),'check_mock_monte_carlo.py')],
        'lvk': [(('density_summary.npz','marginal_integration_check.json'),'summarize_lvk_marginals.py'),
                (('monte_carlo_check.json','monte_carlo_check.npz'),'check_lvk_monte_carlo.py'),
                (('comparison.json','comparison_summary.npz'),'summarize_lvk_comparisons.py')]}
    for domain,directories in (('lisa',lisa),('lvk',lvk)):
        for directory in sorted(directories):
            path = ROOT/directory
            if not (path/'summary.json').exists():
                if args.completed_only:
                    print('Not yet complete:',directory,flush=True)
                    continue
                raise FileNotFoundError(path/'summary.json')
            for outputs,script in tasks[domain]:
                dependencies = [ROOT/'scripts'/script, ROOT/'src/gwpopulation/core.py']
                if domain == 'lvk':
                    dependencies += [ROOT/'src/gwpopulation/smooth_mass.py', ROOT/'scripts/analyze_lvk_population.py']
                if script == 'summarize_lvk_comparisons.py':
                    dependencies += [path/'density_summary.npz', ROOT/'results/analysis/lvk_reference/density_summary.npz']
                if not args.force and products_current(path, outputs, dependencies):
                    continue
                print('Checking',path.name,script,flush=True)
                with (path/(script.removesuffix('.py')+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts'/script),directory],
                                   cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed all available checks',flush=True)


if __name__ == '__main__':
    main()
