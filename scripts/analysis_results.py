"""Resolve completed analysis products without selecting against injected truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTERIOR_STUDY_DIRECTORIES = (
    'importance_correction', 'importance_prediction', 'lisa_correction',
    'lisa_transport_pilot', 'lisa_production', 'lisa_refinement',
    'lisa_scientific_comparison')
EXTERNAL_ANALYSIS_DIRECTORIES = (
    'external_gp_pilot', 'external_gp_symmetric_pilot', 'external_gp_selected',
    'external_gp_selected_bounded', 'external_gp_selected_extended',
    'external_selected_comparison', 'external_gp_control_catalog',
    'external_gp_control_copula_pilot', 'external_gp_control_gp_pilot',
    'external_gp_control_qmc_pilot', 'external_control_production',
    'external_control_gp_extended', 'external_control_copula_repaired',
    'external_control_comparison', 'external_lisa_gp', 'external_lisa_gp_sparse',
    'external_lisa_gp_completed', 'external_lisa_sparse_check',
    'external_lisa_comparison', 'external_lisa_comparison_double',
    'external_lisa_recovery_refinement')
EXTERNAL_ANALYSIS_REGISTRIES = (
    'external_control_posteriors.json', 'external_lisa_gp_posteriors.json')


def configuration():
    return json.loads((ROOT/'configs/analysis_runs.json').read_text())


def read_summary(directory):
    return json.loads((ROOT/directory/'summary.json').read_text())


def lisa_baselines():
    candidates = configuration()['lisa_baseline_candidates']
    selected = {}
    for name, directories in candidates.items():
        summaries = {directory: read_summary(directory) for directory in directories}
        settings = ('family','components','posterior_samples','catalog_seed','quadrature','integration')
        if len({tuple(summary['configuration'][key] for key in settings)
                for summary in summaries.values()}) != 1:
            raise ValueError('Optimization selection requires an identical statistical target')
        selected[name] = min(summaries,key=lambda directory:summaries[directory]['negative_elbo_last_1000'])
    return selected


def main():
    config = configuration()
    selected = lisa_baselines()
    directories = set(selected.values()) | set(config['lisa_sensitivity'])
    directories |= {entry['directory'] for entry in config['lvk_sensitivity']}
    files = {}
    for directory in sorted(directories):
        names = ['summary.json','posterior.npz','flow.pt','density_summary.npz']
        if directory.startswith('results/analysis/morphologies/'):
            names += ['catalog.npz','joint_diagnostics.json','detected_joint_diagnostics.json',
                      'detected_joint_summary.npz','event_integral_check.json']
        else:
            names += ['marginal_integration_check.json','monte_carlo_check.json',
                      'comparison.json','comparison_summary.npz']
        for name in names:
            path = ROOT/directory/name
            files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in (('vi_corrected','lvk_reference','selected_mixture','prior_diagnostics')
                      + POSTERIOR_STUDY_DIRECTORIES + EXTERNAL_ANALYSIS_DIRECTORIES):
        for path in sorted((ROOT/'results/analysis'/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.json','.npz','.pt','.npy'):
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('rate_calibration.json',) + EXTERNAL_ANALYSIS_REGISTRIES:
        path = ROOT/'results/analysis'/name
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'lisa_baselines':selected,'selection_rule':'minimum mean negative ELBO over the final 1000 steps among the same-model completed starts; injected truth is not used',
                'lvk_baseline':config['lvk_baseline'],'sha256':files}
    (ROOT/'results/analysis/result_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:value for key,value in manifest.items() if key!='sha256'},indent=2))


if __name__ == '__main__':
    main()
