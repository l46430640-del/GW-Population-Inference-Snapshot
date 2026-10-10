"""Summarize full-PE covariance and the leverage of individual released rows."""
from __future__ import annotations

import time
STARTED = time.perf_counter()
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def deletion_changes(features, deltas):
    shifted = deltas-deltas.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    change = weights @ features-features.mean(0)
    sd = features.std(0)
    return change, np.divide(change, sd, out=np.zeros_like(change), where=sd>0)


def main():
    config = json.loads((ROOT/'configs/lvk_pe_uncertainty.json').read_text())
    directory = ROOT/config['output']/'attempt_000'
    output = directory/'interpretation.json'
    if output.exists() or (directory/'interpretation_cost.json').exists():
        raise FileExistsError('The interpretation and its allowance cannot be overwritten.')
    cost = json.loads((directory/'cost.json').read_text())
    if cost['status'] != 'complete':
        raise ValueError('The event calculation must be complete.')
    charged = cost['total_charge_seconds']
    sources = {}
    process = psutil.Process()
    peak, status = 0, 'failed'
    def check():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if time.perf_counter()-STARTED+charged > config['maximum_seconds']:
            raise RuntimeError('The shared LVK assessment time allowance is exhausted.')
        if peak > config['maximum_resident_gib']*2**30:
            raise RuntimeError('The LVK assessment memory allowance is exhausted.')
        if sum(p.stat().st_size for p in directory.iterdir()) > config['maximum_output_mib']*2**20:
            raise RuntimeError('The LVK assessment output allowance is exhausted.')
    def source(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Source changed: '+str(path))
        sources[path.relative_to(ROOT).as_posix()] = actual
        return path
    try:
        check()
        source(Path(__file__))
        source(ROOT/'configs/lvk_pe_uncertainty.json')
        report = json.loads(source(directory/'summary.json').read_text())
        meta = json.loads(source(directory/'pe_metadata.json').read_text())
        with source(ROOT/config['pe_input'], config['pe_sha256']).open('rb') as handle:
            events = pickle.load(handle)
        check()
        cases, low_events = [], set()
        for case in report['cases']:
            with np.load(source(directory/(case['label']+'.npz'), case['product_sha256'])) as arrays:
                keep = arrays['representative']
                features = arrays['features'][keep]
                shifts, scaled = deletion_changes(features, arrays['fixed_row_deletion_log_delta'][:, keep])
                most_sensitive = int(np.abs(scaled).max(1).argmax())
                event = str(arrays['event_names'][most_sensitive])
                compressed_sd = arrays['compression_log_delta'][:, keep].std(1)
                compressed_order = np.argsort(compressed_sd)[::-1][:5]
                compression_leaders = [{'event': str(arrays['event_names'][i]),
                                       'log_integral_delta_sd': float(compressed_sd[i])} for i in compressed_order]
            ranking = case['events_by_pair_variance']
            worst = min(ranking, key=lambda r: r['minimum_ess_all_checked'])
            low_events.update(r['event'] for r in ranking if r['low_ess_all_checked_count'])
            cases.append({'label': case['label'], 'representative_draws': case['representative_draws'],
                'evaluated_draws': case['evaluated_draws'],
                'minimum_event_ess_representative_quantiles': case['minimum_event_ess_representative_quantiles'],
                'minimum_ess_all_checked': worst['minimum_ess_all_checked'], 'lowest_ess_event': worst['event'],
                'lowest_ess_saved_draw': worst['worst_saved_draw'],
                'leading_covariance_event': ranking[0]['event'],
                'leading_event_fraction_of_pair_variance': ranking[0]['mean_pair_difference_variance_representative']/case['mean_pair_difference_variance_representative'],
                'mean_pair_difference_variance': case['mean_pair_difference_variance_representative'],
                'clustered_mean_pair_difference_variance': case['clustered_mean_pair_difference_variance'],
                'maximum_conditional_mean_error_over_vi_sd': max(case['scientific_sensitivity']['conditional_mc_se_over_vi_sd']),
                'maximum_compression_mean_change_over_vi_sd': max(abs(x) for x in case['scientific_sensitivity']['mean_change_over_vi_sd']),
                'compression_ratio_ess_fraction': case['scientific_sensitivity']['ratio_ess_fraction'],
                'leading_compression_error_events': compression_leaders,
                'largest_fixed_row_deletion_event': event,
                'fixed_row_deletion_maximum_mean_change_over_vi_sd': float(abs(scaled).max()),
                'fixed_row_deletion_feature_changes': shifts[most_sensitive].tolist(),
                'fixed_row_deletion_feature_changes_over_vi_sd': scaled[most_sensitive].tolist()})
        physical = []
        for name in sorted(low_events):
            e = events[name]
            q = e['m2']/e['m1']
            rows = [next(r for r in c['events_by_pair_variance'] if r['event'] == name) for c in report['cases']]
            dominant = sorted({r['dominating_original_pe_row'] for r in rows})
            physical.append({'event': name, 'pe_rows': len(q), 'quantiles': [.05, .5, .95],
                'm1_quantiles': np.quantile(e['m1'], [.05, .5, .95]).tolist(),
                'z_quantiles': np.quantile(e['z'], [.05, .5, .95]).tolist(),
                'q_quantiles': np.quantile(q, [.05, .5, .95]).tolist(),
                'rows_q_above_0_8': int((q>.8).sum()), 'rows_q_above_0_9': int((q>.9).sum()),
                'dominating_rows': [{'index': int(i), 'm1': float(e['m1'][i]), 'z': float(e['z'][i]), 'q': float(q[i]),
                    'rows_with_q_at_least_this_value': int((q>=q[i]).sum())} for i in dominant]})
        result = {'cases': cases, 'low_ess_events': physical,
            'full_pe_rows': sum(r['rows'] for r in meta['events']),
            'repeated_states': [r for r in meta['events'] if r['maximum_repeat_count']>1],
            'source_sha256': sources, 'selection_held_fixed': True,
            'deletion_is_leverage_stress_only': True, 'pe_dependence_calibrated': False,
            'posterior_certified': False, 'historical_results_superseded': False}
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='ascii', newline='\n')
        check()
        status = 'complete'
    finally:
        seconds = time.perf_counter()-STARTED
        (directory/'interpretation_cost.json').write_text(json.dumps({'status': status, 'seconds': seconds,
            'previous_seconds': charged, 'total_charge_seconds': charged+seconds, 'peak_resident_bytes': peak},
            indent=2)+'\n', encoding='ascii', newline='\n')
    print(json.dumps({'status': status, 'seconds': seconds, 'total_charge_seconds': charged+seconds,
                      'low_ess_events': sorted(low_events)}))


if __name__ == '__main__':
    main()
