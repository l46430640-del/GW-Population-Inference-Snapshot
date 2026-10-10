"""Prepare independent bounded event batches, retaining the declared precision."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()
import run_external_lisa_gp as production

import json
from pathlib import Path
import numpy as np

from correct_selected_posterior import sha256
from lisa_noisy_catalog import posterior_from_measurements as base_sampler
from run_external_gp_pilot import ROOT

CONFIG = ROOT / 'configs/external_lisa_event_batches.json'
ORIGINAL_SOURCES = production.source_manifest
ORIGINAL_FREEZE = production.freeze_target


def batched_event_posterior(y, rho, response, n_samples, rng, batch_samples=2048):
    if n_samples <= 0 or batch_samples <= 0:
        raise ValueError('Positive event and batch sample counts are required')
    points, centers, reports, sizes = [], [], [], []
    for start in range(0, n_samples, batch_samples):
        size = min(batch_samples, n_samples-start)
        block, center, report = base_sampler(y, rho, response, size, rng)
        if len(block) != size or report['proposal_ess'] < max(1024, 4*size):
            raise ValueError('Every batch must meet the original event precision requirement')
        points.append(block)
        centers.append(center)
        reports.append(report)
        sizes.append(size)
    weights = np.asarray(sizes)/n_samples
    combined_ess = 1/np.sum(weights**2/np.asarray([r['proposal_ess'] for r in reports]))
    if combined_ess < 4*n_samples:
        raise ValueError('Combined proposal pools do not meet the declared ESS requirement')
    return np.concatenate(points), weights @ np.asarray(centers), {
        'proposal_ess': float(combined_ess), 'proposal_points': sum(r['proposal_points'] for r in reports),
        'batch_samples': sizes, 'normalized_batch_weights': weights.tolist(), 'batches': reports,
        'interpretation': 'Weight ESS of independent separately normalized proposal pools; not an independent posterior sample count.'}


def main():
    amendment = json.loads(CONFIG.read_text())
    base = ROOT / amendment['base_config']
    if sha256(base) != amendment['base_config_sha256']:
        raise ValueError('Original numerical declaration changed')
    failed_path = ROOT / amendment['failed_job']
    failed = json.loads(failed_path.read_text())
    if failed['status'] != 'stopped' or 'Inverse-SNR importance ESS remains' not in failed['error']:
        raise ValueError('The batch repair must retain its motivating preparation failure')
    if json.loads(base.read_text())['verification_event_samples'] != amendment['total_samples']:
        raise ValueError('Batched preparation cannot change the declared event sample count')
    def sources(config):
        result = ORIGINAL_SOURCES(config)
        paths = [CONFIG, Path(__file__), failed_path]
        result.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
        return result
    def freeze(output, signature):
        signature['event_generation_amendment'] = amendment
        return ORIGINAL_FREEZE(output, signature)
    production.CONFIG = base
    production.PROCESS_STARTED = PROCESS_STARTED
    production.source_manifest = sources
    production.freeze_target = freeze
    production.posterior_from_measurements = lambda y, rho, response, count, rng: batched_event_posterior(
        y, rho, response, count, rng, amendment['batch_samples'])
    production.main()


if __name__ == '__main__':
    main()
