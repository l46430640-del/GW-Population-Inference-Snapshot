"""Use the accepted posterior's double precision in the core equality check."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()
import compare_external_lisa as comparison

import json
from pathlib import Path
import shutil
import numpy as np

from benchmark_selected_mixture import runtime_metadata
from correct_selected_posterior import sha256
from run_lisa_production import write_json

ROOT = comparison.ROOT
CONFIG = ROOT / 'configs/external_lisa_precision.json'
ORIGINAL_CACHED = comparison.cached
ORIGINAL_INTEGRALS = comparison.event_integrals


def matching_write(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Continuation record changed: {path}')
    else:
        write_json(path, value)


def main():
    amendment = json.loads(CONFIG.read_text())
    base = json.loads((ROOT / amendment['base_config']).read_text())
    parent, output = ROOT / amendment['parent_output'], ROOT / amendment['output']
    failure = json.loads((parent / 'job_cost_000.json').read_text())
    target = json.loads((parent / 'copula_target_summary.json').read_text())
    if failure['status'] != 'stopped' or target['decision']['kernel_passed']:
        raise ValueError('The original failed precision check must remain present')
    if amendment['maximum_seconds']+failure['process_elapsed_seconds'] > base['maximum_seconds']:
        raise ValueError('Precision continuation exceeds the original evaluation allowance')
    output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / base['posterior_registry']).read_text())
    registry['source_sha256'].update(target['source_sha256'])
    paths = [CONFIG, Path(__file__), parent / 'job_cost_000.json', parent / 'copula_target_summary.json',
        parent / 'kernel_verification.npz', parent / 'kernel_verification.json']
    paths.extend(parent / (label+'.'+suffix) for label in amendment['reuse_products'] for suffix in ('npz', 'json'))
    registry['source_sha256'].update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    registry['cost']['preceding_lisa_evaluation_attempt_seconds'] = failure['process_elapsed_seconds']
    registry['cost']['total_study_charge_seconds'] += failure['process_elapsed_seconds']
    registry['cost']['remaining_study_seconds'] = base['maximum_total_study_seconds']-registry['cost']['total_study_charge_seconds']
    registry_path = output / 'posterior_registry.json'
    matching_write(registry_path, registry)
    config = dict(base, posterior_registry=registry_path.relative_to(ROOT).as_posix(),
        output=amendment['output'], maximum_seconds=amendment['maximum_seconds'], precision_amendment=amendment)
    config_path = output / 'configuration.json'
    matching_write(config_path, config)
    def cached(directory, label, sources, builder):
        destination = directory / (label+'.npz')
        if label not in amendment['reuse_products'] or destination.with_suffix('.json').exists():
            return ORIGINAL_CACHED(directory, label, sources, builder)
        source = parent / destination.name
        metadata = json.loads(source.with_suffix('.json').read_text())
        if metadata['source_sha256'] != target['source_sha256'] or sha256(source) != metadata['product_sha256']:
            raise ValueError('Original training cache has changed provenance')
        if destination.exists():
            raise FileExistsError('An unverified adopted product exists')
        shutil.copyfile(source, destination)
        adopted = dict(metadata, source_sha256=sources, seconds=0., runtime=runtime_metadata(),
            reused_from=source.relative_to(ROOT).as_posix(),
            parent_metadata_sha256=sha256(source.with_suffix('.json')), original_seconds=metadata['seconds'])
        write_json(destination.with_suffix('.json'), adopted)
        return dict(np.load(destination)), adopted
    comparison.CONFIG = config_path
    comparison.PROCESS_STARTED = PROCESS_STARTED
    comparison.cached = cached
    comparison.event_integrals = lambda model, samples, data: ORIGINAL_INTEGRALS(model, samples, data.double())
    comparison.main()


if __name__ == '__main__':
    main()
