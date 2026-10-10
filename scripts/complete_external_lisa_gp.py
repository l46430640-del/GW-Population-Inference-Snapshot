"""Complete only the fourth high-resolution chain without pooling its prefix."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()
import run_external_lisa_gp_sparse as sparse

import json
from pathlib import Path
import shutil
import numpy as np

from correct_selected_posterior import sha256
from run_external_gp_pilot import ROOT
from run_lisa_production import verify_product, write_json

CONFIG = ROOT / 'configs/external_lisa_gp_completion.json'
ORIGINAL_SOURCES = sparse.sources_and_inputs
ORIGINAL_FREEZE = sparse.freeze_target
ORIGINAL_SAVE = sparse.production.save_chain


def main():
    amendment = json.loads(CONFIG.read_text())
    completion = amendment['completion']
    parent = ROOT / completion['parent']
    parent_target = json.loads((parent / 'target.json').read_text())
    parent_summary = json.loads((parent / 'summary.json').read_text())
    for name, digest in parent_target['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Parent source changed: {name}')
    paths = [CONFIG, Path(__file__), parent / 'target.json', parent / 'summary.json', parent / 'acceptance.json']
    for seed in completion['reuse_seeds']+[completion['replace_seed']]:
        path = parent / f'chain_{seed}.npz'
        report = verify_product(path)
        if sha256(path.with_suffix('.json')) != parent_summary['chain_reports_sha256'][path.with_suffix('.json').name]:
            raise ValueError('A parent chain report changed')
        if seed in completion['reuse_seeds'] and (report['status'] != 'complete' or report['divergences'] != []):
            raise ValueError('Only complete nondivergent chains can be reused')
        if seed == completion['replace_seed'] and (report['status'] != 'time_limit'
                or report['samples_completed'] != completion['expected_parent_retained']):
            raise ValueError('The partial fourth chain differs from the declared continuation')
        paths.extend([path, path.with_suffix('.json')])
    def sources(config):
        source, inputs, path = ORIGINAL_SOURCES(config)
        source.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
        return source, inputs, path
    def freeze(directory, signature):
        if signature['effective_protocol'] != parent_target['effective_protocol']:
            raise ValueError('Continuation must preserve the full target and sampling protocol')
        if signature['configuration']['initial_amplitude_and_lengths'] != parent_target['configuration']['initial_amplitude_and_lengths']:
            raise ValueError('Continuation must preserve every initialization')
        digest = ORIGINAL_FREEZE(directory, signature)
        for seed in completion['reuse_seeds']:
            source = parent / f'chain_{seed}.npz'
            path = directory / source.name
            original = verify_product(source)
            if path.exists() or path.with_suffix('.json').exists():
                report = verify_product(path)
                if sha256(path) != sha256(source) or report['target_product_sha256'] != digest:
                    raise ValueError('Adopted chain differs from its parent')
                continue
            shutil.copyfile(source, path)
            report = dict(original, target_product_sha256=digest,
                parent_target_product_sha256=original['target_product_sha256'],
                reused_from=source.relative_to(ROOT).as_posix(), parent_report_sha256=sha256(source.with_suffix('.json')),
                original_process_elapsed_seconds=original['process_elapsed_seconds'],
                original_sampling_and_compilation_seconds=original['sampling_and_compilation_seconds'],
                process_elapsed_seconds=0., sampling_and_compilation_seconds=0.)
            write_json(path.with_suffix('.json'), report)
        return digest
    def save(path, report, arrays, digest):
        old = np.load(parent / path.name)
        count = len(old['coordinate'])
        exact = (np.array_equal(old['initial_coordinate'], arrays['initial_coordinate'])
            and np.array_equal(old['warmup_coordinate'], arrays['warmup_coordinate'])
            and np.array_equal(old['coordinate'], arrays['coordinate'][:count]))
        ORIGINAL_SAVE(path, report, arrays, digest)
        write_json(path.parent / 'prefix_verification.json', {'seed': completion['replace_seed'],
            'compared_retained_draws': count, 'prefix_identical': bool(exact),
            'prefix_not_counted_as_additional_draws': True, 'parent_chain_sha256': sha256(parent / path.name)})
        if completion['require_identical_prefix'] and not exact:
            raise ValueError('The original warmup and retained prefix did not reproduce')
    sparse.CONFIG = CONFIG
    sparse.PROCESS_STARTED = PROCESS_STARTED
    sparse.sources_and_inputs = sources
    sparse.freeze_target = freeze
    sparse.production.save_chain = save
    sparse.main()


if __name__ == '__main__':
    main()
