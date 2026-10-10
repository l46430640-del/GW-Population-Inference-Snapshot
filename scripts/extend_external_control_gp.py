"""Extend GP chains without changing the fixed control target or warmup."""
from __future__ import annotations

from run_external_gp_pilot import ROOT

import argparse
import json
from pathlib import Path

import numpy as np

from correct_selected_posterior import sha256
from run_lisa_production import verify_product, write_json
import run_external_control_production as production

CONFIG = ROOT / 'configs/external_control_gp_extension.json'


def extended_protocol(config, original):
    result = production_protocol(config, original)
    extension = config['sampling_extension']
    if extension['original_retained_per_chain'] != original['sampling']['retained_per_chain']:
        raise ValueError('Unexpected original chain length')
    if extension['retained_per_chain'] <= extension['original_retained_per_chain']:
        raise ValueError('An extension must add retained samples')
    result['sampling']['retained_per_chain'] = extension['retained_per_chain']
    return result


production_protocol = production.effective_protocol


def verify_prefix(label, config, protocol, directory, root, sources, deadline):
    if label not in ('gp_6', 'gp_12'):
        raise ValueError('This extension is restricted to the two GP resolutions')
    extension = config['sampling_extension']
    parent = ROOT / extension['parent_output'] / label
    parent_target = json.loads((parent / 'target.json').read_text())
    parent_summary = json.loads((parent / 'summary.json').read_text())
    if sha256(parent / 'summary.json') != extension['parent_summaries_sha256'][label]:
        raise ValueError('Parent summary changed')
    for name, expected in parent_target['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Parent source changed: {name}')
    paths = [CONFIG, Path(__file__), parent / 'target.json', parent / 'summary.json']
    for name, expected in parent_summary['chain_reports_sha256'].items():
        if sha256(parent / name) != expected:
            raise ValueError('Parent chain report changed')
        verify_product((parent / name).with_suffix('.npz'))
        paths.extend([parent / name, (parent / name).with_suffix('.npz')])
    sources = dict(sources)
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    production_run_model(label, config, protocol, directory, root, sources, deadline)
    rows = []
    for seed in protocol['sampling']['chain_seeds']:
        child_path = root / label / f'chain_{seed}.npz'
        if not child_path.exists():
            continue
        verify_product(child_path)
        before = np.load(parent / child_path.name)
        after = np.load(child_path)
        count = min(len(before['coordinate']), len(after['coordinate']))
        exact = (np.array_equal(before['initial_coordinate'], after['initial_coordinate'])
            and np.array_equal(before['warmup_coordinate'], after['warmup_coordinate'])
            and np.array_equal(before['coordinate'][:count], after['coordinate'][:count]))
        rows.append({'seed': seed, 'compared_retained_draws': count, 'prefix_identical': exact})
    path = root / label / 'prefix_verification.json'
    if path.exists():
        if json.loads(path.read_text())['chains'] != rows:
            raise ValueError('Prefix verification differs from its saved result')
    else:
        write_json(path, {'chains': rows, 'parent_output': extension['parent_output'],
            'prefixes_are_duplicates_not_additional_posterior_draws': True})
    if len(rows) != 4 or not all(row['prefix_identical'] for row in rows):
        raise ValueError('Replay did not preserve the original starts, warmup and sample prefix')


production_run_model = production.run_model


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', choices=['gp_6', 'gp_12'], required=True)
    args, _ = parser.parse_known_args()
    config = json.loads(CONFIG.read_text())
    extension = config['sampling_extension']
    if extension['source'] != Path(__file__).relative_to(ROOT).as_posix() or extension['source_sha256'] != sha256(Path(__file__)):
        raise ValueError('Sampling extension source differs from its declaration')
    production.CONFIG = CONFIG
    production.effective_protocol = extended_protocol
    production.run_model = verify_prefix
    production.main()


if __name__ == '__main__':
    main()
