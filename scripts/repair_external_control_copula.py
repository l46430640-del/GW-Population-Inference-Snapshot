"""Replace the divergent chain, retaining verified independent trajectories."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()
from run_external_gp_pilot import ROOT

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from correct_selected_posterior import sha256
from run_lisa_production import verify_product, write_json
import run_external_control_production as production

CONFIG = ROOT / 'configs/external_control_copula_repair.json'
ORIGINAL_PROTOCOL = production.effective_protocol
ORIGINAL_FREEZE = production.freeze_target
ORIGINAL_EVALUATE = production.evaluate_clouds
STATE = {}


def repair_protocol(config, original):
    result = ORIGINAL_PROTOCOL(config, original)
    result['sampling']['target_acceptance'] = config['sampling_repair']['target_acceptance']
    return result


def prepare_reuse(config):
    repair = config['sampling_repair']
    parent = ROOT / repair['parent_output'] / 'copula'
    target = json.loads((parent / 'target.json').read_text())
    summary = json.loads((parent / 'summary.json').read_text())
    if sha256(parent / 'summary.json') != repair['parent_summary_sha256']:
        raise ValueError('Parent summary changed')
    for name, expected in target['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Parent source changed: {name}')
    if sha256(parent / 'posterior.npz') != summary['posterior_product_sha256']:
        raise ValueError('Parent numerical evaluations changed')
    seeds = target['effective_protocol']['sampling']['chain_seeds']
    count = target['effective_protocol']['sampling']['retained_per_chain']
    if set(repair['reuse_seeds']) | {repair['replace_seed']} != set(seeds):
        raise ValueError('Exactly one declared chain must be replaced')
    paths = [CONFIG, Path(__file__), parent / 'target.json', parent / 'summary.json', parent / 'posterior.npz']
    for seed in seeds:
        path = parent / f'chain_{seed}.npz'
        report = verify_product(path)
        if sha256(path.with_suffix('.json')) != summary['chain_reports_sha256'][path.with_suffix('.json').name]:
            raise ValueError('Parent chain report changed')
        if seed in repair['reuse_seeds'] and (report['status'] != 'complete' or report['divergences'] != []):
            raise ValueError('Only complete nondivergent chains may be reused')
        if seed == repair['replace_seed'] and not report['divergences']:
            raise ValueError('Replacement must be motivated by the recorded divergence failure')
        paths.extend([path, path.with_suffix('.json')])
    data = dict(np.load(parent / 'posterior.npz'))
    fine_rows = np.full(4*count, -1, dtype=int)
    fine_rows[data['verification_indices']] = np.arange(len(data['verification_indices']))
    keep = np.repeat([seed in repair['reuse_seeds'] for seed in seeds], count)
    pilot = json.loads((ROOT / config['copula_pilot_config']).read_text())
    catalog = json.loads((ROOT / config['catalog_config']).read_text())
    clouds = [np.load(ROOT / pilot['output'] / f'{label}_events.npz')['data'] for label in ('primary', 'verification')]
    clouds.append(np.load(ROOT / catalog['output'] / 'train_verification.npz')['data'])
    STATE.update(config=config, parent=parent, target=target, data=data, fine_rows=fine_rows, keep=keep,
        clouds=clouds, source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}, hits=0, fresh=0)


def freeze_and_reuse(output, signature):
    if output.name != 'copula':
        raise ValueError('This repair applies only to the copula control')
    parent = STATE['parent']
    if signature['initial_coordinates'] != STATE['target']['initial_coordinates']:
        raise ValueError('Repair must retain every original initialization')
    old_protocol, new_protocol = STATE['target']['effective_protocol'], signature['effective_protocol']
    for key in ('copula', 'evaluation'):
        if old_protocol[key] != new_protocol[key]:
            raise ValueError('Repair must retain the population model and numerical requirements')
    for key in ('chains', 'chain_seeds', 'warmup', 'retained_per_chain'):
        if old_protocol['sampling'][key] != new_protocol['sampling'][key]:
            raise ValueError('Repair must retain chain identities and lengths')
    signature['source_sha256'].update(STATE['source_sha256'])
    digest = ORIGINAL_FREEZE(output, signature)
    for seed in STATE['config']['sampling_repair']['reuse_seeds']:
        source = parent / f'chain_{seed}.npz'
        path = output / source.name
        original = verify_product(source)
        if path.exists() or path.with_suffix('.json').exists():
            reused = verify_product(path)
            if sha256(path) != sha256(source) or reused['target_product_sha256'] != digest:
                raise ValueError('Existing adopted chain differs from the parent')
            continue
        shutil.copyfile(source, path)
        report = dict(original, target_product_sha256=digest,
            parent_target_product_sha256=original['target_product_sha256'],
            reused_from=source.relative_to(ROOT).as_posix(),
            parent_report_sha256=sha256(source.with_suffix('.json')),
            original_process_elapsed_seconds=original['process_elapsed_seconds'],
            original_sampling_and_compilation_seconds=original['sampling_and_compilation_seconds'],
            process_elapsed_seconds=0., sampling_and_compilation_seconds=0.)
        write_json(path.with_suffix('.json'), report)
    return digest


def cached_evaluations(sites, clouds, indices, deadline, memory):
    if len(clouds) == 2:
        expected, columns = STATE['clouds'][:2], [0, 1]
    elif len(clouds) == 1:
        expected, columns = STATE['clouds'][2:], [2]
    else:
        raise ValueError('Unexpected integration request')
    if not all(np.array_equal(a, b) for a, b in zip(clouds, expected)):
        raise ValueError('Cached and requested integration clouds differ')
    values = []
    for j, index in enumerate(indices):
        production.enforce_budget(deadline, memory)
        fine_row = STATE['fine_rows'][index]
        reusable = STATE['keep'][index] and (columns != [2] or fine_row >= 0)
        if reusable:
            row = (STATE['data']['verification_log_integrals'][fine_row][None] if columns == [2]
                else STATE['data']['log_integrals'][index])
            STATE['hits'] += len(clouds)
        else:
            row = ORIGINAL_EVALUATE(sites, clouds, [index], deadline, memory)[0]
            STATE['fresh'] += len(clouds)
        values.append(row)
        if (j+1) % 400 == 0:
            print('Repaired likelihood checks', j+1, '/', len(indices), flush=True)
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', choices=['copula'], required=True)
    parser.parse_known_args()
    config = json.loads(CONFIG.read_text())
    repair = config['sampling_repair']
    if repair['source'] != Path(__file__).relative_to(ROOT).as_posix() or repair['source_sha256'] != sha256(Path(__file__)):
        raise ValueError('Repair source differs from its declaration')
    output = ROOT / config['output'] / 'copula'
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed repair is retained')
    prepare_reuse(config)
    production.CONFIG = CONFIG
    production.PROCESS_STARTED = PROCESS_STARTED
    production.effective_protocol = repair_protocol
    production.freeze_target = freeze_and_reuse
    production.evaluate_clouds = cached_evaluations
    production.main()
    write_json(output / 'integration_reuse.json', {'reused_state_cloud_evaluations': STATE['hits'],
        'fresh_state_cloud_evaluations': STATE['fresh'],
        'replaced_seed': repair['replace_seed'], 'reused_seeds': repair['reuse_seeds'],
        'parent_posterior_sha256': sha256(STATE['parent'] / 'posterior.npz'),
        'rule': 'Reuse deterministic integrals only for byte-identical nondivergent parent chains and identical clouds. No posterior state from the rejected chain is reused.'})


if __name__ == '__main__':
    main()
