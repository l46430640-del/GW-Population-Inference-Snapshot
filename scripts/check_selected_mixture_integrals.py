"""Check event integration precision without changing the selected-mixture target."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from benchmark_selected_mixture import ROOT, event_posterior, selected_model
from summarize_selected_mixture import selected_draws


@torch.no_grad()
def event_integrals(model, samples, data):
    points = data[..., :3].reshape(-1, 3)
    values, minimum_ess = [], []
    for index in range(len(samples['weights_logits'])):
        params = {key: torch.tensor(value[index], dtype=data.dtype) for key, value in samples.items()}
        log_weights = model._log_pdf_phys_from_params(params, points).reshape(data.shape[:2]) - data[..., 3].log()
        values.append((torch.logsumexp(log_weights, -1) - np.log(data.shape[1])).numpy())
        weight = (log_weights - log_weights.max(-1, keepdim=True).values).exp()
        minimum_ess.append(float((weight.sum(-1).square() / weight.square().sum(-1)).min()))
    return np.asarray(values), np.asarray(minimum_ess)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default='results/analysis/selected_mixture')
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--draws', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    directory = ROOT / args.directory
    source = directory / 'train_catalog.npz'
    catalog = np.load(source)
    rng = np.random.default_rng(526902)
    points = np.asarray([event_posterior(y, args.samples, rng) for y in catalog['observed']])
    fine = torch.tensor(np.concatenate([points, np.ones((*points.shape[:2], 1))], -1))
    coarse = torch.tensor(catalog['data'])
    paths = sorted(directory.glob('nuts_chain_*.npz'))
    paths += sorted(directory.glob('vi_start_*.npz'))
    paths += sorted(directory.glob('independent_start_*.npz'))
    if not paths:
        raise ValueError('No posterior samples to check')
    results = {}
    for path in paths:
        samples = selected_draws(dict(np.load(path)), args.draws)
        model = selected_model(path.name.startswith('independent'))
        original, original_ess = event_integrals(model, samples, coarse)
        refined, refined_ess = event_integrals(model, samples, fine)
        differences = original - refined
        results[path.stem] = {
            'posterior_draws': len(samples['weights_logits']),
            'event_log_integral_difference_05_50_95': np.quantile(differences, [.05, .5, .95]).tolist(),
            'total_log_integral_difference_05_50_95': np.quantile(differences.sum(-1), [.05, .5, .95]).tolist(),
            'total_log_integral_difference_sd': float(differences.sum(-1).std()),
            'coarse_minimum_event_ess_05_50_95': np.quantile(original_ess, [.05, .5, .95]).tolist(),
            'fine_minimum_event_ess_05_50_95': np.quantile(refined_ess, [.05, .5, .95]).tolist(),
        }
        print(path.stem, json.dumps(results[path.stem]), flush=True)
    checked = paths + [source, ROOT / 'scripts/check_selected_mixture_integrals.py']
    output = {'configuration': vars(args), 'coarse_samples': coarse.shape[1],
              'fine_samples': args.samples, 'event_posterior_seed': 526902,
              'results': results,
              'sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in checked}}
    (directory / 'event_integral_check.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
