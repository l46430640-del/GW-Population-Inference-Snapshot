"""Check compressed likelihood integrals against all available public samples."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analyze_lvk_population import (ROOT, coordinates, estimator, load_inputs,
                                    reference_density, spin_density)
from lvk_injections import covered_mass_pairs


def weights(model, params, points, log_prior):
    unit = (torch.tensor(points, dtype=torch.float32) - model.min_vals) / (model.max_vals - model.min_vals)
    valid = ((unit > 0) & (unit < 1)).all(dim=1)
    mass = 10**np.asarray(points)[:,0]
    valid &= torch.tensor(covered_mass_pairs(mass,3+(mass-3)*np.asarray(points)[:,2]))
    log_weight = model._log_pdf_phys_from_params(params, unit.clamp(1e-7, 1-1e-7)) - torch.tensor(log_prior)
    return log_weight.masked_fill(~valid, -torch.inf).double()


def logmean(log_weights):
    return float(torch.logsumexp(log_weights, 0) - np.log(len(log_weights)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--draws', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = ROOT / args.directory
    state = torch.load(directory / 'flow.pt', weights_only=False)
    config = SimpleNamespace(**state['configuration'])
    arrays = load_inputs(config.spin, config.posterior_samples, config.injections, config.mmax)
    model = estimator(config, arrays)
    model.flow_transforms.load_state_dict(state['state'])
    latent = np.load(directory / 'posterior.npz')['latent'][:args.draws]
    with (ROOT / 'external_data/lvk/analysis_inputs/sampleDict_FAR_1_in_1_yr.pickle').open('rb') as handle:
        events = pickle.load(handle)
    full = []
    for name in arrays['event_names']:
        event = events[name]
        points, jacobian = coordinates(event['m1'], event['m2'], event['z'])
        prior = event['z_prior'] * jacobian / np.maximum(spin_density(event['a1'], event['a2'], config.spin), 1e-30)
        proposal = reference_density(points, config.mmax)
        full.append((points, np.log(prior), float(np.mean(proposal / prior))))
    full_ess, alpha_relative, event_min_ess, event_delta, likelihood_delta = [], [], [], [], []
    for value in latent:
        params = model._unflatten_latent_vector(torch.tensor(value))
        li = weights(model, params, arrays['full_injection_points'], arrays['full_injection_log_p_draw'])
        wi = (li - li.max()).exp()
        full_ess.append(float(wi.sum()**2 / wi.square().sum()))
        full_alpha = float((torch.logsumexp(li, 0) - np.log(arrays['full_n_draw'])).exp())
        compressed_alpha = float(model._selection_fraction_for_params(params))
        alpha_relative.append(compressed_alpha / full_alpha - 1)
        minimum, deltas = [], []
        for i, (points, logprior, reference_mean) in enumerate(full):
            le = weights(model, params, points, logprior)
            we = (le - le.max()).exp()
            minimum.append(float(we.sum()**2 / we.square().sum()))
            data = arrays['data'][i]
            compressed_logmean = logmean(weights(model, params, data[:, :3], np.log(data[:, 3]))) + np.log(reference_mean)
            deltas.append(compressed_logmean - logmean(le))
        event_min_ess.append(min(minimum))
        event_delta.append(deltas)
        likelihood_delta.append(sum(deltas) - len(full) * np.log(compressed_alpha / full_alpha))
    q = [.05, .5, .95]
    result = {'draws': len(latent), 'full_injection_ess_quantiles': np.quantile(full_ess, q).tolist(),
              'compressed_alpha_relative_difference_quantiles': np.quantile(alpha_relative, q).tolist(),
              'full_minimum_event_ess_quantiles': np.quantile(event_min_ess, q).tolist(),
              'event_log_integral_error_quantiles': np.quantile(event_delta, q).tolist(),
              'total_log_likelihood_error_quantiles': np.quantile(likelihood_delta, q).tolist(),
              'total_log_likelihood_error_std': float(np.std(likelihood_delta)),
              'reference_injection_ess': float(arrays['reference_injection_ess'])}
    (directory / 'monte_carlo_check.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(directory / 'monte_carlo_check.npz', full_injection_ess=full_ess,
                        alpha_relative=alpha_relative, full_min_event_ess=event_min_ess,
                        event_log_error=event_delta, log_likelihood_error=likelihood_delta)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
