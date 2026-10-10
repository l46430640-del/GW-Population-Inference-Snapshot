"""Common LISA selected-catalog integrals for the binned GP representation."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
import torch

from binned_gp_benchmark import grid_geometry


def selection_coefficients(response, bins, power, seed, cell_block=8, check=lambda: None):
    lower, _ = grid_geometry(bins)
    uniform = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    result = np.empty(bins**3)
    for start in range(0, len(lower), cell_block):
        check()
        points = lower[start:start+cell_block, None] + uniform[None]/bins
        detected = ndtr(response(points.reshape(-1, 3))-8).reshape(len(points), -1)
        result[start:start+len(points)] = detected.mean(1)/bins**3
    if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result > 1/bins**3):
        raise ValueError('Every LISA bin must retain a finite positive selection integral')
    return result


@torch.no_grad()
def integral_changes(potential, coordinates, fine_weights, fine_selection, check=lambda: None):
    probability = []
    for point in coordinates:
        check()
        probability.append(torch.softmax(potential.field(torch.tensor(point)), -1).numpy())
    probability = np.asarray(probability).reshape(-1, potential.bins**3)
    primary = probability @ np.exp(potential.log_event_weights.numpy()).T
    refined = probability @ np.exp(fine_weights).T
    alpha = probability @ potential.selection.numpy()
    alpha_fine = probability @ fine_selection
    if np.any(primary <= 0) or np.any(refined <= 0) or np.any(alpha <= 0) or np.any(alpha_fine <= 0):
        raise ValueError('An event or selection integral underflows')
    event = (np.log(refined)-np.log(primary)).sum(1)
    selection = -primary.shape[1]*(np.log(alpha_fine)-np.log(alpha))
    return {'event_log_likelihood_change': event, 'selection_log_likelihood_change': selection,
        'combined_log_likelihood_change': event+selection}


def integral_report(arrays, tolerance):
    differences = {name: float(np.std(value)) if len(value) else None for name, value in arrays.items()}
    return {'standard_deviations': differences,
        'passed': bool(all(value is not None and np.isfinite(value) and value <= tolerance for value in differences.values()))}
