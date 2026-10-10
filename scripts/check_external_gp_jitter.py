"""Diagnose covariance regularization in the training-only GP pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from binned_gp_benchmark import BinnedGPPotential
from correct_selected_posterior import sha256
from run_lisa_production import write_json

ROOT = Path(__file__).resolve().parents[1]


def symmetric_field(coordinate, bins, jitter):
    """Independent diagnostic only; not a posterior sampler implementation."""
    centers = (np.arange(bins)+.5)/bins
    distance = centers[:, None]-centers[None, :]
    roots = []
    residuals = []
    for length in np.exp(coordinate[1:4]):
        covariance = np.exp(-.5*(distance/length)**2) + jitter*np.eye(bins)
        values, vectors = np.linalg.eigh(covariance)
        if np.min(values) <= 0:
            raise ValueError("Non-positive diagnostic covariance eigenvalue")
        root = (vectors*np.sqrt(values)) @ vectors.T
        residuals.append(np.max(np.abs(root@root.T-covariance)))
        roots.append(root)
    field = np.einsum("ia,jb,kc,abc->ijk", *roots, coordinate[4:].reshape((bins,)*3), optimize=True)
    return np.exp(coordinate[0])*field.ravel(), max(residuals)


def log_likelihood(field, weights, selection):
    return logsumexp(weights+field, axis=1).sum() - len(weights)*logsumexp(np.log(selection)+field)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    config_path = ROOT / "configs/external_gp_pilot.json"
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT / config["protocol"]).read_text())
    directory = ROOT / config["output"]
    primary_jitter = config["covariance_axis_jitter"]
    fine_jitter = config["verification_covariance_axis_jitter"]
    rows = []
    sources = [config_path, ROOT / config["protocol"], Path(__file__), ROOT / "scripts/binned_gp_benchmark.py"]
    for bins in config["bins_per_axis"]:
        path = directory / f"bins_{bins}.npz"
        report = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != report["draw_product_sha256"]:
            raise ValueError("Pilot draw hash mismatch")
        sources.extend([path, path.with_suffix(".json")])
        with np.load(path) as product:
            coordinates = product["coordinate"]
            weights, selection = product["log_event_weights"], product["selection_coefficients"]
            primary = BinnedGPPotential(bins, weights, selection, protocol["binned_gp"], primary_jitter)
            fine = BinnedGPPotential(bins, weights, selection, protocol["binned_gp"], fine_jitter)
            cholesky_changes, symmetric_changes, field_changes, residuals = [], [], [], []
            with torch.no_grad():
                for coordinate in coordinates:
                    first = primary.field(torch.tensor(coordinate)).numpy()
                    second = fine.field(torch.tensor(coordinate)).numpy()
                    cholesky_changes.append(log_likelihood(second, weights, selection)-log_likelihood(first, weights, selection))
                    a, ra = symmetric_field(coordinate, bins, primary_jitter)
                    b, rb = symmetric_field(coordinate, bins, fine_jitter)
                    symmetric_changes.append(log_likelihood(b, weights, selection)-log_likelihood(a, weights, selection))
                    field_changes.append([np.max(np.abs(first-second)), np.max(np.abs(a-b))])
                    residuals.append(max(ra, rb))
            np.testing.assert_allclose(cholesky_changes, product["covariance_jitter_log_target_difference"], atol=2e-11)
            relative_covariance_change = (1+primary_jitter)**3 - (1+fine_jitter)**3
            rows.append({
                "bins_per_axis": bins, "draws": len(coordinates),
                "length_scale_quantiles_05_50_95": np.quantile(np.exp(coordinates[:, 1:4]), [.05, .5, .95], axis=0).tolist(),
                "maximum_covariance_element_change_over_sigma_squared": relative_covariance_change,
                "maximum_covariance_element_change": float(relative_covariance_change*np.exp(2*coordinates[:, 0]).max()),
                "cholesky_log_likelihood_difference_sd": float(np.std(cholesky_changes)),
                "symmetric_root_log_likelihood_difference_sd": float(np.std(symmetric_changes)),
                "maximum_field_change_cholesky_and_symmetric": np.max(field_changes, axis=0).tolist(),
                "maximum_symmetric_root_covariance_residual": float(max(residuals)),
                "cholesky_coordinate_check_passed": bool(np.std(cholesky_changes) < protocol["evaluation"]["maximum_log_likelihood_difference_sd"]),
            })
    result = {
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
        "primary_jitter": primary_jitter, "verification_jitter": fine_jitter, "resolutions": rows,
        "eligible_for_scientific_comparisons": False,
        "interpretation": "Changing Cholesky jitter changes the field at fixed whitened coordinates. A symmetric covariance square root is an alternative Gaussian prior parameterization. Its values here diagnose numerical continuity only: these coordinates are not posterior draws for that parameterization. Gradients and fresh sampling must be validated before using it.",
    }
    write_json(directory / "jitter_check.json", result)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
