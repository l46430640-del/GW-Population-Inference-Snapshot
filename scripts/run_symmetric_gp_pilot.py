"""Repeat the bounded GP pilot in stable coordinates without changing its prior."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()

# The retained sampler also sets single-threaded library limits before imports.
from run_external_gp_pilot import ROOT, run_pilot

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from binned_gp_benchmark import controlled_log_event_weights, controlled_selection
from correct_selected_posterior import sha256
from run_lisa_production import write_json
from sample_lisa_transport import value_gradient
from symmetric_gp_benchmark import SymmetricBinnedGPPotential


def check_coordinates(potential, reference, coordinates, config):
    points = torch.as_tensor(coordinates, dtype=torch.float64)
    difference = []
    with torch.no_grad():
        for point in points:
            difference.append(float(potential(point)-reference(point)))
    if not len(points):
        return {"derivatives_passed": False, "jitter_difference_sd": None}, {
            "covariance_jitter_log_target_difference": np.empty(0)}
    indices = np.unique(np.r_[np.linspace(0, len(points)-1, min(len(points), config["derivative_check_evenly_spaced_draws"]), dtype=int),
                              coordinates[:, :4].argmin(axis=0), coordinates[:, :4].argmax(axis=0),
                              np.argmax(np.abs(difference))]).astype(int)
    directions = torch.zeros((5, potential.dimension), dtype=torch.float64)
    directions[:4, :4] = torch.eye(4)
    white = torch.sin(torch.arange(potential.dimension-4, dtype=torch.float64)+1)
    directions[4, 4:] = white/torch.linalg.vector_norm(white)
    automatic, numeric = [], []
    step = config["derivative_check_step"]
    for index in indices:
        point = points[index]
        _, gradient = value_gradient(potential, point)
        automatic.append((directions@gradient).numpy())
        row = []
        with torch.no_grad():
            for direction in directions:
                full = (potential(point+step*direction)-potential(point-step*direction))/(2*step)
                half = (potential(point+step/2*direction)-potential(point-step/2*direction))/step
                row.append(float((4*half-full)/3))
        numeric.append(row)
    automatic, numeric = np.asarray(automatic), np.asarray(numeric)
    maximum_error = float(np.max(np.abs(automatic-numeric)))
    finite = np.isfinite(automatic).all() and np.isfinite(numeric).all() and np.isfinite(difference).all()
    report = {"draws_checked": indices.tolist(), "direction_count": 5,
              "maximum_directional_derivative_error": maximum_error if np.isfinite(maximum_error) else None,
              "derivatives_passed": bool(finite and maximum_error < config["maximum_directional_derivative_error"]),
              "jitter_difference_sd": float(np.std(difference)) if np.isfinite(difference).all() else None}
    arrays = {"derivative_check_indices": indices, "directional_derivative_automatic": automatic,
              "directional_derivative_finite_difference": numeric,
              "covariance_jitter_log_target_difference": np.asarray(difference)}
    return report, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/external_gp_symmetric_pilot.json")
    parser.add_argument("--bins", type=int, required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    protocol_path = ROOT / config["protocol"]
    protocol = json.loads(protocol_path.read_text())
    if args.bins not in config["bins_per_axis"]:
        raise ValueError("Undeclared pilot resolution")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    paths = [config_path, protocol_path, ROOT / config["training_catalog"], Path(__file__),
             ROOT / "scripts/run_external_gp_pilot.py", ROOT / "scripts/binned_gp_benchmark.py",
             ROOT / "scripts/symmetric_gp_benchmark.py", ROOT / "scripts/sample_lisa_transport.py",
             ROOT / protocol["source_manifest"]]
    signature = {"source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths},
                 "input_fields": ["observed", "sigma"], "experiment": config["experiment"],
                 "coordinate_basis": config["coordinate_basis"]}
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    signature_path = output / "target.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("Existing symmetric pilot has different sources")
    else:
        write_json(signature_path, signature)
    path = output / f"bins_{args.bins}.npz"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Completed pilot is retained")
    started = time.perf_counter()
    with np.load(ROOT / config["training_catalog"]) as catalog:
        weights = controlled_log_event_weights(catalog["observed"], catalog["sigma"], args.bins, config["quadrature_order"])
        fine_weights = controlled_log_event_weights(catalog["observed"], catalog["sigma"], args.bins, config["verification_quadrature_order"])
    error = float(np.max(np.abs(weights-fine_weights)))
    if error > config["maximum_quadrature_log_weight_difference"]:
        raise ValueError("Event integration requires refinement")
    selection = controlled_selection(args.bins)
    potential = SymmetricBinnedGPPotential(args.bins, weights, selection, protocol["binned_gp"], config["covariance_axis_jitter"])
    reference = SymmetricBinnedGPPotential(args.bins, weights, selection, protocol["binned_gp"], config["verification_covariance_axis_jitter"])
    preparation_seconds = time.perf_counter()-started
    report, arrays = run_pilot(potential, config)
    # Reconstruct refinement in the sampled basis. The legacy helper's
    # Cholesky reference is in different coordinates and is not this check.
    verification_started = time.perf_counter()
    verification, refinement = check_coordinates(potential, reference, arrays["coordinate"], config)
    arrays.update(refinement)
    arrays.update(log_event_weights=weights, selection_coefficients=selection)
    jitter_ok = verification["jitter_difference_sd"] is not None and verification["jitter_difference_sd"] < protocol["evaluation"]["maximum_log_likelihood_difference_sd"]
    report.update(
        source_sha256=signature["source_sha256"], preparation_seconds=preparation_seconds,
        quadrature_maximum_log_weight_change=error, coordinate_basis=config["coordinate_basis"],
        coordinate_verification=verification, covariance_jitter_log_target_difference_sd=verification["jitter_difference_sd"],
        stability_check_passed=bool(jitter_ok and verification["derivatives_passed"]),
        coordinate_verification_seconds=time.perf_counter()-verification_started,
        process_elapsed_seconds=time.perf_counter()-PROCESS_STARTED,
        ready_for_production_sampling=bool(report["status"] == "complete" and report["divergences"] == []
                                           and jitter_ok and verification["derivatives_passed"]),
        eligible_for_scientific_comparisons=False,
    )
    np.savez_compressed(path, **arrays)
    report["draw_product_sha256"] = sha256(path)
    write_json(path.with_suffix(".json"), report)
    print(json.dumps({key: report[key] for key in ("status", "bins_per_axis", "sampling_and_compilation_seconds",
                      "divergences", "covariance_jitter_log_target_difference_sd", "coordinate_verification",
                      "ready_for_production_sampling")}), flush=True)


if __name__ == "__main__":
    main()
