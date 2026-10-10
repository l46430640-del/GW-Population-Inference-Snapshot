"""Run bounded, training-only pilots of the external binned GP adapter."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import arviz as az
import numpy as np
import psutil
import pyro
from pyro.infer import MCMC, NUTS
import torch

from benchmark_selected_mixture import runtime_metadata
from binned_gp_benchmark import (
    BinnedGPPotential, FEATURE_NAMES, controlled_log_event_weights,
    controlled_selection, population_functionals,
)
from correct_selected_posterior import sha256
from run_lisa_production import write_json
from sample_lisa_transport import compile_potential

ROOT = Path(__file__).resolve().parents[1]


class BudgetExceeded(RuntimeError):
    pass


def run_pilot(potential, config):
    initial = potential.initial(config["seed"])
    started = time.perf_counter()
    traced, trace_checks = compile_potential(potential, initial, config["jit_absolute_tolerance"])
    compilation_seconds = time.perf_counter() - started
    process = psutil.Process()
    calls, peak_rss = 0, process.memory_info().rss
    warmup, samples, progress = [], [], []

    def counted(values):
        nonlocal calls, peak_rss
        calls += 1
        if time.perf_counter()-started > config["maximum_seconds"]:
            raise BudgetExceeded("time_limit")
        if calls % 64 == 1:
            memory = process.memory_info()
            peak_rss = max(peak_rss, getattr(memory, "peak_wset", memory.rss))
            if peak_rss > config["maximum_resident_gib"]*1024**3:
                raise BudgetExceeded("memory_limit")
        return traced(values["coordinate"])

    def record(kernel, state, stage, iteration):
        (warmup if stage == "Warmup" else samples).append(state["coordinate"].detach().clone())
        progress.append([int(stage != "Warmup"), iteration, calls,
                         time.perf_counter()-started, float(kernel.step_size)])
        if (iteration+1) % 20 == 0:
            print(potential.bins, stage, iteration+1, "seconds", round(progress[-1][3], 1), flush=True)

    pyro.set_rng_seed(config["seed"])
    kernel = NUTS(potential_fn=counted, full_mass=config["full_mass"],
                  target_accept_prob=config["target_acceptance"], max_tree_depth=config["maximum_tree_depth"])
    sampler = MCMC(kernel, initial_params={"coordinate": initial}, num_chains=1,
                   warmup_steps=config["warmup"], num_samples=config["retained"],
                   disable_progbar=True, hook_fn=record)
    status, error = "complete", None
    try:
        sampler.run()
    except BudgetExceeded as exc:
        status = str(exc)
    except (RuntimeError, ValueError) as exc:
        status, error = "numerical_failure", str(exc)
    seconds = time.perf_counter()-started
    coordinate = torch.stack(samples) if samples else torch.empty((0, potential.dimension))
    feature_names = FEATURE_NAMES + ["sigma", "length_x1", "length_x2", "length_x3", "log_posterior"]
    features, jitter_difference = [], []
    reference = BinnedGPPotential(potential.bins, potential.log_event_weights.numpy(),
                                 potential.selection.numpy(),
                                 json.loads((ROOT / config["protocol"]).read_text())["binned_gp"],
                                 jitter=config["verification_covariance_axis_jitter"])
    with torch.no_grad():
        for point in coordinate:
            features.append(torch.cat([population_functionals(potential.field(point), potential.selection, potential.bins),
                                       point[:4].exp(), -potential(point)[None]]).numpy())
            jitter_difference.append(float(potential(point)-reference(point)))
    features = np.asarray(features).reshape(len(coordinate), len(feature_names))
    diagnostics = sampler.diagnostics() if status == "complete" else None
    pilot_ess = None
    if len(coordinate) >= 4 and np.isfinite(features).all():
        dataset = az.from_dict(posterior={name: features[None, :, i] for i, name in enumerate(feature_names)})
        pilot_ess = {method: {name: float(value) if np.isfinite(float(value)) else None
                              for name, value in az.ess(dataset, method=method).data_vars.items()}
                     for method in ("bulk", "tail")}
    retained_evaluations = np.diff([0] + [row[2] for row in progress])[len(warmup):]
    report = {
        "status": status, "error": error, "bins_per_axis": potential.bins, "dimension": potential.dimension,
        "configuration": config, "sampling_and_compilation_seconds": seconds,
        "compilation_seconds": compilation_seconds, "potential_evaluations": calls,
        "warmup_completed": len(warmup), "samples_completed": len(samples),
        "divergences": diagnostics["divergences"]["chain 0"] if diagnostics else None,
        "acceptance_rate": diagnostics["acceptance rate"]["chain 0"] if diagnostics else None,
        "step_size": float(kernel.step_size), "peak_resident_bytes": peak_rss,
        "retained_potential_evaluation_quantiles": np.quantile(retained_evaluations, [0, .5, 1]).tolist() if samples else None,
        "retained_full_tree_evaluation_count": int(np.sum(retained_evaluations >= 2**config["maximum_tree_depth"]-1)),
        "jit_checks": trace_checks, "single_chain_pilot_ess": pilot_ess,
        "covariance_jitter_log_target_difference_sd": float(np.std(jitter_difference)) if samples else None,
        "eligible_for_scientific_comparisons": False, "runtime": runtime_metadata(),
    }
    arrays = {"coordinate": coordinate.numpy(), "initial_coordinate": initial.numpy(),
              "warmup_coordinate": torch.stack(warmup).numpy() if warmup else np.empty((0, potential.dimension)),
              "features": features, "feature_names": feature_names, "progress": np.asarray(progress),
              "covariance_jitter_log_target_difference": np.asarray(jitter_difference)}
    return report, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/external_gp_pilot.json")
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
    source_paths = [config_path, protocol_path, ROOT / config["training_catalog"],
                    Path(__file__), ROOT / "scripts/binned_gp_benchmark.py",
                    ROOT / "scripts/sample_lisa_transport.py", ROOT / protocol["source_manifest"]]
    signature = {"source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_paths},
                 "input_fields": ["observed", "sigma"], "experiment": config["experiment"]}
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    signature_path = output / "target.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("Existing pilot has different sources")
    else:
        write_json(signature_path, signature)
    path = output / f"bins_{args.bins}.npz"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Completed pilot is retained")
    prepare_started = time.perf_counter()
    with np.load(ROOT / config["training_catalog"]) as catalog:
        weights = controlled_log_event_weights(catalog["observed"], catalog["sigma"], args.bins,
                                               config["quadrature_order"])
        fine = controlled_log_event_weights(catalog["observed"], catalog["sigma"], args.bins,
                                            config["verification_quadrature_order"])
    error = float(np.max(np.abs(weights-fine)))
    if error > config["maximum_quadrature_log_weight_difference"]:
        raise ValueError(f"Event quadrature refinement changes log weights by {error}")
    selection = controlled_selection(args.bins)
    potential = BinnedGPPotential(args.bins, weights, selection, protocol["binned_gp"],
                                 config["covariance_axis_jitter"])
    preparation_seconds = time.perf_counter()-prepare_started
    report, arrays = run_pilot(potential, config)
    arrays.update(log_event_weights=weights, selection_coefficients=selection)
    np.savez_compressed(path, **arrays)
    report.update(source_sha256=signature["source_sha256"], draw_product_sha256=sha256(path),
                  preparation_seconds=preparation_seconds, quadrature_maximum_log_weight_change=error,
                  process_elapsed_seconds=time.perf_counter()-PROCESS_STARTED)
    write_json(path.with_suffix(".json"), report)
    print(json.dumps({key: report[key] for key in ("status", "bins_per_axis", "sampling_and_compilation_seconds",
                                                  "samples_completed", "divergences", "peak_resident_bytes")}), flush=True)


if __name__ == "__main__":
    main()
