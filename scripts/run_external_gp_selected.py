"""Obtain checked four-chain GP posteriors for the selected training catalog."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()

# This import establishes the recorded single-threaded numerical environment.
from run_external_gp_pilot import ROOT, run_pilot

import argparse
import json
from pathlib import Path

import numpy as np
import pyro
import torch

from benchmark_selected_mixture import runtime_metadata
from binned_gp_benchmark import controlled_log_event_weights, controlled_selection, grid_geometry
from correct_selected_posterior import sha256
from run_lisa_production import mixing_summary, verify_product, write_json
from run_symmetric_gp_pilot import check_coordinates
from symmetric_gp_benchmark import SymmetricBinnedGPPotential


class InitializedGPPotential(SymmetricBinnedGPPotential):
    def set_initial_hyperparameters(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Four positive initial kernel parameters are required")
        self.initial_hyperparameters = torch.tensor(np.log(values), dtype=torch.float64)

    def initial(self, seed):
        point = super().initial(seed)
        point[:4] = self.initial_hyperparameters
        return point


def density_functionals(field, bins, cdf_points):
    """Integrate marginal CDFs and joint octant probabilities within bins."""
    lower, upper = grid_geometry(bins)
    probability = torch.softmax(field, -1).detach().numpy()
    names, values = [], []
    for axis in range(3):
        for point in cdf_points:
            fraction = np.clip((point-lower[:, axis])*bins, 0, 1)
            names.append(f"cdf_x{axis+1}_{point:g}")
            values.append(probability @ fraction)
    for octant in np.ndindex(2, 2, 2):
        lo = np.asarray(octant)/2
        hi = lo+.5
        fraction = np.maximum(0., np.minimum(upper, hi)-np.maximum(lower, lo))*bins
        names.append("octant_"+"".join(str(i) for i in octant))
        values.append(probability @ fraction.prod(1))
    return names, np.asarray(values)


def production_decision(protocol, reports, mixing):
    sampling = protocol["sampling"]
    complete = (len(reports) == sampling["chains"]
                and [r["seed"] for r in reports] == sampling["chain_seeds"]
                and all(r["status"] == "complete"
                        and r["warmup_completed"] == sampling["warmup"]
                        and r["samples_completed"] == sampling["retained_per_chain"] for r in reports))
    no_divergences = complete and all(r["divergences"] == [] for r in reports)
    mixing_ok = (mixing is not None and np.isfinite([mixing[k] for k in (
        "maximum_rhat", "minimum_bulk_ess", "minimum_tail_ess")]).all()
        and mixing["maximum_rhat"] <= sampling["maximum_functional_rhat"]
        and mixing["minimum_bulk_ess"] >= sampling["minimum_functional_bulk_ess"]
        and mixing["minimum_tail_ess"] >= sampling["minimum_functional_tail_ess"])
    numerical_ok = complete and all(r["coordinate_verification"]["derivatives_passed"]
        and r["covariance_jitter_log_target_difference_sd"] is not None
        and r["covariance_jitter_log_target_difference_sd"] <= protocol["evaluation"]["maximum_log_likelihood_difference_sd"]
        and r["event_quadrature_passed"] for r in reports)
    resource_ok = complete and (
        sum(r["process_elapsed_seconds"] for r in reports) <= protocol["resources"]["maximum_seconds_per_fit"]
        and max(r["runtime"]["peak_working_set_bytes"] for r in reports)
        <= protocol["resources"]["maximum_resident_gib"]*1024**3)
    decision = {"complete": bool(complete), "divergences_passed": bool(no_divergences),
                "mixing_passed": bool(mixing_ok), "numerical_passed": bool(numerical_ok),
                "resources_passed": bool(resource_ok)}
    decision["posterior_accepted"] = all(decision.values())
    return decision


def checked_reports(output, target_hash, protocol):
    reports = []
    for seed in protocol["sampling"]["chain_seeds"]:
        path = output / f"chain_{seed}.npz"
        if not path.exists() and not path.with_suffix(".json").exists():
            continue
        report = verify_product(path)
        if report["target_product_sha256"] != target_hash or report["seed"] != seed:
            raise ValueError("Saved chain has a different target or seed")
        reports.append(report)
    return reports


def summarize(output, protocol, target_hash):
    reports = checked_reports(output, target_hash, protocol)
    mixing, mixing_error, traces = None, None, {}
    pooled_jitter_sd = None
    if len(reports) == protocol["sampling"]["chains"] and min(r["samples_completed"] for r in reports) >= 4:
        products = []
        for report in reports:
            with np.load(output / f"chain_{report['seed']}.npz") as product:
                products.append({k: product[k] for k in ("diagnostic_names", "diagnostic_values",
                                                       "covariance_jitter_log_target_difference")})
        names = products[0]["diagnostic_names"].tolist()
        if any(p["diagnostic_names"].tolist() != names for p in products):
            raise ValueError("Chains have different diagnostic functionals")
        length = min(len(p["diagnostic_values"]) for p in products)
        values = np.stack([p["diagnostic_values"][:length] for p in products])
        try:
            mixing = mixing_summary(names, values)
        except ValueError as exc:
            mixing_error = str(exc)
        scale = values.reshape(-1, len(names)).std(0)
        for j, name in enumerate(names):
            traces[name] = {"chain_means": values[:, :, j].mean(1).tolist(),
                           "chain_quantiles_05_50_95": np.quantile(values[:, :, j], [.05, .5, .95], axis=1).T.tolist(),
                           "half_chain_means": [[float(v[:length//2, j].mean()), float(v[length//2:, j].mean())]
                                                for v in values],
                           "pooled_sd": float(scale[j])}
        pooled_jitter_sd = float(np.concatenate([p["covariance_jitter_log_target_difference"] for p in products]).std())
    decision = production_decision(protocol, reports, mixing)
    if pooled_jitter_sd is None or pooled_jitter_sd > protocol["evaluation"]["maximum_log_likelihood_difference_sd"]:
        decision["numerical_passed"] = decision["posterior_accepted"] = False
    summary = {"decision": decision, "mixing": mixing, "mixing_error": mixing_error,
               "trace_summaries": traces, "pooled_jitter_log_target_difference_sd": pooled_jitter_sd,
               "sampling_and_compilation_seconds": sum(r["sampling_and_compilation_seconds"] for r in reports),
               "recorded_process_elapsed_seconds": sum(r["process_elapsed_seconds"] for r in reports),
               "retained_full_tree_evaluation_count": sum(r["retained_full_tree_evaluation_count"] for r in reports),
               "maximum_peak_working_set_bytes": max((r["runtime"]["peak_working_set_bytes"] for r in reports), default=None),
               "target_product_sha256": target_hash,
               "chain_reports_sha256": {f"chain_{r['seed']}.json": sha256(output / f"chain_{r['seed']}.json") for r in reports},
               "comparison_results_available": False, "runtime": runtime_metadata()}
    write_json(output / "summary.json", summary)
    print(json.dumps({"decision": decision, "mixing": None if mixing is None else
                      {k: mixing[k] for k in ("maximum_rhat", "minimum_bulk_ess", "minimum_tail_ess")}}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/external_gp_selected.json")
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--chain", type=int, choices=range(4))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT / config["protocol"]).read_text())
    pilot = json.loads((ROOT / config["validated_pilot"]).read_text())
    experiment = next(case for case in protocol["experiments"] if case["name"] == config["experiment"])
    if config["experiment"] != "selected_copula_truth" or args.bins not in (
        protocol["binned_gp"]["primary_bins_per_axis"], protocol["binned_gp"]["sensitivity_bins_per_axis"]):
        raise ValueError("This calculation is restricted to the declared selected-catalog experiment")
    starts = config["initial_amplitude_and_lengths"]
    if np.asarray(starts).shape != (protocol["sampling"]["chains"], 4):
        raise ValueError("One initial kernel configuration per chain is required")
    torch.set_num_threads(protocol["resources"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    output = ROOT / config["output"] / f"bins_{args.bins}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Completed production summary is retained")
    with np.load(ROOT / experiment["train"]) as catalog:
        if len(catalog["observed"]) != experiment["train_count"]:
            raise ValueError("Training count differs from the declared experiment")
        weights = controlled_log_event_weights(catalog["observed"], catalog["sigma"], args.bins, pilot["quadrature_order"])
        fine = controlled_log_event_weights(catalog["observed"], catalog["sigma"], args.bins, pilot["verification_quadrature_order"])
    error = float(np.max(np.abs(weights-fine)))
    if error > pilot["maximum_quadrature_log_weight_difference"]:
        raise ValueError("Production event quadrature needs refinement")
    selection = controlled_selection(args.bins)
    inputs_path = output / "inputs.npz"
    if inputs_path.exists():
        with np.load(inputs_path) as inputs:
            if not np.array_equal(inputs["log_event_weights"], weights) or not np.array_equal(inputs["selection_coefficients"], selection):
                raise ValueError("Saved event or selection coefficients differ")
    else:
        np.savez_compressed(inputs_path, log_event_weights=weights, selection_coefficients=selection)
    paths = [config_path, ROOT / config["protocol"], ROOT / config["validated_pilot"], ROOT / experiment["train"],
             Path(__file__), ROOT / "scripts/run_external_gp_pilot.py", ROOT / "scripts/run_symmetric_gp_pilot.py",
             ROOT / "scripts/binned_gp_benchmark.py", ROOT / "scripts/symmetric_gp_benchmark.py",
             ROOT / "scripts/sample_lisa_transport.py", ROOT / "scripts/run_lisa_production.py",
             ROOT / "scripts/benchmark_selected_mixture.py", ROOT / protocol["source_manifest"]]
    signature = {"configuration": config, "bins_per_axis": args.bins, "experiment": config["experiment"],
                 "input_fields": ["observed", "sigma"], "inputs_product_sha256": sha256(inputs_path),
                 "quadrature_maximum_log_weight_change": error,
                 "quadrature_total_log_likelihood_difference_bound": 2*len(weights)*error,
                 "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}}
    target_path = output / "target.json"
    if target_path.exists():
        if json.loads(target_path.read_text()) != signature:
            raise ValueError("Existing production target differs from the requested sources")
    else:
        write_json(target_path, signature)
    target_hash = sha256(target_path)
    if args.summarize:
        summarize(output, protocol, target_hash)
        return
    previous = checked_reports(output, target_hash, protocol)
    previous_seconds = sum(r["process_elapsed_seconds"] for r in previous)
    previous_seeds = {r["seed"] for r in previous}
    accounted_start = PROCESS_STARTED
    for chain in ([args.chain] if args.chain is not None else range(protocol["sampling"]["chains"])):
        seed = protocol["sampling"]["chain_seeds"][chain]
        if seed in previous_seeds:
            if args.chain is not None:
                raise FileExistsError("Completed production chain is retained")
            continue
        remaining = protocol["resources"]["maximum_seconds_per_fit"]-previous_seconds-(time.perf_counter()-PROCESS_STARTED)
        if remaining <= 0:
            break
        chain_config = dict(pilot, seed=seed, warmup=protocol["sampling"]["warmup"],
                            retained=protocol["sampling"]["retained_per_chain"], maximum_seconds=remaining,
                            target_acceptance=protocol["sampling"]["target_acceptance"],
                            maximum_tree_depth=protocol["sampling"]["maximum_tree_depth"],
                            output=config["output"], decision=config["decision"])
        potential = InitializedGPPotential(args.bins, weights, selection, protocol["binned_gp"], pilot["covariance_axis_jitter"])
        potential.set_initial_hyperparameters(starts[chain])
        reference = SymmetricBinnedGPPotential(args.bins, weights, selection, protocol["binned_gp"], pilot["verification_covariance_axis_jitter"])
        pyro.clear_param_store()
        report, arrays = run_pilot(potential, chain_config)
        verification, refined = check_coordinates(potential, reference, arrays["coordinate"], pilot)
        arrays.update(refined)
        extra_names, extra_values = density_functionals(torch.zeros(args.bins**3), args.bins, config["marginal_cdf_points"])[0], []
        with torch.no_grad():
            for point in arrays["coordinate"]:
                _, values = density_functionals(potential.field(torch.tensor(point)), args.bins, config["marginal_cdf_points"])
                extra_values.append(values)
        arrays["diagnostic_names"] = list(arrays["feature_names"])+extra_names
        arrays["diagnostic_values"] = np.column_stack([arrays["features"], np.asarray(extra_values).reshape(len(extra_values), len(extra_names))])
        path = output / f"chain_{seed}.npz"
        np.savez_compressed(path, **arrays)
        report.update(seed=seed, initial_amplitude_and_lengths=starts[chain],
                      target_product_sha256=target_hash, draw_product_sha256=sha256(path),
                      coordinate_verification=verification, event_quadrature_passed=True,
                      covariance_jitter_log_target_difference_sd=verification["jitter_difference_sd"],
                      process_elapsed_seconds=time.perf_counter()-accounted_start, runtime=runtime_metadata())
        write_json(path.with_suffix(".json"), report)
        accounted_start = time.perf_counter()
        print("Saved", args.bins, seed, report["status"], "sampling seconds", round(report["sampling_and_compilation_seconds"], 2), flush=True)
        del arrays, refined, extra_values
        if report["status"] != "complete":
            break
    if args.chain is None:
        summarize(output, protocol, target_hash)


if __name__ == "__main__":
    main()
