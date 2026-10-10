import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from pyro.ops import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from binned_gp_benchmark import controlled_log_event_weights, controlled_selection, population_functionals
import run_external_gp_selected as production
from run_external_gp_selected import InitializedGPPotential, checked_reports, density_functionals, production_decision
from run_lisa_production import mixing_summary
from run_symmetric_gp_pilot import check_coordinates
from symmetric_gp_benchmark import SymmetricBinnedGPPotential
from run_external_gp_bounded_diagnostics import cumulative_minimum


@pytest.fixture(scope="module", autouse=True)
def precision():
    previous = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous[0])
    torch.set_num_threads(previous[1])


def configuration():
    config = json.loads((ROOT / "configs/external_gp_selected.json").read_text())
    protocol = json.loads((ROOT / config["protocol"]).read_text())
    pilot = json.loads((ROOT / config["validated_pilot"]).read_text())
    return config, protocol, pilot


def test_production_settings_follow_the_fixed_protocol_and_dispersed_starts():
    config, protocol, pilot = configuration()
    assert config["experiment"] == pilot["experiment"] == "selected_copula_truth"
    assert protocol["sampling"]["chains"] == 4
    starts = np.asarray(config["initial_amplitude_and_lengths"])
    assert starts.shape == (4, 4) and np.isfinite(starts).all() and (starts > 0).all()
    assert (np.ptp(starts, axis=0) > 0).all()
    assert protocol["sampling"]["warmup"] == 500
    assert protocol["sampling"]["retained_per_chain"] == 800
    assert set(pilot["bins_per_axis"]) == {6, 12}


@pytest.mark.parametrize("bins", [2, 3, 6, 12])
def test_uniform_cdfs_and_octants_integrate_even_when_boundaries_cross_bins(bins):
    names, values = density_functionals(torch.zeros(bins**3), bins, [.25, .5, .75])
    assert len(names) == len(set(names)) == 17
    np.testing.assert_allclose(values, [.25, .5, .75]*3+[.125]*8, rtol=0, atol=1e-14)


def test_joint_diagnostics_distinguish_dependence_with_identical_marginals():
    field = torch.tensor([2., -2., -2., 2., -2., 2., 2., -2.])
    _, values = density_functionals(field, 2, [.5])
    np.testing.assert_allclose(values[:3], [.5]*3, atol=1e-14)
    np.testing.assert_allclose(values[3:], torch.softmax(field, 0), atol=1e-14)
    assert values[3:].std() > .1
    assert values[3:].sum() == pytest.approx(1., abs=1e-14)


def test_initial_kernel_parameters_change_starts_without_changing_the_target():
    _, protocol, _ = configuration()
    weights = controlled_log_event_weights([[.2, .3, .4]], [.04, .05, .04], 3)
    selected = controlled_selection(3)
    original = SymmetricBinnedGPPotential(3, weights, selected, protocol["binned_gp"])
    potential = InitializedGPPotential(3, weights, selected, protocol["binned_gp"])
    with pytest.raises(ValueError, match="positive"):
        potential.set_initial_hyperparameters([0., 1., 1., 1.])
    potential.set_initial_hyperparameters([2., .1, .2, .3])
    start = potential.initial(923)
    np.testing.assert_allclose(start[:4].exp(), [2., .1, .2, .3], atol=1e-14)
    np.testing.assert_array_equal(start[4:], original.initial(923)[4:])
    assert float(potential(start)) == pytest.approx(float(original(start)), abs=1e-12)


def passing_example():
    _, protocol, _ = configuration()
    reports = [{"seed": seed, "status": "complete", "warmup_completed": 500, "samples_completed": 800,
                "divergences": [], "coordinate_verification": {"derivatives_passed": True},
                "covariance_jitter_log_target_difference_sd": .001, "event_quadrature_passed": True,
                "process_elapsed_seconds": 100., "runtime": {"peak_working_set_bytes": 1024**3}}
               for seed in protocol["sampling"]["chain_seeds"]]
    mixing = {"maximum_rhat": 1.005, "minimum_bulk_ess": 600., "minimum_tail_ess": 700.}
    return protocol, reports, mixing


def test_acceptance_requires_complete_chains_numerics_mixing_and_resources():
    protocol, reports, mixing = passing_example()
    assert production_decision(protocol, reports, mixing)["posterior_accepted"]
    broken = copy.deepcopy(reports)
    broken[0]["samples_completed"] = 799
    assert not production_decision(protocol, broken, mixing)["posterior_accepted"]
    broken = copy.deepcopy(reports)
    broken[0]["seed"] = reports[1]["seed"]
    assert not production_decision(protocol, broken, mixing)["posterior_accepted"]
    broken = copy.deepcopy(reports)
    broken[0]["divergences"] = [2]
    assert not production_decision(protocol, broken, mixing)["divergences_passed"]
    broken = copy.deepcopy(reports)
    broken[0]["coordinate_verification"]["derivatives_passed"] = False
    assert not production_decision(protocol, broken, mixing)["numerical_passed"]
    broken = copy.deepcopy(reports)
    broken[0]["covariance_jitter_log_target_difference_sd"] = .3
    assert not production_decision(protocol, broken, mixing)["numerical_passed"]
    broken = copy.deepcopy(reports)
    broken[0]["event_quadrature_passed"] = False
    assert not production_decision(protocol, broken, mixing)["numerical_passed"]
    broken = copy.deepcopy(reports)
    broken[0]["process_elapsed_seconds"] = 7201
    assert not production_decision(protocol, broken, mixing)["resources_passed"]
    broken = copy.deepcopy(reports)
    broken[0]["runtime"]["peak_working_set_bytes"] = 3*1024**3
    assert not production_decision(protocol, broken, mixing)["resources_passed"]
    for key, value in [("maximum_rhat", 1.02), ("minimum_bulk_ess", 399.),
                       ("minimum_tail_ess", 399.), ("maximum_rhat", float("nan"))]:
        assert not production_decision(protocol, reports, dict(mixing, **{key: value}))["mixing_passed"]
    assert not production_decision(protocol, reports, None)["posterior_accepted"]
    assert not production_decision(protocol, [], mixing)["posterior_accepted"]


def test_saved_chain_validation_rejects_changed_targets_and_products(tmp_path):
    _, protocol, _ = configuration()
    seed = protocol["sampling"]["chain_seeds"][0]
    path = tmp_path / f"chain_{seed}.npz"
    np.savez_compressed(path, coordinate=np.zeros((4, 3)))
    report = {"draw_product_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "target_product_sha256": "target", "seed": seed}
    path.with_suffix(".json").write_text(json.dumps(report))
    assert checked_reports(tmp_path, "target", protocol) == [report]
    with pytest.raises(ValueError, match="different target"):
        checked_reports(tmp_path, "other", protocol)
    np.savez_compressed(path, coordinate=np.ones((4, 3)))
    with pytest.raises(ValueError, match="digest"):
        checked_reports(tmp_path, "target", protocol)


def test_pooled_numerical_differences_are_checked_across_chains(tmp_path, monkeypatch):
    protocol, reports, mixing = passing_example()
    for index, report in enumerate(reports):
        report.update(sampling_and_compilation_seconds=10., retained_full_tree_evaluation_count=0)
        path = tmp_path / f"chain_{report['seed']}.npz"
        np.savez_compressed(path, diagnostic_names=["a"],
                            diagnostic_values=np.arange(8., dtype=float)[:, None],
                            covariance_jitter_log_target_difference=np.full(8, index))
        path.with_suffix(".json").write_text(json.dumps(report))
    monkeypatch.setattr(production, "checked_reports", lambda *args: reports)
    monkeypatch.setattr(production, "mixing_summary", lambda *args: mixing)
    monkeypatch.setattr(production, "runtime_metadata", lambda: {})
    summary = production.summarize(tmp_path, protocol, "target")
    assert summary["decision"]["mixing_passed"]
    assert not summary["decision"]["numerical_passed"]
    assert not summary["decision"]["posterior_accepted"]
    assert summary["pooled_jitter_log_target_difference_sd"] > .2


def test_bounded_diagnostics_preserve_prefix_minima_and_effective_sample_sizes(monkeypatch):
    generator = torch.Generator().manual_seed(819)
    values = torch.randn((32, 2, 3), generator=generator)
    values[5:9] = values[4]
    torch.testing.assert_close(cumulative_minimum(values), stats._cummin(values), rtol=0, atol=0)
    np.testing.assert_array_equal(cumulative_minimum(values), np.minimum.accumulate(values.numpy(), axis=0))
    samples = torch.randn((2, 80, 7), generator=generator)
    expected = stats.effective_sample_size(samples)
    monkeypatch.setattr(stats, "_cummin", cumulative_minimum)
    torch.testing.assert_close(stats.effective_sample_size(samples), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Finite"):
        cumulative_minimum(torch.tensor([float("nan")]))


def test_diagnostic_amendment_preserves_the_failed_attempt_and_target_settings():
    base, _, _ = configuration()
    refined = json.loads((ROOT / "configs/external_gp_selected_bounded.json").read_text())
    for name in ("protocol", "validated_pilot", "experiment", "initial_amplitude_and_lengths", "marginal_cdf_points", "joint_partition"):
        assert refined[name] == base[name]
    adapter = refined["diagnostic_adapter"]
    assert hashlib.sha256((ROOT / adapter["source"]).read_bytes()).hexdigest() == adapter["sha256"]
    assert hashlib.sha256(inspect.getsource(stats._cummin).encode()).hexdigest() == adapter["original_cummin_sha256"]
    amendment = refined["amendment"]
    path = ROOT / amendment["prior_attempt_summary"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == amendment["prior_attempt_summary_sha256"]
    summary = json.loads(path.read_text())
    assert not summary["decision"]["posterior_accepted"]
    assert not summary["decision"]["resources_passed"]
    assert summary["maximum_peak_working_set_bytes"] > 2*1024**3


def test_sampling_extension_changes_only_the_declared_draw_count_and_metadata():
    _, original, _ = configuration()
    config = json.loads((ROOT / "configs/external_gp_selected_extended.json").read_text())
    extended = json.loads((ROOT / config["protocol"]).read_text())
    assert extended["sampling"]["retained_per_chain"] == 2400
    comparison = copy.deepcopy(extended)
    comparison["sampling"]["retained_per_chain"] = original["sampling"]["retained_per_chain"]
    comparison["status"] = original["status"]
    comparison.pop("sampling_extension")
    assert comparison == original
    previous = json.loads((ROOT / extended["sampling_extension"]["previous_summary"]).read_text())
    assert previous["decision"]["complete"]
    assert previous["decision"]["numerical_passed"]
    assert previous["decision"]["resources_passed"]
    assert not previous["decision"]["posterior_accepted"]
    assert previous["mixing"]["minimum_tail_ess"] < original["sampling"]["minimum_functional_tail_ess"]


@pytest.mark.parametrize("bins", [6, 12])
def test_production_artifacts_and_decision_reconstruct(bins):
    config, protocol, pilot = configuration()
    if bins == 12:
        config = json.loads((ROOT / "configs/external_gp_selected_extended.json").read_text())
        protocol = json.loads((ROOT / config["protocol"]).read_text())
    output = ROOT / config["output"] / f"bins_{bins}"
    if not (output / "summary.json").exists():
        pytest.skip("Production calculation has not yet completed")
    target_path = output / "target.json"
    signature = json.loads(target_path.read_text())
    summary = json.loads((output / "summary.json").read_text())
    target_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
    assert summary["target_product_sha256"] == target_hash
    assert not summary["comparison_results_available"]
    assert signature["input_fields"] == ["observed", "sigma"]
    assert signature["configuration"] == config
    if bins == 12:
        record = json.loads((ROOT / config["output"] / "diagnostic_adapter.json").read_text())
        assert record["adapter"] == config["diagnostic_adapter"]
    for name, expected in signature["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        assert "test_catalog" not in name
    assert hashlib.sha256((output / "inputs.npz").read_bytes()).hexdigest() == signature["inputs_product_sha256"]
    reports = checked_reports(output, target_hash, protocol)
    assert len(reports) == 4
    values, differences = [], []
    with np.load(output / "inputs.npz") as inputs, np.load(ROOT / pilot["training_catalog"]) as catalog:
        weights, selection = inputs["log_event_weights"], inputs["selection_coefficients"]
        np.testing.assert_allclose(weights, controlled_log_event_weights(catalog["observed"], catalog["sigma"], bins, 64), rtol=0, atol=1e-9)
        np.testing.assert_allclose(selection, controlled_selection(bins), rtol=0, atol=1e-15)
    potential = SymmetricBinnedGPPotential(bins, weights, selection, protocol["binned_gp"], pilot["covariance_axis_jitter"])
    reference = SymmetricBinnedGPPotential(bins, weights, selection, protocol["binned_gp"], pilot["verification_covariance_axis_jitter"])
    for chain, report in enumerate(reports):
        path = output / f"chain_{report['seed']}.npz"
        assert summary["chain_reports_sha256"][path.with_suffix(".json").name] == hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest()
        with np.load(path) as product:
            retained = protocol["sampling"]["retained_per_chain"]
            assert product["coordinate"].shape == (retained, bins**3+4)
            assert product["warmup_coordinate"].shape == (500, bins**3+4)
            np.testing.assert_allclose(np.exp(product["initial_coordinate"][:4]), config["initial_amplitude_and_lengths"][chain], atol=1e-14)
            names = product["diagnostic_names"].tolist()
            assert len(names) == 34
            values.append(product["diagnostic_values"])
            differences.append(product["covariance_jitter_log_target_difference"])
            for index in (0, retained//2-1, retained-1):
                point = torch.tensor(product["coordinate"][index])
                field = potential.field(point)
                np.testing.assert_allclose(product["features"][index, :12], population_functionals(field, selection, bins), atol=1e-12)
                np.testing.assert_allclose(product["features"][index, 12:16], point[:4].exp(), atol=1e-12)
                assert product["features"][index, -1] == pytest.approx(float(-potential(point)), abs=1e-10)
                extra_names, extra = density_functionals(field, bins, config["marginal_cdf_points"])
                assert names[17:] == extra_names
                np.testing.assert_allclose(product["diagnostic_values"][index, 17:], extra, atol=1e-12)
            verification, arrays = check_coordinates(potential, reference, product["coordinate"], pilot)
            for name, recomputed in arrays.items():
                np.testing.assert_allclose(product[name], recomputed, rtol=0, atol=1e-9)
            assert report["coordinate_verification"]["draws_checked"] == verification["draws_checked"]
            if bins == 12:
                previous_path = ROOT / "results/analysis/external_gp_selected_bounded/bins_12" / path.name
                with np.load(previous_path) as previous:
                    np.testing.assert_array_equal(product["coordinate"][:800], previous["coordinate"])
                    np.testing.assert_array_equal(product["warmup_coordinate"], previous["warmup_coordinate"])
                    np.testing.assert_array_equal(product["features"][:800], previous["features"])
    mixing = mixing_summary(names, np.stack(values))
    for key in ("maximum_rhat", "minimum_bulk_ess", "minimum_tail_ess"):
        assert summary["mixing"][key] == pytest.approx(mixing[key], abs=1e-10)
    assert summary["pooled_jitter_log_target_difference_sd"] == pytest.approx(np.concatenate(differences).std(), abs=1e-12)
    decision = production_decision(protocol, reports, mixing)
    if np.concatenate(differences).std() > protocol["evaluation"]["maximum_log_likelihood_difference_sd"]:
        decision["numerical_passed"] = decision["posterior_accepted"] = False
    assert summary["decision"] == decision
    assert summary["sampling_and_compilation_seconds"] == pytest.approx(sum(r["sampling_and_compilation_seconds"] for r in reports))
    assert summary["recorded_process_elapsed_seconds"] == pytest.approx(sum(r["process_elapsed_seconds"] for r in reports))
    if bins == 12:
        old = ROOT / "results/analysis/external_gp_selected/bins_12/chain_847201.npz"
        with np.load(old) as previous, np.load(output / "chain_847201.npz") as current:
            np.testing.assert_array_equal(current["coordinate"][:800], previous["coordinate"])
            np.testing.assert_array_equal(current["warmup_coordinate"], previous["warmup_coordinate"])
            np.testing.assert_array_equal(current["features"][:800], previous["features"])
