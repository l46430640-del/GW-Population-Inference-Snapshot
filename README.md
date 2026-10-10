# GW Population Inference Snapshot

Analysis code and saved numerical products for **Joint population structure in
gravitational-wave catalogs with bounded copula mixtures**, by Yi-kun Li,
Cheng Cheng, Lang Cui, and Yun Fang.

This snapshot contains the authors' population models, analysis and figure
scripts, configurations, simulated catalogs, posterior samples, numerical
summaries, and figures.
It was assembled from existing files on 2026-10-10. No inference, numerical
validation, or figure reconstruction was rerun for this publication step.

## Contents

| Location | Contents |
| --- | --- |
| `src/gwpopulation/` | Bounded marginal distributions, Gaussian copula mixtures, variational inference, selection and rate calculations, and stable density kernels |
| `scripts/` | Authors' simulation, inference, posterior comparison, integration, and figure routines |
| `configs/` | Fixed configurations for the reported experiments and their numerical refinements |
| `results/analysis/` | Saved posterior draws, simulated observations, derived densities, predictive scores, integration diagnostics, and recorded computation costs |
| `model_assets/lisa_analysis/` | Authors' generated IMRPhenomD SNR grid and its original numerical assessment |
| `figures/` | Population-density figures in PDF and PNG formats |
| `docs/analysis/*.json` | Input provenance and saved numerical assessments |
| `tests/` | Scientific tests for the population models and analysis routines |

This repository distributes software and numerical products.

### Finding the Reported Results

* `configs/analysis_runs.json` specifies the LISA baseline candidates and the
  adopted GWTC-3 analysis. For a fixed LISA model, optimization starts are
  selected by the final mean negative ELBO, without using injected truth.
* `selected_mixture/`, `importance_correction/`, and `importance_prediction/`
  contain the controlled experiment, independence comparison, and corrected
  posterior predictions. `vi_corrected/` contains the simpler common-target
  variational/MCMC comparison.
* `morphologies/`, `lisa_production/`, `lisa_refinement/`, and
  `lisa_scientific_comparison/` contain the LISA catalogs and population,
  posterior, recovery, and prediction calculations. Earlier correction and
  transport attempts are retained to document the inference path and cost.
* `external_selected_comparison/`, `external_control_comparison/`,
  `external_lisa_comparison_double/`, and `external_lisa_recovery_refinement/`
  contain the comparisons with the binned Gaussian-process model. The
  registries `external_control_posteriors.json` and
  `external_lisa_gp_posteriors.json` identify the retained chains. Earlier
  attempts and both GP resolutions remain available. The final comparisons
  are in `external_selected_comparison/partition_refinement/summary.json`,
  `external_control_comparison/summary.json`, and
  `external_lisa_recovery_refinement/summary.json`.
* `lvk_smooth/`, `lvk_mixture/`, and `lvk_reference/` contain the GWTC-3
  population results and our reconstructed reference-density summaries.
  The small `lvk/` and `lvk_refined/` records document initialization; they
  are not substitutes for the mixture-injection analysis.
* `density_impact/`, `lvk_pe_uncertainty/`, `lvk_k2_compression/`, and
  `prior_diagnostics/` contain the sensitivity assessments. The first three
  store their final summaries in `attempt_000/summary.json`.

These names are relative to `results/analysis/` unless a full path is given.
Pilot outputs and intermediate comparisons should be interpreted through the
final summaries, not treated as additional independent experiments.

### Interpretation and Model Versions

The population representation combines finite mixtures of bounded marginals
with Gaussian copulas. The GP comparison is the authors' implementation of
the published binned-density model in common unit coordinates, with the same
observation likelihood and independent scale-invariant rate prior as the
copula experiment. The GP log-height amplitude has a half-normal prior of
scale 10; independent lognormal length scales have median 0.3 and log standard
deviation 1. These adaptations are specified in `configs/external_benchmark.json`.
This is a comparison of population representations, not a reproduction of
the external authors' LVK inference or software runtime.

The saved comparison includes all three populations and both GP resolutions,
including outcomes favoring the GP. Single-catalog intervals and their scores
do not establish repeated-catalog coverage. The LISA observations use a
controlled, nonspinning, averaged-response experiment. The GWTC-3 results use
variational inference with the fixed spin assumptions in their configurations.

Historical fits use the original estimator in `core.py` and the original
analysis runners. `gwpopulation.stable_model.CorrectedPopulationEstimator`
implements the stable bounded-density definition with consistent log tails.
The saved-draw impact assessment compares these definitions without replacing
the original posterior samples with newly fitted ones. Keeping both versions
preserves the connection between the saved results and their actual estimator.
Do not relabel the historical checkpoints as fresh corrected-target fits.

## External Inputs: Obtain Separately

No downloaded external data, third-party source files, journal templates, or
literature PDFs/source archives are included. The fetchers in `scripts/` are
the authors' own download utilities; they retrieve external inputs into the
ignored `external_data/` directory when explicitly run.

| Input | Public source | Acquisition and provenance |
| --- | --- | --- |
| GWTC-3 event PE samples and official population hyperposterior | [Callister and Farr data release, Zenodo 8087858](https://doi.org/10.5281/zenodo.8087858) | `scripts/fetch_lvk_analysis_inputs.py`; member names and SHA-256 values in `docs/analysis/lvk_input_manifest.json` |
| O1--O3 mixture search injections | [LVK sensitivity release, Zenodo 5636816](https://doi.org/10.5281/zenodo.5636816) | `scripts/fetch_lvk_mixture.py`; filename and checksums in `docs/analysis/lvk_mixture_manifest.json` |
| IMRPhenomD waveform implementation | [PyIMRPhenomD](https://github.com/XGI-MSU/PyIMRPhenomD) | Commit `0a008e776fe6f92aeca18ebba0ba6ec7216b6b16`; `scripts/fetch_lisa_waveform.py` and `docs/analysis/waveform_manifest.json` |
| Published GP implementation consulted for the comparator | [gppop release, Zenodo 14691458](https://zenodo.org/records/14691458) | `gppop_spin_dev_ee5ffc421.tar.gz`; URLs and checksums in `docs/analysis/external_benchmark_sources.json` |

The Callister-Farr fetcher also retrieves the smaller historical injection
dictionary, whose filename is in the manifest. The reported GWTC-3 analysis
uses the wider mixture campaign in the second row. Prepared PE/injection
caches are also excluded. The retained `lvk_reference/density_summary.npz`
contains our computed marginal-density summaries, not a copy of the downloaded
official hyperposterior. Likewise, the LISA SNR grid is a generated numerical
product; the waveform source itself is excluded and retains its own GPL license.

The GP implementation is described in [Ray et al., ApJ 957, 37 (2023)](https://doi.org/10.3847/1538-4357/acf452)
and [Ray et al., ApJ 991, 17 (2025)](https://doi.org/10.3847/1538-4357/adf22a).
`scripts/fetch_external_benchmark_sources.py` retrieves their reference
literature and released code for inspection. The local comparator routines
do not import the external `gppop` package.

## Environment and Entry Points

Run commands from the repository root in a separate Python environment.
`requirements-analysis.txt` records the package versions used for the saved
CPU calculations with Python 3.11.7. The recorded experiments used one CPU
thread. Installation is optional for reading the saved JSON summaries and figures.

```console
python -m pip install -e . -r requirements-analysis.txt
```

The following commands obtain external inputs when they are needed. They were
not executed to assemble this snapshot:

```console
python scripts/fetch_lvk_analysis_inputs.py
python scripts/fetch_lvk_mixture.py
```

The existing LISA grid can be used directly. To rebuild it, first fetch the
pinned waveform code and then run the asset builder. The waveform dependency
also needs `interpolation==2.2.7`, as recorded in the waveform manifest.

```console
python scripts/fetch_lisa_waveform.py
python scripts/build_lisa_analysis_assets.py
```

The saved figures can be inspected without running anything. Their generation
entry points are listed below; these commands overwrite their corresponding
figure files:

```console
python scripts/plot_analysis_results.py
python scripts/plot_analysis_results.py --lvk-directory results/analysis/lvk_smooth/smooth_k3_uniform_m150_s812301_l0.35 --output figures/lvk_marginals.pdf
python scripts/plot_selected_mixture.py
```

Inference runners and refinements are retained in `scripts/`; their settings
and output directories are recorded in `configs/` and each run's summary.
Reusing the saved products avoids the substantial cost of fitting again.
Some diagnostic scripts require separately downloaded LVK inputs. Legacy
entry points in the shared package may refer to older assets that are outside
this snapshot; use the analysis scripts and configurations above.
Load serialized inputs and checkpoints only from trusted sources.

## File Integrity

`FILE_MANIFEST.json` records the size and SHA-256 digest of each distributed
file. Embedded hashes in saved computational records identify the original
calculation inputs; they are historical provenance, not checksums of the
current code and path metadata. Saved numerical arrays and checkpoint tensor
storage retain their original bytes. The snapshot does not claim a new run.

The authors' code is distributed under the BSD 3-Clause license in `LICENSE`.
External inputs remain subject to their original licenses and attribution
requirements. Please cite the associated paper when using this analysis
and the relevant public releases when using their data or waveform code.
