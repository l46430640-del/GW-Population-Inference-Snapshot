from __future__ import annotations

import csv
import shutil
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyro
import scipy.stats as stats
import torch
from astropy.cosmology import Planck15
from matplotlib.ticker import AutoMinorLocator
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

from .core import BayesianMixtureEstimator_NF_Correlated, BoundedLogitNormal, SafeKumaraswamy, TruncatedNormal, TruncatedPareto, load_trained_model, save_trained_model


MIN_M2_SOLAR_MASS = 3.0
MIN_Q_THRESHOLD = 0.2

ALPHA_M1 = 3.78
M_MIN = 5.0
M_MAX = 112.5
LAMBDA_PEAK = 0.04
MU_M = 34.0
SIGMA_M = 3.6
DELTA_M = 4.8
BETA_Q = 0.81
KAPPA = 2.9

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "data": "#333333",
    "ci_90": "#5DA9E9",
    "ci_50": "#2E86AB",
    "reference": "#D1495B",
}

PLOT_BOUNDS = {"m1": (np.log10(3.0), np.log10(300.0)), "z": (0.0, 3.0), "q": (MIN_Q_THRESHOLD, 1.0)}
LINEAR_PLOT_BOUNDS = {"m1": (3.0, 120.0), "z": (0.0, 3.0), "q": (MIN_Q_THRESHOLD, 1.0)}
PLOT_LABELS = [r"$m_1\,[M_\odot]$", r"$z$", r"$q$"]


def _mass_jacobian_logm1_q(m1_source: np.ndarray) -> np.ndarray:
    return np.log(10.0) * np.square(m1_source)


@dataclass
class LVKInjectionSummary:
    n_draw: int
    n_detected_total: int
    n_detected_kept: int
    observation_years: float
    detection_efficiency_all_draws: float
    detection_efficiency_after_cuts: float
    min_q: float
    min_m2_solar_mass: float
    min_m1_solar_mass: float
    max_m1_solar_mass: float
    max_redshift: float
    semianalytic_snr_threshold: float
    o3_ifar_threshold: float
    o3_ifar_keys: tuple[str, ...]


def load_gwtc3_posterior_data(
    pickle_path: str | Path,
    *,
    target_size: int = 5000,
    resample_seed: int | None = None,
) -> list[torch.Tensor]:
    """Load GWTC-3 posterior samples and apply the paper's BBH cuts."""
    pickle_path = Path(pickle_path)
    print(f"Loading GWTC-3 posterior samples from {pickle_path}...")
    print(f"Applying cuts: m2 >= {MIN_M2_SOLAR_MASS} M_sun, q >= {MIN_Q_THRESHOLD}")
    rng = np.random.default_rng(resample_seed) if resample_seed is not None else None
    with open(pickle_path, "rb") as handle:
        events_data_list = pickle.load(handle)[0]

    formatted_data_list: list[torch.Tensor] = []
    kept_event_count = 0
    for df in events_data_list:
        m1 = df["m1"].values.copy()
        m2 = df["m2"].values.copy()
        z = df["z"].values.copy()
        prior = df["prior"].values.copy()

        mask_swap = m1 < m2
        if np.any(mask_swap):
            m1[mask_swap], m2[mask_swap] = m2[mask_swap], m1[mask_swap].copy()

        q = np.clip(m2 / m1, 1e-6, 1.0)
        valid_mask = (m2 >= MIN_M2_SOLAR_MASS) & (q >= MIN_Q_THRESHOLD)
        if not np.any(valid_mask):
            continue

        m1 = m1[valid_mask]
        z = z[valid_mask]
        q = q[valid_mask]
        prior = prior[valid_mask]

        current_size = len(m1)
        if current_size < target_size:
            if rng is None:
                indices = np.random.choice(current_size, target_size, replace=True)
            else:
                indices = rng.choice(current_size, target_size, replace=True)
            m1 = m1[indices]
            z = z[indices]
            q = q[indices]
            prior = prior[indices]
        elif current_size > target_size:
            m1 = m1[:target_size]
            z = z[:target_size]
            q = q[:target_size]
            prior = prior[:target_size]

        transformed_prior = np.clip(prior * _mass_jacobian_logm1_q(m1), 1e-300, None)
        event_samples = np.stack([np.log10(m1), z, q, transformed_prior], axis=1)
        formatted_data_list.append(torch.tensor(event_samples, dtype=torch.float32))
        kept_event_count += 1

    print(f"Posterior cleaning complete: kept {kept_event_count} / {len(events_data_list)} BBH events.")
    if formatted_data_list:
        print(f"Per-event sample shape: {tuple(formatted_data_list[0].shape)}")
    return formatted_data_list


def _coerce_optional_target_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _posterior_field(samples: np.ndarray, *names: str) -> np.ndarray | None:
    fields = samples.dtype.names or ()
    for name in names:
        if name in fields:
            return np.asarray(samples[name], dtype=float)
    return None


def _valid_lvk_bbh_mask(
    m1: np.ndarray,
    m2: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    log_prior: np.ndarray | None,
    *,
    min_m2_solar_mass: float = MIN_M2_SOLAR_MASS,
    min_q: float = MIN_Q_THRESHOLD,
    min_m1_solar_mass: float = 3.0,
    max_m1_solar_mass: float = 300.0,
    max_redshift: float = 3.0,
) -> np.ndarray:
    mask = (
        np.isfinite(m1)
        & np.isfinite(m2)
        & np.isfinite(q)
        & np.isfinite(z)
        & (m2 >= min_m2_solar_mass)
        & (q >= min_q)
        & (q <= 1.0)
        & (z > 0.0)
        & (z < max_redshift)
        & (m1 > min_m1_solar_mass)
        & (m1 < max_m1_solar_mass)
    )
    if log_prior is not None:
        mask = mask & np.isfinite(log_prior)
    return mask


def _extract_gwosc_group_arrays(samples: np.ndarray) -> dict[str, np.ndarray] | None:
    m1 = _posterior_field(samples, "mass_1_source")
    m2 = _posterior_field(samples, "mass_2_source")
    z = _posterior_field(samples, "redshift")
    q = _posterior_field(samples, "mass_ratio")
    if m1 is None or z is None:
        return None
    if q is None:
        inverted_q = _posterior_field(samples, "inverted_mass_ratio")
        if inverted_q is not None:
            q = 1.0 / np.clip(inverted_q, 1e-12, None)
    if m2 is None and q is not None:
        m2 = m1 * q
    if m2 is None or q is None:
        return None
    q = np.where(q > 1.0, 1.0 / np.clip(q, 1e-12, None), q)
    return {"m1": m1, "m2": m2, "q": q, "z": z, "log_prior": _posterior_field(samples, "log_prior")}


def _score_gwosc_group(
    handle: h5py.File,
    group_name: str,
    *,
    require_log_prior: bool,
) -> tuple[int, bool]:
    key = f"{group_name}/posterior_samples"
    if key not in handle:
        return -1, False
    samples = handle[key][()]
    arrays = _extract_gwosc_group_arrays(samples)
    if arrays is None:
        return -1, False
    has_log_prior = arrays["log_prior"] is not None
    if require_log_prior and not has_log_prior:
        return -1, False
    mask = _valid_lvk_bbh_mask(arrays["m1"], arrays["m2"], arrays["q"], arrays["z"], arrays["log_prior"])
    return int(np.sum(mask)), has_log_prior


def _choose_gwosc_posterior_group(
    handle: h5py.File,
    *,
    preferred_group: str | None,
    require_log_prior: bool,
) -> tuple[str | None, int, bool]:
    candidates = [
        name
        for name in handle.keys()
        if name not in {"history", "version"} and f"{name}/posterior_samples" in handle
    ]
    if preferred_group:
        if preferred_group not in candidates:
            return None, 0, False
        n_valid, has_log_prior = _score_gwosc_group(handle, preferred_group, require_log_prior=require_log_prior)
        return (preferred_group, n_valid, has_log_prior) if n_valid > 0 else (None, 0, has_log_prior)
    scored = []
    for name in candidates:
        n_valid, has_log_prior = _score_gwosc_group(handle, name, require_log_prior=require_log_prior)
        if n_valid > 0:
            # Keep the mixed group as a secondary preference only when it has
            # the required prior information; otherwise use the largest valid
            # single-waveform posterior group.
            mixed_bonus = 1 if name.endswith("Mixed") else 0
            scored.append((n_valid, mixed_bonus, name, has_log_prior))
    if not scored:
        return None, 0, False
    n_valid, _, name, has_log_prior = max(scored, key=lambda row: (row[0], row[1]))
    return name, n_valid, has_log_prior


def load_gwosc_gwtc3_pe_data(
    manifest_path: str | Path,
    data_dir: str | Path,
    *,
    subset: str = "confident",
    posterior_group: str | None = None,
    require_log_prior: bool = True,
    allow_unity_prior_if_missing: bool = False,
    target_size: int | None = None,
    resample_seed: int | None = None,
    min_samples_per_event: int = 1,
) -> list[torch.Tensor]:
    """Load GWOSC per-event GWTC-3-era PE files into the LVK event-tensor format.

    This path is intended for the newly downloaded GWOSC PE HDF5 files. The
    default keeps the current event-prior correction semantics by requiring a
    posterior group with ``log_prior``. Falling back to a unity prior is
    explicit because it changes the likelihood reweighting path.
    """
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    subset_key = "subset_confident" if subset == "confident" else "subset_confident_plus_marginal"
    selected = [
        row
        for row in rows
        if bool(row.get(subset_key)) and row.get("download_filename")
    ]
    selected = sorted(selected, key=lambda row: (float(row.get("gps") or 0.0), str(row.get("name") or "")))
    rng = np.random.default_rng(resample_seed) if resample_seed is not None else None
    formatted_data_list: list[torch.Tensor] = []
    skipped: list[dict[str, Any]] = []
    for row in selected:
        path = data_dir / str(row["download_filename"])
        if not path.exists():
            skipped.append({"event": row.get("name"), "reason": "missing_file", "file": str(path)})
            continue
        with h5py.File(path, "r") as handle:
            group, _, has_log_prior = _choose_gwosc_posterior_group(
                handle,
                preferred_group=posterior_group,
                require_log_prior=require_log_prior,
            )
            if group is None and allow_unity_prior_if_missing:
                group, _, has_log_prior = _choose_gwosc_posterior_group(
                    handle,
                    preferred_group=posterior_group,
                    require_log_prior=False,
                )
            if group is None:
                skipped.append({"event": row.get("name"), "reason": "no_valid_group", "file": str(path)})
                continue
            samples = handle[f"{group}/posterior_samples"][()]
        arrays = _extract_gwosc_group_arrays(samples)
        if arrays is None:
            skipped.append({"event": row.get("name"), "reason": "missing_required_fields", "file": str(path), "group": group})
            continue
        log_prior = arrays["log_prior"]
        if log_prior is None:
            if not allow_unity_prior_if_missing:
                skipped.append({"event": row.get("name"), "reason": "missing_log_prior", "file": str(path), "group": group})
                continue
            log_prior = np.zeros_like(arrays["m1"])
        mask = _valid_lvk_bbh_mask(arrays["m1"], arrays["m2"], arrays["q"], arrays["z"], log_prior)
        if int(np.sum(mask)) < int(min_samples_per_event):
            skipped.append(
                {
                    "event": row.get("name"),
                    "reason": "insufficient_valid_samples",
                    "file": str(path),
                    "group": group,
                    "valid_samples": int(np.sum(mask)),
                }
            )
            continue
        m1 = arrays["m1"][mask]
        z = arrays["z"][mask]
        q = arrays["q"][mask]
        lp = log_prior[mask]
        current_size = len(m1)
        if target_size is not None:
            if current_size < target_size:
                if rng is None:
                    indices = np.random.choice(current_size, target_size, replace=True)
                else:
                    indices = rng.choice(current_size, target_size, replace=True)
                m1 = m1[indices]
                z = z[indices]
                q = q[indices]
                lp = lp[indices]
            elif current_size > target_size:
                m1 = m1[:target_size]
                z = z[:target_size]
                q = q[:target_size]
                lp = lp[:target_size]
        # A per-event constant in the PE prior does not affect the population
        # likelihood; centering keeps the tensor finite across heterogeneous
        # PESummary files.
        centered_prior = np.exp(lp - np.nanmedian(lp))
        transformed_prior = np.clip(centered_prior * _mass_jacobian_logm1_q(m1), 1e-300, None)
        event_samples = np.stack([np.log10(m1), z, q, transformed_prior], axis=1)
        formatted_data_list.append(torch.tensor(event_samples, dtype=torch.float32))
    print(
        "GWOSC PE posterior loading complete:",
        f"kept {len(formatted_data_list)} / {len(selected)} events,",
        f"skipped {len(skipped)}.",
    )
    if skipped:
        print("Skipped events:", ", ".join(str(item["event"]) for item in skipped[:20]))
    if formatted_data_list:
        sizes = [int(tensor.shape[0]) for tensor in formatted_data_list]
        print(f"Per-event sample count: min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}")
    return formatted_data_list


def load_lvk_posterior_events_from_config(config: dict[str, Any]) -> list[torch.Tensor]:
    source = str(config.get("posterior_source", "pickle")).lower()
    target_size = _coerce_optional_target_size(config.get("posterior_target_size", 5000))
    if source in {"gwosc_pe", "gwosc_gwtc3_pe", "gwosc_manifest"}:
        return load_gwosc_gwtc3_pe_data(
            config["posterior_gwosc_pe_manifest"],
            config["posterior_gwosc_pe_dir"],
            subset=str(config.get("posterior_gwosc_subset", "confident")),
            posterior_group=config.get("posterior_gwosc_group"),
            require_log_prior=bool(config.get("posterior_gwosc_require_log_prior", True)),
            allow_unity_prior_if_missing=bool(config.get("posterior_gwosc_allow_unity_prior_if_missing", False)),
            target_size=target_size,
            resample_seed=config.get("posterior_resample_seed"),
            min_samples_per_event=int(config.get("posterior_min_samples_per_event", 1)),
        )
    if target_size is None:
        raise ValueError("The legacy pickle loader requires an integer posterior_target_size.")
    return load_gwtc3_posterior_data(
        config["posterior_pickle"],
        target_size=target_size,
        resample_seed=config.get("posterior_resample_seed"),
    )


def load_lvk_injection_data(
    hdf5_path: str | Path,
    *,
    min_m2_solar_mass: float = MIN_M2_SOLAR_MASS,
    min_q: float = MIN_Q_THRESHOLD,
    min_m1_solar_mass: float = 3.0,
    max_m1_solar_mass: float = 300.0,
    max_redshift: float = 3.0,
    semianalytic_snr_threshold: float = 10.0,
    o3_ifar_threshold: float = 1.0,
) -> dict[str, Any]:
    """Load the official O1+O2+O3 injection file and apply the BBH cuts.

    The GWTC-3 sensitivity release specifies SNR thresholding for O1/O2
    semianalytic injections and FAR/IFAR thresholding for O3 real-search
    injections.  We therefore OR over all available O3 IFAR search columns,
    matching the public gwpopulation_pipe loader semantics.
    """
    hdf5_path = Path(hdf5_path)
    print(f"Loading LVK injections from {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as handle:
        analysis_time_s = float(handle.attrs["analysis_time_s"])
        total_generated = int(handle.attrs["total_generated"])
        injections = handle["injections"]
        m1 = injections["mass1_source"][()]
        m2 = injections["mass2_source"][()]
        z = injections["redshift"][()]
        sampling_pdf = injections["sampling_pdf"][()]
        optimal_snr_net = injections["optimal_snr_net"][()]
        names = injections["name"][()].astype(str)
        ifar_keys = tuple(sorted(key for key in injections.keys() if "ifar" in key.lower()))
        if not ifar_keys:
            raise KeyError("No IFAR columns found in the O1+O2+O3 injection file.")
        ifar_arrays = [injections[key][()] for key in ifar_keys]

    q = np.clip(m2 / m1, 1e-6, 1.0)
    detected_mask = np.zeros_like(z, dtype=bool)
    o3_mask = np.char.find(names.astype(str), "o3") >= 0
    o3_detected = np.zeros_like(z, dtype=bool)
    for ifar_values in ifar_arrays:
        o3_detected |= ifar_values > o3_ifar_threshold
    detected_mask[o3_mask] = o3_detected[o3_mask]
    detected_mask[~o3_mask] = optimal_snr_net[~o3_mask] > semianalytic_snr_threshold

    physical_mask = (
        (m2 >= min_m2_solar_mass)
        & (q >= min_q)
        & (q <= 1.0)
        & (z > 0.0)
        & (z < max_redshift)
        & (m1 > min_m1_solar_mass)
        & (m1 < max_m1_solar_mass)
    )
    kept_mask = detected_mask & physical_mask
    summary = LVKInjectionSummary(
        n_draw=total_generated,
        n_detected_total=int(np.sum(detected_mask)),
        n_detected_kept=int(np.sum(kept_mask)),
        observation_years=analysis_time_s / (365.25 * 24.0 * 3600.0),
        detection_efficiency_all_draws=float(np.sum(detected_mask) / total_generated),
        detection_efficiency_after_cuts=float(np.sum(kept_mask) / total_generated),
        min_q=min_q,
        min_m2_solar_mass=min_m2_solar_mass,
        min_m1_solar_mass=min_m1_solar_mass,
        max_m1_solar_mass=max_m1_solar_mass,
        max_redshift=max_redshift,
        semianalytic_snr_threshold=semianalytic_snr_threshold,
        o3_ifar_threshold=o3_ifar_threshold,
        o3_ifar_keys=ifar_keys,
    )
    print(
        "LVK injection summary:",
        f"Ndraw={summary.n_draw:,},",
        f"detected={summary.n_detected_total:,},",
        f"kept={summary.n_detected_kept:,},",
        f"Tobs={summary.observation_years:.3f} yr",
    )
    transformed_sampling_pdf = np.clip(sampling_pdf[kept_mask] * _mass_jacobian_logm1_q(m1[kept_mask]), 1e-300, None)
    return {
        "log_m1": torch.tensor(np.log10(m1[kept_mask]), dtype=torch.float32),
        "z": torch.tensor(z[kept_mask], dtype=torch.float32),
        "q": torch.tensor(q[kept_mask], dtype=torch.float32),
        "log_p_draw": torch.tensor(np.log(transformed_sampling_pdf), dtype=torch.float32),
        "Ndraw": total_generated,
        "N_det": int(np.sum(kept_mask)),
        "Tobs": summary.observation_years,
        "det_eff": summary.detection_efficiency_after_cuts,
        "log_det_eff": float(np.log(max(summary.detection_efficiency_after_cuts, 1e-12))),
        "summary": asdict(summary),
        "draw_density_space": "log10_m1_q_z",
    }


def set_publication_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
        }
    )


class LVKPowerLawPeak:
    """Reference GWTC-3 Power Law + Peak model."""

    def __init__(
        self,
        alpha: float = ALPHA_M1,
        m_min: float = M_MIN,
        m_max: float = M_MAX,
        lambda_peak: float = LAMBDA_PEAK,
        mu_m: float = MU_M,
        sigma_m: float = SIGMA_M,
        delta_m: float = DELTA_M,
        beta_q: float = BETA_Q,
        kappa: float = KAPPA,
        z_max: float = 3.0,
    ):
        self.alpha = alpha
        self.m_min = m_min
        self.m_max = m_max
        self.lambda_peak = lambda_peak
        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.delta_m = delta_m
        self.beta_q = beta_q
        self.kappa = kappa
        self.z_max = z_max

    def _smoothing_function(self, m: np.ndarray, m_min: float, delta_m: float) -> np.ndarray:
        if delta_m == 0:
            return np.where(m >= m_min, 1.0, 0.0)
        x = np.clip((m - m_min) / delta_m, -50, 50)
        return np.where(m < m_min, 0.0, np.where(m >= m_min + delta_m, 1.0, 1.0 / (1.0 + np.exp(-4.0 * x + 2.0))))

    def _power_law_pdf(self, m: np.ndarray) -> np.ndarray:
        if self.alpha == 1.0:
            norm = np.log(self.m_max / self.m_min)
        else:
            norm = (self.m_max ** (1 - self.alpha) - self.m_min ** (1 - self.alpha)) / (1 - self.alpha)
        return np.where((m >= self.m_min) & (m <= self.m_max), m ** (-self.alpha) / norm, 0.0)

    def _gaussian_pdf(self, m: np.ndarray) -> np.ndarray:
        return stats.norm.pdf(m, loc=self.mu_m, scale=self.sigma_m)

    def pdf_m1_unnorm(self, m: np.ndarray) -> np.ndarray:
        smoothing = self._smoothing_function(m, self.m_min, self.delta_m)
        return smoothing * ((1 - self.lambda_peak) * self._power_law_pdf(m) + self.lambda_peak * self._gaussian_pdf(m))

    def pdf_m1(self, m: np.ndarray) -> np.ndarray:
        m_grid = np.linspace(self.m_min, self.m_max, 10000)
        norm = np.trapz(self.pdf_m1_unnorm(m_grid), m_grid)
        return self.pdf_m1_unnorm(m) / (norm + 1e-12)

    def pdf_q(self, q: np.ndarray, m1: float | None = None) -> np.ndarray:
        q_min = MIN_Q_THRESHOLD if m1 is None else max(self.m_min / m1, MIN_Q_THRESHOLD)
        mask = (q >= q_min) & (q <= 1.0)
        if self.beta_q == -1.0:
            norm = np.log(1.0 / q_min)
        else:
            norm = (1.0 ** (self.beta_q + 1) - q_min ** (self.beta_q + 1)) / (self.beta_q + 1)
        return np.where(mask, q ** self.beta_q / norm, 0.0)

    def pdf_z(self, z: np.ndarray) -> np.ndarray:
        dvc_dz = Planck15.differential_comoving_volume(z).value
        pdf_unnorm = dvc_dz * (1 + z) ** (self.kappa - 1)
        z_grid = np.linspace(0.001, self.z_max, 4000)
        norm = np.trapz(Planck15.differential_comoving_volume(z_grid).value * (1 + z_grid) ** (self.kappa - 1), z_grid)
        return np.where((z > 0.0) & (z <= self.z_max), pdf_unnorm / norm, 0.0)

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        batch_size = max(5000, n_samples * 5)

        m1_grid = np.linspace(self.m_min, self.m_max, 1000)
        m1_max = self.pdf_m1(m1_grid).max() * 1.1
        m1_samples: list[float] = []
        while len(m1_samples) < n_samples:
            proposal = np.random.uniform(self.m_min, self.m_max, batch_size)
            accepted = np.random.uniform(0, m1_max, batch_size) < self.pdf_m1(proposal)
            m1_samples.extend(proposal[accepted])
        m1_samples = np.asarray(m1_samples[:n_samples])

        z_grid = np.linspace(0.01, self.z_max, 1000)
        z_max = self.pdf_z(z_grid).max() * 1.1
        z_samples: list[float] = []
        while len(z_samples) < n_samples:
            proposal = np.random.uniform(0.01, self.z_max, batch_size)
            accepted = np.random.uniform(0, z_max, batch_size) < self.pdf_z(proposal)
            z_samples.extend(proposal[accepted])
        z_samples = np.asarray(z_samples[:n_samples])

        q_samples = []
        for m1 in m1_samples:
            q_min_m1 = max(self.m_min / m1, MIN_Q_THRESHOLD)
            u = np.random.uniform(0, 1)
            if self.beta_q == -1.0:
                q = q_min_m1 * np.exp(u * np.log(1.0 / q_min_m1))
            else:
                q = (q_min_m1 ** (self.beta_q + 1) + u * (1.0 - q_min_m1 ** (self.beta_q + 1))) ** (1.0 / (self.beta_q + 1))
            q_samples.append(q)
        return np.stack([m1_samples, z_samples, np.asarray(q_samples)], axis=1)


def calculate_js_divergence_1d(samples_p: np.ndarray, samples_q: np.ndarray, x_min: float, x_max: float, n_grid: int = 500) -> float:
    samples_p = samples_p[(samples_p >= x_min) & (samples_p <= x_max)]
    samples_q = samples_q[(samples_q >= x_min) & (samples_q <= x_max)]
    if len(samples_p) < 100 or len(samples_q) < 100:
        return float("nan")
    x_grid = np.linspace(x_min, x_max, n_grid)
    kde_p = stats.gaussian_kde(samples_p)
    kde_q = stats.gaussian_kde(samples_q)
    pdf_p = np.clip(kde_p(x_grid), 1e-12, None)
    pdf_q = np.clip(kde_q(x_grid), 1e-12, None)
    pdf_p /= np.trapz(pdf_p, x_grid)
    pdf_q /= np.trapz(pdf_q, x_grid)
    return float(jensenshannon(pdf_p, pdf_q) ** 2)


def calculate_js_divergence(bnn_samples: np.ndarray, lvk_samples: np.ndarray, bounds: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    bounds = bounds or LINEAR_PLOT_BOUNDS
    return {
        "m1": calculate_js_divergence_1d(bnn_samples[:, 0], lvk_samples[:, 0], bounds["m1"][0], bounds["m1"][1]),
        "z": calculate_js_divergence_1d(bnn_samples[:, 1], lvk_samples[:, 1], bounds["z"][0], bounds["z"][1]),
        "q": calculate_js_divergence_1d(bnn_samples[:, 2], lvk_samples[:, 2], bounds["q"][0], bounds["q"][1]),
    }


def run_comparison(
    bnn_samples: np.ndarray,
    n_lvk_samples: int | None = None,
    output_table: bool = True,
    plot: bool = True,
    save_path: str | Path | None = None,
) -> dict[str, float]:
    if bnn_samples[:, 0].max() < 10:
        bnn_samples = bnn_samples.copy()
        bnn_samples[:, 0] = 10 ** bnn_samples[:, 0]
    n_lvk_samples = n_lvk_samples or len(bnn_samples)
    lvk_samples = LVKPowerLawPeak().sample(n_lvk_samples, seed=42)
    results = calculate_js_divergence(bnn_samples, lvk_samples)
    if output_table:
        print("JS divergence against the LVK Power Law + Peak reference:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for idx, label in enumerate(PLOT_LABELS):
            axes[idx].hist(bnn_samples[:, idx], bins=50, density=True, alpha=0.5, color=COLORS["primary"], label="This work")
            axes[idx].hist(lvk_samples[:, idx], bins=50, density=True, alpha=0.5, color=COLORS["reference"], label="Reference")
            axes[idx].set_xlabel(label)
            axes[idx].set_ylabel(r"$p(x)$")
            axes[idx].legend(fontsize=9)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return results


def get_reflected_data_1d(data: np.ndarray, limits: tuple[float | None, float | None]) -> np.ndarray:
    low, high = limits
    data_aug = data.copy()
    if low is not None:
        data_aug = np.concatenate([data_aug, 2 * low - data])
    if high is not None:
        data_aug = np.concatenate([data_aug, 2 * high - data])
    return data_aug


def kde_1d_reflected_scipy(
    data: np.ndarray,
    limits: tuple[float | None, float | None],
    x_grid: np.ndarray,
    bandwidth_factor: float = 0.33,
) -> np.ndarray:
    data_aug = get_reflected_data_1d(data, limits)
    kde = gaussian_kde(data_aug)
    kde.set_bandwidth(kde.factor * bandwidth_factor)
    factor = 1.0 + float(limits[0] is not None) + float(limits[1] is not None)
    return kde(x_grid) * factor


def kde_2d_reflected_scipy(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_limits: tuple[float | None, float | None],
    y_limits: tuple[float | None, float | None],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth_factor: float = 0.33,
) -> np.ndarray:
    x_parts = [x_data]
    y_parts = [y_data]
    if x_limits[0] is not None:
        x_parts.append(2 * x_limits[0] - x_data)
        y_parts.append(y_data)
    if x_limits[1] is not None:
        x_parts.append(2 * x_limits[1] - x_data)
        y_parts.append(y_data)
    if y_limits[0] is not None:
        x_parts.append(x_data)
        y_parts.append(2 * y_limits[0] - y_data)
    if y_limits[1] is not None:
        x_parts.append(x_data)
        y_parts.append(2 * y_limits[1] - y_data)
    values = np.vstack([np.concatenate(x_parts), np.concatenate(y_parts)])
    kernel = gaussian_kde(values)
    kernel.set_bandwidth(kernel.factor * bandwidth_factor)
    return np.reshape(kernel(np.vstack([x_grid.ravel(), y_grid.ravel()])).T, x_grid.shape)


def _convert_samples_to_linear_m1(samples: np.ndarray) -> np.ndarray:
    converted = samples.copy()
    if np.nanmax(converted[:, 0]) < 10.0:
        converted[:, 0] = 10 ** converted[:, 0]
    return converted


def sample_posterior_predictive(estimator: Any, num_draws: int = 200, samples_per_draw: int = 5000) -> list[np.ndarray]:
    samples_list: list[np.ndarray] = []
    with torch.no_grad():
        for _ in tqdm(range(num_draws), desc="Posterior predictive draws"):
            samples_list.append(_convert_samples_to_linear_m1(estimator.sample(samples_per_draw).cpu().numpy()))
    return samples_list


def _compute_log_pdf_phys_for_params(estimator: Any, params: dict[str, torch.Tensor], points_phys: torch.Tensor) -> torch.Tensor:
    points_norm = (points_phys - estimator.min_vals) / (estimator.max_vals - estimator.min_vals)
    points_norm = points_norm.clamp(min=1e-6, max=1.0 - 1e-6).unsqueeze(0)
    log_parts = []
    if estimator.num_corr_kumaraswamy > 0:
        a = torch.exp(params["log_k_a"]) + 0.0001
        b = torch.exp(params["log_k_b"]) + 0.0001
        L = estimator._build_cholesky_from_unconstrained(params["k_cholesky_params"], estimator.data_dim, estimator.device)
        log_parts.append(estimator._copula_log_prob(SafeKumaraswamy(a, b), L, points_norm))
    if estimator.num_corr_tnormal > 0:
        locs = params["tn_locs"]
        scales = torch.exp(params["log_tn_scales"]) + 0.0001
        L = estimator._build_cholesky_from_unconstrained(params["tnormal_cholesky_params"], estimator.data_dim, estimator.device)
        log_parts.append(estimator._copula_log_prob(TruncatedNormal(locs, scales, low=0.0001, high=1.0 - 0.0001), L, points_norm))
    if estimator.num_corr_tpareto > 0:
        alphas = torch.exp(params["log_tp_alphas"]) + 0.0001
        L = estimator._build_cholesky_from_unconstrained(params["tpareto_cholesky_params"], estimator.data_dim, estimator.device)
        log_parts.append(estimator._copula_log_prob(TruncatedPareto(alphas, low=0.0001, high=1.0), L, points_norm))
    if getattr(estimator, "num_corr_logitnormal", 0) > 0:
        locs = params["ln_locs"]
        scales = torch.exp(params["log_ln_scales"]) + 0.0001
        L = estimator._build_cholesky_from_unconstrained(params["logitnormal_cholesky_params"], estimator.data_dim, estimator.device)
        log_parts.append(estimator._copula_log_prob(BoundedLogitNormal(locs, scales), L, points_norm))
    if getattr(estimator, "num_corr_axis_lntnormal", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["axis_lntn_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._axis_lntnormal_marginal_dist(
            params["axis_lntn_ln_locs"],
            params["axis_lntn_log_ln_scales"],
            params["axis_lntn_tn_locs"],
            params["axis_lntn_log_tn_scales"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_axis_lnkumaraswamy", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["axis_lnk_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._axis_lnkumaraswamy_marginal_dist(
            params["axis_lnk_ln_locs"],
            params["axis_lnk_log_ln_scales"],
            params["axis_lnk_log_k_a"],
            params["axis_lnk_log_k_b"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_axis_bernstein_tnormal", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["axis_btn_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._axis_bernstein_tnormal_marginal_dist(
            params["axis_btn_m1_logits"],
            params["axis_btn_tn_locs"],
            params["axis_btn_log_tn_scales"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_axis_bernstein_kumaraswamy", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["axis_bk_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._axis_bernstein_kumaraswamy_marginal_dist(
            params["axis_bk_m1_logits"],
            params["axis_bk_log_k_a"],
            params["axis_bk_log_k_b"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_axis_spline_tnormal", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["axis_stn_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._axis_spline_tnormal_marginal_dist(
            params["axis_stn_m1_logits"],
            params["axis_stn_tn_locs"],
            params["axis_stn_log_tn_scales"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_axis_spline_kumaraswamy", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["axis_sk_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._axis_spline_kumaraswamy_marginal_dist(
            params["axis_sk_m1_logits"],
            params["axis_sk_log_k_a"],
            params["axis_sk_log_k_b"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_decoupled_spline_tnormal_copula", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["dec_stn_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._decoupled_spline_tnormal_marginal_dist(
            params["dec_stn_m1_logits"],
            params["dec_stn_tn_locs"],
            params["dec_stn_log_tn_scales"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    if getattr(estimator, "num_corr_decoupled_spline_kumaraswamy_copula", 0) > 0:
        L = estimator._build_cholesky_from_unconstrained(params["dec_sk_cholesky_params"], estimator.data_dim, estimator.device)
        marginal_dist = estimator._decoupled_spline_kumaraswamy_marginal_dist(
            params["dec_sk_m1_logits"],
            params["dec_sk_log_k_a"],
            params["dec_sk_log_k_b"],
        )
        log_parts.append(estimator._copula_log_prob(marginal_dist, L, points_norm))
    log_probs_all = torch.cat(log_parts, dim=2).squeeze(0)
    log_weights = torch.log_softmax(params["weights_logits"], dim=-1)
    log_vol = torch.sum(torch.log(estimator.max_vals - estimator.min_vals))
    return torch.logsumexp(log_weights + log_probs_all, dim=1) - log_vol


def _compute_selection_fraction_from_params(estimator: Any, params: dict[str, torch.Tensor], injection_data: dict[str, Any]) -> dict[str, Any]:
    inj_points = torch.stack([injection_data["log_m1"], injection_data["z"], injection_data["q"]], dim=1).to(estimator.device)
    inj_log_p_draw = injection_data["log_p_draw"].to(estimator.device)
    log_pdf_phys = _compute_log_pdf_phys_for_params(estimator, params, inj_points)
    log_weights = log_pdf_phys - inj_log_p_draw
    log_selection_fraction = torch.logsumexp(log_weights, dim=0) - np.log(injection_data["Ndraw"])
    weights = torch.exp(log_weights - torch.max(log_weights))
    neff = float((torch.sum(weights) ** 2 / (torch.sum(weights ** 2) + 1e-12)).item())
    return {
        "selection_fraction": float(torch.exp(log_selection_fraction).item()),
        "log_selection_fraction": float(log_selection_fraction.item()),
        "log_weight_mean": float(log_weights.mean().item()),
        "log_weight_std": float(log_weights.std().item()),
        "log_weight_min": float(log_weights.min().item()),
        "log_weight_max": float(log_weights.max().item()),
        "neff": neff,
        "log_pdf_minus_log_draw_quantiles": [float(x) for x in torch.quantile(log_weights, torch.tensor([0.05, 0.5, 0.95], device=log_weights.device)).cpu().tolist()],
    }


def run_lvk_selection_diagnostics(
    estimator: Any,
    injection_data: dict[str, Any],
    output_dir: str | Path,
    num_posterior_draws: int = 32,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    with torch.no_grad():
        latent_vectors = estimator.flow_dist.sample(torch.Size([num_posterior_draws]))
        for draw_id, latent in enumerate(latent_vectors):
            params = estimator._unflatten_latent_vector(latent)
            stats_dict = _compute_selection_fraction_from_params(estimator, params, injection_data)
            stats_dict["draw_id"] = draw_id
            diagnostics.append(stats_dict)
    selection_fractions = np.array([row["selection_fraction"] for row in diagnostics], dtype=float)
    aggregate = {
        "num_draws": num_posterior_draws,
        "selection_fraction_quantiles": [float(x) for x in np.quantile(selection_fractions, [0.05, 0.5, 0.95])],
        "selection_fraction_mean": float(np.mean(selection_fractions)),
        "selection_fraction_max": float(np.max(selection_fractions)),
        "selection_fraction_min": float(np.min(selection_fractions)),
        "posterior_mean_selection_fraction": getattr(estimator, "posterior_mean_selection_fraction", None),
    }
    with open(output_dir / "selection_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump({"aggregate": aggregate, "draws": diagnostics}, handle, indent=2)
    with open(output_dir / "selection_diagnostics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "draw_id",
                "selection_fraction",
                "log_selection_fraction",
                "log_weight_mean",
                "log_weight_std",
                "log_weight_min",
                "log_weight_max",
                "neff",
                "log_pdf_minus_log_draw_quantiles",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostics)
    lines = [
        "# LVK Selection Diagnostics",
        "",
        f"- posterior_mean_selection_fraction: {aggregate['posterior_mean_selection_fraction']}",
        f"- draw-level selection_fraction quantiles (5/50/95%): {aggregate['selection_fraction_quantiles']}",
        f"- draw-level selection_fraction min/max: {aggregate['selection_fraction_min']} / {aggregate['selection_fraction_max']}",
    ]
    (output_dir / "selection_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    return {"aggregate": aggregate, "draws": diagnostics}


def plot_publication_marginals(samples_list: list[np.ndarray], save_path: str | Path = "lvk_marginal.pdf") -> None:
    set_publication_style()
    reference_model = LVKPowerLawPeak()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bounds_list = [LINEAR_PLOT_BOUNDS["m1"], LINEAR_PLOT_BOUNDS["z"], LINEAR_PLOT_BOUNDS["q"]]
    for idx, (ax, limits, label) in enumerate(zip(axes, bounds_list, PLOT_LABELS)):
        x_grid = np.linspace(limits[0], limits[1], 250)
        densities = []
        for draw in samples_list:
            densities.append(kde_1d_reflected_scipy(draw[:, idx], limits, x_grid))
        density_stack = np.asarray(densities)
        ax.fill_between(
            x_grid,
            np.percentile(density_stack, 5, axis=0),
            np.percentile(density_stack, 95, axis=0),
            color=COLORS["ci_90"],
            alpha=0.35,
            label="90% band",
        )
        ax.plot(x_grid, np.median(density_stack, axis=0), color=COLORS["primary"], lw=2, label="Posterior median")
        if idx == 0:
            reference = reference_model.pdf_m1(x_grid)
        elif idx == 1:
            reference = reference_model.pdf_z(x_grid)
        else:
            reference = reference_model.pdf_q(x_grid)
        ax.plot(x_grid, reference, color=COLORS["reference"], lw=1.8, ls="--", label="LVK reference")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$p(x)$")
        ax.set_xlim(limits)
        ax.set_ylim(bottom=0.0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_publication_corner(samples: np.ndarray, save_path: str | Path = "lvk_distribution_2d.pdf") -> None:
    set_publication_style()
    reference = LVKPowerLawPeak().sample(len(samples), seed=7)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        (0, 1, LINEAR_PLOT_BOUNDS["m1"], LINEAR_PLOT_BOUNDS["z"]),
        (0, 2, LINEAR_PLOT_BOUNDS["m1"], LINEAR_PLOT_BOUNDS["q"]),
        (1, 2, LINEAR_PLOT_BOUNDS["z"], LINEAR_PLOT_BOUNDS["q"]),
    ]
    for idx, (x_idx, y_idx, x_bounds, y_bounds) in enumerate(pairs):
        ax = axes[idx]
        x_grid = np.linspace(x_bounds[0], x_bounds[1], 120)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], 120)
        xx, yy = np.meshgrid(x_grid, y_grid)
        zz = kde_2d_reflected_scipy(samples[:, x_idx], samples[:, y_idx], x_bounds, y_bounds, xx, yy)
        z_ref = kde_2d_reflected_scipy(reference[:, x_idx], reference[:, y_idx], x_bounds, y_bounds, xx, yy)

        z_flat = np.sort(zz.ravel())[::-1]
        z_cdf = np.cumsum(z_flat)
        levels = []
        for conf in (0.68, 0.95, 0.99):
            levels.append(z_flat[np.searchsorted(z_cdf, conf * z_cdf[-1])])
        levels = sorted(levels)

        ax.contourf(xx, yy, zz, levels=[levels[2], zz.max()], colors=[COLORS["ci_50"]], alpha=0.75)
        ax.contourf(xx, yy, zz, levels=[levels[1], levels[2]], colors=[COLORS["ci_50"]], alpha=0.45)
        ax.contourf(xx, yy, zz, levels=[levels[0], levels[1]], colors=[COLORS["ci_50"]], alpha=0.20)
        ax.contour(xx, yy, zz, levels=levels, colors=[COLORS["primary"]], linewidths=1.2)
        ax.contour(xx, yy, z_ref, levels=3, colors=[COLORS["reference"]], linewidths=1.0, linestyles="--")
        ax.set_xlabel(PLOT_LABELS[x_idx])
        ax.set_ylabel(PLOT_LABELS[y_idx])
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_results(estimator: Any, output_dir: str | Path = ".") -> list[np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_list = sample_posterior_predictive(estimator)
    plot_publication_marginals(samples_list, output_dir / "lvk_marginal_morphology.pdf")
    stacked = np.concatenate(samples_list, axis=0)
    if len(stacked) > 20000:
        stacked = stacked[np.random.choice(len(stacked), 20000, replace=False)]
    np.savez_compressed(output_dir / "posterior_predictive_samples.npz", samples=stacked.astype(np.float32))
    plot_publication_corner(stacked, output_dir / "lvk_2d_morphology.pdf")
    return samples_list


def build_lvk_estimator(config: dict[str, Any]) -> BayesianMixtureEstimator_NF_Correlated:
    estimator_cfg = config["estimator"]
    bounds_cfg = config.get("physical_bounds", {})
    m1_min = float(bounds_cfg.get("m1_min_solar_mass", 3.0))
    m1_max = float(bounds_cfg.get("m1_max_solar_mass", 300.0))
    z_min = float(bounds_cfg.get("z_min", 0.0))
    z_max = float(bounds_cfg.get("z_max", 3.0))
    q_min = float(bounds_cfg.get("q_min", MIN_Q_THRESHOLD))
    q_max = float(bounds_cfg.get("q_max", 1.0))
    return BayesianMixtureEstimator_NF_Correlated(
        data_dim=3,
        num_corr_kumaraswamy=estimator_cfg["num_corr_kumaraswamy"],
        num_corr_tnormal=estimator_cfg["num_corr_tnormal"],
        num_corr_tpareto=estimator_cfg["num_corr_tpareto"],
        hidden_dim_flow=estimator_cfg.get("hidden_dim_flow", 128),
        num_flow_transforms=estimator_cfg.get("num_flow_transforms", 8),
        device=config.get("device"),
        physical_bounds=(
            [np.log10(m1_min), z_min, q_min],
            [np.log10(m1_max), z_max, q_max],
        ),
        num_corr_logitnormal=estimator_cfg.get("num_corr_logitnormal", 0),
        num_corr_axis_lntnormal=estimator_cfg.get("num_corr_axis_lntnormal", 0),
        num_corr_axis_lnkumaraswamy=estimator_cfg.get("num_corr_axis_lnkumaraswamy", 0),
        num_corr_axis_bernstein_tnormal=estimator_cfg.get("num_corr_axis_bernstein_tnormal", 0),
        num_corr_axis_bernstein_kumaraswamy=estimator_cfg.get("num_corr_axis_bernstein_kumaraswamy", 0),
        axis_bernstein_degree=estimator_cfg.get("axis_bernstein_degree", 8),
        num_corr_axis_spline_tnormal=estimator_cfg.get("num_corr_axis_spline_tnormal", 0),
        num_corr_axis_spline_kumaraswamy=estimator_cfg.get("num_corr_axis_spline_kumaraswamy", 0),
        axis_spline_num_bins=estimator_cfg.get("axis_spline_num_bins", 12),
        num_corr_decoupled_spline_tnormal_copula=estimator_cfg.get("num_corr_decoupled_spline_tnormal_copula", 0),
        num_corr_decoupled_spline_kumaraswamy_copula=estimator_cfg.get("num_corr_decoupled_spline_kumaraswamy_copula", 0),
        m1_roughness_lambda=estimator_cfg.get("m1_roughness_lambda", 0.0),
        m1_basis_coordinate=estimator_cfg.get("m1_basis_coordinate", "log10"),
        m1_basis_logit_epsilon=estimator_cfg.get("m1_basis_logit_epsilon", 0.02),
        m1_basis_type=estimator_cfg.get("m1_basis_type", "single"),
        m1_two_resolution_global_bins=estimator_cfg.get("m1_two_resolution_global_bins", 5),
        m1_two_resolution_local_bins=estimator_cfg.get("m1_two_resolution_local_bins", 16),
        m1_two_resolution_local_min_solar_mass=estimator_cfg.get("m1_two_resolution_local_min_solar_mass", 5.0),
        m1_two_resolution_local_max_solar_mass=estimator_cfg.get("m1_two_resolution_local_max_solar_mass", 50.0),
    )


def summarize_posterior_predictive(samples_list: list[np.ndarray], rate_samples: torch.Tensor | None = None) -> dict[str, Any]:
    stacked = np.concatenate(samples_list, axis=0)
    summary = {
        "posterior_predictive_size": int(len(stacked)),
        "m1_median": float(np.median(stacked[:, 0])),
        "m1_ci90": [float(x) for x in np.percentile(stacked[:, 0], [5, 95])],
        "z_median": float(np.median(stacked[:, 1])),
        "z_ci90": [float(x) for x in np.percentile(stacked[:, 1], [5, 95])],
        "q_median": float(np.median(stacked[:, 2])),
        "q_ci90": [float(x) for x in np.percentile(stacked[:, 2], [5, 95])],
    }
    if rate_samples is not None:
        summary["rate_mean_per_year"] = float(rate_samples.mean().item())
        summary["rate_ci90_per_year"] = [float(x) for x in torch.quantile(rate_samples, torch.tensor([0.05, 0.95], device=rate_samples.device)).cpu().tolist()]
    return summary


def _compute_edge_fraction(samples: np.ndarray, lower: float, upper: float, edge_fraction: float = 0.01) -> dict[str, float]:
    width = upper - lower
    low_edge = lower + width * edge_fraction
    high_edge = upper - width * edge_fraction
    return {
        "low": float(np.mean(samples <= low_edge)),
        "high": float(np.mean(samples >= high_edge)),
    }


def evaluate_lvk_quality(summary: dict[str, Any], stacked_samples: np.ndarray) -> dict[str, Any]:
    js_results = calculate_js_divergence(stacked_samples, LVKPowerLawPeak().sample(len(stacked_samples), seed=123))
    edge_fractions = {
        "m1": _compute_edge_fraction(stacked_samples[:, 0], LINEAR_PLOT_BOUNDS["m1"][0], LINEAR_PLOT_BOUNDS["m1"][1]),
        "z": _compute_edge_fraction(stacked_samples[:, 1], LINEAR_PLOT_BOUNDS["z"][0], LINEAR_PLOT_BOUNDS["z"][1]),
        "q": _compute_edge_fraction(stacked_samples[:, 2], LINEAR_PLOT_BOUNDS["q"][0], LINEAR_PLOT_BOUNDS["q"][1]),
    }
    anomalies: list[str] = []
    selection_fraction = summary.get("selection_fraction_mean")
    if selection_fraction is None or not np.isfinite(selection_fraction) or selection_fraction <= 0.0 or selection_fraction > 1.0:
        anomalies.append(f"selection_fraction_mean={selection_fraction} is outside (0, 1]")
    rate_mean = summary.get("rate_mean_per_year")
    if rate_mean is None or not np.isfinite(rate_mean) or rate_mean <= 0.0 or rate_mean > 1.0e3:
        anomalies.append(f"rate_mean_per_year={rate_mean} looks non-physical")
    for param, edges in edge_fractions.items():
        if max(edges["low"], edges["high"]) > 0.50:
            anomalies.append(f"{param} posterior predictive samples pile up at a support boundary")
    if np.isnan(np.mean(list(js_results.values()))) or np.mean(list(js_results.values())) > 0.25:
        anomalies.append(f"average JS divergence to LVK reference is too large: {np.mean(list(js_results.values())):.3f}")
    if max(js_results.values()) > 0.35:
        anomalies.append(f"at least one marginal strongly disagrees with LVK reference: {js_results}")
    return {
        "passed": len(anomalies) == 0,
        "selection_fraction_mean": selection_fraction,
        "js_divergence": js_results,
        "edge_fractions": edge_fractions,
        "anomalies": anomalies,
        "allow_report_replacement": len(anomalies) == 0,
    }


def write_acceptance_summary(output_dir: Path, quality_gate: dict[str, Any], summary: dict[str, Any]) -> None:
    with open(output_dir / "acceptance_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"quality_gate": quality_gate, "summary": summary}, handle, indent=2)
    lines = [
        "# LVK Acceptance Summary",
        "",
        f"- Gate status: {'PASS' if quality_gate['passed'] else 'FAIL'}",
        f"- Allow report replacement: {quality_gate['allow_report_replacement']}",
        f"- Selection fraction mean: {quality_gate['selection_fraction_mean']}",
        f"- Rate mean per year: {summary.get('rate_mean_per_year')}",
        f"- JS divergence: {json.dumps(quality_gate['js_divergence'])}",
        "",
        "## Key anomalies",
    ]
    if quality_gate["anomalies"]:
        lines.extend([f"- {item}" for item in quality_gate["anomalies"]])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Report action",
            "- Replace LVK report figures only if gate status is PASS.",
            "- If FAIL, freeze LVK report replacement and retain conservative LVK text.",
        ]
    )
    (output_dir / "acceptance_summary.md").write_text("\n".join(lines), encoding="utf-8")


def sync_lvk_report_assets(output_dir: Path, quality_gate: dict[str, Any], report_dir: str | Path | None) -> list[str]:
    if not quality_gate["allow_report_replacement"] or report_dir is None:
        return []
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    mapping = {
        output_dir / "lvk_2d_morphology.pdf": report_dir / "lvk_2d_morphology.pdf",
        output_dir / "lvk_marginal_morphology.pdf": report_dir / "lvk_marginal_morphology.pdf",
    }
    for src, dst in mapping.items():
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def write_summary_table(summary: dict[str, Any], output_path: str | Path) -> None:
    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, (list, dict)):
            flat_rows.append({"metric": key, "value": json.dumps(value)})
        else:
            flat_rows.append({"metric": key, "value": value})
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(flat_rows)


def run_lvk_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if "training_seed" in config:
        seed = int(config["training_seed"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
    events = load_lvk_posterior_events_from_config(config)
    injections = load_lvk_injection_data(config["injection_hdf5"])
    estimator = build_lvk_estimator(config)

    training_cfg = config["training"]
    estimator.train(
        events,
        epochs=training_cfg["epochs"],
        lr=training_cfg.get("lr", 1e-3),
        param_names=["log10_m1", "z", "q"],
        correct_selection_effects=True,
        injection_data=injections,
        observation_years=injections["Tobs"],
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
    )
    estimator.infer_rate(
        num_observed_events=len(events),
        num_mcmc_samples=config["rate_inference"].get("num_mcmc_samples", 1500),
        warmup_steps=config["rate_inference"].get("warmup_steps", 500),
    )

    checkpoint_path = output_dir / config.get("checkpoint_name", "lvk_gwtc3_retrained.pt")
    save_trained_model(estimator, checkpoint_path)
    samples_list = plot_results(estimator, output_dir=output_dir)
    stacked_samples = np.concatenate(samples_list, axis=0)
    summary = summarize_posterior_predictive(samples_list, estimator.yearly_rate_posterior_samples)
    summary["n_events"] = len(events)
    summary["checkpoint"] = str(checkpoint_path)
    summary["injection_summary"] = injections["summary"]
    summary["selection_fraction_mean"] = getattr(estimator, "posterior_mean_selection_fraction", None)
    quality_gate = evaluate_lvk_quality(summary, stacked_samples)
    summary["quality_gate"] = quality_gate
    summary["report_assets_synced"] = sync_lvk_report_assets(output_dir, quality_gate, config.get("report_dir"))

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_table(summary, output_dir / "summary.csv")
    write_acceptance_summary(output_dir, quality_gate, summary)
    with open(output_dir / "run.log", "w", encoding="utf-8") as handle:
        handle.write("LVK GWTC-3 run completed.\n")
        handle.write(f"Events kept: {len(events)}\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"Injection file: {config['injection_hdf5']}\n")
        handle.write(f"Selection fraction mean: {summary['selection_fraction_mean']}\n")
        handle.write(f"Gate status: {'PASS' if quality_gate['passed'] else 'FAIL'}\n")
        if quality_gate["anomalies"]:
            handle.write("Anomalies:\n")
            for item in quality_gate["anomalies"]:
                handle.write(f"- {item}\n")
    return summary


__all__ = [
    "MIN_M2_SOLAR_MASS",
    "MIN_Q_THRESHOLD",
    "LVKPowerLawPeak",
    "LVKInjectionSummary",
    "build_lvk_estimator",
    "calculate_js_divergence",
    "calculate_js_divergence_1d",
    "kde_1d_reflected_scipy",
    "kde_2d_reflected_scipy",
    "load_gwtc3_posterior_data",
    "load_gwosc_gwtc3_pe_data",
    "load_lvk_posterior_events_from_config",
    "load_lvk_injection_data",
    "load_trained_model",
    "plot_publication_corner",
    "plot_publication_marginals",
    "plot_results",
    "run_comparison",
    "run_lvk_pipeline",
    "run_lvk_selection_diagnostics",
    "sample_posterior_predictive",
    "sync_lvk_report_assets",
    "save_trained_model",
]
