"""Build an IMRPhenomD LISA SNR grid and verify quadrature/interpolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'external_data/literature/python_deps'))
sys.path.insert(0, str(ROOT / 'external_data/literature/PyIMRPhenomD'))
from gwpopulation.lisa import cosmo, lisa_psd_robson_et_al_2018, calculate_snr_for_event_IMR

LOW = np.array([5., .21, .1])
HIGH = np.array([9., 7.5, 1.])
constants = None
IMRPhenDAmplitude = None


def load_waveform():
    global constants, IMRPhenDAmplitude
    if constants is None:
        import IMRPhenomD_const
        from IMRPhenomD_internals import IMRPhenDAmplitude as amplitude_function
        constants = IMRPhenomD_const
        IMRPhenDAmplitude = amplitude_function


def amplitude(frequency, m1_detector, m2_detector, distance_m):
    load_waveform()
    mass = m1_detector + m2_detector
    eta = m1_detector * m2_detector / mass**2
    prefactor = (2 * np.sqrt(5 / (64 * np.pi)) * mass**2
                 * constants.MRSUN_SI * constants.MTSUN_SI / distance_m)
    return IMRPhenDAmplitude(frequency * mass * constants.MTSUN_SI, eta, 0., 0.,
                             len(frequency), amp_mult=prefactor)


def snr_phenom_d(point, n_frequency=768, distance_m=None):
    load_waveform()
    log_m1, z, q = point
    m1 = 10**log_m1 * (1 + z)
    m2 = m1 * q
    chirp_seconds = (m1 * m2)**(.6) / (m1 + m2)**(.2) * constants.MTSUN_SI
    four_years = 4 * 365.25 * 86400
    first = max(1e-5, (5 / (256 * four_years))**(3 / 8) / (np.pi * chirp_seconds**(5 / 8)))
    last = min(1., .2 / ((m1 + m2) * constants.MTSUN_SI))
    if last <= first:
        return 0.
    if distance_m is None:
        distance_m = cosmo.luminosity_distance_Mpc(z) * 1e6 * constants.PC_SI
    frequency = np.geomspace(first, last, n_frequency)
    amp = amplitude(frequency, m1, m2, distance_m)
    psd = lisa_psd_robson_et_al_2018(frequency, True)
    return float(np.sqrt(np.trapz((16 / 5) * amp**2 / psd, frequency)))


class LISASNR:
    def __init__(self, path=None):
        path = path or ROOT / 'model_assets/lisa_analysis/phenom_d_grid.npz'
        data = np.load(path)
        self.low, self.high = data['low'], data['high']
        self.interpolator = RegularGridInterpolator(tuple(data[f'axis_{i}'] for i in range(3)),
                                                    np.log1p(data['rho']), bounds_error=True)
        self.log_rho_grid = np.log1p(data['rho'])
        self._torch_grid = None

    def __call__(self, points_unit):
        physical = self.low + np.asarray(points_unit) * (self.high - self.low)
        return np.maximum(np.expm1(self.interpolator(physical)), 0.)

    def redshift_curve(self, mass_ratio_points):
        """Bilinear interpolation on mass and ratio, retaining redshift knots."""
        coordinates = np.asarray(mass_ratio_points) * (np.array(self.log_rho_grid.shape)[[0, 2]] - 1)
        lower = np.floor(coordinates).astype(int)
        lower = np.minimum(lower, np.array(self.log_rho_grid.shape)[[0, 2]] - 2)
        fraction = coordinates - lower
        curve = np.zeros((len(coordinates), self.log_rho_grid.shape[1]))
        for mass_offset in (0, 1):
            for ratio_offset in (0, 1):
                weight = ((fraction[:, 0] if mass_offset else 1 - fraction[:, 0])
                          * (fraction[:, 1] if ratio_offset else 1 - fraction[:, 1]))
                curve += weight[:, None] * self.log_rho_grid[
                    lower[:, 0] + mass_offset, :, lower[:, 1] + ratio_offset]
        return curve

    def torch(self, points_unit):
        import torch
        import torch.nn.functional as functional
        if self._torch_grid is None or self._torch_grid.dtype != points_unit.dtype:
            self._torch_grid = torch.tensor(self.log_rho_grid, dtype=points_unit.dtype,
                                            device=points_unit.device)[None, None]
        grid = (2 * points_unit[:, [2, 1, 0]] - 1).reshape(1, -1, 1, 1, 3)
        value = functional.grid_sample(self._torch_grid, grid, mode='bilinear',
                                        padding_mode='border', align_corners=True)
        return value.reshape(-1).expm1().clamp(min=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', nargs=3, type=int, default=[161, 97, 49])
    parser.add_argument('--frequency-points', type=int, default=768)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    load_waveform()
    started = time.perf_counter()
    # Verify the face-on amplitude normalization against the Newtonian limit.
    f = np.geomspace(1e-5, 2e-5, 64)
    m1, m2, distance = 35., 30., 1e9 * constants.PC_SI
    amp = amplitude(f, m1, m2, distance)
    mc = (m1 * m2)**.6 / (m1 + m2)**.2 * constants.MTSUN_SI
    newtonian = np.sqrt(5 / 24) * np.pi**(-2 / 3) * mc**(5 / 6) * f**(-7 / 6) / (distance / constants.CLIGHT)
    ratio = float(np.median(amp / newtonian))
    if not abs(ratio - 1) < .002:
        raise AssertionError(f'Waveform normalization mismatch: {ratio}')
    if args.smoke:
        print({'amplitude_to_newtonian_ratio': ratio,
               'snr': [snr_phenom_d(x) for x in [[6., 2., .5], [8., 5., .3], [9., 7., .1]]],
               'seconds': time.perf_counter() - started})
        return
    output = ROOT / 'model_assets/lisa_analysis'
    output.mkdir(parents=True, exist_ok=True)
    axes = [np.linspace(low, high, n) for low, high, n in zip(LOW, HIGH, args.grid)]
    distances = cosmo.luminosity_distance_Mpc(axes[1]) * 1e6 * constants.PC_SI
    rho = np.zeros(tuple(args.grid))
    for i, mass in enumerate(axes[0]):
        for j, z in enumerate(axes[1]):
            for k, q in enumerate(axes[2]):
                rho[i, j, k] = snr_phenom_d([mass, z, q], args.frequency_points, distances[j])
        if (i + 1) % 10 == 0:
            print('mass planes', i + 1, '/', args.grid[0], flush=True)
    np.savez_compressed(output / 'phenom_d_grid.npz', low=LOW, high=HIGH, rho=rho,
                        **{f'axis_{i}': axis for i, axis in enumerate(axes)})
    interpolator = LISASNR()
    rng = np.random.default_rng(7731)
    points = rng.uniform(size=(300, 3))
    candidates = rng.uniform(size=(20000, 3))
    approximate = interpolator(candidates)
    near_threshold = candidates[(approximate > 5) & (approximate < 11)][:200]
    points = np.concatenate([points, near_threshold])
    physical = LOW + points * (HIGH - LOW)
    direct = np.array([snr_phenom_d(x, 3072) for x in physical])
    low_resolution = np.array([snr_phenom_d(x, args.frequency_points) for x in physical])
    interpolated = interpolator(points)
    old = np.array([calculate_snr_for_event_IMR(*x, num_freq_bins=3072) for x in physical])
    valid = direct > .01
    result = {'configuration': vars(args), 'model': 'IMRPhenomD, nonspinning, inclination averaged',
              'frequency_bounds': 'max(1e-5 Hz, Newtonian four-year start) to min(1 Hz, Mf=0.2)',
              'amplitude_to_newtonian_ratio': ratio, 'validation_points': len(points),
              'threshold_validation_points': len(near_threshold),
              'relative_frequency_error_quantiles': np.quantile(abs(low_resolution[valid] / direct[valid] - 1), [.5, .9, .99, 1]).tolist(),
              'absolute_detection_error_quantiles': np.quantile(abs(ndtr(interpolated - 8) - ndtr(direct - 8)), [.5, .9, .99, 1]).tolist(),
              'relative_snr_interpolation_error_quantiles': np.quantile(abs(interpolated[valid] / direct[valid] - 1), [.5, .9, .99, 1]).tolist(),
              'seconds': time.perf_counter() - started,
              'waveform_source_sha256': hashlib.sha256((ROOT / 'external_data/literature/PyIMRPhenomD/IMRPhenomD_internals.py').read_bytes()).hexdigest()}
    np.savez_compressed(output / 'validation.npz', physical_points=physical, direct_rho=direct,
                        interpolated_rho=interpolated, original_phenom_a_rho=old)
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
