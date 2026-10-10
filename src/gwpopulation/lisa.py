from __future__ import annotations

import copy
import os
import pickle
import random
from pathlib import Path

import astropy.units as u
import corner
import dcor
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
from numpy import trapz
from scipy.integrate import dblquad, quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, ks_2samp, multivariate_normal, norm as scipy_norm, pearsonr, qmc
from tqdm.auto import tqdm, trange


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

H0_DEFAULT = 67.4

OMEGA_M_DEFAULT = 0.315

OMEGA_L_DEFAULT = 0.685

C_LIGHT_KM_S = 299792.458

MPC_TO_CM = 3.086e+24

GYR_TO_S = 31540000.0 * 1000000000.0

YR_TO_S = 31540000.0

M_SUN_G = 1.989e+33

G_CONST_CGS = 6.6743e-08

G_CONST_SI = 6.6743e-11

C_LIGHT_SI = 299792458.0

M_SUN_KG = 1.98847e+30

MPC_TO_M = 3.08567758e+22

PI = np.pi

PHENOM_A_COEFFS = {'f0': {'ak': 0.2974, 'bk': 0.04481, 'ck': 0.09556}, 'f1': {'ak': 0.59411, 'bk': 0.089794, 'ck': 0.19111}, 'f2': {'ak': 0.50801, 'bk': 0.077515, 'ck': 0.022369}, 'f3': {'ak': 0.84845, 'bk': 0.12848, 'ck': 0.27299}}

_gsmf_interpolators = None

class Cosmology:

    def __init__(self, H0=H0_DEFAULT, Omega_M=OMEGA_M_DEFAULT, Omega_L=OMEGA_L_DEFAULT):
        """
        Cosmology 类，作为 astropy.cosmology 的封装。
        接受 H0, Omega_M 和 Omega_L。
        """
        self._H0_input = H0
        self._Omega_M_input = Omega_M
        self._Omega_L_input = Omega_L
        self._astro_cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_M)
        self.H0 = self._astro_cosmo.H0.value
        self.h = self._astro_cosmo.h
        self.Omega_M = self._astro_cosmo.Om0
        self.Omega_L = self._astro_cosmo.Ode0
        self.Omega_k = self._astro_cosmo.Ok0
        self.H0_s = self._astro_cosmo.H0.to_value(1 / u.s)
        self.DH = self._astro_cosmo.hubble_distance.to_value(u.Mpc)

    def E_z(self, z):
        """
        无量纲哈勃参数 E(z) = H(z)/H0。
        """
        return self._astro_cosmo.efunc(z)

    def H_z(self, z):
        """
        作为红移函数 H(z) 的哈勃参数 (单位 km/s/Mpc)。
        """
        return self._astro_cosmo.H(z).to_value(u.km / u.s / u.Mpc)

    def comoving_distance_Mpc(self, z):
        """
        共动距离 (单位 Mpc)。
        """
        result = self._astro_cosmo.comoving_distance(z).to_value(u.Mpc)
        if isinstance(z, (list, np.ndarray)) and np.asarray(z).ndim > 0 and (np.asarray(z).size == 1):
            if isinstance(result, np.ndarray):
                return result.reshape(np.asarray(z).shape)
        return result

    def luminosity_distance_Mpc(self, z):
        """
        光度距离 (单位 Mpc)。
        """
        result = self._astro_cosmo.luminosity_distance(z).to_value(u.Mpc)
        if isinstance(z, (list, np.ndarray)) and np.asarray(z).ndim > 0 and (np.asarray(z).size == 1):
            if isinstance(result, np.ndarray):
                return result.reshape(np.asarray(z).shape)
        return result

    def dV_c_dz_fullsky_Gpc3(self, z):
        """
        全天每单位红移的差分共动体积 (单位 Gpc^3)。
        dV/dz = d(Vc)/dz。
        """
        z_arr = np.asanyarray(z)
        results = np.zeros_like(z_arr, dtype=float)
        mask_positive_z = z_arr > 0
        if np.any(mask_positive_z):
            diff_vol_per_sr_positive_z = self._astro_cosmo.differential_comoving_volume(z_arr[mask_positive_z])
            results[mask_positive_z] = (diff_vol_per_sr_positive_z * 4 * np.pi * u.sr).to_value(u.Gpc ** 3)
        if np.isscalar(z):
            return results.item()
        return results.reshape(np.asarray(z).shape)

    def lookback_time_integrand(self, z_prime):
        """
        回视时间的被积函数
        """
        hz_in_gyr_inv = self._astro_cosmo.H(z_prime).to_value(1 / u.Gyr)
        if hz_in_gyr_inv == 0:
            return np.inf
        return 1.0 / ((1.0 + z_prime) * hz_in_gyr_inv)

    def lookback_time(self, z):
        """
        到红移 z 的回溯时间 (单位 Gyr)。
        """
        result = self._astro_cosmo.lookback_time(z).to_value(u.Gyr)
        if isinstance(z, (list, np.ndarray)) and np.asarray(z).ndim > 0 and (np.asarray(z).size == 1):
            if isinstance(result, np.ndarray):
                return result.reshape(np.asarray(z).shape)
        return result

    def dt_dz_Gyr(self, z):
        """
        差分回溯时间: dt_L/dz = dt_proper/dz = 1 / (H(z)(1+z)) (单位 Gyr)。
        """
        z_arr = np.asanyarray(z)
        results = np.empty_like(z_arr, dtype=float)
        mask_valid_z = z_arr >= 0
        if np.any(mask_valid_z):
            hz_val_valid = self._astro_cosmo.H(z_arr[mask_valid_z])
            hz_per_gyr_valid = hz_val_valid.to_value(1 / u.Gyr)
            results[mask_valid_z] = np.where(hz_per_gyr_valid == 0, np.inf, 1.0 / (hz_per_gyr_valid * (1.0 + z_arr[mask_valid_z])))
        mask_invalid_z = z_arr < 0
        if np.any(mask_invalid_z):
            results[mask_invalid_z] = np.nan
        if np.isscalar(z):
            return results.item()
        return results.reshape(np.asarray(z).shape)

def get_phenomA_transition_freq_source_hz(M_total_source_kg, q):
    """
    计算PhenomA模型的源坐标系转换频率 (f0, f1, f2, f3)。
    
    参数:
        M_total_source_kg (float): 双黑洞总质量 (kg)。
        q (float): 质量比 (m2/m1, q <= 1)。

    返回:
        dict: 包含 f0, f1, f2, f3 的字典 (单位 Hz)。
    """
    eta = q / (1.0 + q) ** 2
    GM_over_c3_s = G_CONST_SI * M_total_source_kg / C_LIGHT_SI ** 3
    freqs = {}
    for key, coeffs in PHENOM_A_COEFFS.items():
        fk_hat = coeffs['ak'] * eta ** 2 + coeffs['bk'] * eta + coeffs['ck']
        freqs[key] = fk_hat / (PI * GM_over_c3_s)
    return freqs

def h_f_squared_IMR_phenomA_times_dL_squared(f_obs_hz, Mc_z_kg, M_total_source_kg, q, z):
    """
    计算基于PhenomA模型的IMR波形 |h(f)|^2 * D_L^2。
    
    参数:
        f_obs_hz (np.ndarray or float): 观测频率 (Hz)。
        Mc_z_kg (float): 红移后的啁啾质量 (kg)。
        M_total_source_kg (float): 源坐标系总质量 (kg)。
        q (float): 质量比。
        z (float): 红移。

    返回:
        np.ndarray or float: |h(f)|^2 * D_L^2 的值。
    """
    f_obs_hz = np.asarray(f_obs_hz, dtype=np.float64)
    results = np.zeros_like(f_obs_hz)
    positive_freq_mask = f_obs_hz > 0
    if not np.any(positive_freq_mask):
        return results
    trans_freqs_source = get_phenomA_transition_freq_source_hz(M_total_source_kg, q)
    f0_s, f1_s, f2_s, f3_s = (trans_freqs_source['f0'], trans_freqs_source['f1'], trans_freqs_source['f2'], trans_freqs_source['f3'])
    f0_o, f1_o, f3_o = (f0_s / (1 + z), f1_s / (1 + z), f3_s / (1 + z))
    f_sigma_o = f2_s / (1 + z)
    inspiral_mask = positive_freq_mask & (f_obs_hz < f0_o)
    merger_mask = positive_freq_mask & (f_obs_hz >= f0_o) & (f_obs_hz < f1_o)
    ringdown_mask = positive_freq_mask & (f_obs_hz >= f1_o) & (f_obs_hz < f3_o)
    if np.any(inspiral_mask):
        f_inspiral = f_obs_hz[inspiral_mask]
        term_const = 5.0 / 24.0 * G_CONST_SI ** (5.0 / 3.0) * C_LIGHT_SI ** (-10.0 / 3.0) * PI ** (-4.0 / 3.0)
        term_mass_freq = Mc_z_kg ** (5.0 / 3.0) * f_inspiral ** (-7.0 / 3.0)
        results[inspiral_mask] = term_const * term_mass_freq
    if np.any(merger_mask):
        term_const_insp = 5.0 / 24.0 * G_CONST_SI ** (5.0 / 3.0) * C_LIGHT_SI ** (-10.0 / 3.0) * PI ** (-4.0 / 3.0)
        term_mass_freq_insp_at_f0 = Mc_z_kg ** (5.0 / 3.0) * f0_o ** (-7.0 / 3.0)
        A_sq_at_f0 = term_const_insp * term_mass_freq_insp_at_f0
        f_merger = f_obs_hz[merger_mask]
        results[merger_mask] = A_sq_at_f0 * (f_merger / f0_o) ** (-4.0 / 3.0)
    if np.any(ringdown_mask):
        term_const_insp = 5.0 / 24.0 * G_CONST_SI ** (5.0 / 3.0) * C_LIGHT_SI ** (-10.0 / 3.0) * PI ** (-4.0 / 3.0)
        term_mass_freq_insp_at_f0 = Mc_z_kg ** (5.0 / 3.0) * f0_o ** (-7.0 / 3.0)
        A_sq_at_f0 = term_const_insp * term_mass_freq_insp_at_f0
        A_sq_at_f1 = A_sq_at_f0 * (f1_o / f0_o) ** (-4.0 / 3.0)
        f_ringdown = f_obs_hz[ringdown_mask]
        sigma_o = f_sigma_o / 2.0
        lorentzian = sigma_o ** 2 / ((f_ringdown - f1_o) ** 2 + sigma_o ** 2)
        results[ringdown_mask] = A_sq_at_f1 * lorentzian
    if np.isscalar(f_obs_hz):
        return results.item()
    return results

def lisa_psd_robson_et_al_2018(f_hz, include_galactic_confusion=True):
    """
    计算LISA灵敏度功率谱密度 S_n(f) [strain^2 Hz^-1].
    """
    f_hz = np.asarray(f_hz, dtype=np.float64)
    psd_values = np.full_like(f_hz, np.inf)
    mask = f_hz > 0
    if not np.any(mask):
        return psd_values
    f = f_hz[mask]
    L_arm = 2500000000.0
    f_star = C_LIGHT_SI / (2.0 * PI * L_arm)
    P_oms_f = 1.5e-11 ** 2 * (1.0 + (0.002 / f) ** 4)
    P_acc_f = 3e-15 ** 2 * (1.0 + (0.0004 / f) ** 2) * (1.0 + (f / 0.008) ** 4)
    Pn_f = P_oms_f / L_arm ** 2 + 2.0 * (1.0 + np.cos(f / f_star) ** 2) * P_acc_f / ((2 * PI * f) ** 4 * L_arm ** 2)
    R_f = 3.0 / 10.0 / (1.0 + 0.6 * (f / f_star) ** 2)
    Sn_instr_f = Pn_f / R_f
    if include_galactic_confusion:
        A = 9e-45
        alpha = 0.138
        beta = -221
        kappa = 521
        gamma = 1680
        f_k = 0.00113
        Sc_f = A * f ** (-7.0 / 3.0) * np.exp(-f ** alpha + beta * f * np.sin(kappa * f)) * (1.0 + np.tanh(gamma * (f_k - f)))
        psd_values[mask] = Sn_instr_f + Sc_f
    else:
        psd_values[mask] = Sn_instr_f
    if np.isscalar(f_hz):
        return psd_values.item()
    return psd_values

def get_chirp_mass_source_kg(log_M_bh1_Msun, q):
    """计算源坐标系下的啁啾质量 (kg)"""
    M_bh1_Msun = 10 ** log_M_bh1_Msun
    M_bh2_Msun = q * M_bh1_Msun
    M1_kg = M_bh1_Msun * M_SUN_KG
    M2_kg = M_bh2_Msun * M_SUN_KG
    Mc_source_kg = (M1_kg * M2_kg) ** (3.0 / 5.0) / (M1_kg + M2_kg) ** (1.0 / 5.0)
    return Mc_source_kg

def get_total_mass_source_kg(log_M_bh1_Msun, q):
    """计算源坐标系下的总质量 (kg)"""
    M_bh1_Msun = 10 ** log_M_bh1_Msun
    M_bh2_Msun = q * M_bh1_Msun
    return (M_bh1_Msun + M_bh2_Msun) * M_SUN_KG

def gw_snr_isco_obs(M_total_source_kg, z):
    """计算非自旋双黑洞最内稳定圆轨道 (ISCO) 对应的引力波频率的红移后观测频率 (Hz)"""
    f_isco_source_hz = C_LIGHT_SI ** 3 / (PI * G_CONST_SI * M_total_source_kg * 6.0 ** (3.0 / 2.0))
    return f_isco_source_hz / (1.0 + z)

def h_f_squared_inspiral_optimal_times_dL_squared(f_obs_hz, Mc_z_kg):
    """
    计算旋进阶段最佳朝向下的引力波信号的傅里叶变换模平方乘以光度距离平方: |h_opt(f)|^2 * D_L^2。
    注意：此函数是旋进近似
    """
    if np.isscalar(f_obs_hz):
        if f_obs_hz <= 0:
            return 0.0
    else:
        results = np.zeros_like(f_obs_hz, dtype=float)
        valid_mask = f_obs_hz > 0
        if np.any(valid_mask):
            f_valid = f_obs_hz[valid_mask]
            term_const = G_CONST_SI ** (5.0 / 3.0) / C_LIGHT_SI ** 3 * (5.0 / (24.0 * PI ** (4.0 / 3.0)))
            term_mass_freq = Mc_z_kg ** (5.0 / 3.0) * f_valid ** (-7.0 / 3.0)
            results[valid_mask] = term_const * term_mass_freq
        return results
    term_const = G_CONST_SI ** (5.0 / 3.0) / C_LIGHT_SI ** 3 * (5.0 / (24.0 * PI ** (7.0 / 3.0)))
    term_mass_freq = Mc_z_kg ** (5.0 / 3.0) * f_obs_hz ** (-7.0 / 3.0)
    return term_const * term_mass_freq

def calculate_snr_for_event_IMR(log_M_bh1_Msun, z, q, f_min_obs_hz=1e-05, f_max_obs_cap_hz=1.0, include_galactic_confusion_in_psd=True, num_freq_bins=20000):
    """
    为单个SMBH并合事件计算信噪比 (SNR)，使用IMR (PhenomA) 模型。
    """
    Mc_source_kg = get_chirp_mass_source_kg(log_M_bh1_Msun, q)
    M_total_source_kg = get_total_mass_source_kg(log_M_bh1_Msun, q)
    Mc_z_kg = Mc_source_kg * (1.0 + z)
    D_L_mpc = cosmo.luminosity_distance_Mpc(z)
    if D_L_mpc == 0:
        return 0.0
    D_L_m = D_L_mpc * MPC_TO_M
    trans_freqs_source = get_phenomA_transition_freq_source_hz(M_total_source_kg, q)
    f_cutoff_source_hz = trans_freqs_source['f3']
    f_cutoff_obs_hz = f_cutoff_source_hz / (1.0 + z)
    f_max_integrate_hz = min(f_cutoff_obs_hz, f_max_obs_cap_hz)
    if f_min_obs_hz >= f_max_integrate_hz:
        return 0.0
    snr_sq_avg_integral = 0.0
    try:
        freqs = np.logspace(np.log10(f_min_obs_hz), np.log10(f_max_integrate_hz), num=num_freq_bins)
        hf_sq_dl_sq = h_f_squared_IMR_phenomA_times_dL_squared(freqs, Mc_z_kg, M_total_source_kg, q, z)
        hf_sq_optimal = hf_sq_dl_sq / D_L_m ** 2
        Sn_f = lisa_psd_robson_et_al_2018(freqs, include_galactic_confusion_in_psd)
        integrand = np.zeros_like(freqs)
        safe_mask = (Sn_f > 1e-60) & ~np.isinf(Sn_f) & ~np.isnan(Sn_f)
        integrand[safe_mask] = 16.0 / 5.0 * hf_sq_optimal[safe_mask] / Sn_f[safe_mask]
        snr_sq_avg_integral = trapz(integrand, freqs)
    except Exception as e:
        print(f'Warning: SNR integration failed for params ({log_M_bh1_Msun}, {z}, {q}). Error: {e}. SNR set to 0.')
        snr_sq_avg_integral = 0.0
    if snr_sq_avg_integral < 0:
        snr_sq_avg_integral = 0.0
    snr_averaged = np.sqrt(snr_sq_avg_integral)
    return snr_averaged

def sigmoid_p_det(snr, snr_threshold=8.0, width=1):
    """
    平滑的探测概率函数，替代硬性SNR截断。
    snr: numpy数组或torch张量。
    """
    return 1.0 / (1.0 + np.exp(-(snr - snr_threshold) / width))

def _create_gsmf_parameter_interpolators():
    """
    内部函数：创建GSMF参数关于红移z的平滑插值器。
    """
    z_centers = np.array([0.35, 0.65, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 4.0, 5.0, 6.0, 7.0])
    params = {'log_M_char_star': np.array([10.93, 10.98, 11.01, 11.01, 10.88, 10.8, 11.0, 10.85, 10.46, 10.33, 10.19, 10.5]), 'log_phi_star1': np.log10(0.001 * np.array([0.6, 0.48, 0.7, 0.69, 0.23, 0.26, 0.22, 0.2, 0.2, 0.13, 0.05, 0.01])), 'alpha_star1': np.array([-1.45, -1.44, -1.35, -1.34, -1.52, -1.46, -1.46, -1.46, -1.46, -1.46, -1.46, -1.46]), 'log_phi_star2': np.log10(0.001 * np.array([1.07, 0.92, 0.77, 0.32, 0.67, 0.29, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06])), 'alpha_star2': np.array([-0.66, -0.79, -0.72, -0.56, -0.55, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    interpolators = {key: interp1d(z_centers, val, kind='cubic', bounds_error=False, fill_value=(val[0], val[-1])) for key, val in params.items()}
    return interpolators

def _get_smoothed_gsmf_params(z):
    global _gsmf_interpolators
    if _gsmf_interpolators is None:
        _gsmf_interpolators = _create_gsmf_parameter_interpolators()
    z_array = np.asarray(z)
    params = {key: interp(z_array) for key, interp in _gsmf_interpolators.items()}
    if z_array.ndim == 0:
        return {key: val.item() for key, val in params.items()}
    return params

def _schechter_component(log_M_star, log_M_char_star, log_phi_star, alpha_star):
    phi_star_val = 10 ** log_phi_star
    M_char_star_val = 10 ** log_M_char_star
    M_star_val = 10 ** log_M_star
    if M_char_star_val <= 0:
        return 0.0
    ratio_M = M_star_val / M_char_star_val
    if np.any(ratio_M <= 0):
        return 0.0
    return np.log(10) * phi_star_val * ratio_M ** (alpha_star + 1) * np.exp(-ratio_M)

def schechter_gsmf(log_M_star, z):
    params = _get_smoothed_gsmf_params(z)
    total_phi = _schechter_component(log_M_star, params['log_M_char_star'], params['log_phi_star1'], params['alpha_star1'])
    total_phi += _schechter_component(log_M_star, params['log_M_char_star'], params['log_phi_star2'], params['alpha_star2'])
    return total_phi

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_m_bh_relation_params(z):
    kappa_kh = 0.49
    b_kh = 1.17
    a_kh = np.log10(kappa_kh) + 9.0 - 11.0 * b_kh
    epsilon_kh = 0.28
    a_jwst = -2.43
    b_jwst = 1.06
    epsilon_jwst = 0.69
    transition_center = 4.0
    transition_width = 0.1
    weight_jwst = _sigmoid((z - transition_center) / transition_width)
    a_val = a_kh * (1 - weight_jwst) + a_jwst * weight_jwst
    b_val = b_kh * (1 - weight_jwst) + b_jwst * weight_jwst
    epsilon_val = epsilon_kh * (1 - weight_jwst) + epsilon_jwst * weight_jwst
    return (a_val, b_val, epsilon_val)

def get_log_M_bh(log_M_star, z, include_scatter=True):
    a_val, b_val, epsilon_val = get_m_bh_relation_params(z)
    log_M_bh_mean = a_val + b_val * log_M_star
    if include_scatter:
        safe_epsilon = np.maximum(epsilon_val, 1e-10)
        return scipy_norm.rvs(loc=log_M_bh_mean, scale=safe_epsilon)
    return log_M_bh_mean

def galaxy_merger_rate_illustris_fit(M_star, q, z):
    """
    根据 Rodriguez-Gomez et al. 2015 (Illustris Simulation), Table 1 的拟合函数，
    计算微分星系并合率。

    此函数计算的是 dN_mergers / (dμ dt)，单位为 Gyr^-1。

    参数:
        M_star (float or np.ndarray): 子代星系的恒星质量 (单位 M_sun)。
        q (float or np.ndarray): 前身星系的质量比 (论文中为 μ)，q <= 1。
        z (float or np.ndarray): 红移。

    返回:
        float or np.ndarray: 每个星系的微分并合率 [Gyr^-1]。
    """
    LOG_A0_GYR_INV = -2.2287
    ETA = 2.4644
    ALPHA0 = 0.2241
    ALPHA1 = -1.1759
    BETA0 = -1.2595
    BETA1 = 0.0611
    GAMMA = -0.0477
    DELTA0 = 0.7668
    DELTA1 = -0.4695
    M0_MSUN = 200000000000.0
    A0 = 10 ** LOG_A0_GYR_INV
    one_plus_z = 1.0 + np.asarray(z)
    A_z = A0 * one_plus_z ** ETA
    alpha_z = ALPHA0 * one_plus_z ** ALPHA1
    beta_z = BETA0 * one_plus_z ** BETA1
    delta_z = DELTA0 * one_plus_z ** DELTA1
    M_star = np.asarray(M_star)
    q = np.asarray(q)
    q = np.maximum(q, 1e-09)
    mass_term1 = (M_star / 10000000000.0) ** alpha_z
    mass_term2 = (1.0 + M_star / M0_MSUN) ** delta_z
    q_exponent = beta_z + GAMMA * np.log10(np.maximum(M_star, 100000.0) / 10000000000.0)
    q_term = q ** q_exponent
    rate = A_z * mass_term1 * mass_term2 * q_term
    if np.isscalar(M_star) and np.isscalar(q) and np.isscalar(z):
        return rate.item()
    return rate

def get_smbh_merger_rate_q_dependent(log_M_bh1_array, q_array, z_array):
    """
    计算超大质量黑洞（SMBH）的并合率，此版本依赖于质量比q。
    它使用了来自 Rodriguez-Gomez et al. 2015 的拟合函数。
    
    此函数旨在替换旧的 `get_smbh_merger_rate` 函数。

    参数:
        log_M_bh1_array (np.ndarray or float): 主黑洞质量的对数 (log10)。
        q_array (np.ndarray or float): 并合事件的质量比。
        z_array (np.ndarray or float): 并合事件的红移。

    返回:
        np.ndarray: SMBH并合率。
    """
    log_M_bh1_array = np.asarray(log_M_bh1_array)
    q_array = np.asarray(q_array)
    z_array = np.asarray(z_array)
    try:
        output_shape = np.broadcast(log_M_bh1_array, q_array, z_array).shape
    except ValueError:
        raise ValueError('输入的 log_M_bh1, q, 和 z 数组形状不兼容。')
    results = np.zeros(output_shape, dtype=float)
    for multi_idx in np.ndindex(output_shape):
        log_M_bh1_val = log_M_bh1_array[multi_idx] if log_M_bh1_array.ndim > 0 else log_M_bh1_array.item()
        q_val = q_array[multi_idx] if q_array.ndim > 0 else q_array.item()
        z_val = z_array[multi_idx] if z_array.ndim > 0 else z_array.item()
        a_z_val, b_z_val, epsilon_z_val = get_m_bh_relation_params(z_val)
        if epsilon_z_val <= 1e-10:
            integral_val = 0.0
        else:

            def integrand_all_terms(log_M_s_integrate):
                M_star_integrate = 10 ** log_M_s_integrate
                phi_gsmf_val = schechter_gsmf(log_M_s_integrate, z_val)
                R_gal_val = galaxy_merger_rate_illustris_fit(M_star_integrate, q_val, z_val)
                log_M_bh_mean_at_M_s = a_z_val + b_z_val * log_M_s_integrate
                prob_density_M_bh_given_M_s = scipy_norm.pdf(log_M_bh1_val, loc=log_M_bh_mean_at_M_s, scale=epsilon_z_val)
                return phi_gsmf_val * R_gal_val * prob_density_M_bh_given_M_s
            log_M_s_min_integrate, log_M_s_max_integrate = (7.0, 12.5)
            integral_val, _ = quad(integrand_all_terms, log_M_s_min_integrate, log_M_s_max_integrate, epsabs=1.49e-09, epsrel=1.49e-09, limit=100)
        results[multi_idx] = integral_val
    return results

def _get_char_amplitude_sq(freqs, log_M_bh1_Msun, z, q):
    """
    一个内部辅助函数，用于计算特征应变振幅的平方 |h(f)|^2。
    这本质上是在信噪比计算中使用的 hf_sq_optimal。
    """
    Mc_source_kg = get_chirp_mass_source_kg(log_M_bh1_Msun, q)
    M_total_source_kg = get_total_mass_source_kg(log_M_bh1_Msun, q)
    Mc_z_kg = Mc_source_kg * (1.0 + z)
    D_L_mpc = cosmo.luminosity_distance_Mpc(z)
    if D_L_mpc == 0:
        return np.zeros_like(freqs, dtype=float)
    D_L_m = D_L_mpc * MPC_TO_M
    hf_sq_dl_sq = h_f_squared_IMR_phenomA_times_dL_squared(freqs, Mc_z_kg, M_total_source_kg, q, z)
    hf_sq_optimal = hf_sq_dl_sq / D_L_m ** 2
    return hf_sq_optimal

def calculate_fisher_covariance_matrix(log_M_bh1_Msun, z, q, fallback_cov, f_min_obs_hz=1e-05, f_max_obs_cap_hz=1.0, num_freq_bins=5000):
    """
    通过费雪信息矩阵 (FIM) 计算后验协方差矩阵。
    此函数包含了多重保护机制，以确保生成的矩阵是物理且表现良好的。
    如果计算失败，它将返回一个预先提供的备用协方差矩阵 (fallback_cov)。

    参数:
        log_M_bh1_Msun, z, q (float): 事件的真实物理参数。
        fallback_cov (np.ndarray): 当计算失败时返回的备用协方差矩阵。
        f_min_obs_hz, f_max_obs_cap_hz (float): 积分的频率上下限。
        num_freq_bins (int): 用于数值积分的频率采样点数量。

    返回:
        np.ndarray: 一个3x3的协方差矩阵。
    """
    params = np.array([log_M_bh1_Msun, z, q])
    h_steps = np.array([0.001, 0.001, 0.001])
    param_deltas = np.maximum(params * h_steps, 1e-05)
    try:
        M_total_source_kg = get_total_mass_source_kg(log_M_bh1_Msun, q)
        trans_freqs_source = get_phenomA_transition_freq_source_hz(M_total_source_kg, q)
        f_cutoff_source_hz = trans_freqs_source['f3']
        f_cutoff_obs_hz = f_cutoff_source_hz / (1.0 + z)
        f_max_integrate_hz = min(f_cutoff_obs_hz, f_max_obs_cap_hz)
        if f_min_obs_hz >= f_max_integrate_hz:
            return fallback_cov
        freqs = np.logspace(np.log10(f_min_obs_hz), np.log10(f_max_integrate_hz), num=num_freq_bins)
        derivatives = []
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += param_deltas[i]
            params_minus = params.copy()
            params_minus[i] -= param_deltas[i]
            h_sq_plus = _get_char_amplitude_sq(freqs, *params_plus)
            h_sq_minus = _get_char_amplitude_sq(freqs, *params_minus)
            h_sq_center = _get_char_amplitude_sq(freqs, *params)
            h_center = np.sqrt(np.maximum(h_sq_center, 1e-99))
            d_h_sq_d_theta_i = (h_sq_plus - h_sq_minus) / (2 * param_deltas[i])
            safe_mask = h_center > 1e-50
            d_h_d_theta_i = np.zeros_like(freqs)
            d_h_d_theta_i[safe_mask] = 1.0 / (2.0 * h_center[safe_mask]) * d_h_sq_d_theta_i[safe_mask]
            derivatives.append(d_h_d_theta_i)
        Sn_f = lisa_psd_robson_et_al_2018(freqs, include_galactic_confusion=True)
        safe_mask_sn = (Sn_f > 1e-60) & ~np.isinf(Sn_f) & ~np.isnan(Sn_f)
        fim = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                integrand = np.zeros_like(freqs)
                integrand[safe_mask_sn] = 16.0 / 5.0 * derivatives[i][safe_mask_sn] * derivatives[j][safe_mask_sn] / Sn_f[safe_mask_sn]
                fim[i, j] = trapz(integrand, freqs)
                if i != j:
                    fim[j, i] = fim[i, j]
        cov = np.linalg.pinv(fim, rcond=1e-15)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < -1e-09):
            return fallback_cov
        variances = np.diag(cov)
        if np.any(variances <= 0) or np.any(np.isnan(variances)):
            return fallback_cov
        min_stds = np.array([0.01, 0.005, 0.01])
        max_stds = np.array([0.5, 0.5, 0.4])
        min_vars, max_vars = (min_stds ** 2, max_stds ** 2)
        stds = np.sqrt(variances)
        if np.any(stds < min_stds) or np.any(stds > max_stds):
            clamped_variances = np.clip(variances, min_vars, max_vars)
            scaling_factors = np.sqrt(clamped_variances) / stds
            scaling_matrix = np.outer(scaling_factors, scaling_factors)
            clamped_cov = cov * scaling_matrix
            eigvals_clamped = np.linalg.eigvalsh(clamped_cov)
            if np.any(eigvals_clamped < -1e-09):
                return fallback_cov
            return clamped_cov
        return cov
    except Exception:
        return fallback_cov

def event_generator_smbh_intrinsic_and_observable(num_intrinsic_events_to_generate=2000, n_samples_per_event_posterior=500, sigma_log_M_posterior=0.1, sigma_z_posterior=0.05, sigma_q_posterior=0.08, rho_logM_z_posterior=0.2, rho_logM_q_posterior=0.1, rho_z_q_posterior=-0.1, q_min=0.1, q_max=1.0, seed=None, grid_shape=(50, 50, 30), cache_dir=None, use_fisher_covariance=True, visualize=False, observable_event_mode='measurement_anchor'):
    """
    根据超大质量黑洞并合率生成模拟事件。
    *** 版本 4: 同时生成并返回内禀事件集和可观测事件集，并缓存P_max计算。***
    
    返回:
        tuple: 包含四个元素的元组:
               - intrinsic_true_params (np.ndarray): 内禀事件的真实参数 [N_intrinsic, 3]
               - intrinsic_post_samples (list): 内禀事件的后验样本列表 [list of np.ndarray]
               - observable_true_params (np.ndarray): 可观测事件的真实参数 [N_observable, 3]
               - observable_post_samples (list): 可观测事件的后验样本列表 [list of np.ndarray]
    """
    print('Running Upgraded 3D Event Generator (V4 - Intrinsic & Observable)...')
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    SNR_REF = 20.0
    SIGMA_LOG_M_BASE = sigma_log_M_posterior
    SIGMA_Z_BASE = sigma_z_posterior
    SIGMA_Q_BASE = sigma_q_posterior
    SIGMA_LOG_M_MIN, SIGMA_Z_MIN, SIGMA_Q_MIN = (0.01, 0.005, 0.01)
    max_posterior_sampling_tries_multiplier = 80

    def stabilize_covariance_3d(cov_matrix):
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        if cov_matrix.ndim != 2:
            cov_matrix = np.diag(np.full((3,), 1e-6, dtype=np.float64))
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.clip(eigvals, 1e-6, None)
        stabilized = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (stabilized + stabilized.T)

    def sample_truncated_gaussian_cloud(mean, cov_matrix, num_samples):
        stabilized_cov = stabilize_covariance_3d(cov_matrix)
        samples_list = []
        tries = 0
        max_tries = max(num_samples * max_posterior_sampling_tries_multiplier, num_samples)
        batch_size = max(32, min(512, num_samples))
        while len(samples_list) < num_samples and tries < max_tries:
            draw_count = min(batch_size, max_tries - tries)
            candidates = np.atleast_2d(multivariate_normal.rvs(mean=mean, cov=stabilized_cov, size=draw_count)).astype(np.float64)
            mask = (
                (candidates[:, 0] >= log_M_bh_min)
                & (candidates[:, 0] <= log_M_bh_max)
                & (candidates[:, 1] >= z_min)
                & (candidates[:, 1] <= z_max)
                & (candidates[:, 2] >= q_min)
                & (candidates[:, 2] <= q_max)
            )
            valid = candidates[mask]
            if len(valid):
                samples_list.extend(valid[: num_samples - len(samples_list)])
            tries += draw_count
        if len(samples_list) < num_samples:
            raise RuntimeError(f'Failed to generate {num_samples} bounded posterior samples; only obtained {len(samples_list)}.')
        return np.asarray(samples_list, dtype=np.float64), stabilized_cov

    def get_snr_dependent_covariance_3d(snr):
        if snr < 1.0:
            snr = 1.0
        scaling_factor = SNR_REF / snr
        sigma_log_m_scaled = max(SIGMA_LOG_M_MIN, SIGMA_LOG_M_BASE * scaling_factor)
        sigma_z_scaled = max(SIGMA_Z_MIN, SIGMA_Z_BASE * scaling_factor)
        sigma_q_scaled = max(SIGMA_Q_MIN, SIGMA_Q_BASE * scaling_factor)
        cov_matrix = np.array([[sigma_log_m_scaled ** 2, rho_logM_z_posterior * sigma_log_m_scaled * sigma_z_scaled, rho_logM_q_posterior * sigma_log_m_scaled * sigma_q_scaled], [rho_logM_z_posterior * sigma_log_m_scaled * sigma_z_scaled, sigma_z_scaled ** 2, rho_z_q_posterior * sigma_z_scaled * sigma_q_scaled], [rho_logM_q_posterior * sigma_log_m_scaled * sigma_q_scaled, rho_z_q_posterior * sigma_z_scaled * sigma_q_scaled, sigma_q_scaled ** 2]])
        return cov_matrix
    log_M_bh_min, log_M_bh_max = (5.0, 10.0)
    z_min, z_max = (0.21, 7.5)
    grid_points_M, grid_points_z, grid_points_q = grid_shape
    cache_root = Path(cache_dir) if cache_dir is not None else Path.cwd()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_filename = cache_root / f'pdf_grid_cache_{grid_points_M}x{grid_points_z}x{grid_points_q}.npz'
    if cache_filename.exists():
        print(f"Loading pre-calculated PDF grid from '{cache_filename}'...")
        with np.load(cache_filename) as data:
            pdf_values_on_grid_3d = data['pdf_grid']
    else:
        print('Pre-calculating 3D PDF grid (logM, z, q) to find P_max (first time run)...')
        log_M_grid = np.linspace(log_M_bh_min, log_M_bh_max, grid_points_M)
        z_grid = np.linspace(z_min, z_max, grid_points_z)
        q_grid = np.linspace(q_min, q_max, grid_points_q)
        log_M_mesh_3d, z_mesh_3d, q_mesh_3d = np.meshgrid(log_M_grid, z_grid, q_grid, indexing='ij')
        print(f'Grid dimensions: M({grid_points_M}) x z({grid_points_z}) x q({grid_points_q}) = {log_M_mesh_3d.size} points')
        rates_on_grid_3d = get_smbh_merger_rate_q_dependent(log_M_mesh_3d, q_mesh_3d, z_mesh_3d)
        dVc_dz_on_grid_3d = cosmo.dV_c_dz_fullsky_Gpc3(z_mesh_3d) * 1000000000.0
        time_dilation_on_grid_3d = 1.0 / (1.0 + z_mesh_3d)
        pdf_values_on_grid_3d = rates_on_grid_3d * dVc_dz_on_grid_3d * time_dilation_on_grid_3d
        pdf_values_on_grid_3d[pdf_values_on_grid_3d < 0] = 0
        pdf_values_on_grid_3d[np.isnan(pdf_values_on_grid_3d)] = 0
        print(f"Saving PDF grid to '{cache_filename}' for future runs...")
        np.savez_compressed(cache_filename, pdf_grid=pdf_values_on_grid_3d)
    P_max_3D = np.max(pdf_values_on_grid_3d)
    if P_max_3D <= 0:
        raise RuntimeError(f"Error: P_max_3D <= 0. Cannot generate events. Check the model or cache file '{cache_filename}'.")
    print(f'Starting Acceptance-Rejection sampling with P_max_3D = {P_max_3D:.4e}.')
    print(f'Target: {num_intrinsic_events_to_generate} intrinsic events...')
    intrinsic_true_params_list = []
    intrinsic_posterior_samples_list = []
    observable_true_params_list = []
    observable_posterior_samples_list = []
    total_candidates = 0
    max_candidates = num_intrinsic_events_to_generate * 1000
    pbar = tqdm(total=num_intrinsic_events_to_generate, desc='Generating Intrinsic Events')
    while len(intrinsic_true_params_list) < num_intrinsic_events_to_generate and total_candidates < max_candidates:
        total_candidates += 1
        log_M_cand_true = np.random.uniform(log_M_bh_min, log_M_bh_max)
        z_cand_true = np.random.uniform(z_min, z_max)
        q_cand_true = np.random.uniform(q_min, q_max)
        Rc_scalar = get_smbh_merger_rate_q_dependent(log_M_cand_true, q_cand_true, z_cand_true).item()
        if Rc_scalar < 0:
            Rc_scalar = 0.0
        dVc_dz_scalar = cosmo.dV_c_dz_fullsky_Gpc3(z_cand_true) * 1000000000.0
        time_dilation_factor = 1.0 / (1.0 + z_cand_true)
        P_cand = Rc_scalar * dVc_dz_scalar * time_dilation_factor
        u = np.random.uniform(0, P_max_3D)
        if u < P_cand:
            true_params_current_event = np.array([log_M_cand_true, z_cand_true, q_cand_true])
            event_snr = calculate_snr_for_event_IMR(log_M_bh1_Msun=log_M_cand_true, z=z_cand_true, q=q_cand_true)
            fallback_cov = get_snr_dependent_covariance_3d(event_snr)
            event_specific_cov = fallback_cov
            if use_fisher_covariance:
                event_specific_cov = calculate_fisher_covariance_matrix(log_M_bh1_Msun=log_M_cand_true, z=z_cand_true, q=q_cand_true, fallback_cov=fallback_cov)
            posterior_samples_current_event, stabilized_cov = sample_truncated_gaussian_cloud(true_params_current_event, event_specific_cov, n_samples_per_event_posterior)
            if posterior_samples_current_event.shape[0] == 0:
                print(f'\nWarning: Failed to generate valid posterior samples for intrinsic event {true_params_current_event}. Discarding.')
                continue
            intrinsic_true_params_list.append(true_params_current_event)
            intrinsic_posterior_samples_list.append(posterior_samples_current_event)
            pbar.update(1)
            prob_of_detection = sigmoid_p_det(event_snr)
            if np.random.uniform(0, 1) < prob_of_detection:
                if observable_event_mode == 'reuse_intrinsic_posterior':
                    observable_posterior_samples_current_event = posterior_samples_current_event.copy()
                elif observable_event_mode == 'measurement_anchor':
                    measurement_anchor_samples, _ = sample_truncated_gaussian_cloud(true_params_current_event, stabilized_cov, 1)
                    measurement_anchor = measurement_anchor_samples[0]
                    observable_posterior_samples_current_event, _ = sample_truncated_gaussian_cloud(measurement_anchor, stabilized_cov, n_samples_per_event_posterior)
                else:
                    raise ValueError(f'Unsupported observable_event_mode: {observable_event_mode}')
                observable_true_params_list.append(true_params_current_event)
                observable_posterior_samples_list.append(observable_posterior_samples_current_event)
    pbar.close()
    intrinsic_true_params_array = np.array(intrinsic_true_params_list)
    observable_true_params_array = np.array(observable_true_params_list)
    print(f'\nGeneration complete.')
    print(f'Generated {len(intrinsic_true_params_array)} intrinsic events from {total_candidates} candidates.')
    print(f'Out of these, {len(observable_true_params_array)} events were observable.')
    if total_candidates >= max_candidates and len(intrinsic_true_params_array) < num_intrinsic_events_to_generate:
        print(f'Warning: Reached max candidates ({max_candidates}).')
    if visualize and corner is not None and len(intrinsic_true_params_array) > intrinsic_true_params_array.shape[1]:
        print('Visualizing the 3D distribution of INTRINSIC events...')
        labels = ['$\\log_{10}(M_{BH} / M_{\\odot})$', 'Redshift (z)', 'Mass Ratio (q)']
        fig = corner.corner(intrinsic_true_params_array, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={'fontsize': 12}, color='darkslateblue')
        plt.suptitle('Intrinsic Event Distribution', fontsize=16)
        plt.show()
    return (intrinsic_true_params_array, intrinsic_posterior_samples_list, observable_true_params_array, observable_posterior_samples_list)

def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('default')
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'text.usetex': False, 'mathtext.fontset': 'cm', 'lines.linewidth': 1.2, 'lines.markersize': 5, 'axes.linewidth': 0.8, 'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.minor.size': 2, 'ytick.minor.size': 2, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '0.8', 'legend.handlelength': 1.5, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.format': 'pdf', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

def load_or_generate_lisa_events(intrinsic_params=None, observable_params=None, observable_posteriors=None, n_events=50):
    """
    Load LISA events from arrays or generate mock data for visualization testing.
    
    If arrays are provided, use them directly.
    Otherwise, generate simple mock data for testing the plotting code.
    
    Parameters:
        intrinsic_params: np.ndarray of shape [N_int, 3] - [log_M, z, q]
        observable_params: np.ndarray of shape [N_obs, 3]
        observable_posteriors: list of np.ndarray, each [n_samples, 3]
        n_events: number of mock events to generate if no data provided
    
    Returns:
        dict with 'intrinsic', 'observable', 'posteriors' keys
    """
    if intrinsic_params is not None and observable_params is not None:
        data = {'intrinsic': intrinsic_params, 'observable': observable_params, 'posteriors': observable_posteriors}
        print(f'Loaded {len(intrinsic_params)} intrinsic and {len(observable_params)} observable events')
        return data
    print(f'Generating {n_events} mock LISA events for visualization testing...')
    np.random.seed(42)
    log_M_int = np.random.normal(7.0, 1.2, n_events)
    log_M_int = np.clip(log_M_int, 5.0, 10.0)
    z_int = np.random.exponential(1.5, n_events) + 0.3
    z_int = np.clip(z_int, 0.21, 7.5)
    q_int = np.random.beta(2, 2, n_events) * 0.9 + 0.1
    intrinsic = np.column_stack([log_M_int, z_int, q_int])
    detection_prob = 0.5 * (1 + np.tanh((log_M_int - 6.5) / 1.0)) * np.exp(-z_int / 4)
    detected_mask = np.random.random(n_events) < detection_prob
    observable = intrinsic[detected_mask]
    posteriors = []
    for params in observable:
        sigma_scale = 0.1 * (1 + (8 - params[0]) * 0.1)
        cov = np.diag([0.15 ** 2, (sigma_scale * 0.4) ** 2, 0.08 ** 2])
        samples = np.random.multivariate_normal(params, cov, 500)
        samples[:, 0] = np.clip(samples[:, 0], 5.0, 10.0)
        samples[:, 1] = np.clip(samples[:, 1], 0.21, 7.5)
        samples[:, 2] = np.clip(samples[:, 2], 0.1, 1.0)
        posteriors.append(samples)
    print(f'  Generated: {len(intrinsic)} intrinsic, {len(observable)} observable events')
    return {'intrinsic': intrinsic, 'observable': observable, 'posteriors': posteriors}

def plot_lisa_events_showcase(data, figsize=(7.5, 5.5), save_path=None):
    """
    Create a publication-quality showcase figure for LISA SMBH events.
    
    Layout: 2x2 grid
      (a) log_M vs z - observable events with error bars
      (b) log_M vs q - observable events with error bars
      (c) log_M histogram comparing intrinsic vs observable
      (d) z histogram comparing intrinsic vs observable
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.28)
    intrinsic = data['intrinsic']
    observable = data['observable']
    posteriors = data.get('posteriors', None)
    log_M_obs = observable[:, 0]
    z_obs = observable[:, 1]
    q_obs = observable[:, 2]
    norm = plt.Normalize(z_obs.min(), z_obs.max())
    colors = EVENT_CMAP(norm(z_obs))
    ax_a = fig.add_subplot(gs[0, 0])
    if posteriors is not None:
        log_M_err_low = np.array([log_M_obs[i] - np.percentile(post[:, 0], 16) for i, post in enumerate(posteriors)])
        log_M_err_high = np.array([np.percentile(post[:, 0], 84) - log_M_obs[i] for i, post in enumerate(posteriors)])
        z_err_low = np.array([z_obs[i] - np.percentile(post[:, 1], 16) for i, post in enumerate(posteriors)])
        z_err_high = np.array([np.percentile(post[:, 1], 84) - z_obs[i] for i, post in enumerate(posteriors)])
        q_err_low = np.array([q_obs[i] - np.percentile(post[:, 2], 16) for i, post in enumerate(posteriors)])
        q_err_high = np.array([np.percentile(post[:, 2], 84) - q_obs[i] for i, post in enumerate(posteriors)])
    else:
        log_M_err_low = log_M_err_high = np.ones_like(log_M_obs) * 0.1
        z_err_low = z_err_high = np.ones_like(z_obs) * 0.1
        q_err_low = q_err_high = np.ones_like(q_obs) * 0.05
    n_with_errorbars = min(15, len(observable))
    np.random.seed(42)
    errbar_indices = np.random.choice(len(observable), n_with_errorbars, replace=False)
    ax_a.scatter(log_M_obs, z_obs, s=30, c=colors, edgecolors='white', linewidths=0.4, zorder=5, alpha=0.85)
    for i in errbar_indices:
        ax_a.errorbar(log_M_obs[i], z_obs[i], xerr=[[log_M_err_low[i]], [log_M_err_high[i]]], yerr=[[z_err_low[i]], [z_err_high[i]]], fmt='none', ecolor=colors[i], elinewidth=0.8, capsize=0, alpha=0.6, zorder=4)
    ax_a.set_xlabel('$\\log_{10}(M_\\bullet / M_\\odot)$')
    ax_a.set_ylabel('Redshift $z$')
    ax_a.set_xlim(5, 10)
    ax_a.set_ylim(0, 8)
    ax_a.xaxis.set_minor_locator(AutoMinorLocator())
    ax_a.yaxis.set_minor_locator(AutoMinorLocator())
    ax_a.text(0.05, 0.95, '(a)', transform=ax_a.transAxes, fontweight='bold', va='top', fontsize=11)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(log_M_obs, q_obs, s=30, c=colors, edgecolors='white', linewidths=0.4, zorder=5, alpha=0.85)
    for i in errbar_indices:
        ax_b.errorbar(log_M_obs[i], q_obs[i], xerr=[[log_M_err_low[i]], [log_M_err_high[i]]], yerr=[[q_err_low[i]], [q_err_high[i]]], fmt='none', ecolor=colors[i], elinewidth=0.8, capsize=0, alpha=0.6, zorder=4)
    ax_b.set_xlabel('$\\log_{10}(M_\\bullet / M_\\odot)$')
    ax_b.set_ylabel('Mass Ratio $q$')
    ax_b.set_xlim(5, 10)
    ax_b.set_ylim(0.05, 1.05)
    ax_b.xaxis.set_minor_locator(AutoMinorLocator())
    ax_b.yaxis.set_minor_locator(AutoMinorLocator())
    ax_b.text(0.05, 0.95, '(b)', transform=ax_b.transAxes, fontweight='bold', va='top', fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=EVENT_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_b, pad=0.02, aspect=25)
    cbar.set_label('Redshift $z$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax_c = fig.add_subplot(gs[1, 0])
    bins_M = np.linspace(5, 10, 15)
    ax_c.hist(intrinsic[:, 0], bins=bins_M, color=COLORS['intrinsic'], alpha=0.5, label='Intrinsic', edgecolor='white', lw=0.5)
    ax_c.hist(observable[:, 0], bins=bins_M, color=COLORS['observable'], alpha=0.7, label='Observable', edgecolor='white', lw=0.5)
    ax_c.set_xlabel('$\\log_{10}(M_\\bullet / M_\\odot)$')
    ax_c.set_ylabel('Number of Events')
    ax_c.set_xlim(5, 10)
    ax_c.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_c.xaxis.set_minor_locator(AutoMinorLocator())
    ax_c.legend(loc='upper right', frameon=True)
    ax_c.text(0.05, 0.95, '(c)', transform=ax_c.transAxes, fontweight='bold', va='top', fontsize=11)
    ax_d = fig.add_subplot(gs[1, 1])
    bins_z = np.linspace(0, 8, 15)
    ax_d.hist(intrinsic[:, 1], bins=bins_z, color=COLORS['intrinsic'], alpha=0.5, label='Intrinsic', edgecolor='white', lw=0.5)
    ax_d.hist(observable[:, 1], bins=bins_z, color=COLORS['observable'], alpha=0.7, label='Observable', edgecolor='white', lw=0.5)
    ax_d.set_xlabel('Redshift $z$')
    ax_d.set_ylabel('Number of Events')
    ax_d.set_xlim(0, 8)
    ax_d.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_d.xaxis.set_minor_locator(AutoMinorLocator())
    ax_d.legend(loc='upper right', frameon=True)
    ax_d.text(0.05, 0.95, '(d)', transform=ax_d.transAxes, fontweight='bold', va='top', fontsize=11)
    ax_d.annotate('LISA (Simulated)', xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved: {save_path}')
    plt.show()
    return fig

def plot_intrinsic_vs_observable(data, figsize=(7, 3), save_path=None):
    """
    Create a side-by-side comparison of intrinsic vs observable populations.
    
    This clearly shows the selection effect: LISA preferentially detects
    higher-mass, lower-redshift events.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    intrinsic = data['intrinsic']
    observable = data['observable']
    ax = axes[0]
    scatter = ax.scatter(intrinsic[:, 0], intrinsic[:, 1], c=intrinsic[:, 2], cmap='viridis', s=25, alpha=0.7, edgecolors='white', lw=0.3)
    ax.set_xlabel('$\\log_{10}(M_\\bullet / M_\\odot)$')
    ax.set_ylabel('Redshift $z$')
    ax.set_xlim(5, 10)
    ax.set_ylim(0, 8)
    ax.set_title(f'Intrinsic Population ($N = {len(intrinsic)}$)', fontsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax = axes[1]
    scatter = ax.scatter(observable[:, 0], observable[:, 1], c=observable[:, 2], cmap='viridis', s=35, alpha=0.8, edgecolors='white', lw=0.3)
    ax.set_xlabel('$\\log_{10}(M_\\bullet / M_\\odot)$')
    ax.set_ylabel('Redshift $z$')
    ax.set_xlim(5, 10)
    ax.set_ylim(0, 8)
    ax.set_title(f'Observable Population ($N = {len(observable)}$)', fontsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    cbar = fig.colorbar(scatter, ax=axes, pad=0.02, aspect=30)
    cbar.set_label('Mass Ratio $q$', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved: {save_path}')
    plt.show()
    return fig

def plot_3d_population(data, view_angles=(30, -60), figsize=(6, 5), save_path=None):
    """
    Create a 3D scatter plot of the SMBH population in parameter space.
    
    Shows both intrinsic (faint) and observable (bright) populations.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    intrinsic = data['intrinsic']
    observable = data['observable']
    ax.scatter(intrinsic[:, 0], intrinsic[:, 1], intrinsic[:, 2], c=COLORS['intrinsic'], alpha=0.2, s=15, label='Intrinsic')
    scatter = ax.scatter(observable[:, 0], observable[:, 1], observable[:, 2], c=observable[:, 1], cmap=EVENT_CMAP, s=40, alpha=0.9, edgecolors='white', linewidths=0.3, label='Observable')
    ax.set_xlabel('$\\log_{10}(M_\\bullet / M_\\odot)$', labelpad=8)
    ax.set_ylabel('Redshift $z$', labelpad=8)
    ax.set_zlabel('Mass Ratio $q$', labelpad=8)
    ax.set_xlim(5, 10)
    ax.set_ylim(0, 8)
    ax.set_zlim(0.1, 1.0)
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    ax.legend(loc='upper left', frameon=True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Redshift $z$', fontsize=9)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved: {save_path}')
    plt.show()
    return fig

def confidence_ellipse(cov, mean, ax, n_std=1.5, facecolor='none', edgecolor='black', **kwargs):
    """
    绘制误差椭圆
    """
    if cov is None or mean is None:
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = np.clip(pearson, -0.999, 0.999)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def compute_real_snr_list(params_array):
    """
    使用用户提供的物理模型批量重新计算 SNR。
    params_array: [N, 3] -> (logM, z, q)
    """
    snr_list = []
    print(f'Recalculating SNR for {len(params_array)} observable events for plotting...')
    for i in tqdm(range(len(params_array)), desc='Computing SNRs'):
        log_M, z, q = params_array[i]
        snr = calculate_snr_for_event_IMR(log_M_bh1_Msun=log_M, z=z, q=q, f_min_obs_hz=1e-05, f_max_obs_cap_hz=1.0, include_galactic_confusion_in_psd=True, num_freq_bins=20000)
        snr_list.append(snr)
    return np.array(snr_list)

def plot_lisa_selection_effects(intrinsic_params, observable_params, observable_post_samples, output_filename='LISA_Intrinsic_vs_Observable_RealSNR.pdf'):
    labels = ['$\\log_{10}(M_{\\rm BH}/M_{\\odot})$', 'Redshift ($z$)', 'Mass Ratio ($q$)']
    snr_vals = compute_real_snr_list(observable_params)
    snr_log_vals = np.log10(np.maximum(snr_vals, 1.0))
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.2, 1], figure=fig)
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ['Mass - Redshift', 'Mass - Mass Ratio', 'Redshift - Mass Ratio']
    axes_main = []
    for i, (idx_x, idx_y) in enumerate(pairs):
        ax_hist = fig.add_subplot(gs[0, i])
        ax_main = fig.add_subplot(gs[1, i], sharex=ax_hist)
        axes_main.append(ax_main)
        x_int = intrinsic_params[:, idx_x]
        y_int = intrinsic_params[:, idx_y]
        sns.kdeplot(x=x_int, y=y_int, ax=ax_main, fill=True, color='slategrey', alpha=0.2, levels=[0.05, 0.32, 1.0], thresh=0.05)
        sns.kdeplot(x=x_int, y=y_int, ax=ax_main, color='slategrey', alpha=0.5, linewidths=0.8, levels=[0.05, 0.32, 1.0], thresh=0.05)
        x_obs = observable_params[:, idx_x]
        y_obs = observable_params[:, idx_y]
        sc = ax_main.scatter(x_obs, y_obs, c=snr_log_vals, cmap='plasma', s=15, alpha=0.9, edgecolors='none', label='Observable')
        sorted_idx = np.argsort(snr_vals)
        if len(sorted_idx) > 10:
            demo_indices = [sorted_idx[int(len(sorted_idx) * 0.1)], sorted_idx[int(len(sorted_idx) * 0.5)], sorted_idx[int(len(sorted_idx) * 0.9)]]
            for d_idx in demo_indices:
                samples = observable_post_samples[d_idx]
                samples_2d = samples[:, [idx_x, idx_y]]
                cov_2d = np.cov(samples_2d.T)
                mean_2d = np.mean(samples_2d, axis=0)
                confidence_ellipse(cov_2d, mean_2d, ax_main, n_std=1.5, edgecolor='black', linestyle='--', linewidth=1.2, zorder=10)
                ax_main.scatter(mean_2d[0], mean_2d[1], marker='+', color='black', s=40, zorder=10)
        ax_main.set_xlabel(labels[idx_x])
        ax_main.set_ylabel(labels[idx_y])
        ax_hist.hist(x_int, bins=30, density=True, histtype='stepfilled', color='slategrey', alpha=0.3, label='Intrinsic')
        ax_hist.hist(x_obs, bins=30, density=True, histtype='step', color='#C83E73', linewidth=2, label='Observable')
        ax_hist.set_title(titles[i], pad=10, fontsize=14)
        ax_hist.tick_params(axis='x', labelbottom=False)
        ax_hist.set_yticks([])
        if i == 0:
            ax_hist.legend(loc='upper left', frameon=False, fontsize=10)
    cbar = fig.colorbar(sc, ax=axes_main, orientation='horizontal', fraction=0.06, pad=0.1, aspect=40)
    cbar.set_label('LISA SNR ($\\log_{10}$)', fontsize=14)
    fig.suptitle('LISA Horizon: Intrinsic Population vs. Observable Events', fontsize=20, y=1.03)
    print(f'Saving high-quality figure to {output_filename}...')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()

def plot_lisa_square_layout(intrinsic_params, observable_params, observable_post_samples, output_filename='LISA_Selection_Square.pdf'):
    labels = ['$\\log_{10}(M_{\\rm BH}/M_{\\odot})$', 'Redshift ($z$)', 'Mass Ratio ($q$)']
    num_ellipses = 20
    total_obs = len(observable_params)
    if total_obs > num_ellipses:
        rng = np.random.default_rng(42)
        ellipse_indices = rng.choice(total_obs, size=num_ellipses, replace=False)
    else:
        ellipse_indices = np.arange(total_obs)
    fig = plt.figure(constrained_layout=True)
    sub_gs = gridspec.GridSpec(2, 3, height_ratios=[0.15, 1], figure=fig)
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ['Mass - Redshift', 'Mass - Mass Ratio', 'Redshift - Mass Ratio']
    color_intrinsic = 'slategrey'
    color_observable = '#C83E73'
    for i, (idx_x, idx_y) in enumerate(pairs):
        ax_hist = fig.add_subplot(sub_gs[0, i])
        ax_main = fig.add_subplot(sub_gs[1, i], sharex=ax_hist)
        ax_main.set_box_aspect(1)
        x_int = intrinsic_params[:, idx_x]
        y_int = intrinsic_params[:, idx_y]
        sns.kdeplot(x=x_int, y=y_int, ax=ax_main, fill=True, color=color_intrinsic, alpha=0.15, levels=[0.05, 0.32, 1.0], thresh=0.05)
        sns.kdeplot(x=x_int, y=y_int, ax=ax_main, color=color_intrinsic, alpha=0.4, linewidths=1.0, levels=[0.05, 0.32, 1.0], thresh=0.05)
        x_obs = observable_params[:, idx_x]
        y_obs = observable_params[:, idx_y]
        ax_main.scatter(x_obs, y_obs, c=color_observable, s=12, alpha=0.5, edgecolors='none', label='Observable')
        for idx in ellipse_indices:
            samples = observable_post_samples[idx]
            samples_2d = samples[:, [idx_x, idx_y]]
            cov_2d = np.cov(samples_2d.T)
            mean_2d = np.mean(samples_2d, axis=0)
            confidence_ellipse(cov_2d, mean_2d, ax_main, n_std=1.5, edgecolor='black', facecolor='black', linestyle='-', linewidth=0.8, alpha=0.15)
            confidence_ellipse(cov_2d, mean_2d, ax_main, n_std=1.5, edgecolor='k', facecolor='none', linestyle='-', linewidth=0.8, alpha=0.4)
        ax_main.set_xlabel(labels[idx_x])
        ax_main.set_ylabel(labels[idx_y])
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()
        ax_main.set_xlim(xlim[0] - (xlim[1] - xlim[0]) * 0.05, xlim[1] + (xlim[1] - xlim[0]) * 0.05)
        ax_main.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.05, ylim[1] + (ylim[1] - ylim[0]) * 0.05)
        ax_hist.hist(x_int, bins=30, density=True, histtype='stepfilled', color=color_intrinsic, alpha=0.3)
        ax_hist.hist(x_obs, bins=30, density=True, histtype='step', color=color_observable, linewidth=1.5)
        ax_hist.set_title(titles[i], pad=8, fontsize=13)
        ax_hist.tick_params(axis='x', labelbottom=False)
        ax_hist.set_yticks([])
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['left'].set_visible(False)
        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=color_intrinsic, lw=4, alpha=0.3, label='Intrinsic Pop.'), Line2D([0], [0], color=color_observable, marker='o', lw=0, markersize=5, label='Observable Events'), Line2D([0], [0], color='black', lw=1, alpha=0.6, linestyle='-', label='1.5-$\\sigma$ Errors')]
            ax_main.legend(handles=legend_elements, loc='best', frameon=True, fontsize=9, framealpha=0.8)
    fig.suptitle('LISA Horizon and Measurement Precision', fontsize=18, y=1.05)
    print(f'Saving square-aspect figure to {output_filename}...')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()

def permutation_test_energy_distance(x, y, num_permutations=500):
    """
    Performs a permutation test to get a p-value for the energy distance.
    """
    observed_distance = dcor.energy_distance(x, y)
    combined_samples = np.vstack((x, y))
    count_greater_or_equal = 0
    pbar = trange(num_permutations, desc='Running Permutation Test')
    for _ in pbar:
        permuted_indices = np.random.permutation(len(combined_samples))
        permuted_samples = combined_samples[permuted_indices]
        new_x = permuted_samples[:len(x)]
        new_y = permuted_samples[len(x):]
        permuted_distance = dcor.energy_distance(new_x, new_y)
        if permuted_distance >= observed_distance:
            count_greater_or_equal += 1
    p_value = (count_greater_or_equal + 1) / (num_permutations + 1)
    return (observed_distance, p_value)

def pairgrid_plot_lower(x, y, label=None, color=None, **kwargs):
    if label == 'Ground Truth':
        sns.kdeplot(x=x, y=y, color=color, levels=4, fill=False, linewidths=1.2, alpha=0.8, **kwargs)
    else:
        sns.kdeplot(x=x, y=y, color=color, levels=4, fill=False, linewidths=1.2, linestyles='--', alpha=0.9, **kwargs)
        plt.scatter(x, y, s=5, color=color, alpha=0.3, rasterized=True)

def perform_dual_validation(estimator, observation_years, true_intrinsic_count, true_observed_count, num_posterior_draws=1000, num_mc_points=50000):
    """
    执行双重验证：
    1. 参数恢复 (Parameter Recovery): 比较推断出的内禀事件总数 vs 真实生成的内禀总数。
    2. 闭环测试 (Closed-Loop): 比较预测的可观测事件总数 vs 实际观测到的总数。
    """
    if not estimator.is_trained or not estimator.is_rate_inferred:
        raise RuntimeError('Estimator must be fully trained and rate must be inferred.')
    print('\n' + '=' * 60)
    print('Starting DUAL VALIDATION: Intrinsic Recovery & Observable Consistency')
    print('=' * 60 + '\n')
    rate_samples = estimator.yearly_rate_posterior_samples.cpu().numpy()
    predicted_intrinsic_distribution = rate_samples * observation_years
    min_vals, max_vals = (estimator.min_vals.cpu().numpy(), estimator.max_vals.cpu().numpy())
    logM_min, logM_max = (min_vals[0], max_vals[0])
    z_min, z_max = (min_vals[1], max_vals[1])
    q_min, q_max = (min_vals[2], max_vals[2])
    param_volume = (logM_max - logM_min) * (z_max - z_min) * (q_max - q_min)
    latent_samples = estimator.flow_dist.sample(torch.Size([num_posterior_draws]))
    predicted_observed_distribution = []
    pbar = tqdm(range(num_posterior_draws), desc='Calculating Observable Predictions')
    for i in pbar:
        latent_vector = latent_samples[i]
        params = estimator._unflatten_latent_vector(latent_vector)
        lambda_yr_sample = np.random.choice(rate_samples)
        mc_logM = np.random.uniform(logM_min, logM_max, num_mc_points)
        mc_z = np.random.uniform(z_min, z_max, num_mc_points)
        mc_q = np.random.uniform(q_min, q_max, num_mc_points)
        mc_points = torch.tensor(np.stack([mc_logM, mc_z, mc_q], axis=-1), dtype=torch.float32, device=estimator.device)
        with torch.no_grad():
            grid_norm = (mc_points - estimator.min_vals) / (estimator.max_vals - estimator.min_vals)
            grid_norm = grid_norm.clamp(min=0.001, max=1.0 - 0.001)
            jacobian_det = torch.prod(estimator.max_vals - estimator.min_vals)
            grid_norm_expanded = grid_norm.view(1, -1, estimator.data_dim)
            log_probs_per_comp = []
            if estimator.num_corr_kumaraswamy > 0:
                a = torch.exp(params['log_k_a']) + 0.0001
                b = torch.exp(params['log_k_b']) + 0.0001
                L = estimator._build_cholesky_from_unconstrained(params['k_cholesky_params'], estimator.data_dim, estimator.device)
                marginal_dist = SafeKumaraswamy(a, b)
                log_probs_per_comp.append(estimator._copula_log_prob(marginal_dist, L, grid_norm_expanded))
            log_probs_all = torch.cat(log_probs_per_comp, dim=2).squeeze(0)
            weights = torch.softmax(params['weights_logits'], dim=-1)
            total_prob_norm = torch.exp(torch.logsumexp(torch.log(weights) + log_probs_all, dim=1))
            pdf_values = total_prob_norm / jacobian_det
            p_det_values = estimator.selection_function(mc_points)
        integrand = lambda_yr_sample * pdf_values * p_det_values
        yearly_detectable_rate = param_volume * torch.mean(integrand).item()
        predicted_observed_distribution.append(yearly_detectable_rate * observation_years)
    predicted_observed_distribution = np.array(predicted_observed_distribution)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    mean_int = np.mean(predicted_intrinsic_distribution)
    cred_int = np.percentile(predicted_intrinsic_distribution, [5, 95])
    ax.hist(predicted_intrinsic_distribution, bins=30, density=True, color='skyblue', alpha=0.7, label='Posterior')
    ax.axvline(true_intrinsic_count, color='crimson', linestyle='-', linewidth=3, label=f'TRUE Intrinsic = {true_intrinsic_count}')
    ax.axvline(mean_int, color='black', linestyle='--', linewidth=2, label=f'Inferred Mean = {mean_int:.1f}')
    ax.fill_betweenx([0, 1], cred_int[0], cred_int[1], transform=ax.get_xaxis_transform(), color='gray', alpha=0.2, label='90% CI')
    ax.set_title('Test 1: Parameter Recovery (Intrinsic)\n(Did we find the hidden truth?)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Intrinsic Events (N_int)', fontsize=12)
    ax.legend()
    status_int = 'PASSED' if cred_int[0] <= true_intrinsic_count <= cred_int[1] else 'TENSION'
    print(f'\n[Test 1] Intrinsic Recovery: {status_int}')
    print(f'  True Count: {true_intrinsic_count}')
    print(f'  Inferred:   {mean_int:.2f} (90% CI: [{cred_int[0]:.2f}, {cred_int[1]:.2f}])')
    ax = axes[1]
    mean_obs = np.mean(predicted_observed_distribution)
    cred_obs = np.percentile(predicted_observed_distribution, [5, 95])
    ax.hist(predicted_observed_distribution, bins=30, density=True, color='lightgreen', alpha=0.7, label='Posterior Prediction')
    ax.axvline(true_observed_count, color='crimson', linestyle='-', linewidth=3, label=f'ACTUAL Observed = {true_observed_count}')
    ax.axvline(mean_obs, color='black', linestyle='--', linewidth=2, label=f'Predicted Mean = {mean_obs:.1f}')
    ax.fill_betweenx([0, 1], cred_obs[0], cred_obs[1], transform=ax.get_xaxis_transform(), color='gray', alpha=0.2, label='90% CI')
    ax.set_title('Test 2: Closed-Loop Check (Observable)\n(Is the model self-consistent?)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Observed Events (N_obs)', fontsize=12)
    ax.legend()
    status_obs = 'PASSED' if cred_obs[0] <= true_observed_count <= cred_obs[1] else 'TENSION'
    print(f'\n[Test 2] Closed-Loop Check:  {status_obs}')
    print(f'  Actual Obs: {true_observed_count}')
    print(f'  Predicted:  {mean_obs:.2f} (90% CI: [{cred_obs[0]:.2f}, {cred_obs[1]:.2f}])')
    plt.tight_layout()
    plt.show()

def plot_publication_quality(predicted_int, predicted_obs, true_int, true_obs, obs_years):
    """
    绘制期刊级质量的双重验证图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    color_int = '#004488'
    color_obs = '#009988'
    color_truth = '#CC3311'
    color_ci_fill = '#EEF3F8'

    def draw_panel(ax, data, true_val, main_color, title_text, x_label):
        mean_val = np.mean(data)
        ci_low, ci_high = np.percentile(data, [5, 95])
        density = gaussian_kde(data)
        xs = np.linspace(min(data) * 0.9, max(data) * 1.1, 500)
        ys = density(xs)
        ax.hist(data, bins=30, density=True, color=main_color, alpha=0.15, edgecolor='none', zorder=1)
        ax.plot(xs, ys, color=main_color, lw=2.5, label='Posterior PDF', zorder=3)
        ax.axvline(true_val, color=color_truth, ls='-', lw=2.5, alpha=0.9, zorder=5, label='Ground Truth')
        ax.axvline(mean_val, color='k', ls='--', lw=1.5, alpha=0.8, zorder=4, label='Inferred Mean')
        ax.fill_between(xs, 0, ys, where=(xs >= ci_low) & (xs <= ci_high), color=main_color, alpha=0.1, zorder=2)
        ax.set_title(title_text, fontweight='bold', pad=15)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Probability Density')
        ax.set_xlim(min(data) * 0.9, max(data) * 1.1)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        stats_text = f'$\\bf{{Ground\\ Truth}}$: {int(true_val)}\n$\\bf{{Inferred}}$: {mean_val:.1f}\n$\\bf{{90\\%\\ CI}}$: [{ci_low:.1f}, {ci_high:.1f}]\n$\\bf{{Precision}}$: {abs(mean_val - true_val) / true_val * 100:.1f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#DDDDDD')
        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props, zorder=10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='upper right', frameon=False, fontsize=10)
    draw_panel(axes[0], predicted_int, true_int, color_int, title_text='Test 1: Intrinsic Parameter Recovery', x_label='Total Intrinsic Events ($N_{\\mathrm{int}}$)')
    axes[0].text(0.5, 1.02, 'Are we recovering the unseen truth?', transform=axes[0].transAxes, ha='center', va='bottom', fontsize=10, style='italic', color='#555555')
    draw_panel(axes[1], predicted_obs, true_obs, color_obs, title_text='Test 2: Observable Closed-Loop Check', x_label='Total Observed Events ($N_{\\mathrm{obs}}$)')
    axes[1].text(0.5, 1.02, 'Is the model consistent with data?', transform=axes[1].transAxes, ha='center', va='bottom', fontsize=10, style='italic', color='#555555')
    plt.savefig('dual_validation_paper_quality.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('dual_validation_paper_quality.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_dual_validation_upgraded(estimator, observation_years, true_intrinsic_count, true_observed_count, num_posterior_draws=1000, num_mc_points=50000):
    print('Generating validation data...')
    rate_samples = estimator.yearly_rate_posterior_samples.cpu().numpy()
    predicted_intrinsic_distribution = rate_samples * observation_years
    min_vals, max_vals = (estimator.min_vals.cpu().numpy(), estimator.max_vals.cpu().numpy())
    logM_min, logM_max = (min_vals[0], max_vals[0])
    z_min, z_max = (min_vals[1], max_vals[1])
    q_min, q_max = (min_vals[2], max_vals[2])
    param_volume = (logM_max - logM_min) * (z_max - z_min) * (q_max - q_min)
    latent_samples = estimator.flow_dist.sample(torch.Size([num_posterior_draws]))
    predicted_observed_distribution = []
    for i in range(num_posterior_draws):
        latent_vector = latent_samples[i]
        params = estimator._unflatten_latent_vector(latent_vector)
        lambda_yr_sample = np.random.choice(rate_samples)
        mc_logM = np.random.uniform(logM_min, logM_max, num_mc_points)
        mc_z = np.random.uniform(z_min, z_max, num_mc_points)
        mc_q = np.random.uniform(q_min, q_max, num_mc_points)
        mc_points = torch.tensor(np.stack([mc_logM, mc_z, mc_q], axis=-1), dtype=torch.float32, device=estimator.device)
        with torch.no_grad():
            grid_norm = (mc_points - estimator.min_vals) / (estimator.max_vals - estimator.min_vals)
            grid_norm = grid_norm.clamp(min=0.001, max=1.0 - 0.001)
            jacobian_det = torch.prod(estimator.max_vals - estimator.min_vals)
            grid_norm_expanded = grid_norm.view(1, -1, estimator.data_dim)
            log_probs_per_comp = []
            if estimator.num_corr_kumaraswamy > 0:
                a = torch.exp(params['log_k_a']) + 0.0001
                b = torch.exp(params['log_k_b']) + 0.0001
                L = estimator._build_cholesky_from_unconstrained(params['k_cholesky_params'], estimator.data_dim, estimator.device)
                marginal_dist = SafeKumaraswamy(a, b)
                log_probs_per_comp.append(estimator._copula_log_prob(marginal_dist, L, grid_norm_expanded))
            log_probs_all = torch.cat(log_probs_per_comp, dim=2).squeeze(0)
            weights = torch.softmax(params['weights_logits'], dim=-1)
            total_prob_norm = torch.exp(torch.logsumexp(torch.log(weights) + log_probs_all, dim=1))
            pdf_values = total_prob_norm / jacobian_det
            p_det_values = estimator.selection_function(mc_points)
        integrand = lambda_yr_sample * pdf_values * p_det_values
        yearly_detectable_rate = param_volume * torch.mean(integrand).item()
        predicted_observed_distribution.append(yearly_detectable_rate * observation_years)
    predicted_observed_distribution = np.array(predicted_observed_distribution)
    plot_publication_quality(predicted_intrinsic_distribution, predicted_observed_distribution, true_intrinsic_count, true_observed_count, observation_years)

def get_smbh_merger_rate_q_dependent_vectorized(log_M_bh1_array, q_array, z_array):
    """
    get_smbh_merger_rate_q_dependent 的矢量化版本。
    避免显式的Python循环，利用numpy.vectorize来大幅提升计算速度。
    """

    def _integrate_single_point(log_M_bh1_val, q_val, z_val):
        a_z_val, b_z_val, epsilon_z_val = get_m_bh_relation_params(z_val)
        if epsilon_z_val <= 1e-10:
            return 0.0

        def integrand_all_terms(log_M_s_integrate):
            M_star_integrate = 10 ** log_M_s_integrate
            phi_gsmf_val = schechter_gsmf(log_M_s_integrate, z_val)
            R_gal_val = galaxy_merger_rate_illustris_fit(M_star_integrate, q_val, z_val)
            log_M_bh_mean_at_M_s = a_z_val + b_z_val * log_M_s_integrate
            prob_density_M_bh_given_M_s = scipy_norm.pdf(log_M_bh1_val, loc=log_M_bh_mean_at_M_s, scale=epsilon_z_val)
            return phi_gsmf_val * R_gal_val * prob_density_M_bh_given_M_s
        log_M_s_min_integrate, log_M_s_max_integrate = (7.0, 12.5)
        integral_val, _ = quad(integrand_all_terms, log_M_s_min_integrate, log_M_s_max_integrate, epsabs=1.49e-09, epsrel=1.49e-09, limit=100)
        return integral_val
    vectorized_integrator = np.vectorize(_integrate_single_point, otypes=[float])
    return vectorized_integrator(log_M_bh1_array, q_array, z_array)


cosmo = Cosmology(H0=H0_DEFAULT, Omega_M=OMEGA_M_DEFAULT, Omega_L=OMEGA_L_DEFAULT)

__all__ = [
    'Cosmology',
    'calculate_snr_for_event_IMR',
    'sigmoid_p_det',
    'event_generator_smbh_intrinsic_and_observable',
    'load_or_generate_lisa_events',
    'perform_dual_validation',
    'perform_dual_validation_upgraded',
    'plot_lisa_events_showcase',
    'plot_lisa_selection_effects',
    'plot_lisa_square_layout',
    'plot_publication_quality',
    'get_smbh_merger_rate_q_dependent',
    'get_smbh_merger_rate_q_dependent_vectorized',
    'cosmo',
]

