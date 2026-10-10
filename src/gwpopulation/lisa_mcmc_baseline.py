from __future__ import annotations

import os
import random
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import trapz
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm

from .lisa import (
    Cosmology,
    calculate_snr_for_event_IMR,
    get_m_bh_relation_params,
    get_smbh_merger_rate_q_dependent,
    schechter_gsmf,
    sigmoid_p_det,
)


H0_DEFAULT = 67.4

OMEGA_M_DEFAULT = 0.315

OMEGA_L_DEFAULT = 0.685

C_LIGHT_KM_S = 299792.458

MPC_TO_CM = 3.086e+24

GYR_TO_S = 31540000.0 * 1000000000.0

YR_TO_S = 31540000.0

M_SUN_G = 1.989e+33

G_CONST_CGS = 6.6743e-08

cosmo = Cosmology(H0=H0_DEFAULT, Omega_M=OMEGA_M_DEFAULT, Omega_L=OMEGA_L_DEFAULT)

def galaxy_merger_rate_per_galaxy(log_M_star, z, gal_merger_params={'f0': 4.93, 'alpha0': 0.2, 'alpha1': -0.01, 'beta_z': 2.4}):
    M_star = 10 ** log_M_star
    rate_Gyr_inv = (1 + z) ** (gal_merger_params['beta_z'] - 1) * gal_merger_params['f0'] * (M_star / 100000000000.0) ** (gal_merger_params['alpha0'] + gal_merger_params['alpha1'] * (1 + z))
    return rate_Gyr_inv

def get_smbh_merger_rate(log_M_bh1_array, z_array, gal_merger_params={'f0': 4.93, 'alpha0': 0.2, 'alpha1': -0.01, 'beta_z': 2.4}):
    """
    计算超大质量黑洞（SMBH）的并合率。
    此函数现在调用的是平滑的GSMF和M_bh-M_star关系，因此其输出也是平滑的。
    """
    log_M_bh1_array = np.asarray(log_M_bh1_array)
    z_array = np.asarray(z_array)
    try:
        output_shape = np.broadcast(log_M_bh1_array, z_array).shape
    except ValueError:
        raise ValueError('log_M_bh1_array 和 z_array 的形状不兼容，无法广播。')
    results = np.zeros(output_shape, dtype=float)
    for multi_idx in np.ndindex(output_shape):
        if log_M_bh1_array.ndim > 0:
            log_M_bh1_val = log_M_bh1_array[multi_idx]
        else:
            log_M_bh1_val = log_M_bh1_array.item()
        if z_array.ndim > 0:
            z_val = z_array[multi_idx]
        else:
            z_val = z_array.item()
        a_z_val, b_z_val, epsilon_z_val = get_m_bh_relation_params(z_val)
        if epsilon_z_val <= 1e-10:
            integral_val = 0.0
        else:

            def integrand_all_terms(log_M_s_integrate):
                phi_gsmf_val = schechter_gsmf(log_M_s_integrate, z_val)
                R_gal_val = galaxy_merger_rate_per_galaxy(log_M_s_integrate, z_val, gal_merger_params)
                log_M_bh_mean_at_M_s = a_z_val + b_z_val * log_M_s_integrate
                prob_density_M_bh_given_M_s = scipy_norm.pdf(log_M_bh1_val, loc=log_M_bh_mean_at_M_s, scale=epsilon_z_val)
                return phi_gsmf_val * R_gal_val * prob_density_M_bh_given_M_s
            log_M_s_min_integrate, log_M_s_max_integrate = (7.0, 12.5)
            integral_val, _ = quad(integrand_all_terms, log_M_s_min_integrate, log_M_s_max_integrate, epsabs=1.49e-09, epsrel=1.49e-09)
        results[multi_idx] = integral_val
    return results

def event_generator_smbh(num_events_to_generate=2000, n_samples_per_event_posterior=500, sigma_log_M_posterior=0.1, sigma_z_posterior=0.05, rho_posterior=0.2, seed=None):
    """
    根据超大质量黑洞并合率的约束生成模拟事件。
    参数:
        num_events_to_generate (int): 期望生成的事件数量。
        n_samples_per_event_posterior (int): 每个事件的后验样本数量。
        sigma_log_M_posterior (float): 在参考SNR下的后验logM标准差 (基准值)。
        sigma_z_posterior (float): 在参考SNR下的后验z的标准差 (基准值)。
        rho_posterior (float): 后验logM和z之间的相关系数。
        seed (int): 随机数种子。
        
    返回:
        generated_events_array (np.ndarray): 生成的引力波事件真实参数 [logM, z]。
        posterior_samples_for_events_list (list): 每个事件的后验样本列表。
        pdf_values_on_grid：超大质量黑洞pdf网格值。
        log_pdf: 超大质量黑洞对数pdf网格值。
    """
    print('Running Event Generator (with SNR-Dependent Posteriors)...')
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    SNR_REF = 20.0
    SIGMA_LOG_M_BASE = sigma_log_M_posterior
    SIGMA_Z_BASE = sigma_z_posterior
    RHO_POSTERIOR = rho_posterior
    SIGMA_LOG_M_MIN = 0.01
    SIGMA_Z_MIN = 0.005

    def get_snr_dependent_covariance(snr):
        """内部辅助函数：根据SNR计算协方差矩阵"""
        if snr < 1.0:
            snr = 1.0
        scaling_factor = SNR_REF / snr
        sigma_log_m_scaled = SIGMA_LOG_M_BASE * scaling_factor
        sigma_z_scaled = SIGMA_Z_BASE * scaling_factor
        final_sigma_log_m = max(SIGMA_LOG_M_MIN, sigma_log_m_scaled)
        final_sigma_z = max(SIGMA_Z_MIN, sigma_z_scaled)
        cov_matrix = np.array([[final_sigma_log_m ** 2, RHO_POSTERIOR * final_sigma_log_m * final_sigma_z], [RHO_POSTERIOR * final_sigma_log_m * final_sigma_z, final_sigma_z ** 2]])
        return cov_matrix
    log_M_bh_min, log_M_bh_max = (5.0, 10.0)
    z_min, z_max = (0.21, 7.5)
    grid_points_M, grid_points_z = (50, 50)
    log_M_grid = np.linspace(log_M_bh_min, log_M_bh_max, grid_points_M)
    z_grid = np.linspace(z_min, z_max, grid_points_z)
    log_M_mesh, z_mesh = np.meshgrid(log_M_grid, z_grid)
    print('Pre-calculating PDF grid to find P_max...')
    rates_on_grid = get_smbh_merger_rate(log_M_mesh, z_mesh)
    dVc_dz_on_grid = cosmo.dV_c_dz_fullsky_Gpc3(z_mesh) * 1000000000.0
    pdf_values_on_grid = rates_on_grid * dVc_dz_on_grid
    pdf_values_on_grid[pdf_values_on_grid < 0] = 0
    pdf_values_on_grid[np.isnan(pdf_values_on_grid)] = 0
    P_max = np.max(pdf_values_on_grid)
    if P_max <= 0:
        print('Error: P_max <= 0. Cannot generate events.')
        return (np.array([]), [], pdf_values_on_grid, np.log10(pdf_values_on_grid + 1e-09))
    print(f'Starting Acceptance-Rejection sampling with P_max = {P_max:.4e}. Target: {num_events_to_generate} events...')
    true_event_params_list = []
    posterior_samples_for_events_list = []
    total_candidates = 0
    max_candidates = num_events_to_generate * 200
    pbar = tqdm(total=num_events_to_generate, desc='Generating Events')
    while len(true_event_params_list) < num_events_to_generate and total_candidates < max_candidates:
        total_candidates += 1
        log_M_cand_true = np.random.uniform(log_M_bh_min, log_M_bh_max)
        z_cand_true = np.random.uniform(z_min, z_max)
        '\n        rate_scalar = get_smbh_merger_rate(log_M_cand_true, z_cand_true).item()\n        if rate_scalar < 0: rate_scalar = 0.0\n        dVc_dz_scalar = cosmo.dV_c_dz_fullsky_Gpc3(z_cand_true) * 1e9\n        P_cand = rate_scalar * dVc_dz_scalar\n        '
        Rc_scalar = get_smbh_merger_rate(log_M_cand_true, z_cand_true).item()
        if Rc_scalar < 0:
            Rc_scalar = 0.0
        dVc_dz_scalar = cosmo.dV_c_dz_fullsky_Gpc3(z_cand_true) * 1000000000.0
        time_dilation_factor = 1.0 / (1.0 + z_cand_true)
        P_cand = Rc_scalar * dVc_dz_scalar * time_dilation_factor
        u = np.random.uniform(0, P_max)
        if u < P_cand:
            q_for_posterior = 1.0
            event_snr = calculate_snr_for_event_IMR(log_M_bh1_Msun=log_M_cand_true, z=z_cand_true, q=q_for_posterior)
            prob_of_detection = sigmoid_p_det(event_snr)
            if np.random.uniform(0, 1) < prob_of_detection:
                true_params_current_event = np.array([log_M_cand_true, z_cand_true])
                true_event_params_list.append(true_params_current_event)
                event_specific_cov = get_snr_dependent_covariance(event_snr)
                posterior_samples_current_event = multivariate_normal.rvs(mean=true_params_current_event, cov=event_specific_cov, size=n_samples_per_event_posterior, random_state=None)
                posterior_samples_current_event[:, 0] = np.clip(posterior_samples_current_event[:, 0], log_M_bh_min, log_M_bh_max)
                posterior_samples_current_event[:, 1] = np.clip(posterior_samples_current_event[:, 1], z_min, z_max)
                posterior_samples_for_events_list.append(posterior_samples_current_event)
                pbar.update(1)
    pbar.close()
    true_event_params_array = np.array(true_event_params_list)
    print(f'\nGeneration complete.')
    if total_candidates >= max_candidates and len(true_event_params_array) < num_events_to_generate:
        print(f'Warning: Reached max candidates ({max_candidates}). Generated {len(true_event_params_array)} events.')
    else:
        print(f'Generated {len(true_event_params_array)} events from {total_candidates} candidates.')
        acceptance_rate = len(true_event_params_array) / total_candidates if total_candidates > 0 else 0
        print(f'Overall acceptance rate: {acceptance_rate:.4f}')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    if len(true_event_params_array) > 0:
        sc = ax.scatter(true_event_params_array[:, 1], true_event_params_array[:, 0], alpha=0.6, s=20, cmap='viridis', c=true_event_params_array[:, 0])
        cb = plt.colorbar(sc, ax=ax, label='$log_{10}(M_{BH1} / M_{\\odot})$')
    log_pdf = np.log10(pdf_values_on_grid + 1e-09)
    log_pdf_max = np.max(log_pdf[np.isfinite(log_pdf)])
    contour_levels = np.linspace(log_pdf_max - 3, log_pdf_max, 7)
    contour = ax.contour(z_mesh, log_M_mesh, log_pdf, levels=contour_levels, cmap='Greys', alpha=0.7, linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.1f')
    print('Calculating SNR=8 contour for visualization...')
    snr_grid_values = np.zeros_like(log_M_mesh)
    q_for_contour = 1.0
    snr_contour_pbar = tqdm(total=log_M_mesh.size, desc='SNR Contour')
    for i in range(log_M_mesh.shape[0]):
        for j in range(log_M_mesh.shape[1]):
            snr_grid_values[i, j] = calculate_snr_for_event_IMR(log_M_bh1_Msun=log_M_mesh[i, j], z=z_mesh[i, j], q=q_for_contour)
            snr_contour_pbar.update(1)
    snr_contour_pbar.close()
    if np.any(snr_grid_values) and np.nanstd(snr_grid_values) > 1e-06:
        snr_contour_level = [8.0]
        contour_snr = ax.contour(z_mesh, log_M_mesh, snr_grid_values, levels=snr_contour_level, colors='cyan', linestyles='--', linewidths=2.0, alpha=0.9)
        ax.clabel(contour_snr, inline=True, fontsize=10, fmt=f'SNR={snr_contour_level[0]:.0f}')
    else:
        print('Warning: SNR grid values are constant, unable to draw SNR contour.')
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('$log_{10}(M_{BH} / M_{\\odot})$', fontsize=14)
    ax.set_title(f'Mock Events (N={len(true_event_params_array)}) with SNR-dependent Uncertainty', fontsize=16)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(log_M_bh_min, log_M_bh_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()
    return (true_event_params_array, posterior_samples_for_events_list, pdf_values_on_grid, log_pdf)

def build_and_load_4d_integral_proxy(grid_resolutions={'logM': 60, 'z': 60, 'a0': 40, 'a1': 40}, cache_filename='integral_proxy_4d_cache.npy', force_rebuild=False):
    """
    构建或加载一个物理上正确的四维代理器 (Surrogate 1)。
    代理器用于处理 (log_M_bh, z, alpha0, alpha1) 之间的依赖关系。
    """
    print('\nStep 2/7: Building/Loading the 4D integral surrogate (Surrogate 1)...')
    if os.path.exists(cache_filename) and (not force_rebuild):
        print(f"Found cache file. Loading pre-computed integral values from '{cache_filename}'...")
        integral_values_4d = np.load(cache_filename)
    else:
        print(f'Cache not found or rebuild forced. Starting pre-computation for the 4D proxy...')
        log_M_bh_grid = np.linspace(5.0, 10.0, grid_resolutions['logM'])
        z_grid = np.linspace(0.21, 7.5, grid_resolutions['z'])
        alpha0_grid = np.linspace(-1.0, 1.0, grid_resolutions['a0'])
        alpha1_grid = np.linspace(-0.5, 0.5, grid_resolutions['a1'])
        integral_values_4d = np.zeros((len(log_M_bh_grid), len(z_grid), len(alpha0_grid), len(alpha1_grid)))
        total_points = np.prod([len(g) for g in [log_M_bh_grid, z_grid, alpha0_grid, alpha1_grid]])
        pbar = tqdm(total=total_points, desc='Surrogate 1 (4D) pre-calculating')
        for i, log_M_bh_val in enumerate(log_M_bh_grid):
            for j, z_val in enumerate(z_grid):
                a_z, b_z, eps_z = get_m_bh_relation_params(z_val)
                if eps_z <= 1e-10:
                    pbar.update(len(alpha0_grid) * len(alpha1_grid))
                    continue
                for k, a0_val in enumerate(alpha0_grid):
                    for l, a1_val in enumerate(alpha1_grid):

                        def surrogate_integrand(log_M_s):
                            alpha_eff = a0_val + a1_val * (1 + z_val)
                            phi_gsmf = schechter_gsmf(log_M_s, z_val)
                            log_M_bh_mean = a_z + b_z * log_M_s
                            prob_dens = scipy_norm.pdf(log_M_bh_val, loc=log_M_bh_mean, scale=eps_z)
                            mass_term = (10 ** (log_M_s - 11.0)) ** alpha_eff
                            return phi_gsmf * prob_dens * mass_term
                        integral_val, _ = quad(surrogate_integrand, 7.0, 12.5, epsabs=1e-08, epsrel=1e-08)
                        integral_values_4d[i, j, k, l] = integral_val
                        pbar.update(1)
        pbar.close()
        print(f"Saving computed integral values to '{cache_filename}'...")
        np.save(cache_filename, integral_values_4d)
    log_M_bh_grid = np.linspace(5.0, 10.0, integral_values_4d.shape[0])
    z_grid = np.linspace(0.21, 7.5, integral_values_4d.shape[1])
    alpha0_grid = np.linspace(-1.0, 1.0, integral_values_4d.shape[2])
    alpha1_grid = np.linspace(-0.5, 0.5, integral_values_4d.shape[3])
    proxy = RegularGridInterpolator((log_M_bh_grid, z_grid, alpha0_grid, alpha1_grid), integral_values_4d, bounds_error=False, fill_value=0.0)
    print('Surrogate 1 Done!')
    return proxy

def log_likelihood_final_corrected(theta, data, proxies):
    """
    对数似然函数。依赖于`lambda_proxy`。
    """
    f0, alpha0, alpha1, beta_z = theta
    proxy_integral, proxy_lambda, proxy_snr, cosmo_obj = proxies
    T_obs_gyr = 4 * 1e-09
    log_M_bh_obs = data[:, 0]
    z_obs = data[:, 1]
    points_to_interpolate = np.vstack((log_M_bh_obs, z_obs, np.full_like(log_M_bh_obs, alpha0), np.full_like(log_M_bh_obs, alpha1))).T
    internal_integral_results = proxy_integral(points_to_interpolate)
    R_comoving_at_obs_gyr = f0 * (1 + z_obs) ** (beta_z - 1.0) * internal_integral_results
    dVc_dz_at_obs_mpc3 = cosmo_obj.dV_c_dz_fullsky_Gpc3(z_obs) * 1000000000.0
    rate_intrinsic_density_gyr = R_comoving_at_obs_gyr / (1 + z_obs) * dVc_dz_at_obs_mpc3
    rate_intrinsic_density_yr = rate_intrinsic_density_gyr / 1000000000.0
    snr_at_obs = proxy_snr(log_M_bh_obs, z_obs, grid=False)
    p_det_at_obs = sigmoid_p_det(snr_at_obs)
    rate_detectable_density_yr = rate_intrinsic_density_yr * p_det_at_obs
    if np.any(rate_detectable_density_yr <= 0):
        return -np.inf
    sum_log_R_detectable = np.sum(np.log(rate_detectable_density_yr))
    point_lambda = np.array([alpha0, alpha1, beta_z])
    lambda_integral_part = proxy_lambda(point_lambda)
    if lambda_integral_part <= 0:
        return -np.inf
    lambda_det = f0 * lambda_integral_part
    return sum_log_R_detectable - lambda_det * T_obs_gyr

def log_probability_corrected(theta, data, proxies):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_final_corrected(theta, data, proxies)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def log_prior(theta):
    f0, alpha0, alpha1, beta_z = theta
    if 0.0 < f0 < 20.0 and -1.0 < alpha0 < 1.0 and (-0.5 < alpha1 < 0.5) and (0.0 < beta_z < 6.0):
        return 0.0
    return -np.inf

def calculate_rate_density(params, log_M_mesh, z_mesh, cosmo_obj):
    """一个辅助函数，用于根据给定的参数计算物理模型的率密度。"""
    f0, alpha0, alpha1, beta_z = params
    gal_merger_p = {'f0': f0, 'alpha0': alpha0, 'alpha1': alpha1, 'beta_z': beta_z}
    smbh_merger_rate_Rc = get_smbh_merger_rate(log_M_mesh, z_mesh, gal_merger_params=gal_merger_p)
    dVc_dz = cosmo_obj.dV_c_dz_fullsky_Gpc3(z_mesh) * 1000000000.0
    time_dilation = 1.0 / (1.0 + z_mesh)
    rate_density = smbh_merger_rate_Rc * dVc_dz * time_dilation / 1000000000.0
    rate_density = np.asarray(rate_density)
    return np.where(np.isnan(rate_density) | (rate_density < 0), 0.0, rate_density)

def get_marginal_distributions(density_field, mask, logM_grid, z_grid):
    full_field_masked = np.zeros_like(density_field)
    full_field_masked[mask] = density_field[mask]
    norm_factor = np.sum(full_field_masked)
    if norm_factor == 0:
        norm_factor = 1.0
    full_field_norm = full_field_masked / norm_factor
    marginal_logM = np.sum(full_field_norm, axis=0)
    marginal_z = np.sum(full_field_norm, axis=1)
    return (marginal_logM, marginal_z)

def build_total_intrinsic_rate_integral_proxy(integral_proxy_4d):
    """
    构建一个3D代理器，输入 (alpha0, alpha1, beta_z)，输出总内禀率中
    除去f0外的积分部分。
    """
    alpha0_grid = np.linspace(-1.0, 1.0, 25)
    alpha1_grid = np.linspace(-0.5, 0.5, 25)
    beta_z_grid = np.linspace(0.0, 6.0, 25)
    log_M_min, log_M_max = (5.0, 10.0)
    z_min, z_max = (0.21, 7.5)
    log_M_int_grid = np.linspace(log_M_min, log_M_max, 100)
    z_int_grid = np.linspace(z_min, z_max, 100)
    log_M_mesh, z_mesh = np.meshgrid(log_M_int_grid, z_int_grid)
    dVc_dz_mesh = cosmo.dV_c_dz_fullsky_Gpc3(z_mesh) * 1000000000.0
    time_dilation_factor = 1.0 / (1.0 + z_mesh)
    integral_values = np.zeros((len(alpha0_grid), len(alpha1_grid), len(beta_z_grid)))
    pbar = tqdm(total=len(alpha0_grid) * len(alpha1_grid) * len(beta_z_grid), desc='总速率代理器构建')
    for i, a0 in enumerate(alpha0_grid):
        for j, a1 in enumerate(alpha1_grid):
            points_to_interpolate = np.stack([log_M_mesh.ravel(), z_mesh.ravel(), np.full_like(log_M_mesh.ravel(), a0), np.full_like(log_M_mesh.ravel(), a1)], axis=-1)
            internal_integral_mesh = integral_proxy_4d(points_to_interpolate).reshape(z_mesh.shape)
            for k, bz in enumerate(beta_z_grid):
                integrand = (1 + z_mesh) ** (bz - 1.0) * internal_integral_mesh * dVc_dz_mesh * time_dilation_factor / 1000000000.0
                integral_over_logM = trapz(integrand, log_M_int_grid, axis=1)
                final_integral_val = trapz(integral_over_logM, z_int_grid, axis=0)
                integral_values[i, j, k] = final_integral_val
                pbar.update(1)
    pbar.close()
    proxy = RegularGridInterpolator((alpha0_grid, alpha1_grid, beta_z_grid), integral_values, bounds_error=False, fill_value=0.0)
    return proxy

def get_summary_stats(samples, name):
    mean = np.mean(samples)
    median = np.median(samples)
    std = np.std(samples)
    p16, p84 = np.percentile(samples, [16, 84])
    return {'Model': name, 'Mean': f'{mean:.2f}', 'Median': f'{median:.2f}', 'Std. Dev.': f'{std:.2f}', '68% Credible Interval': f'[{p16:.2f}, {p84:.2f}]'}

def alternative_rate_density(log_M_mesh, z_mesh, params=None):
    """
    Alternative physics model: Bimodal Mass + Cosmic Noon Peak
    
    This model has structure that the parameterized MCMC CANNOT capture:
    1. Bimodal mass distribution (two Gaussian peaks)
    2. Non-monotonic redshift evolution (peak + decline)
    
    Parameters (can be customized):
        R0: Overall normalization (rate scale)
        mu1, sigma1, w1: Light seed peak parameters
        mu2, sigma2: Heavy seed peak parameters (w2 = 1 - w1)
        gamma: Low-z power-law slope
        z_peak: Redshift of maximum rate
        z_width: Width of the redshift peak
    """
    if params is None:
        params = {'R0': 50.0, 'mu1': 5.5, 'sigma1': 0.5, 'w1': 0.6, 'mu2': 7.5, 'sigma2': 0.6, 'gamma': 2.0, 'z_peak': 2.5, 'z_width': 1.5}
    R0 = params['R0']
    mu1, sigma1, w1 = (params['mu1'], params['sigma1'], params['w1'])
    mu2, sigma2 = (params['mu2'], params['sigma2'])
    w2 = 1.0 - w1
    gamma = params['gamma']
    z_peak, z_width = (params['z_peak'], params['z_width'])
    gauss1 = np.exp(-0.5 * ((log_M_mesh - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
    gauss2 = np.exp(-0.5 * ((log_M_mesh - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
    mass_term = w1 * gauss1 + w2 * gauss2
    z_evolution = (1 + z_mesh) ** gamma * np.exp(-((z_mesh - z_peak) / z_width) ** 2)
    dVc_dz = cosmo.dV_c_dz_fullsky_Gpc3(z_mesh) * 1000000000.0
    time_dilation = 1.0 / (1.0 + z_mesh)
    rate_density = R0 * mass_term * z_evolution * dVc_dz * time_dilation / 1000000000.0
    rate_density = np.asarray(rate_density)
    return np.where(np.isnan(rate_density) | (rate_density < 0), 0.0, rate_density)

def event_generator_alternative(num_events_to_generate=400, n_samples_per_event_posterior=1000, sigma_log_M_posterior=0.1, sigma_z_posterior=0.05, rho_posterior=0.2, seed=None, snr_grid_shape=(20, 20), show_plot=False):
    """
    Generate events from the alternative physics model with LISA selection effects.
    """
    print('Running Alternative Model Event Generator...')
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    SNR_REF = 20.0
    SIGMA_LOG_M_BASE = sigma_log_M_posterior
    SIGMA_Z_BASE = sigma_z_posterior
    RHO_POSTERIOR = rho_posterior
    SIGMA_LOG_M_MIN = 0.01
    SIGMA_Z_MIN = 0.005

    def get_snr_dependent_covariance(snr):
        if snr < 1.0:
            snr = 1.0
        scaling_factor = SNR_REF / snr
        sigma_log_m_scaled = SIGMA_LOG_M_BASE * scaling_factor
        sigma_z_scaled = SIGMA_Z_BASE * scaling_factor
        final_sigma_log_m = max(SIGMA_LOG_M_MIN, sigma_log_m_scaled)
        final_sigma_z = max(SIGMA_Z_MIN, sigma_z_scaled)
        cov_matrix = np.array([[final_sigma_log_m ** 2, RHO_POSTERIOR * final_sigma_log_m * final_sigma_z], [RHO_POSTERIOR * final_sigma_log_m * final_sigma_z, final_sigma_z ** 2]])
        return cov_matrix
    log_M_bh_min, log_M_bh_max = (5.0, 10.0)
    z_min, z_max = (0.21, 7.5)
    grid_points_M, grid_points_z = (80, 80)
    log_M_grid = np.linspace(log_M_bh_min, log_M_bh_max, grid_points_M)
    z_grid = np.linspace(z_min, z_max, grid_points_z)
    log_M_mesh_gen, z_mesh_gen = np.meshgrid(log_M_grid, z_grid)
    print('Pre-calculating Alternative Model PDF grid...')
    pdf_values_on_grid = alternative_rate_density(log_M_mesh_gen, z_mesh_gen)
    P_max = np.max(pdf_values_on_grid)
    if P_max <= 0:
        print('Error: P_max <= 0 for alternative model. Cannot generate events.')
        return (np.array([]), [], pdf_values_on_grid, np.log10(pdf_values_on_grid + 1e-09))
    snr_log_m_grid = np.linspace(log_M_bh_min, log_M_bh_max, snr_grid_shape[0])
    snr_z_grid = np.linspace(z_min, z_max, snr_grid_shape[1])
    snr_values = np.zeros((len(snr_log_m_grid), len(snr_z_grid)))
    for i, log_m in enumerate(snr_log_m_grid):
        for j, z_val in enumerate(snr_z_grid):
            snr_values[i, j] = calculate_snr_for_event_IMR(log_m, z_val, q=1.0)
    snr_proxy = RectBivariateSpline(snr_log_m_grid, snr_z_grid, snr_values)
    print(f'P_max = {P_max:.4e}. Generating {num_events_to_generate} events...')
    true_event_params_list = []
    posterior_samples_for_events_list = []
    total_candidates = 0
    max_candidates = num_events_to_generate * 300
    pbar = tqdm(total=num_events_to_generate, desc='Generating Alt Model Events')
    while len(true_event_params_list) < num_events_to_generate and total_candidates < max_candidates:
        total_candidates += 1
        log_M_cand = np.random.uniform(log_M_bh_min, log_M_bh_max)
        z_cand = np.random.uniform(z_min, z_max)
        P_cand = alternative_rate_density(np.array([[log_M_cand]]), np.array([[z_cand]]))[0, 0]
        u = np.random.uniform(0, P_max)
        if u < P_cand:
            event_snr = snr_proxy(log_M_cand, z_cand, grid=False).item()
            prob_of_detection = sigmoid_p_det(event_snr)
            if np.random.uniform(0, 1) < prob_of_detection:
                true_params = np.array([log_M_cand, z_cand])
                true_event_params_list.append(true_params)
                event_specific_cov = get_snr_dependent_covariance(event_snr)
                posterior_samples = multivariate_normal.rvs(mean=true_params, cov=event_specific_cov, size=n_samples_per_event_posterior)
                posterior_samples[:, 0] = np.clip(posterior_samples[:, 0], log_M_bh_min, log_M_bh_max)
                posterior_samples[:, 1] = np.clip(posterior_samples[:, 1], z_min, z_max)
                posterior_samples_for_events_list.append(posterior_samples)
                pbar.update(1)
    pbar.close()
    true_event_params_array = np.array(true_event_params_list)
    print(f'\nGenerated {len(true_event_params_array)} events from alternative model.')
    if show_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        if len(true_event_params_array) > 0:
            sc = ax.scatter(true_event_params_array[:, 1], true_event_params_array[:, 0], alpha=0.6, s=20, cmap='plasma', c=true_event_params_array[:, 0])
            plt.colorbar(sc, ax=ax, label='$log_{10}(M_{BH} / M_{\\odot})$')
        log_pdf = np.log10(pdf_values_on_grid + 1e-09)
        log_pdf_max = np.max(log_pdf[np.isfinite(log_pdf)])
        contour_levels = np.linspace(log_pdf_max - 3, log_pdf_max, 7)
        contour = ax.contour(z_mesh_gen, log_M_mesh_gen, log_pdf, levels=contour_levels, cmap='Greys', alpha=0.7, linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=9, fmt='%.1f')
        snr_on_grid = snr_proxy(log_M_mesh_gen, z_mesh_gen, grid=False)
        ax.contour(z_mesh_gen, log_M_mesh_gen, snr_on_grid, levels=[8.0], colors='cyan', linestyles='--', linewidths=2.0)
        ax.set_xlabel('Redshift (z)', fontsize=14)
        ax.set_ylabel('$log_{10}(M_{BH} / M_{\\odot})$', fontsize=14)
        ax.set_title(f'Alternative Model Events (N={len(true_event_params_array)}): Bimodal Mass + Cosmic Noon Peak', fontsize=14)
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(log_M_bh_min, log_M_bh_max)
        plt.tight_layout()
        plt.show()
    log_pdf = np.log10(pdf_values_on_grid + 1e-09)
    return (true_event_params_array, posterior_samples_for_events_list, pdf_values_on_grid, log_pdf)

def normalize_field(field):
    """归一化到[0,1]范围用于形状比较"""
    field_min = np.min(field)
    field_max = np.max(field)
    if field_max - field_min > 1e-10:
        return (field - field_min) / (field_max - field_min)
    return np.zeros_like(field)

def plot_density_alt(ax, data, mask, title, cmap='viridis'):
    masked_data = np.where(mask, data, np.nan)
    im = ax.pcolormesh(z_eval_mesh_alt, log_M_eval_mesh_alt, masked_data, shading='auto', cmap=cmap, vmin=0, vmax=vmax_alt)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Redshift (z)', fontsize=11)
    ax.set_ylabel('$log_{10}(M_{BH} / M_{\\odot})$', fontsize=11)
    ax.contour(z_eval_mesh_alt, log_M_eval_mesh_alt, snr_on_eval_grid_alt, levels=[8.0], colors='red', linestyles='--', linewidths=1.5)
    return im

def get_marginal_1d(density_field, mask, axis):
    masked = np.where(mask, density_field, 0)
    marginal = np.sum(masked, axis=axis)
    if np.sum(marginal) > 0:
        marginal = marginal / np.sum(marginal)
    return marginal


__all__ = [
    'build_and_load_4d_integral_proxy',
    'log_likelihood_final_corrected',
    'log_probability_corrected',
    'log_prior',
    'calculate_rate_density',
    'get_marginal_distributions',
    'build_total_intrinsic_rate_integral_proxy',
    'get_summary_stats',
    'alternative_rate_density',
    'event_generator_alternative',
    'event_generator_smbh',
    'normalize_field',
    'plot_density_alt',
    'get_marginal_1d',
    'cosmo',
]

