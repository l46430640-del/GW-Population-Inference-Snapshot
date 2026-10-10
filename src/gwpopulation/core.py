from __future__ import annotations

from pathlib import Path
from numbers import Number
from typing import Any
import hashlib
import sys
import traceback

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import torch.nn as nn
from pyro.distributions import constraints
from pyro.distributions.distribution import Distribution as TorchDistribution
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from torch.distributions.utils import broadcast_all
from torch.distributions.transforms import CorrCholeskyTransform
from tqdm.auto import trange
from .rate import conditional_rate_draws


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_standard_normal = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))


class NumericalStabilityError(RuntimeError):
    """Raised when training or inference produces non-finite intermediate values."""


def _tensor_stats_dict(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    stats: dict[str, Any] = {
        'shape': tuple(detached.shape),
        'numel': int(detached.numel()),
        'num_finite': int(finite.sum().item()),
    }
    if finite.any():
        finite_vals = detached[finite]
        stats.update(
            {
                'min': float(finite_vals.min().item()),
                'max': float(finite_vals.max().item()),
                'mean': float(finite_vals.mean().item()),
                'std': float(finite_vals.std().item()) if finite_vals.numel() > 1 else 0.0,
            }
        )
    return stats

def compute_neff(weights):
    """
    计算有效样本数 (Effective Sample Size, Neff)
    
    参数:
        weights: 形状为 (N,) 或 (num_events, num_samples_per_event) 的张量
                 代表重要性采样权重 (可以是未归一化的)
    
    返回:
        neff: 标量张量，有效样本数
        
    公式: Neff = (sum(w))^2 / sum(w^2)
    """
    weights = torch.clamp(weights, min=1e-100)
    sum_w = torch.sum(weights, dim=-1)
    sum_w2 = torch.sum(weights ** 2, dim=-1)
    neff = sum_w ** 2 / (sum_w2 + 1e-100)
    return neff

class TruncatedNormal(TorchDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'low': constraints.real, 'high': constraints.real}
    has_rsample = True

    def __init__(self, loc, scale, low=0.0, high=1.0, validate_args=None):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        self._validate_args = validate_args
        if (self.low >= self.high).any():
            raise ValueError('`low` must be smaller than `high`.')
        self.low_std = (self.low - self.loc) / self.scale
        self.high_std = (self.high - self.loc) / self.scale
        self.log_Z = torch.log(_standard_normal.cdf(self.high_std) - _standard_normal.cdf(self.low_std) + 1e-06)
        self.Z = _standard_normal.cdf(self.high_std) - _standard_normal.cdf(self.low_std) + 1e-08
        self.log_Z = self.Z.log()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p_low = _standard_normal.cdf(self.low_std)
        p_high = _standard_normal.cdf(self.high_std)
        u = p_low + (p_high - p_low) * torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        u = u.clamp(min=1e-07, max=1 - 1e-07)
        x_std = _standard_normal.icdf(u)
        return self.loc + self.scale * x_std

    def log_prob(self, value):
        is_in_bounds = (value >= self.low) & (value <= self.high)
        value_std = (value - self.loc) / self.scale
        log_phi = _standard_normal.log_prob(value_std) - torch.log(self.scale)
        log_p = log_phi - self.log_Z
        return torch.where(is_in_bounds, log_p, -torch.inf)

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def event_shape(self):
        return torch.Size()

    def cdf(self, value):
        """计算累积分布函数 (CDF)。"""
        is_in_bounds = (value >= self.low) & (value <= self.high)
        value_std = (value - self.loc) / self.scale
        cdf_val = _standard_normal.cdf(value_std)
        cdf_low = _standard_normal.cdf(self.low_std)
        normalized_cdf = (cdf_val - cdf_low) / self.Z
        return torch.where(is_in_bounds, normalized_cdf.clamp(0.0, 1.0), torch.where(value < self.low, 0.0, 1.0))

    def icdf(self, value):
        """计算逆累积分布函数 (Inverse CDF)。"""
        p_low = _standard_normal.cdf(self.low_std)
        u = value * self.Z + p_low
        u = u.clamp(min=1e-07, max=1 - 1e-07)
        x_std = _standard_normal.icdf(u)
        return self.loc + self.scale * x_std

class TruncatedPareto(TorchDistribution):
    arg_constraints = {'alpha': constraints.positive, 'low': constraints.positive, 'high': constraints.positive}
    has_rsample = True

    def __init__(self, alpha, low=0.0001, high=1.0, validate_args=None):
        self.alpha, self.low, self.high = broadcast_all(alpha, low, high)
        self._validate_args = validate_args
        if (self.low >= self.high).any():
            raise ValueError('`low` must be smaller than `high`.')
        self.cdf_high = 1 - (self.low / self.high) ** self.alpha
        self.log_Z = torch.log(self.cdf_high + 1e-06)
        self.Z = 1 - (self.low / self.high) ** self.alpha + 1e-08
        self.log_Z = self.Z.log()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u_prime = torch.rand(shape, dtype=self.alpha.dtype, device=self.alpha.device)
        u = u_prime * self.cdf_high
        return self.low / torch.pow(1.0 - u, 1.0 / self.alpha)

    def log_prob(self, value):
        is_in_bounds = (value >= self.low) & (value <= self.high)
        unnormalized_log_prob = torch.log(self.alpha) + self.alpha * torch.log(self.low) - (self.alpha + 1) * torch.log(value)
        log_p = unnormalized_log_prob - self.log_Z
        return torch.where(is_in_bounds, log_p, -torch.inf)

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape

    @property
    def batch_shape(self):
        return self.alpha.shape

    @property
    def event_shape(self):
        return torch.Size()

    def cdf(self, value):
        """计算累积分布函数 (CDF)。"""
        is_in_bounds = (value >= self.low) & (value <= self.high)
        cdf_val = 1 - (self.low / value) ** self.alpha
        normalized_cdf = cdf_val / self.Z
        return torch.where(is_in_bounds, normalized_cdf.clamp(0.0, 1.0), torch.where(value < self.low, 0.0, 1.0))

    def icdf(self, value):
        """计算逆累积分布函数 (Inverse CDF)。"""
        u = value * self.Z
        return self.low / torch.pow(1.0 - u, 1.0 / self.alpha)

class BoundedLogitNormal(TorchDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self._validate_args = validate_args

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value.clamp(1e-09, 1.0 - 1e-09)
        y = torch.logit(x)
        log_jac = torch.log(x) + torch.log1p(-x)
        log_prob = _standard_normal.log_prob((y - self.loc) / self.scale) - torch.log(self.scale) - log_jac
        return torch.where((value >= 0.0) & (value <= 1.0), log_prob, -torch.inf)

    def cdf(self, value):
        x = value.clamp(1e-09, 1.0 - 1e-09)
        y = torch.logit(x)
        cdf = _standard_normal.cdf((y - self.loc) / self.scale)
        return torch.where(value <= 0.0, torch.zeros_like(cdf), torch.where(value >= 1.0, torch.ones_like(cdf), cdf))

    def icdf(self, value):
        p = value.clamp(1e-09, 1.0 - 1e-09)
        y = self.loc + self.scale * _standard_normal.icdf(p)
        return torch.sigmoid(y).clamp(1e-09, 1.0 - 1e-09)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        y = self.loc + self.scale * torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return torch.sigmoid(y).clamp(1e-09, 1.0 - 1e-09)

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def event_shape(self):
        return torch.Size()

class BoundedBernsteinMixture(TorchDistribution):
    arg_constraints = {'logits': constraints.real}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, logits, validate_args=None):
        if logits.shape[-1] < 2:
            raise ValueError('BoundedBernsteinMixture requires at least two basis functions.')
        self.logits = logits
        self.num_basis = int(logits.shape[-1])
        self._validate_args = validate_args

    @property
    def batch_shape(self):
        return self.logits.shape[:-1]

    @property
    def event_shape(self):
        return torch.Size()

    def _basis_parameters(self, device, dtype):
        idx = torch.arange(self.num_basis, device=device, dtype=dtype)
        alpha = idx + 1.0
        beta = self.num_basis - idx
        return alpha, beta

    def _log_beta_basis_pdf(self, x):
        x_basis = x.unsqueeze(-1)
        alpha, beta = self._basis_parameters(x.device, x.dtype)
        log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        return (
            (alpha - 1.0) * torch.log(x_basis)
            + (beta - 1.0) * torch.log1p(-x_basis)
            - log_norm
        )

    def _beta_basis_cdf(self, x):
        x_basis = x.unsqueeze(-1)
        log_x = torch.log(x_basis)
        log_1mx = torch.log1p(-x_basis)
        total_order = self.num_basis
        j_all = torch.arange(total_order + 1, device=x.device, dtype=x.dtype)
        total_order_tensor = torch.tensor(float(total_order), device=x.device, dtype=x.dtype)
        log_binom = (
            torch.lgamma(total_order_tensor + 1.0)
            - torch.lgamma(j_all + 1.0)
            - torch.lgamma(total_order_tensor - j_all + 1.0)
        )
        basis_cdfs = []
        for k in range(self.num_basis):
            j = j_all[k + 1:]
            terms = log_binom[k + 1:] + j * log_x + (total_order_tensor - j) * log_1mx
            basis_cdfs.append(torch.exp(torch.logsumexp(terms, dim=-1, keepdim=True)))
        return torch.cat(basis_cdfs, dim=-1).clamp(0.0, 1.0)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        in_bounds = (value >= 0.0) & (value <= 1.0)
        x = value.clamp(1e-09, 1.0 - 1e-09)
        log_weights = torch.log_softmax(self.logits, dim=-1)
        log_basis = self._log_beta_basis_pdf(x)
        log_prob = torch.logsumexp(log_weights + log_basis, dim=-1)
        return torch.where(in_bounds, log_prob, -torch.inf)

    def cdf(self, value):
        x = value.clamp(1e-09, 1.0 - 1e-09)
        weights = torch.softmax(self.logits, dim=-1)
        basis_cdf = self._beta_basis_cdf(x)
        cdf = torch.sum(weights * basis_cdf, dim=-1)
        return torch.where(value <= 0.0, torch.zeros_like(cdf), torch.where(value >= 1.0, torch.ones_like(cdf), cdf.clamp(0.0, 1.0)))

    def icdf(self, value):
        p = value.clamp(1e-09, 1.0 - 1e-09)
        low = torch.zeros_like(p)
        high = torch.ones_like(p)
        for _ in range(48):
            mid = 0.5 * (low + high)
            cdf_mid = self.cdf(mid)
            low = torch.where(cdf_mid < p, mid, low)
            high = torch.where(cdf_mid < p, high, mid)
        return (0.5 * (low + high)).clamp(1e-09, 1.0 - 1e-09)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        return self.icdf(u)

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape

class BoundedMonotoneSpline(TorchDistribution):
    arg_constraints = {'logits': constraints.real}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(
        self,
        logits,
        validate_args=None,
        coordinate='log10',
        log10_min=None,
        log10_max=None,
        logit_epsilon=0.02,
    ):
        if logits.shape[-1] < 3:
            raise ValueError('BoundedMonotoneSpline requires at least three knots.')
        self.logits = logits
        self.num_knots = int(logits.shape[-1])
        self.num_bins = self.num_knots - 1
        self._validate_args = validate_args
        self.coordinate = str(coordinate or 'log10').lower()
        self.logit_epsilon = float(logit_epsilon)
        self.log10_min = None if log10_min is None else torch.as_tensor(log10_min, dtype=logits.dtype, device=logits.device)
        self.log10_max = None if log10_max is None else torch.as_tensor(log10_max, dtype=logits.dtype, device=logits.device)
        if self.coordinate not in ('log10', 'normalized', 'identity', 'm1', 'm1_logit', 'shifted_logit_m1'):
            raise ValueError(f'Unsupported BoundedMonotoneSpline coordinate: {coordinate}')
        if self.coordinate in ('m1', 'm1_logit', 'shifted_logit_m1') and (self.log10_min is None or self.log10_max is None):
            raise ValueError('m1-coordinate BoundedMonotoneSpline requires log10_min and log10_max.')

    @property
    def batch_shape(self):
        return self.logits.shape[:-1]

    @property
    def event_shape(self):
        return torch.Size()

    def _normalized_heights(self):
        heights = torch.nn.functional.softplus(self.logits) + 1.0e-04
        dx = 1.0 / float(self.num_bins)
        area = dx * (0.5 * heights[..., 0] + heights[..., 1:-1].sum(dim=-1) + 0.5 * heights[..., -1])
        return heights / area.unsqueeze(-1).clamp_min(1.0e-12)

    def _bin_tensors(self, value):
        x = value.clamp(1.0e-09, 1.0 - 1.0e-09)
        dx = torch.as_tensor(1.0 / float(self.num_bins), dtype=x.dtype, device=x.device)
        left = torch.arange(self.num_bins, dtype=x.dtype, device=x.device) * dx
        right = left + dx
        x_expanded = x.unsqueeze(-1)
        in_bin = (x_expanded >= left) & (x_expanded < right)
        in_bin = torch.where((x_expanded >= left[-1]) & (x_expanded <= right[-1]), torch.nn.functional.one_hot(torch.full_like(x, self.num_bins - 1, dtype=torch.long), num_classes=self.num_bins).bool(), in_bin)
        local = ((x_expanded - left) / dx).clamp(0.0, 1.0)
        return x, dx, left, in_bin, local

    def _to_basis_coordinate(self, value):
        x = value.clamp(1.0e-09, 1.0 - 1.0e-09)
        if self.coordinate in ('log10', 'normalized', 'identity'):
            return x, torch.zeros_like(x)
        log_range = (self.log10_max - self.log10_min).clamp_min(1.0e-12)
        log_m = self.log10_min + x * log_range
        ten = torch.as_tensor(10.0, dtype=x.dtype, device=x.device)
        mass = torch.pow(ten, log_m)
        mass_min = torch.pow(ten, self.log10_min)
        mass_max = torch.pow(ten, self.log10_max)
        mass_range = (mass_max - mass_min).clamp_min(1.0e-12)
        y = ((mass - mass_min) / mass_range).clamp(1.0e-09, 1.0 - 1.0e-09)
        dy_dx = (torch.log(ten) * log_range * mass / mass_range).clamp_min(1.0e-30)
        if self.coordinate == 'm1':
            return y, torch.log(dy_dx)
        eps = min(max(self.logit_epsilon, 1.0e-06), 0.49)
        eps_t = torch.as_tensor(eps, dtype=x.dtype, device=x.device)
        one = torch.as_tensor(1.0, dtype=x.dtype, device=x.device)
        y_eps = (eps_t + (one - 2.0 * eps_t) * y).clamp(eps_t, one - eps_t)
        z_min = torch.logit(eps_t)
        z_max = torch.logit(one - eps_t)
        z_range = (z_max - z_min).clamp_min(1.0e-12)
        t = ((torch.logit(y_eps) - z_min) / z_range).clamp(1.0e-09, 1.0 - 1.0e-09)
        dt_dy = ((one - 2.0 * eps_t) / (y_eps * (one - y_eps)) / z_range).clamp_min(1.0e-30)
        return t, torch.log(dt_dy) + torch.log(dy_dx)

    def _from_basis_coordinate(self, basis_value):
        t = basis_value.clamp(1.0e-09, 1.0 - 1.0e-09)
        if self.coordinate in ('log10', 'normalized', 'identity'):
            return t
        ten = torch.as_tensor(10.0, dtype=t.dtype, device=t.device)
        mass_min = torch.pow(ten, self.log10_min)
        mass_max = torch.pow(ten, self.log10_max)
        mass_range = (mass_max - mass_min).clamp_min(1.0e-12)
        log_range = (self.log10_max - self.log10_min).clamp_min(1.0e-12)
        if self.coordinate == 'm1':
            y = t
        else:
            eps = min(max(self.logit_epsilon, 1.0e-06), 0.49)
            eps_t = torch.as_tensor(eps, dtype=t.dtype, device=t.device)
            one = torch.as_tensor(1.0, dtype=t.dtype, device=t.device)
            z_min = torch.logit(eps_t)
            z_max = torch.logit(one - eps_t)
            y_eps = torch.sigmoid(z_min + t * (z_max - z_min))
            y = ((y_eps - eps_t) / (one - 2.0 * eps_t)).clamp(1.0e-09, 1.0 - 1.0e-09)
        mass = (mass_min + y * mass_range).clamp_min(1.0e-30)
        return ((torch.log10(mass) - self.log10_min) / log_range).clamp(1.0e-09, 1.0 - 1.0e-09)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        in_support = (value >= 0.0) & (value <= 1.0)
        basis_value, log_dt_dx = self._to_basis_coordinate(value)
        heights = self._normalized_heights()
        x, _ = torch.broadcast_tensors(basis_value, heights[..., 0])
        position = x * self.num_bins
        index = position.long().clamp(0, self.num_bins - 1)
        expanded = heights.expand(*x.shape, self.num_knots)
        h0 = expanded.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        h1 = expanded.gather(-1, (index + 1).unsqueeze(-1)).squeeze(-1)
        density = (h0 + (h1 - h0) * (position - index)).clamp_min(1.0e-12)
        return torch.where(in_support, torch.log(density) + log_dt_dx, -torch.inf)

    def cdf(self, value):
        basis_value, _ = self._to_basis_coordinate(value)
        heights = self._normalized_heights()
        x, _ = torch.broadcast_tensors(basis_value, heights[..., 0])
        position = x * self.num_bins
        index = position.long().clamp(0, self.num_bins - 1)
        local = position - index
        dx = 1.0 / self.num_bins
        bin_area = dx * 0.5 * (heights[..., :-1] + heights[..., 1:])
        zero = torch.zeros_like(bin_area[..., :1])
        cum_left = torch.cat([zero, torch.cumsum(bin_area, dim=-1)[..., :-1]], dim=-1)
        expanded = heights.expand(*x.shape, self.num_knots)
        h0 = expanded.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        h1 = expanded.gather(-1, (index + 1).unsqueeze(-1)).squeeze(-1)
        left = cum_left.expand(*x.shape, self.num_bins).gather(-1, index.unsqueeze(-1)).squeeze(-1)
        cdf = (left + dx * (h0 * local + 0.5 * (h1 - h0) * local.square())).clamp(0.0, 1.0)
        return torch.where(value <= 0.0, torch.zeros_like(cdf), torch.where(value >= 1.0, torch.ones_like(cdf), cdf))

    def icdf(self, value):
        p = value.clamp(1.0e-09, 1.0 - 1.0e-09)
        heights = self._normalized_heights()
        h0 = heights[..., :-1]
        h1 = heights[..., 1:]
        dx = torch.as_tensor(1.0 / float(self.num_bins), dtype=p.dtype, device=p.device)
        left = torch.arange(self.num_bins, dtype=p.dtype, device=p.device) * dx
        bin_area = dx * 0.5 * (h0 + h1)
        cum_edges = torch.cat([torch.zeros_like(bin_area[..., :1]), torch.cumsum(bin_area, dim=-1)], dim=-1)
        p_expanded = p.unsqueeze(-1)
        in_bin = (p_expanded >= cum_edges[..., :-1]) & (p_expanded < cum_edges[..., 1:])
        in_bin = torch.where((p_expanded >= cum_edges[..., -2:-1]) & (p_expanded <= cum_edges[..., -1:]), torch.nn.functional.one_hot(torch.full_like(p, self.num_bins - 1, dtype=torch.long), num_classes=self.num_bins).bool(), in_bin)
        target = (p_expanded - cum_edges[..., :-1]).clamp_min(0.0)
        a = 0.5 * dx * (h1 - h0)
        b = dx * h0
        disc = (b * b + 4.0 * a * target).clamp_min(1.0e-18)
        safe_denom = torch.where(torch.abs(a) < 1.0e-10, torch.ones_like(a), 2.0 * a)
        local_quad = (-b + torch.sqrt(disc)) / safe_denom
        local_linear = target / b.clamp_min(1.0e-12)
        local = torch.where(torch.abs(a) < 1.0e-10, local_linear, local_quad).clamp(0.0, 1.0)
        x_bins = left + dx * local
        x = torch.sum(torch.where(in_bin, x_bins, torch.zeros_like(x_bins)), dim=-1)
        return self._from_basis_coordinate(x.clamp(1.0e-09, 1.0 - 1.0e-09))

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        return self.icdf(u)

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape


class BoundedTwoResolutionSpline(TorchDistribution):
    arg_constraints = {
        'global_logits': constraints.real,
        'local_logits': constraints.real,
        'local_weight_logit': constraints.real,
    }
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, global_logits, local_logits, local_weight_logit, local_low, local_high, validate_args=None):
        if global_logits.shape[-1] < 3 or local_logits.shape[-1] < 3:
            raise ValueError('BoundedTwoResolutionSpline requires at least three knots in both global and local bases.')
        self.global_logits = global_logits
        self.local_logits = local_logits
        self.local_weight_logit = local_weight_logit
        self._validate_args = validate_args
        self.global_dist = BoundedMonotoneSpline(global_logits)
        self.local_dist = BoundedMonotoneSpline(local_logits)
        self.local_low = torch.as_tensor(local_low, dtype=global_logits.dtype, device=global_logits.device)
        self.local_high = torch.as_tensor(local_high, dtype=global_logits.dtype, device=global_logits.device)
        if torch.any(self.local_low >= self.local_high):
            raise ValueError('BoundedTwoResolutionSpline local_low must be smaller than local_high.')
        self.local_width = (self.local_high - self.local_low).clamp_min(1.0e-12)
        self._batch_shape = torch.broadcast_shapes(
            self.global_dist.batch_shape,
            self.local_dist.batch_shape,
            self.local_weight_logit.shape,
        )

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return torch.Size()

    def _local_coordinate(self, value):
        return ((value - self.local_low) / self.local_width).clamp(1.0e-09, 1.0 - 1.0e-09)

    def _log_mix_weights(self):
        log_w_local = -torch.nn.functional.softplus(-self.local_weight_logit)
        log_w_global = -torch.nn.functional.softplus(self.local_weight_logit)
        return log_w_global, log_w_local

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        in_support = (value >= 0.0) & (value <= 1.0)
        log_w_global, log_w_local = self._log_mix_weights()
        global_log_prob = self.global_dist.log_prob(value)
        local_x = self._local_coordinate(value)
        local_log_prob = self.local_dist.log_prob(local_x) - torch.log(self.local_width)
        in_local = (value >= self.local_low) & (value <= self.local_high)
        local_log_prob = torch.where(in_local, local_log_prob, torch.full_like(local_log_prob, -torch.inf))
        log_prob = torch.logaddexp(log_w_global + global_log_prob, log_w_local + local_log_prob)
        return torch.where(in_support, log_prob, torch.full_like(log_prob, -torch.inf))

    def cdf(self, value):
        w_local = torch.sigmoid(self.local_weight_logit)
        global_cdf = self.global_dist.cdf(value)
        local_x = self._local_coordinate(value)
        local_cdf = self.local_dist.cdf(local_x)
        local_cdf = torch.where(
            value <= self.local_low,
            torch.zeros_like(local_cdf),
            torch.where(value >= self.local_high, torch.ones_like(local_cdf), local_cdf),
        )
        cdf = (1.0 - w_local) * global_cdf + w_local * local_cdf
        return torch.where(value <= 0.0, torch.zeros_like(cdf), torch.where(value >= 1.0, torch.ones_like(cdf), cdf.clamp(0.0, 1.0)))

    def icdf(self, value):
        p = value.clamp(1.0e-09, 1.0 - 1.0e-09)
        low = torch.zeros_like(p)
        high = torch.ones_like(p)
        for _ in range(56):
            mid = 0.5 * (low + high)
            cdf_mid = self.cdf(mid)
            low = torch.where(cdf_mid < p, mid, low)
            high = torch.where(cdf_mid < p, high, mid)
        return (0.5 * (low + high)).clamp(1.0e-09, 1.0 - 1.0e-09)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.global_logits.dtype, device=self.global_logits.device)
        return self.icdf(u)

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape


class AxisSpecificMarginals(TorchDistribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, marginal_dists, validate_args=None):
        if not marginal_dists:
            raise ValueError('AxisSpecificMarginals requires at least one marginal distribution.')
        self.marginal_dists = list(marginal_dists)
        self._validate_args = validate_args
        batch_shape = torch.Size()
        for marginal in self.marginal_dists:
            batch_shape = torch.broadcast_shapes(batch_shape, marginal.batch_shape)
        self._batch_shape = batch_shape

    def log_prob(self, value):
        return torch.stack(
            [marginal.log_prob(value[..., axis]) for axis, marginal in enumerate(self.marginal_dists)],
            dim=-1,
        )

    def cdf(self, value):
        return torch.stack(
            [marginal.cdf(value[..., axis]) for axis, marginal in enumerate(self.marginal_dists)],
            dim=-1,
        )

    def icdf(self, value):
        return torch.stack(
            [marginal.icdf(value[..., axis]) for axis, marginal in enumerate(self.marginal_dists)],
            dim=-1,
        )

    def sample(self, sample_shape=torch.Size()):
        return torch.stack(
            [marginal.sample(sample_shape) for marginal in self.marginal_dists],
            dim=-1,
        )

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return torch.Size()

class SafeKumaraswamy(TorchDistribution):
    """
    从头开始编写的的 Kumaraswamy 分布。
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration2': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration2, validate_args=None):
        """
        构造函数。
        """
        self.concentration1, self.concentration2 = broadcast_all(concentration1, concentration2)
        self._validate_args = validate_args

    def log_prob(self, value):
        """
        计算对数概率密度，内置数值稳定性。
        """
        if self._validate_args:
            self._validate_sample(value)
        a = self.concentration1
        b = self.concentration2
        value = value.clamp(1e-09, 1.0 - 1e-09)
        log_a = torch.log(a)
        log_b = torch.log(b)
        log_prob_part1 = log_a + log_b + (a - 1.0) * torch.log(value)
        # exp(a*log(x)) can round to one near the upper boundary in float32.
        log_one_minus_power = torch.log(-torch.expm1(a * torch.log(value)))
        log_prob_part2 = (b - 1.0) * log_one_minus_power
        return log_prob_part1 + log_prob_part2

    def cdf(self, value):
        """
        计算累积分布函数 (CDF)，内置数值稳定性。
        F(x) = 1 - (1 - x^a)^b
        """
        if self._validate_args:
            self._validate_sample(value)
        a = self.concentration1
        b = self.concentration2
        value = value.clamp(0.0, 1.0)
        log_inner_term = b * torch.log(-torch.expm1(a * torch.log(value)))
        return -torch.expm1(log_inner_term)

    def icdf(self, p):
        """
        计算逆累积分布函数 (Inverse CDF)，用于采样，内置数值稳定性。
        x = (1 - (1-p)^(1/b))^(1/a)
        """
        a = self.concentration1
        b = self.concentration2
        p = p.clamp(0.0, 1.0)
        log_1_minus_p = torch.log1p(-p)
        log_main_term = torch.log(-torch.expm1(log_1_minus_p / b))
        return torch.exp(log_main_term / a)

    def sample(self, sample_shape=torch.Size()):
        """
        从分布中采样。
        """
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.concentration1.dtype, device=self.concentration1.device)
        return self.icdf(u)

    def _extended_shape(self, sample_shape=torch.Size()):
        """
        辅助函数，用于确定采样形状。
        """
        return sample_shape + self.batch_shape

    @property
    def batch_shape(self):
        """
        定义分布的批次形状。
        """
        return self.concentration1.shape

    @property
    def event_shape(self):
        """
        定义分布的事件形状 (对于单变量分布，为空)。
        """
        return torch.Size()

class CouplingNet(nn.Module):
    """
    一个自定义神经网络模块，用于参数化仿射耦合层。
    它的 forward 方法返回两个张量 (loc, log_scale)，以满足 AffineCoupling 的 API 要求。
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim * 2))
        self.diagnostic_name = None
        self.diagnostic_callback = None

    def forward(self, x):
        params = self.net(x)
        loc, log_scale_unconstrained = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale_unconstrained)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(self.diagnostic_name, x, loc, log_scale_unconstrained, log_scale)
        return (loc, log_scale)

class BayesianMixtureEstimator_NF_Correlated:
    """
    贝叶斯混合密度网络
    """

    def __init__(self, data_dim, num_corr_kumaraswamy, num_corr_tnormal, num_corr_tpareto, hidden_dim_flow=128, num_flow_transforms=8, device=None, physical_bounds=None, num_corr_logitnormal=0, num_corr_axis_lntnormal=0, num_corr_axis_lnkumaraswamy=0, num_corr_axis_bernstein_tnormal=0, num_corr_axis_bernstein_kumaraswamy=0, axis_bernstein_degree=8, num_corr_axis_spline_tnormal=0, num_corr_axis_spline_kumaraswamy=0, axis_spline_num_bins=12, num_corr_decoupled_spline_tnormal_copula=0, num_corr_decoupled_spline_kumaraswamy_copula=0, m1_roughness_lambda=0.0, m1_basis_coordinate='log10', m1_basis_logit_epsilon=0.02, m1_basis_type='single', m1_two_resolution_global_bins=5, m1_two_resolution_local_bins=16, m1_two_resolution_local_min_solar_mass=5.0, m1_two_resolution_local_max_solar_mass=50.0, m1_smooth_length=.35):
        if data_dim <= 1:
            raise ValueError('data_dim must be > 1 for correlation to be meaningful.')
        self.data_dim = data_dim
        self.num_corr_kumaraswamy = num_corr_kumaraswamy
        self.num_corr_tnormal = num_corr_tnormal
        self.num_corr_tpareto = num_corr_tpareto
        self.num_corr_logitnormal = num_corr_logitnormal
        self.num_corr_axis_lntnormal = num_corr_axis_lntnormal
        self.num_corr_axis_lnkumaraswamy = num_corr_axis_lnkumaraswamy
        self.num_corr_axis_bernstein_tnormal = num_corr_axis_bernstein_tnormal
        self.num_corr_axis_bernstein_kumaraswamy = num_corr_axis_bernstein_kumaraswamy
        self.axis_bernstein_degree = int(axis_bernstein_degree)
        self.axis_bernstein_num_basis = self.axis_bernstein_degree + 1
        self.num_corr_axis_spline_tnormal = num_corr_axis_spline_tnormal
        self.num_corr_axis_spline_kumaraswamy = num_corr_axis_spline_kumaraswamy
        self.axis_spline_num_bins = int(axis_spline_num_bins)
        self.axis_spline_num_knots = self.axis_spline_num_bins + 1
        self.num_corr_decoupled_spline_tnormal_copula = int(num_corr_decoupled_spline_tnormal_copula)
        self.num_corr_decoupled_spline_kumaraswamy_copula = int(num_corr_decoupled_spline_kumaraswamy_copula)
        self.m1_roughness_lambda = float(m1_roughness_lambda)
        self.m1_basis_coordinate = str(m1_basis_coordinate or 'log10')
        self.m1_basis_logit_epsilon = float(m1_basis_logit_epsilon)
        self.m1_basis_type = str(m1_basis_type or 'single').lower()
        self.m1_smooth_length = float(m1_smooth_length)
        if self.m1_smooth_length <= 0:
            raise ValueError('Smooth mass correlation length must be positive.')
        self.m1_two_resolution_global_bins = int(m1_two_resolution_global_bins)
        self.m1_two_resolution_local_bins = int(m1_two_resolution_local_bins)
        self.m1_two_resolution_local_min_solar_mass = float(m1_two_resolution_local_min_solar_mass)
        self.m1_two_resolution_local_max_solar_mass = float(m1_two_resolution_local_max_solar_mass)
        if self.m1_basis_type not in ('single', 'smooth', 'two_resolution', 'two-resolution'):
            raise ValueError(f'Unsupported m1_basis_type: {m1_basis_type}')
        if self.m1_two_resolution_global_bins < 2 or self.m1_two_resolution_local_bins < 2:
            raise ValueError('Two-resolution m1 basis requires at least two global and local bins.')
        if self.m1_two_resolution_local_min_solar_mass <= 0.0 or self.m1_two_resolution_local_min_solar_mass >= self.m1_two_resolution_local_max_solar_mass:
            raise ValueError('Invalid two-resolution local m1 interval.')
        self.m1_two_resolution_global_knots = self.m1_two_resolution_global_bins + 1
        self.m1_two_resolution_local_knots = self.m1_two_resolution_local_bins + 1
        self.m1_spline_num_params = (
            self.axis_spline_num_knots
            if self.m1_basis_type in ('single', 'smooth')
            else self.m1_two_resolution_global_knots + self.m1_two_resolution_local_knots + 1
        )
        self.K = num_corr_kumaraswamy + num_corr_tnormal + num_corr_tpareto + num_corr_logitnormal + num_corr_axis_lntnormal + num_corr_axis_lnkumaraswamy + num_corr_axis_bernstein_tnormal + num_corr_axis_bernstein_kumaraswamy + num_corr_axis_spline_tnormal + num_corr_axis_spline_kumaraswamy + self.num_corr_decoupled_spline_tnormal_copula + self.num_corr_decoupled_spline_kumaraswamy_copula
        self.injection_data = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.is_rate_inferred = False
        self.provided_bounds = physical_bounds
        self.min_vals, self.max_vals = (None, None)
        if self.provided_bounds is not None:
            self.min_vals = torch.tensor(self.provided_bounds[0], dtype=torch.float32, device=self.device)
            self.max_vals = torch.tensor(self.provided_bounds[1], dtype=torch.float32, device=self.device)
            if (self.min_vals >= self.max_vals).any():
                raise ValueError('Physical bounds min must be strictly less than max.')
            print(f'模式: [物理边界约束]. 模型将定义在用户指定的范围内:\n  Min: {self.min_vals.cpu().numpy()}\n  Max: {self.max_vals.cpu().numpy()}')
        else:
            print('模式: [数据驱动边界]. 模型范围将在训练开始时由输入数据决定。')
        self.param_names = [f'Parameter {i + 1}' for i in range(data_dim)]
        self.correct_selection_effects = False
        self.selection_function = None
        self.observation_years = None
        self.posterior_mean_selection_fraction = None
        self.training_diagnostics = {'status': 'not_started', 'phases': [], 'failure': None}
        self.rate_diagnostics = {'status': 'not_started'}
        self._latest_model_diagnostics = {}
        self.lambda_det_mc_samples = 40000
        self.obs_neff_threshold = 10.0
        self.selection_neff_multiplier = 4.0
        self.q_monotonicity_lambda = 50.0
        self.q_monotonicity_num_points = 25
        self.selection_mc_mode = 'random'
        self.selection_mc_seed = 12345
        self.selection_integration = 'uniform'
        self._selection_mc_cache = {}
        self.copula_cdf_clamp_eps = 1.0e-7
        self.copula_cdf_double = False
        self.flow_param_clip_max = None
        self.flow_init_scale = None
        self.flow_grad_scale = 1.0
        self.flow_grad_warmup_epochs = 0
        self.flow_grad_warmup_scale = None
        self.kumaraswamy_log_param_clip = (-8.0, 8.0)
        self.reset_flow_optimizer_state_on_sanitize = False
        self._flow_grad_hooks = []
        self._current_training_phase = None
        self._current_training_epoch = None
        self._current_flow_grad_stats = None
        self._current_flow_head_forward_stats = None
        self._current_guide_flow_stats = None
        self._current_flow_numerics_stats = None
        self._current_flow_forward_context = None
        self.diagnostic_flow_head_enabled = False
        self.diagnostic_flow_numerics_enabled = False
        self.diagnostic_optimizer_state_enabled = False
        self.diagnostic_objective_terms_enabled = False
        self.diagnostic_trace_optimizer_enabled = False
        self.diagnostic_trace_epochs_by_phase = {}
        self.diagnostic_torch_internals_enabled = False
        self.diagnostic_saved_tensor_max_records = 80
        self.diagnostic_max_flow_params = 6
        self.cholesky_params_dim = self.data_dim * (self.data_dim - 1) // 2
        self.dims = {
            'weights': self.K,
            'k_marginals': self.num_corr_kumaraswamy * self.data_dim * 2,
            'k_cholesky': self.num_corr_kumaraswamy * self.cholesky_params_dim,
            'tnormal_marginals': self.num_corr_tnormal * self.data_dim * 2,
            'tnormal_cholesky': self.num_corr_tnormal * self.cholesky_params_dim,
            'tpareto_marginals': self.num_corr_tpareto * self.data_dim,
            'tpareto_cholesky': self.num_corr_tpareto * self.cholesky_params_dim,
            'logitnormal_marginals': self.num_corr_logitnormal * self.data_dim * 2,
            'logitnormal_cholesky': self.num_corr_logitnormal * self.cholesky_params_dim,
            'axis_lntnormal_marginals': self.num_corr_axis_lntnormal * (2 + 2 * (self.data_dim - 1)),
            'axis_lntnormal_cholesky': self.num_corr_axis_lntnormal * self.cholesky_params_dim,
            'axis_lnkumaraswamy_marginals': self.num_corr_axis_lnkumaraswamy * (2 + 2 * (self.data_dim - 1)),
            'axis_lnkumaraswamy_cholesky': self.num_corr_axis_lnkumaraswamy * self.cholesky_params_dim,
            'axis_bernstein_tnormal_marginals': self.num_corr_axis_bernstein_tnormal * (self.axis_bernstein_num_basis + 2 * (self.data_dim - 1)),
            'axis_bernstein_tnormal_cholesky': self.num_corr_axis_bernstein_tnormal * self.cholesky_params_dim,
            'axis_bernstein_kumaraswamy_marginals': self.num_corr_axis_bernstein_kumaraswamy * (self.axis_bernstein_num_basis + 2 * (self.data_dim - 1)),
            'axis_bernstein_kumaraswamy_cholesky': self.num_corr_axis_bernstein_kumaraswamy * self.cholesky_params_dim,
            'axis_spline_tnormal_marginals': self.num_corr_axis_spline_tnormal * (self.m1_spline_num_params + 2 * (self.data_dim - 1)),
            'axis_spline_tnormal_cholesky': self.num_corr_axis_spline_tnormal * self.cholesky_params_dim,
            'axis_spline_kumaraswamy_marginals': self.num_corr_axis_spline_kumaraswamy * (self.m1_spline_num_params + 2 * (self.data_dim - 1)),
            'axis_spline_kumaraswamy_cholesky': self.num_corr_axis_spline_kumaraswamy * self.cholesky_params_dim,
            'decoupled_spline_tnormal_marginals': (self.m1_spline_num_params + 2 * (self.data_dim - 1)) if self.num_corr_decoupled_spline_tnormal_copula > 0 else 0,
            'decoupled_spline_tnormal_cholesky': self.num_corr_decoupled_spline_tnormal_copula * self.cholesky_params_dim,
            'decoupled_spline_kumaraswamy_marginals': (self.m1_spline_num_params + 2 * (self.data_dim - 1)) if self.num_corr_decoupled_spline_kumaraswamy_copula > 0 else 0,
            'decoupled_spline_kumaraswamy_cholesky': self.num_corr_decoupled_spline_kumaraswamy_copula * self.cholesky_params_dim,
        }
        self.latent_dim = sum(self.dims.values())
        if self.latent_dim == 0:
            raise ValueError('Model must have at least one mixture component.')
        self.base_dist = dist.Normal(torch.zeros(self.latent_dim, device=self.device), torch.ones(self.latent_dim, device=self.device))
        transforms = []
        for transform_index in range(num_flow_transforms):
            split_dim = self.latent_dim // 2
            param_net_output_dim = self.latent_dim - split_dim
            param_net = CouplingNet(input_dim=split_dim, output_dim=param_net_output_dim, hidden_dim=hidden_dim_flow)
            param_net.diagnostic_name = str(transform_index)
            param_net.diagnostic_callback = self._record_flow_head_forward_stats
            transforms.append(T.AffineCoupling(split_dim, param_net))
            # Reversing halves guarantees every coordinate is transformed.
            # Independent random permutations can leave a coordinate unchanged.
            transforms.append(T.Permute(torch.arange(self.latent_dim - 1, -1, -1, device=self.device)))
        self.flow_transforms = T.ComposeTransformModule(transforms).to(self.device)
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.flow_transforms)
        self._standard_normal = dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))

    def _apply_flow_initialization(self):
        if self.flow_init_scale is None:
            return
        scale = float(self.flow_init_scale)
        with torch.no_grad():
            for param in self.flow_transforms.parameters():
                param.mul_(scale)

    def _configure_flow_gradient_hooks(self):
        for hook in self._flow_grad_hooks:
            hook.remove()
        self._flow_grad_hooks = []
        for name, param in self.flow_transforms.named_parameters():
            self._flow_grad_hooks.append(param.register_hook(lambda grad, n=name: self._record_and_scale_flow_grad(n, grad)))

    def _current_flow_grad_scale(self):
        scale = float(self.flow_grad_scale)
        if (
            self.flow_grad_warmup_scale is not None
            and self._current_training_phase == 'warmup'
            and self._current_training_epoch is not None
            and int(self._current_training_epoch) < int(self.flow_grad_warmup_epochs)
        ):
            return float(self.flow_grad_warmup_scale)
        return scale

    def _record_and_scale_flow_grad(self, name, grad):
        if grad is None:
            return grad
        if self._current_flow_grad_stats is None:
            self._current_flow_grad_stats = {}
        grad_detached = grad.detach()
        abs_max = float(grad_detached.abs().max().item()) if grad_detached.numel() else 0.0
        l2 = float(torch.linalg.vector_norm(grad_detached).item()) if grad_detached.numel() else 0.0
        self._current_flow_grad_stats[name] = {
            'abs_max': abs_max,
            'l2_norm': l2,
            'shape': list(grad_detached.shape),
        }
        head_stats = self._flow_head_param_stats(name, grad_detached)
        if head_stats:
            self._current_flow_grad_stats[name]['head_stats'] = head_stats
        scale = self._current_flow_grad_scale()
        if abs(scale - 1.0) < 1e-12:
            return grad
        return grad * scale

    @staticmethod
    def _flow_head_param_stats(name, tensor, previous_tensor=None):
        if '.nn.net.2.' not in str(name):
            return []
        if tensor.ndim < 1 or tensor.shape[0] < 2 or tensor.shape[0] % 2 != 0:
            return []
        split = tensor.shape[0] // 2
        groups = [('loc', tensor[:split]), ('log_scale_unconstrained', tensor[split:])]
        previous_groups = {}
        if previous_tensor is not None and previous_tensor.shape == tensor.shape:
            previous_groups = {
                'loc': previous_tensor[:split],
                'log_scale_unconstrained': previous_tensor[split:],
            }
        stats = []
        for head_name, values in groups:
            detached = values.detach()
            finite = torch.isfinite(detached)
            item = {
                'head': head_name,
                'shape': list(detached.shape),
                'numel': int(detached.numel()),
                'num_finite': int(finite.sum().item()),
            }
            if finite.any():
                finite_vals = detached[finite]
                item.update(
                    {
                        'abs_max': float(finite_vals.abs().max().item()),
                        'l2_norm': float(torch.linalg.vector_norm(finite_vals).item()),
                        'min': float(finite_vals.min().item()),
                        'max': float(finite_vals.max().item()),
                    }
                )
            else:
                item.update({'abs_max': float('nan'), 'l2_norm': float('nan'), 'min': float('nan'), 'max': float('nan')})
            if head_name in previous_groups:
                delta = detached - previous_groups[head_name].detach()
                finite_delta = torch.isfinite(delta)
                item['delta_num_finite'] = int(finite_delta.sum().item())
                if finite_delta.any():
                    finite_delta_vals = delta[finite_delta]
                    item['delta_abs_max'] = float(finite_delta_vals.abs().max().item())
                    item['delta_l2_norm'] = float(torch.linalg.vector_norm(finite_delta_vals).item())
                else:
                    item['delta_abs_max'] = float('nan')
                    item['delta_l2_norm'] = float('nan')
            stats.append(item)
        return stats

    @staticmethod
    def _merge_tensor_summary(target, tensor):
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        target['num_calls'] = int(target.get('num_calls', 0)) + 1
        target['num_nonfinite_calls'] = int(target.get('num_nonfinite_calls', 0)) + (0 if bool(finite.all().item()) else 1)
        target['numel'] = int(detached.numel())
        target['num_finite_last'] = int(finite.sum().item())
        if finite.any():
            finite_vals = detached[finite]
            min_v = float(finite_vals.min().item())
            max_v = float(finite_vals.max().item())
            abs_v = float(finite_vals.abs().max().item())
            l2_v = float(torch.linalg.vector_norm(finite_vals).item())
            target['min'] = min(float(target.get('min', min_v)), min_v)
            target['max'] = max(float(target.get('max', max_v)), max_v)
            target['abs_max'] = max(float(target.get('abs_max', abs_v)), abs_v)
            target['l2_norm_max'] = max(float(target.get('l2_norm_max', l2_v)), l2_v)
        else:
            target['min'] = float('nan')
            target['max'] = float('nan')
            target['abs_max'] = float('nan')
            target['l2_norm_max'] = float('nan')

    def _record_flow_head_forward_stats(self, name, x, loc, log_scale_unconstrained, log_scale):
        if self._current_flow_head_forward_stats is None:
            return
        key = str(name)
        context = str(self._current_flow_forward_context or 'unspecified')
        context_store = self._current_flow_head_forward_stats.setdefault(context, {})
        summary_store = self._current_flow_head_forward_stats.setdefault('all', {})
        for store in [context_store, summary_store]:
            entry = store.setdefault(
                key,
                {
                    'transform': key,
                    'context': context if store is context_store else 'all',
                    'num_calls': 0,
                    'input': {},
                    'loc': {},
                    'log_scale_unconstrained': {},
                    'log_scale': {},
                },
            )
            entry['num_calls'] += 1
            for field_name, tensor in [
                ('input', x),
                ('loc', loc),
                ('log_scale_unconstrained', log_scale_unconstrained),
                ('log_scale', log_scale),
            ]:
                self._merge_tensor_summary(entry[field_name], tensor)

    def _summarize_flow_inverse_path(self, latent_vector):
        transforms = list(getattr(self.flow_transforms, 'parts', []))
        y = latent_vector.detach()
        entries = []
        with torch.no_grad():
            for reverse_index, transform in enumerate(reversed(transforms)):
                transform_index = len(transforms) - 1 - reverse_index
                entry = {
                    'transform_index': int(transform_index),
                    'transform_type': type(transform).__name__,
                    'input': _tensor_stats_dict(y),
                }
                try:
                    x = transform.inv(y)
                    entry['output'] = _tensor_stats_dict(x)
                    try:
                        ladj = transform.log_abs_det_jacobian(x, y)
                        entry['log_abs_det_jacobian'] = _tensor_stats_dict(ladj)
                    except Exception as exc:
                        entry['log_abs_det_jacobian_error'] = {
                            'error_type': type(exc).__name__,
                            'message': str(exc),
                        }
                    y = x.detach()
                except Exception as exc:
                    entry['inverse_error'] = {
                        'error_type': type(exc).__name__,
                        'message': str(exc),
                    }
                    entries.append(entry)
                    break
                entries.append(entry)
        return entries

    def _summarize_flow_optimizer_state(self, scheduler, previous_state=None, max_params=8):
        entries = []
        total_exp_avg_sq = 0.0
        total_exp_avg_sq_sq = 0.0
        total_step_sq = 0.0
        max_effective_step_abs = 0.0
        total_exp_avg_nonfinite = 0
        total_exp_avg_sq_nonfinite = 0
        total_effective_step_nonfinite = 0

        def finite_tensor_stats(tensor):
            detached = tensor.detach()
            finite_mask = torch.isfinite(detached)
            nonfinite_count = int((~finite_mask).sum().item())
            finite_values = detached[finite_mask]
            if finite_values.numel():
                abs_max = float(finite_values.abs().max().item())
                l2_norm = float(torch.linalg.vector_norm(finite_values).item())
            else:
                abs_max = float('nan') if detached.numel() else 0.0
                l2_norm = float('nan') if detached.numel() else 0.0
            return abs_max, l2_norm, nonfinite_count

        def finite_sort_value(item, key):
            value = item.get(key, 0.0)
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                return -float('inf')
            return value_float if np.isfinite(value_float) else -float('inf')

        for name, param in self.flow_transforms.named_parameters():
            sched_obj = getattr(scheduler, 'optim_objs', {}).get(param)
            optimizer = getattr(sched_obj, 'optimizer', None)
            if optimizer is None:
                continue
            state = optimizer.state.get(param, {})
            exp_avg = state.get('exp_avg')
            exp_avg_sq = state.get('exp_avg_sq')
            if exp_avg is None or exp_avg_sq is None:
                continue
            group = optimizer.param_groups[0]
            lr = float(group.get('lr', 0.0))
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            eps = float(group.get('eps', 1e-8))
            step_value = state.get('step', 0)
            if isinstance(step_value, torch.Tensor):
                step_float = float(step_value.detach().cpu().item())
            else:
                step_float = float(step_value)
            bias_correction1 = 1.0 - float(beta1) ** max(step_float, 1.0)
            bias_correction2 = 1.0 - float(beta2) ** max(step_float, 1.0)
            step_size = lr * (bias_correction2 ** 0.5) / max(bias_correction1, 1e-12)
            effective_step = step_size * exp_avg.detach() / (exp_avg_sq.detach().sqrt() + eps)
            exp_avg_abs, exp_avg_l2, exp_avg_nonfinite = finite_tensor_stats(exp_avg)
            exp_avg_sq_abs, exp_avg_sq_l2, exp_avg_sq_nonfinite = finite_tensor_stats(exp_avg_sq)
            effective_step_abs, effective_step_l2, effective_step_nonfinite = finite_tensor_stats(effective_step)
            if np.isfinite(exp_avg_l2):
                total_exp_avg_sq += exp_avg_l2 * exp_avg_l2
            if np.isfinite(exp_avg_sq_l2):
                total_exp_avg_sq_sq += exp_avg_sq_l2 * exp_avg_sq_l2
            if np.isfinite(effective_step_l2):
                total_step_sq += effective_step_l2 * effective_step_l2
            total_exp_avg_nonfinite += exp_avg_nonfinite
            total_exp_avg_sq_nonfinite += exp_avg_sq_nonfinite
            total_effective_step_nonfinite += effective_step_nonfinite
            max_effective_step_abs = max(max_effective_step_abs, effective_step_abs)
            param_tensor = param.detach()
            param_abs = float(param_tensor.abs().max().item()) if param_tensor.numel() else 0.0
            update_abs = None
            update_l2 = None
            previous_tensor = previous_state.get(name) if previous_state is not None and name in previous_state else None
            if previous_tensor is not None:
                update = param_tensor - previous_tensor
                update_abs = float(update.abs().max().item()) if update.numel() else 0.0
                update_l2 = float(torch.linalg.vector_norm(update).item()) if update.numel() else 0.0
            item = {
                'name': name,
                'shape': list(param_tensor.shape),
                'adam_step': step_float,
                'lr': lr,
                'beta1': float(beta1),
                'beta2': float(beta2),
                'eps': eps,
                'exp_avg_abs_max': exp_avg_abs,
                'exp_avg_l2_norm': exp_avg_l2,
                'exp_avg_nonfinite_count': exp_avg_nonfinite,
                'exp_avg_sq_abs_max': exp_avg_sq_abs,
                'exp_avg_sq_l2_norm': exp_avg_sq_l2,
                'exp_avg_sq_nonfinite_count': exp_avg_sq_nonfinite,
                'effective_step_abs_max': effective_step_abs,
                'effective_step_l2_norm': effective_step_l2,
                'effective_step_nonfinite_count': effective_step_nonfinite,
                'param_abs_max': param_abs,
                'update_abs_max': update_abs,
                'update_l2_norm': update_l2,
                'effective_step_to_param_abs': effective_step_abs / max(param_abs, 1e-12),
                'effective_step_to_update_abs': None if update_abs is None else effective_step_abs / max(update_abs, 1e-12),
            }
            item['head_stats'] = []
            if '.nn.net.2.' in str(name):
                for stats_name, tensor in [
                    ('exp_avg', exp_avg.detach()),
                    ('exp_avg_sq', exp_avg_sq.detach()),
                    ('effective_step', effective_step.detach()),
                    ('param', param_tensor),
                ]:
                    for head_item in self._flow_head_param_stats(name, tensor, previous_tensor=None):
                        head_copy = dict(head_item)
                        head_copy['source'] = stats_name
                        item['head_stats'].append(head_copy)
            entries.append(item)
        top_effective_step = sorted(entries, key=lambda x: finite_sort_value(x, 'effective_step_abs_max'), reverse=True)[:max_params]
        top_exp_avg = sorted(entries, key=lambda x: finite_sort_value(x, 'exp_avg_abs_max'), reverse=True)[:max_params]
        top_exp_avg_sq = sorted(entries, key=lambda x: finite_sort_value(x, 'exp_avg_sq_abs_max'), reverse=True)[:max_params]
        nonfinite_optimizer_params = [
            {
                'name': item['name'],
                'exp_avg_nonfinite_count': item['exp_avg_nonfinite_count'],
                'exp_avg_sq_nonfinite_count': item['exp_avg_sq_nonfinite_count'],
                'effective_step_nonfinite_count': item['effective_step_nonfinite_count'],
            }
            for item in entries
            if item['exp_avg_nonfinite_count'] or item['exp_avg_sq_nonfinite_count'] or item['effective_step_nonfinite_count']
        ][:max_params]
        return {
            'num_params': len(entries),
            'exp_avg_l2_total': float(np.sqrt(total_exp_avg_sq)),
            'exp_avg_sq_l2_total': float(np.sqrt(total_exp_avg_sq_sq)),
            'effective_step_l2_total': float(np.sqrt(total_step_sq)),
            'effective_step_abs_max': float(max_effective_step_abs),
            'exp_avg_nonfinite_count': int(total_exp_avg_nonfinite),
            'exp_avg_sq_nonfinite_count': int(total_exp_avg_sq_nonfinite),
            'effective_step_nonfinite_count': int(total_effective_step_nonfinite),
            'nonfinite_optimizer_params': nonfinite_optimizer_params,
            'top_effective_step_params': top_effective_step,
            'top_exp_avg_params': top_exp_avg,
            'top_exp_avg_sq_params': top_exp_avg_sq,
        }

    def _summarize_adam_reconstruction(self, scheduler, previous_state=None, max_params=8):
        entries = []
        family_sq = {}
        layer_sq = {}
        total_residual_sq = 0.0
        mismatch_count = 0
        max_rel_residual = 0.0
        optim_objs = getattr(scheduler, 'optim_objs', {})
        for name, param in self.flow_transforms.named_parameters():
            previous_tensor = previous_state.get(name) if previous_state is not None and name in previous_state else None
            if previous_tensor is None:
                continue
            sched_obj = optim_objs.get(param)
            optimizer = getattr(sched_obj, 'optimizer', None)
            if optimizer is None:
                continue
            state = optimizer.state.get(param, {})
            exp_avg = state.get('exp_avg')
            exp_avg_sq = state.get('exp_avg_sq')
            if exp_avg is None or exp_avg_sq is None:
                continue
            group = optimizer.param_groups[0]
            lr = float(group.get('lr', 0.0))
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            eps = float(group.get('eps', 1e-8))
            step_value = state.get('step', 0)
            step_float = float(step_value.detach().cpu().item()) if torch.is_tensor(step_value) else float(step_value)
            bias_correction1 = 1.0 - float(beta1) ** max(step_float, 1.0)
            bias_correction2 = 1.0 - float(beta2) ** max(step_float, 1.0)
            step_size = lr * (bias_correction2 ** 0.5) / max(bias_correction1, 1e-12)
            reconstructed_delta = -step_size * exp_avg.detach() / (exp_avg_sq.detach().sqrt() + eps)
            actual_delta = param.detach() - previous_tensor
            residual = actual_delta - reconstructed_delta
            actual_l2 = float(torch.linalg.vector_norm(actual_delta).item()) if actual_delta.numel() else 0.0
            reconstructed_l2 = float(torch.linalg.vector_norm(reconstructed_delta).item()) if reconstructed_delta.numel() else 0.0
            residual_l2 = float(torch.linalg.vector_norm(residual).item()) if residual.numel() else 0.0
            residual_abs = float(residual.abs().max().item()) if residual.numel() else 0.0
            actual_abs = float(actual_delta.abs().max().item()) if actual_delta.numel() else 0.0
            rel_residual = residual_l2 / max(actual_l2, reconstructed_l2, 1e-12)
            finite = bool(torch.isfinite(actual_delta).all().item() and torch.isfinite(reconstructed_delta).all().item() and torch.isfinite(residual).all().item())
            mismatch = bool((not finite) or rel_residual > 5e-3 or residual_abs > 5e-6)
            if mismatch:
                mismatch_count += 1
            family, layer = self._flow_family_name(name)
            if np.isfinite(residual_l2):
                total_residual_sq += residual_l2 * residual_l2
                family_sq[family] = family_sq.get(family, 0.0) + residual_l2 * residual_l2
                layer_sq[layer] = layer_sq.get(layer, 0.0) + residual_l2 * residual_l2
            if np.isfinite(rel_residual):
                max_rel_residual = max(max_rel_residual, rel_residual)
            entries.append(
                {
                    'name': name,
                    'family': family,
                    'layer_class': layer,
                    'shape': list(param.shape),
                    'adam_step': step_float,
                    'lr': lr,
                    'beta1': float(beta1),
                    'beta2': float(beta2),
                    'eps': eps,
                    'actual_delta_l2': actual_l2,
                    'actual_delta_abs_max': actual_abs,
                    'reconstructed_delta_l2': reconstructed_l2,
                    'residual_l2': residual_l2,
                    'residual_abs_max': residual_abs,
                    'relative_residual': rel_residual,
                    'finite': finite,
                    'mismatch': mismatch,
                }
            )
        total_residual_l2 = float(np.sqrt(total_residual_sq))
        top_residual_params = sorted(entries, key=lambda item: item.get('residual_l2', 0.0), reverse=True)[:max_params]
        top_families = sorted(
            [
                {
                    'family': family,
                    'residual_l2': float(np.sqrt(value)),
                    'share': float(value / max(total_residual_sq, 1e-30)),
                }
                for family, value in family_sq.items()
            ],
            key=lambda item: item['share'],
            reverse=True,
        )[:max_params]
        top_layers = sorted(
            [
                {
                    'layer_class': layer,
                    'residual_l2': float(np.sqrt(value)),
                    'share': float(value / max(total_residual_sq, 1e-30)),
                }
                for layer, value in layer_sq.items()
            ],
            key=lambda item: item['share'],
            reverse=True,
        )
        return {
            'num_params': len(entries),
            'mismatch_count': int(mismatch_count),
            'max_relative_residual': float(max_rel_residual),
            'total_residual_l2': total_residual_l2,
            'top_residual_params': top_residual_params,
            'top_residual_families': top_families,
            'top_residual_layers': top_layers,
        }

    @staticmethod
    def _tensor_data_ptr(tensor):
        try:
            return int(tensor.data_ptr())
        except Exception:
            return None

    def _summarize_flow_grad_lifecycle(self, scheduler, stage, max_params=8):
        rows = []
        fault_count = 0
        optim_objs = getattr(scheduler, 'optim_objs', {})
        store_values = []
        for store_name, value in pyro.get_param_store().items():
            if torch.is_tensor(value):
                store_values.append((str(store_name), self._tensor_data_ptr(value), id(value)))
        for name, param in self.flow_transforms.named_parameters():
            grad = getattr(param, 'grad', None)
            sched_obj = optim_objs.get(param)
            optimizer = getattr(sched_obj, 'optimizer', None)
            state_present = bool(optimizer is not None and param in getattr(optimizer, 'state', {}))
            param_ptr = self._tensor_data_ptr(param)
            matching_store_names = [store_name for store_name, ptr, _ in store_values if ptr == param_ptr]
            grad_present = grad is not None
            grad_finite = bool(torch.isfinite(grad).all().item()) if torch.is_tensor(grad) else None
            grad_nonfinite = int((~torch.isfinite(grad)).sum().item()) if torch.is_tensor(grad) else 0
            grad_l2 = float(torch.linalg.vector_norm(grad.detach()).item()) if torch.is_tensor(grad) and grad.numel() else 0.0
            has_fault = bool((grad_present and not grad_finite) or (optimizer is not None and not state_present))
            if has_fault:
                fault_count += 1
            rows.append(
                {
                    'stage': stage,
                    'name': name,
                    'family': self._flow_family_name(name)[0],
                    'layer_class': self._flow_family_name(name)[1],
                    'param_object_id': int(id(param)),
                    'param_data_ptr': param_ptr,
                    'param_version': int(getattr(param, '_version', -1)),
                    'grad_present': bool(grad_present),
                    'grad_object_id': None if grad is None else int(id(grad)),
                    'grad_data_ptr': None if grad is None else self._tensor_data_ptr(grad),
                    'grad_version': None if grad is None else int(getattr(grad, '_version', -1)),
                    'grad_l2_norm': grad_l2,
                    'grad_nonfinite_count': grad_nonfinite,
                    'grad_finite': grad_finite,
                    'optimizer_bound': optimizer is not None,
                    'optimizer_state_present': state_present,
                    'param_store_same_data_ptr_count': len(matching_store_names),
                    'param_store_same_data_ptr_names': matching_store_names[:3],
                    'has_fault': has_fault,
                }
            )
        top_rows = sorted(rows, key=lambda item: item.get('grad_l2_norm', 0.0), reverse=True)[:max_params]
        return {
            'stage': stage,
            'num_params': len(rows),
            'fault_count': int(fault_count),
            'top_params': top_rows,
        }

    def _summarize_saved_tensor_hooks(self, records, max_records=40, counts=None):
        counts = counts or {}
        pack_count = int(counts.get('pack_count', sum(1 for item in records if item.get('action') == 'pack')))
        unpack_count = int(counts.get('unpack_count', sum(1 for item in records if item.get('action') == 'unpack')))
        nonfinite_count = int(counts.get('nonfinite_count', sum(1 for item in records if item.get('finite') is False)))
        version_fault_count = int(counts.get('version_fault_count', 0))
        packed_versions = {}
        for item in records:
            if item.get('action') == 'pack':
                packed_versions.setdefault(item.get('tensor_object_id'), item.get('version'))
            elif item.get('action') == 'unpack':
                packed_version = packed_versions.get(item.get('tensor_object_id'))
                if packed_version is not None and packed_version != item.get('version'):
                    item['version_mismatch'] = True
                else:
                    item['version_mismatch'] = False
        return {
            'pack_count': int(pack_count),
            'unpack_count': int(unpack_count),
            'nonfinite_count': int(nonfinite_count),
            'version_fault_count': int(version_fault_count),
            'sample_records': records[:max_records],
        }

    @staticmethod
    def _summarize_flow_grad_stats(grad_stats, max_params=6):
        if not grad_stats:
            return {
                'num_params': 0,
                'l2_norm_total': 0.0,
                'max_abs': 0.0,
                'top_abs_params': [],
            }
        items = []
        total_sq = 0.0
        max_abs = 0.0
        for name, item in grad_stats.items():
            l2 = float(item.get('l2_norm', 0.0))
            abs_v = float(item.get('abs_max', 0.0))
            total_sq += l2 * l2
            max_abs = max(max_abs, abs_v)
            items.append({
                'name': name,
                'shape': item.get('shape', []),
                'abs_max': abs_v,
                'l2_norm': l2,
                'head_stats': item.get('head_stats', []),
            })
        items_sorted = sorted(items, key=lambda x: x['abs_max'], reverse=True)
        return {
            'num_params': len(items),
            'l2_norm_total': float(np.sqrt(total_sq)),
            'max_abs': float(max_abs),
            'top_abs_params': items_sorted[:max_params],
        }

    def _clone_flow_param_state(self):
        state = {}
        for name, param in self.flow_transforms.named_parameters():
            state[name] = param.detach().clone()
        return state

    def _summarize_flow_state(self, previous_state=None, max_params=6):
        items = []
        total_sq = 0.0
        max_abs = 0.0
        delta_sq = 0.0
        delta_max_abs = 0.0
        for name, param in self.flow_transforms.named_parameters():
            tensor = param.detach()
            abs_max = float(tensor.abs().max().item()) if tensor.numel() else 0.0
            l2 = float(torch.linalg.vector_norm(tensor).item()) if tensor.numel() else 0.0
            total_sq += l2 * l2
            max_abs = max(max_abs, abs_max)
            item = {
                'name': name,
                'shape': list(tensor.shape),
                'abs_max': abs_max,
                'l2_norm': l2,
            }
            previous_tensor = previous_state.get(name) if previous_state is not None and name in previous_state else None
            if previous_state is not None and name in previous_state:
                delta = tensor - previous_tensor
                delta_abs_max = float(delta.abs().max().item()) if delta.numel() else 0.0
                delta_l2 = float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0
                delta_sq += delta_l2 * delta_l2
                delta_max_abs = max(delta_max_abs, delta_abs_max)
                item['delta_abs_max'] = delta_abs_max
                item['delta_l2_norm'] = delta_l2
            head_stats = self._flow_head_param_stats(name, tensor, previous_tensor=previous_tensor)
            if head_stats:
                item['head_stats'] = head_stats
            items.append(item)
        items_sorted = sorted(items, key=lambda x: x['abs_max'], reverse=True)
        summary = {
            'num_params': len(items),
            'l2_norm': float(np.sqrt(total_sq)),
            'max_abs': float(max_abs),
            'top_abs_params': items_sorted[:max_params],
        }
        if previous_state is not None:
            summary['delta_l2_norm'] = float(np.sqrt(delta_sq))
            summary['delta_max_abs'] = float(delta_max_abs)
        return summary

    @staticmethod
    def _tensor_digest(tensor):
        detached = tensor.detach().cpu().contiguous()
        arr = detached.numpy()
        digest = hashlib.sha256()
        digest.update(str(tuple(arr.shape)).encode('utf-8'))
        digest.update(str(arr.dtype).encode('utf-8'))
        digest.update(arr.tobytes())
        return digest.hexdigest()

    def _hash_flow_state(self):
        digest = hashlib.sha256()
        for name, param in sorted(self.flow_transforms.named_parameters(), key=lambda item: item[0]):
            digest.update(str(name).encode('utf-8'))
            digest.update(self._tensor_digest(param).encode('utf-8'))
        return digest.hexdigest()

    def _hash_param_store_state(self):
        digest = hashlib.sha256()
        for name, value in sorted(pyro.get_param_store().items(), key=lambda item: str(item[0])):
            digest.update(str(name).encode('utf-8'))
            digest.update(self._tensor_digest(value).encode('utf-8'))
        return digest.hexdigest()

    def _hash_optimizer_state(self, scheduler):
        digest = hashlib.sha256()
        optim_objs = getattr(scheduler, 'optim_objs', {})
        for name, param in sorted(self.flow_transforms.named_parameters(), key=lambda item: item[0]):
            digest.update(str(name).encode('utf-8'))
            sched_obj = optim_objs.get(param)
            optimizer = getattr(sched_obj, 'optimizer', None)
            if optimizer is None:
                digest.update(b'no_optimizer')
                continue
            state = optimizer.state.get(param, {})
            for key in ['step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                value = state.get(key)
                digest.update(str(key).encode('utf-8'))
                if torch.is_tensor(value):
                    digest.update(self._tensor_digest(value).encode('utf-8'))
                elif value is None:
                    digest.update(b'none')
                else:
                    digest.update(str(value).encode('utf-8'))
        return digest.hexdigest()

    @staticmethod
    def _capture_rng_state():
        state = {
            'torch_cpu': torch.get_rng_state(),
            'cuda': None,
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            return
        torch.set_rng_state(state['torch_cpu'])
        if torch.cuda.is_available() and state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(state['cuda'])

    def _clear_flow_grads(self):
        for _, param in self.flow_transforms.named_parameters():
            param.grad = None
        for _, value in pyro.get_param_store().items():
            if getattr(value, 'grad', None) is not None:
                value.grad = None

    @staticmethod
    def _flow_family_name(name):
        parts = str(name).split('.')
        transform = parts[0] if parts and parts[0].isdigit() else 'unknown'
        if '.nn.net.0.' in str(name):
            layer = 'nn.net.0'
        elif '.nn.net.2.' in str(name):
            layer = 'nn.net.2'
        else:
            layer = 'other'
        return f'flow_transform_{transform}:{layer}', layer

    def _summarize_current_flow_grads(self, clip_norm=None):
        rows = []
        total_sq = 0.0
        max_abs = 0.0
        nonfinite_count = 0
        family_sq = {}
        layer_sq = {}
        for name, param in self.flow_transforms.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            detached = grad.detach()
            finite = torch.isfinite(detached)
            nonfinite = int((~finite).sum().item())
            nonfinite_count += nonfinite
            finite_vals = detached[finite]
            if finite_vals.numel():
                abs_max = float(finite_vals.abs().max().item())
                l2 = float(torch.linalg.vector_norm(finite_vals).item())
            else:
                abs_max = float('nan') if detached.numel() else 0.0
                l2 = float('nan') if detached.numel() else 0.0
            if np.isfinite(l2):
                total_sq += l2 * l2
                family, layer = self._flow_family_name(name)
                family_sq[family] = family_sq.get(family, 0.0) + l2 * l2
                layer_sq[layer] = layer_sq.get(layer, 0.0) + l2 * l2
            if np.isfinite(abs_max):
                max_abs = max(max_abs, abs_max)
            rows.append(
                {
                    'name': name,
                    'family': self._flow_family_name(name)[0],
                    'layer_class': self._flow_family_name(name)[1],
                    'shape': list(detached.shape),
                    'grad_abs_max': abs_max,
                    'grad_l2_norm': l2,
                    'grad_nonfinite_count': nonfinite,
                }
            )
        total_l2 = float(np.sqrt(total_sq))
        clip_factor = 1.0
        if clip_norm is not None and total_l2 > 0.0:
            clip_factor = min(1.0, float(clip_norm) / max(total_l2, 1e-12))
        top_params = sorted(rows, key=lambda item: item.get('grad_abs_max', 0.0), reverse=True)[: int(self.diagnostic_max_flow_params)]
        top_families = sorted(
            [
                {
                    'family': family,
                    'grad_l2_norm': float(np.sqrt(value)),
                    'share': float(value / max(total_sq, 1e-30)),
                }
                for family, value in family_sq.items()
            ],
            key=lambda item: item['share'],
            reverse=True,
        )[: int(self.diagnostic_max_flow_params)]
        top_layers = sorted(
            [
                {
                    'layer_class': layer,
                    'grad_l2_norm': float(np.sqrt(value)),
                    'share': float(value / max(total_sq, 1e-30)),
                }
                for layer, value in layer_sq.items()
            ],
            key=lambda item: item['share'],
            reverse=True,
        )
        return {
            'num_params': len(rows),
            'raw_grad_l2_total': total_l2,
            'raw_grad_abs_max': float(max_abs),
            'raw_grad_nonfinite_count': int(nonfinite_count),
            'clip_norm': None if clip_norm is None else float(clip_norm),
            'clip_factor': float(clip_factor),
            'clipped_grad_l2_total_proxy': float(total_l2 * clip_factor),
            'top_params': top_params,
            'top_families': top_families,
            'top_layers': top_layers,
        }

    @staticmethod
    def _summarize_trace_sites(trace, trace_type):
        rows = []
        for name, node in trace.nodes.items():
            if node.get('type') not in {'sample', 'param'}:
                continue
            value = node.get('value')
            log_prob = node.get('log_prob')
            log_prob_sum = node.get('log_prob_sum')
            scale = node.get('scale')
            row = {
                'trace_type': trace_type,
                'site_name': str(name),
                'node_type': node.get('type'),
                'fn_type': type(node.get('fn')).__name__ if node.get('fn') is not None else None,
                'is_observed': bool(node.get('is_observed', False)),
                'value_shape': list(value.shape) if torch.is_tensor(value) else None,
                'log_prob_shape': list(log_prob.shape) if torch.is_tensor(log_prob) else None,
                'scale': float(scale) if isinstance(scale, Number) else None,
                'finite_value': bool(torch.isfinite(value).all().item()) if torch.is_tensor(value) else None,
                'finite_log_prob': bool(torch.isfinite(log_prob).all().item()) if torch.is_tensor(log_prob) else None,
            }
            if torch.is_tensor(log_prob_sum):
                row['log_prob_sum'] = float(log_prob_sum.detach().cpu().item())
            elif log_prob_sum is not None:
                row['log_prob_sum'] = float(log_prob_sum)
            rows.append(row)
        return rows

    def _trace_epoch_enabled(self, phase_name, epoch):
        epochs_by_phase = getattr(self, 'diagnostic_trace_epochs_by_phase', {}) or {}
        phase_epochs = epochs_by_phase.get(phase_name, [])
        return int(epoch) in {int(item) for item in phase_epochs}

    def _capture_trace_optimizer_pre_step(self, scheduler, data_samples_on_device, phase_name, epoch, clip_norm):
        rng_state = self._capture_rng_state()
        pre_flow_state = self._clone_flow_param_state()
        entry = {
            'epoch': int(epoch),
            'phase': phase_name,
            'pre_flow_hash': self._hash_flow_state(),
            'pre_param_store_hash': self._hash_param_store_state(),
            'pre_optimizer_hash': self._hash_optimizer_state(scheduler),
            'pre_flow_state_summary': self._summarize_flow_state(max_params=int(self.diagnostic_max_flow_params)),
            'pre_optimizer_summary': self._summarize_flow_optimizer_state(
                scheduler,
                previous_state=pre_flow_state,
                max_params=int(self.diagnostic_max_flow_params),
            ),
            'trace_sites': [],
            'backward_summary': {},
        }
        try:
            self._restore_rng_state(rng_state)
            trace_elbo = Trace_ELBO()
            model_trace, guide_trace = next(
                trace_elbo._get_traces(self._pyro_model, self._pyro_guide, (data_samples_on_device,), {})
            )
            entry['trace_sites'] = self._summarize_trace_sites(model_trace, 'model') + self._summarize_trace_sites(
                guide_trace, 'guide'
            )
            entry['trace_ok'] = True
        except Exception as exc:
            entry['trace_ok'] = False
            entry['trace_error'] = {'error_type': type(exc).__name__, 'message': str(exc)}

        try:
            self._clear_flow_grads()
            self._current_flow_grad_stats = {}
            self._latest_model_diagnostics = {}
            self._restore_rng_state(rng_state)
            saved_tensor_records = []
            saved_tensor_counts = {
                'pack_count': 0,
                'unpack_count': 0,
                'nonfinite_count': 0,
                'version_fault_count': 0,
            }
            saved_tensor_versions = {}
            max_saved_records = int(getattr(self, 'diagnostic_saved_tensor_max_records', 80))

            def pack_hook(tensor):
                finite = bool(torch.isfinite(tensor).all().item()) if tensor.numel() else True
                saved_tensor_counts['pack_count'] += 1
                if not finite:
                    saved_tensor_counts['nonfinite_count'] += 1
                saved_tensor_versions[int(id(tensor))] = int(getattr(tensor, '_version', -1))
                if len(saved_tensor_records) < max_saved_records:
                    saved_tensor_records.append(
                        {
                            'action': 'pack',
                            'tensor_object_id': int(id(tensor)),
                            'data_ptr': self._tensor_data_ptr(tensor),
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'device': str(tensor.device),
                            'version': int(getattr(tensor, '_version', -1)),
                            'finite': finite,
                        }
                    )
                return tensor

            def unpack_hook(tensor):
                finite = bool(torch.isfinite(tensor).all().item()) if tensor.numel() else True
                saved_tensor_counts['unpack_count'] += 1
                if not finite:
                    saved_tensor_counts['nonfinite_count'] += 1
                packed_version = saved_tensor_versions.get(int(id(tensor)))
                version_mismatch = packed_version is not None and packed_version != int(getattr(tensor, '_version', -1))
                if version_mismatch:
                    saved_tensor_counts['version_fault_count'] += 1
                if len(saved_tensor_records) < max_saved_records:
                    saved_tensor_records.append(
                        {
                            'action': 'unpack',
                            'tensor_object_id': int(id(tensor)),
                            'data_ptr': self._tensor_data_ptr(tensor),
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'device': str(tensor.device),
                            'version': int(getattr(tensor, '_version', -1)),
                            'finite': finite,
                            'version_mismatch': version_mismatch,
                        }
                    )
                return tensor

            if getattr(self, 'diagnostic_torch_internals_enabled', False):
                with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                    loss = Trace_ELBO().differentiable_loss(self._pyro_model, self._pyro_guide, data_samples_on_device)
            else:
                loss = Trace_ELBO().differentiable_loss(self._pyro_model, self._pyro_guide, data_samples_on_device)
            if torch.is_tensor(loss):
                loss_value = float(loss.detach().cpu().item())
                loss.backward()
            else:
                loss_value = float(loss)
            entry['backward_summary'] = {
                'backward_ok': True,
                'differentiable_loss': loss_value,
                'latest_model_diagnostics': dict(self._latest_model_diagnostics or {}),
                'flow_grad_summary': self._summarize_current_flow_grads(clip_norm=clip_norm),
            }
            if getattr(self, 'diagnostic_torch_internals_enabled', False):
                entry['backward_summary']['grad_lifecycle_summary'] = self._summarize_flow_grad_lifecycle(
                    scheduler,
                    stage='diagnostic_backward',
                    max_params=int(self.diagnostic_max_flow_params),
                )
                entry['backward_summary']['saved_tensor_summary'] = self._summarize_saved_tensor_hooks(
                    saved_tensor_records,
                    max_records=max_saved_records,
                    counts=saved_tensor_counts,
                )
        except Exception as exc:
            entry['backward_summary'] = {
                'backward_ok': False,
                'error_type': type(exc).__name__,
                'message': str(exc),
            }
        finally:
            self._clear_flow_grads()
            self._current_flow_grad_stats = {}
            self._restore_rng_state(rng_state)
        return entry, pre_flow_state

    @staticmethod
    def _clamp_log_param(values, low=-8.0, high=8.0):
        return torch.clamp(values, min=low, max=high)

    def _maybe_clamp_kumaraswamy_log_param(self, values):
        clip = self.kumaraswamy_log_param_clip
        if clip is None or clip is False:
            return values
        low, high = clip
        return self._clamp_log_param(values, low=float(low), high=float(high))

    @staticmethod
    def _clamp_prob_param(values, eps=1e-4):
        return values.clamp(min=eps, max=1.0 - eps)

    def _axis_lntnormal_marginal_dist(self, ln_locs, log_ln_scales, tn_locs, log_tn_scales):
        if self.data_dim < 2:
            raise ValueError('Axis-specific marginal family requires data_dim >= 2.')
        ln_dist = BoundedLogitNormal(ln_locs, torch.exp(log_ln_scales) + 0.0001)
        tail_dist = TruncatedNormal(
            tn_locs,
            torch.exp(log_tn_scales) + 0.0001,
            low=0.0001,
            high=1.0 - 0.0001,
        )
        return AxisSpecificMarginals(
            [ln_dist] + [
                TruncatedNormal(
                    tail_dist.loc[..., axis],
                    tail_dist.scale[..., axis],
                    low=0.0001,
                    high=1.0 - 0.0001,
                )
                for axis in range(self.data_dim - 1)
            ]
        )

    def _axis_lnkumaraswamy_marginal_dist(self, ln_locs, log_ln_scales, log_k_a, log_k_b):
        if self.data_dim < 2:
            raise ValueError('Axis-specific marginal family requires data_dim >= 2.')
        ln_dist = BoundedLogitNormal(ln_locs, torch.exp(log_ln_scales) + 0.0001)
        a = torch.exp(log_k_a) + 0.0001
        b = torch.exp(log_k_b) + 0.0001
        return AxisSpecificMarginals(
            [ln_dist] + [
                SafeKumaraswamy(a[..., axis], b[..., axis])
                for axis in range(self.data_dim - 1)
            ]
        )

    def _axis_bernstein_tnormal_marginal_dist(self, m1_logits, tn_locs, log_tn_scales):
        if self.data_dim < 2:
            raise ValueError('Axis-specific marginal family requires data_dim >= 2.')
        m1_dist = BoundedBernsteinMixture(m1_logits)
        tail_dist = TruncatedNormal(
            tn_locs,
            torch.exp(log_tn_scales) + 0.0001,
            low=0.0001,
            high=1.0 - 0.0001,
        )
        return AxisSpecificMarginals(
            [m1_dist] + [
                TruncatedNormal(
                    tail_dist.loc[..., axis],
                    tail_dist.scale[..., axis],
                    low=0.0001,
                    high=1.0 - 0.0001,
                )
                for axis in range(self.data_dim - 1)
            ]
        )

    def _axis_bernstein_kumaraswamy_marginal_dist(self, m1_logits, log_k_a, log_k_b):
        if self.data_dim < 2:
            raise ValueError('Axis-specific marginal family requires data_dim >= 2.')
        m1_dist = BoundedBernsteinMixture(m1_logits)
        a = torch.exp(log_k_a) + 0.0001
        b = torch.exp(log_k_b) + 0.0001
        return AxisSpecificMarginals(
            [m1_dist] + [
                SafeKumaraswamy(a[..., axis], b[..., axis])
                for axis in range(self.data_dim - 1)
            ]
        )

    def _axis_spline_tnormal_marginal_dist(self, m1_logits, tn_locs, log_tn_scales):
        if self.data_dim < 2:
            raise ValueError('Axis-specific marginal family requires data_dim >= 2.')
        m1_dist = self._m1_spline_dist(m1_logits)
        tail_dist = TruncatedNormal(
            tn_locs,
            torch.exp(log_tn_scales) + 0.0001,
            low=0.0001,
            high=1.0 - 0.0001,
        )
        return AxisSpecificMarginals(
            [m1_dist] + [
                TruncatedNormal(
                    tail_dist.loc[..., axis],
                    tail_dist.scale[..., axis],
                    low=0.0001,
                    high=1.0 - 0.0001,
                )
                for axis in range(self.data_dim - 1)
            ]
        )

    def _axis_spline_kumaraswamy_marginal_dist(self, m1_logits, log_k_a, log_k_b):
        if self.data_dim < 2:
            raise ValueError('Axis-specific marginal family requires data_dim >= 2.')
        m1_dist = self._m1_spline_dist(m1_logits)
        a = torch.exp(log_k_a) + 0.0001
        b = torch.exp(log_k_b) + 0.0001
        return AxisSpecificMarginals(
            [m1_dist] + [
                SafeKumaraswamy(a[..., axis], b[..., axis])
                for axis in range(self.data_dim - 1)
            ]
        )

    def _decoupled_spline_tnormal_marginal_dist(self, m1_logits, tn_locs, log_tn_scales):
        return self._axis_spline_tnormal_marginal_dist(m1_logits, tn_locs, log_tn_scales)

    def _decoupled_spline_kumaraswamy_marginal_dist(self, m1_logits, log_k_a, log_k_b):
        return self._axis_spline_kumaraswamy_marginal_dist(m1_logits, log_k_a, log_k_b)

    def _m1_spline_dist(self, m1_logits):
        basis_type = str(getattr(self, 'm1_basis_type', 'single') or 'single').lower()
        if basis_type == 'smooth':
            from .smooth_mass import SmoothLogMassDensity
            if self.min_vals is None or self.max_vals is None:
                raise RuntimeError('Smooth mass density requires physical bounds.')
            return SmoothLogMassDensity(m1_logits, self.min_vals[0], self.max_vals[0],
                                        length_scale=self.m1_smooth_length)
        if basis_type in ('two_resolution', 'two-resolution'):
            global_knots = int(getattr(self, 'm1_two_resolution_global_knots', int(getattr(self, 'm1_two_resolution_global_bins', 5)) + 1))
            local_knots = int(getattr(self, 'm1_two_resolution_local_knots', int(getattr(self, 'm1_two_resolution_local_bins', 16)) + 1))
            expected = global_knots + local_knots + 1
            if m1_logits.shape[-1] != expected:
                raise ValueError(f'Two-resolution m1 basis expected {expected} parameters, got {m1_logits.shape[-1]}.')
            if self.min_vals is None or self.max_vals is None:
                raise RuntimeError('Two-resolution m1 basis requires physical bounds before model evaluation.')
            global_logits = m1_logits[..., :global_knots]
            local_logits = m1_logits[..., global_knots:global_knots + local_knots]
            local_weight_logit = m1_logits[..., global_knots + local_knots]
            log_range = (self.max_vals[0] - self.min_vals[0]).clamp_min(1.0e-12)
            local_min = torch.log10(torch.as_tensor(float(getattr(self, 'm1_two_resolution_local_min_solar_mass', 5.0)), dtype=m1_logits.dtype, device=m1_logits.device))
            local_max = torch.log10(torch.as_tensor(float(getattr(self, 'm1_two_resolution_local_max_solar_mass', 50.0)), dtype=m1_logits.dtype, device=m1_logits.device))
            local_low = ((local_min - self.min_vals[0]) / log_range).clamp(1.0e-06, 1.0 - 1.0e-06)
            local_high = ((local_max - self.min_vals[0]) / log_range).clamp(1.0e-06, 1.0 - 1.0e-06)
            if torch.any(local_low >= local_high):
                raise RuntimeError('Two-resolution m1 local interval does not overlap the current m1 support.')
            return BoundedTwoResolutionSpline(global_logits, local_logits, local_weight_logit, local_low, local_high)
        coordinate = str(getattr(self, 'm1_basis_coordinate', 'log10') or 'log10').lower()
        log10_min = None
        log10_max = None
        if coordinate in ('m1', 'm1_logit', 'shifted_logit_m1'):
            if self.min_vals is None or self.max_vals is None:
                raise RuntimeError('m1-coordinate spline basis requires physical bounds before model evaluation.')
            log10_min = self.min_vals[0]
            log10_max = self.max_vals[0]
        return BoundedMonotoneSpline(
            m1_logits,
            coordinate=coordinate,
            log10_min=log10_min,
            log10_max=log10_max,
            logit_epsilon=float(getattr(self, 'm1_basis_logit_epsilon', 0.02)),
        )

    def _m1_basis_roughness_penalty(self, logits):
        if logits is None or logits.shape[-1] < 3:
            return torch.tensor(0.0, device=self.device)
        def _single_penalty(single_logits):
            heights = torch.nn.functional.softplus(single_logits) + 1.0e-04
            log_heights = torch.log(heights / heights.mean(dim=-1, keepdim=True).clamp_min(1.0e-12))
            second_diff = log_heights[..., 2:] - 2.0 * log_heights[..., 1:-1] + log_heights[..., :-2]
            return torch.sum(second_diff * second_diff)
        basis_type = str(getattr(self, 'm1_basis_type', 'single') or 'single').lower()
        if basis_type in ('two_resolution', 'two-resolution'):
            global_knots = int(getattr(self, 'm1_two_resolution_global_knots', int(getattr(self, 'm1_two_resolution_global_bins', 5)) + 1))
            local_knots = int(getattr(self, 'm1_two_resolution_local_knots', int(getattr(self, 'm1_two_resolution_local_bins', 16)) + 1))
            return (
                _single_penalty(logits[..., :global_knots])
                + _single_penalty(logits[..., global_knots:global_knots + local_knots])
            )
        return _single_penalty(logits)

    def _apply_m1_roughness_factor(self, named_logits):
        lambda_rough = float(getattr(self, 'm1_roughness_lambda', 0.0))
        if lambda_rough <= 0.0:
            return 0.0
        terms = []
        for logits in named_logits.values():
            terms.append(self._m1_basis_roughness_penalty(logits))
        if not terms:
            return 0.0
        penalty = torch.stack([term.reshape(()) for term in terms]).sum()
        factor_value = -lambda_rough * penalty
        pyro.factor('m1_roughness_penalty', factor_value)
        penalty_value = self._safe_float(penalty) or 0.0
        self._latest_model_diagnostics['m1_roughness_lambda'] = lambda_rough
        self._latest_model_diagnostics['m1_roughness_penalty_value'] = penalty_value
        self._latest_model_diagnostics['m1_roughness_factor_value'] = self._safe_float(factor_value)
        return penalty_value

    def _ensure_finite(self, name, tensor):
        if not torch.isfinite(tensor).all():
            raise NumericalStabilityError(f'{name} became non-finite: {_tensor_stats_dict(tensor)}')
        return tensor

    def _update_failure_diagnostics(self, stage, exc):
        self.training_diagnostics['status'] = 'failed'
        self.training_diagnostics['failure'] = {
            'stage': stage,
            'error_type': type(exc).__name__,
            'message': str(exc),
            'traceback': traceback.format_exc(),
        }

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().float().mean().item()
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _capture_model_diagnostics(self, *, selection_mode, min_neff_obs, selection_fraction=None, neff_sel=None):
        self._latest_model_diagnostics = {
            'selection_mode': selection_mode,
            'min_neff_obs': self._safe_float(min_neff_obs),
            'selection_fraction': self._safe_float(selection_fraction),
            'selection_neff': self._safe_float(neff_sel),
        }

    @staticmethod
    def _tensor_summary_dict(tensor):
        detached = tensor.detach().float().reshape(-1)
        return {
            'min': float(detached.min().item()),
            'max': float(detached.max().item()),
            'mean': float(detached.mean().item()),
            'std': float(detached.std(unbiased=False).item()) if detached.numel() > 1 else 0.0,
        }

    def summarize_posterior_parameters(self, num_draws=64):
        if not self.is_trained:
            return {'status': 'not_trained'}
        with torch.no_grad():
            latent_samples = self.flow_dist.sample(torch.Size([int(num_draws)]))
            params_list = [self._unflatten_latent_vector(latent) for latent in latent_samples]
            weights = torch.stack([torch.softmax(params['weights_logits'], dim=-1) for params in params_list], dim=0)
            summary = {
                'status': 'completed',
                'num_draws': int(num_draws),
                'weights': {
                    'overall': self._tensor_summary_dict(weights),
                    'component_means': weights.mean(dim=0).detach().cpu().tolist(),
                    'component_stds': weights.std(dim=0, unbiased=False).detach().cpu().tolist(),
                    'max_component_weight_mean': float(weights.max(dim=-1).values.mean().item()),
                },
                'families': {},
            }
            if self.num_corr_kumaraswamy > 0:
                a = torch.stack([torch.exp(params['log_k_a']) + 0.0001 for params in params_list], dim=0)
                b = torch.stack([torch.exp(params['log_k_b']) + 0.0001 for params in params_list], dim=0)
                cholesky_params = torch.stack([params['k_cholesky_params'] for params in params_list], dim=0)
                summary['families']['kumaraswamy'] = {
                    'a': self._tensor_summary_dict(a),
                    'b': self._tensor_summary_dict(b),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_tnormal > 0:
                locs = torch.stack([params['tn_locs'] for params in params_list], dim=0)
                log_scales = torch.stack([params['log_tn_scales'] for params in params_list], dim=0)
                scales = torch.exp(log_scales)
                cholesky_params = torch.stack([params['tnormal_cholesky_params'] for params in params_list], dim=0)
                summary['families']['tnormal'] = {
                    'locs': self._tensor_summary_dict(locs),
                    'scales': self._tensor_summary_dict(scales),
                    'log_scales': self._tensor_summary_dict(log_scales),
                    'locs_near_lower_frac': float((locs < 0.05).float().mean().item()),
                    'locs_near_upper_frac': float((locs > 0.95).float().mean().item()),
                    'log_scales_near_upper_clip_frac': float((log_scales > 4.5).float().mean().item()),
                    'log_scales_near_lower_clip_frac': float((log_scales < -7.5).float().mean().item()),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_tpareto > 0:
                log_alphas = torch.stack([params['log_tp_alphas'] for params in params_list], dim=0)
                alphas = torch.exp(log_alphas)
                cholesky_params = torch.stack([params['tpareto_cholesky_params'] for params in params_list], dim=0)
                summary['families']['tpareto'] = {
                    'alphas': self._tensor_summary_dict(alphas),
                    'log_alphas': self._tensor_summary_dict(log_alphas),
                    'log_alphas_near_upper_clip_frac': float((log_alphas > 7.5).float().mean().item()),
                    'log_alphas_near_lower_clip_frac': float((log_alphas < -7.5).float().mean().item()),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_logitnormal > 0:
                locs = torch.stack([params['ln_locs'] for params in params_list], dim=0)
                log_scales = torch.stack([params['log_ln_scales'] for params in params_list], dim=0)
                scales = torch.exp(log_scales)
                cholesky_params = torch.stack([params['logitnormal_cholesky_params'] for params in params_list], dim=0)
                summary['families']['logitnormal'] = {
                    'locs': self._tensor_summary_dict(locs),
                    'scales': self._tensor_summary_dict(scales),
                    'log_scales': self._tensor_summary_dict(log_scales),
                    'locs_near_upper_clip_frac': float((locs > 7.5).float().mean().item()),
                    'locs_near_lower_clip_frac': float((locs < -7.5).float().mean().item()),
                    'log_scales_near_upper_clip_frac': float((log_scales > 4.5).float().mean().item()),
                    'log_scales_near_lower_clip_frac': float((log_scales < -7.5).float().mean().item()),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_axis_bernstein_tnormal > 0:
                logits = torch.stack([params['axis_btn_m1_logits'] for params in params_list], dim=0)
                weights = torch.softmax(logits, dim=-1)
                log_scales = torch.stack([params['axis_btn_log_tn_scales'] for params in params_list], dim=0)
                cholesky_params = torch.stack([params['axis_btn_cholesky_params'] for params in params_list], dim=0)
                summary['families']['axis_bernstein_tnormal'] = {
                    'm1_basis_weights': self._tensor_summary_dict(weights),
                    'tail_log_scales': self._tensor_summary_dict(log_scales),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_axis_bernstein_kumaraswamy > 0:
                logits = torch.stack([params['axis_bk_m1_logits'] for params in params_list], dim=0)
                weights = torch.softmax(logits, dim=-1)
                log_a = torch.stack([params['axis_bk_log_k_a'] for params in params_list], dim=0)
                log_b = torch.stack([params['axis_bk_log_k_b'] for params in params_list], dim=0)
                cholesky_params = torch.stack([params['axis_bk_cholesky_params'] for params in params_list], dim=0)
                summary['families']['axis_bernstein_kumaraswamy'] = {
                    'm1_basis_weights': self._tensor_summary_dict(weights),
                    'tail_log_k_a': self._tensor_summary_dict(log_a),
                    'tail_log_k_b': self._tensor_summary_dict(log_b),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_axis_spline_tnormal > 0:
                logits = torch.stack([params['axis_stn_m1_logits'] for params in params_list], dim=0)
                heights = torch.nn.functional.softplus(logits) + 1.0e-04
                log_scales = torch.stack([params['axis_stn_log_tn_scales'] for params in params_list], dim=0)
                cholesky_params = torch.stack([params['axis_stn_cholesky_params'] for params in params_list], dim=0)
                summary['families']['axis_spline_tnormal'] = {
                    'm1_knot_heights': self._tensor_summary_dict(heights),
                    'tail_log_scales': self._tensor_summary_dict(log_scales),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_axis_spline_kumaraswamy > 0:
                logits = torch.stack([params['axis_sk_m1_logits'] for params in params_list], dim=0)
                heights = torch.nn.functional.softplus(logits) + 1.0e-04
                log_a = torch.stack([params['axis_sk_log_k_a'] for params in params_list], dim=0)
                log_b = torch.stack([params['axis_sk_log_k_b'] for params in params_list], dim=0)
                cholesky_params = torch.stack([params['axis_sk_cholesky_params'] for params in params_list], dim=0)
                summary['families']['axis_spline_kumaraswamy'] = {
                    'm1_knot_heights': self._tensor_summary_dict(heights),
                    'tail_log_k_a': self._tensor_summary_dict(log_a),
                    'tail_log_k_b': self._tensor_summary_dict(log_b),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_decoupled_spline_tnormal_copula > 0:
                logits = torch.stack([params['dec_stn_m1_logits'] for params in params_list], dim=0)
                heights = torch.nn.functional.softplus(logits) + 1.0e-04
                log_scales = torch.stack([params['dec_stn_log_tn_scales'] for params in params_list], dim=0)
                cholesky_params = torch.stack([params['dec_stn_cholesky_params'] for params in params_list], dim=0)
                summary['families']['decoupled_spline_tnormal_copula'] = {
                    'm1_knot_heights': self._tensor_summary_dict(heights),
                    'tail_log_scales': self._tensor_summary_dict(log_scales),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
                logits = torch.stack([params['dec_sk_m1_logits'] for params in params_list], dim=0)
                heights = torch.nn.functional.softplus(logits) + 1.0e-04
                log_a = torch.stack([params['dec_sk_log_k_a'] for params in params_list], dim=0)
                log_b = torch.stack([params['dec_sk_log_k_b'] for params in params_list], dim=0)
                cholesky_params = torch.stack([params['dec_sk_cholesky_params'] for params in params_list], dim=0)
                summary['families']['decoupled_spline_kumaraswamy_copula'] = {
                    'm1_knot_heights': self._tensor_summary_dict(heights),
                    'tail_log_k_a': self._tensor_summary_dict(log_a),
                    'tail_log_k_b': self._tensor_summary_dict(log_b),
                    'cholesky_unconstrained': self._tensor_summary_dict(cholesky_params),
                }
            return summary

    def _sanitize_param_store(self):
        sanitized = {}
        for name, value in pyro.get_param_store().items():
            clip_max = 20.0
            if self.flow_param_clip_max is not None and str(name).startswith('flow_transforms$$$'):
                clip_max = float(self.flow_param_clip_max)
            cleaned = torch.nan_to_num(value.data, nan=0.0, posinf=clip_max, neginf=-clip_max).clamp(min=-clip_max, max=clip_max)
            if not torch.equal(cleaned, value.data):
                value.data.copy_(cleaned)
                sanitized[name] = {**_tensor_stats_dict(cleaned), 'clip_max': float(clip_max)}
        return sanitized

    def _reset_flow_optimizer_state_for_sanitized_params(self, scheduler, sanitized_param_names):
        if not self.reset_flow_optimizer_state_on_sanitize or not sanitized_param_names:
            return {}
        sanitized_names = {str(name) for name in sanitized_param_names}
        reset = {}
        optim_objs = getattr(scheduler, 'optim_objs', {})
        for name, param in self.flow_transforms.named_parameters():
            full_name = f'flow_transforms$$${name}'
            if name not in sanitized_names and full_name not in sanitized_names:
                continue
            sched_obj = optim_objs.get(param)
            optimizer = getattr(sched_obj, 'optimizer', None)
            if optimizer is None:
                continue
            state = optimizer.state.get(param, {})
            reset_tensors = []
            for state_key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                tensor = state.get(state_key)
                if torch.is_tensor(tensor):
                    tensor.zero_()
                    reset_tensors.append(state_key)
            if reset_tensors:
                reset[full_name] = {'reset_tensors': reset_tensors}
        return reset

    @staticmethod
    def _build_cholesky_from_unconstrained(unconstrained_params, D, device):
        """Use the same bijection for evaluation and guide-density correction."""
        if unconstrained_params.shape[-1] != D * (D - 1) // 2:
            raise ValueError('Incorrect number of correlation parameters.')
        return CorrCholeskyTransform()(unconstrained_params)

    def _latent_log_abs_det_jacobian(self, latent_vector):
        """Jacobian from flow coordinates to the model's constrained sites.

        The model places LKJ priors on Cholesky factors and Uniform priors on
        normal locations. Their densities use different measures from the flow.
        """
        result = torch.zeros(latent_vector.shape[:-1], device=latent_vector.device,
                             dtype=latent_vector.dtype)
        offset = 0
        tail_dim = self.data_dim - 1
        for name, size in self.dims.items():
            raw = latent_vector[..., offset:offset + size]
            offset += size
            if not size:
                continue
            if name.endswith('_cholesky'):
                raw = raw.reshape(*raw.shape[:-1], -1, self.cholesky_params_dim)
                transform = CorrCholeskyTransform()
                result = result + transform.log_abs_det_jacobian(raw, transform(raw)).sum(-1)
                continue
            locations = None
            if name == 'tnormal_marginals':
                locations = raw.reshape(*raw.shape[:-1], -1, self.data_dim, 2)[..., 0]
            elif name == 'axis_lntnormal_marginals':
                locations = raw.reshape(*raw.shape[:-1], -1, 2 + 2 * tail_dim)[..., 2:2 + tail_dim]
            elif name in {'axis_bernstein_tnormal_marginals', 'axis_spline_tnormal_marginals',
                          'decoupled_spline_tnormal_marginals'}:
                n_basis = self.axis_bernstein_num_basis if 'bernstein' in name else self.m1_spline_num_params
                locations = raw.reshape(*raw.shape[:-1], -1, n_basis + 2 * tail_dim)[..., n_basis:n_basis + tail_dim]
            if locations is not None:
                log_jac = -torch.nn.functional.softplus(-locations) - torch.nn.functional.softplus(locations)
                result = result + log_jac.sum(dim=(-2, -1))
        return result

    def _unflatten_latent_vector(self, latent_vector, apply_numerical_limits=True):
        """Unpacks the flat latent vector into a structured dictionary of parameters."""
        # The model may saturate extreme values in its likelihood evaluation.
        # Guide sample sites must retain the continuous prior coordinates.
        def clamp(values, *args, **kwargs):
            return torch.clamp(values, *args, **kwargs) if apply_numerical_limits else values

        def clamp_log(values, *args, **kwargs):
            return self._clamp_log_param(values, *args, **kwargs) if apply_numerical_limits else values

        def clamp_prob(values):
            return self._clamp_prob_param(values) if apply_numerical_limits else values

        def clamp_kumaraswamy(values):
            return self._maybe_clamp_kumaraswamy_log_param(values) if apply_numerical_limits else values

        params = {}
        current_idx = 0
        end_idx = current_idx + self.dims['weights']
        params['weights_logits'] = clamp(latent_vector[..., current_idx:end_idx], min=-20.0, max=20.0)
        current_idx = end_idx
        if self.num_corr_kumaraswamy > 0:
            end_idx = current_idx + self.dims['k_marginals']
            k_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_kumaraswamy, self.data_dim, 2)
            params['log_k_a'] = clamp_kumaraswamy(k_params[..., 0])
            params['log_k_b'] = clamp_kumaraswamy(k_params[..., 1])
            current_idx = end_idx
            end_idx = current_idx + self.dims['k_cholesky']
            params['k_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_kumaraswamy, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_tnormal > 0:
            end_idx = current_idx + self.dims['tnormal_marginals']
            tn_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_tnormal, self.data_dim, 2)
            params['tn_locs'] = clamp_prob(torch.sigmoid(tn_params[..., 0]))
            params['log_tn_scales'] = clamp_log(tn_params[..., 1], low=-8.0, high=5.0)
            current_idx = end_idx
            end_idx = current_idx + self.dims['tnormal_cholesky']
            params['tnormal_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_tnormal, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_tpareto > 0:
            end_idx = current_idx + self.dims['tpareto_marginals']
            params['log_tp_alphas'] = clamp_log(latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_tpareto, self.data_dim))
            current_idx = end_idx
            end_idx = current_idx + self.dims['tpareto_cholesky']
            params['tpareto_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_tpareto, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_logitnormal > 0:
            end_idx = current_idx + self.dims['logitnormal_marginals']
            ln_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_logitnormal, self.data_dim, 2)
            params['ln_locs'] = clamp(ln_params[..., 0], min=-8.0, max=8.0)
            params['log_ln_scales'] = clamp_log(ln_params[..., 1], low=-8.0, high=5.0)
            current_idx = end_idx
            end_idx = current_idx + self.dims['logitnormal_cholesky']
            params['logitnormal_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_logitnormal, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_axis_lntnormal > 0:
            tail_dim = self.data_dim - 1
            end_idx = current_idx + self.dims['axis_lntnormal_marginals']
            axis_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_lntnormal, 2 + 2 * tail_dim)
            params['axis_lntn_ln_locs'] = clamp(axis_params[..., 0], min=-8.0, max=8.0)
            params['axis_lntn_log_ln_scales'] = clamp_log(axis_params[..., 1], low=-8.0, high=5.0)
            params['axis_lntn_tn_locs'] = clamp_prob(torch.sigmoid(axis_params[..., 2:2 + tail_dim]))
            params['axis_lntn_log_tn_scales'] = clamp_log(axis_params[..., 2 + tail_dim:2 + 2 * tail_dim], low=-8.0, high=5.0)
            current_idx = end_idx
            end_idx = current_idx + self.dims['axis_lntnormal_cholesky']
            params['axis_lntn_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_lntnormal, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_axis_lnkumaraswamy > 0:
            tail_dim = self.data_dim - 1
            end_idx = current_idx + self.dims['axis_lnkumaraswamy_marginals']
            axis_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_lnkumaraswamy, 2 + 2 * tail_dim)
            params['axis_lnk_ln_locs'] = clamp(axis_params[..., 0], min=-8.0, max=8.0)
            params['axis_lnk_log_ln_scales'] = clamp_log(axis_params[..., 1], low=-8.0, high=5.0)
            params['axis_lnk_log_k_a'] = clamp_kumaraswamy(axis_params[..., 2:2 + tail_dim])
            params['axis_lnk_log_k_b'] = clamp_kumaraswamy(axis_params[..., 2 + tail_dim:2 + 2 * tail_dim])
            current_idx = end_idx
            end_idx = current_idx + self.dims['axis_lnkumaraswamy_cholesky']
            params['axis_lnk_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_lnkumaraswamy, self.cholesky_params_dim)
            current_idx = end_idx
        m1_roughness_logits = {}
        if self.num_corr_axis_bernstein_tnormal > 0:
            tail_dim = self.data_dim - 1
            basis = self.axis_bernstein_num_basis
            end_idx = current_idx + self.dims['axis_bernstein_tnormal_marginals']
            axis_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_bernstein_tnormal, basis + 2 * tail_dim)
            params['axis_btn_m1_logits'] = clamp(axis_params[..., :basis], min=-12.0, max=12.0)
            params['axis_btn_tn_locs'] = clamp_prob(torch.sigmoid(axis_params[..., basis:basis + tail_dim]))
            params['axis_btn_log_tn_scales'] = clamp_log(axis_params[..., basis + tail_dim:basis + 2 * tail_dim], low=-8.0, high=5.0)
            current_idx = end_idx
            end_idx = current_idx + self.dims['axis_bernstein_tnormal_cholesky']
            params['axis_btn_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_bernstein_tnormal, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_axis_bernstein_kumaraswamy > 0:
            tail_dim = self.data_dim - 1
            basis = self.axis_bernstein_num_basis
            end_idx = current_idx + self.dims['axis_bernstein_kumaraswamy_marginals']
            axis_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_bernstein_kumaraswamy, basis + 2 * tail_dim)
            params['axis_bk_m1_logits'] = clamp(axis_params[..., :basis], min=-12.0, max=12.0)
            params['axis_bk_log_k_a'] = clamp_kumaraswamy(axis_params[..., basis:basis + tail_dim])
            params['axis_bk_log_k_b'] = clamp_kumaraswamy(axis_params[..., basis + tail_dim:basis + 2 * tail_dim])
            current_idx = end_idx
            end_idx = current_idx + self.dims['axis_bernstein_kumaraswamy_cholesky']
            params['axis_bk_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_bernstein_kumaraswamy, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_axis_spline_tnormal > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            end_idx = current_idx + self.dims['axis_spline_tnormal_marginals']
            axis_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_spline_tnormal, m1_params + 2 * tail_dim)
            params['axis_stn_m1_logits'] = clamp(axis_params[..., :m1_params], min=-12.0, max=12.0)
            params['axis_stn_tn_locs'] = clamp_prob(torch.sigmoid(axis_params[..., m1_params:m1_params + tail_dim]))
            params['axis_stn_log_tn_scales'] = clamp_log(axis_params[..., m1_params + tail_dim:m1_params + 2 * tail_dim], low=-8.0, high=5.0)
            current_idx = end_idx
            end_idx = current_idx + self.dims['axis_spline_tnormal_cholesky']
            params['axis_stn_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_spline_tnormal, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_axis_spline_kumaraswamy > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            end_idx = current_idx + self.dims['axis_spline_kumaraswamy_marginals']
            axis_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_spline_kumaraswamy, m1_params + 2 * tail_dim)
            params['axis_sk_m1_logits'] = clamp(axis_params[..., :m1_params], min=-12.0, max=12.0)
            params['axis_sk_log_k_a'] = clamp_kumaraswamy(axis_params[..., m1_params:m1_params + tail_dim])
            params['axis_sk_log_k_b'] = clamp_kumaraswamy(axis_params[..., m1_params + tail_dim:m1_params + 2 * tail_dim])
            current_idx = end_idx
            end_idx = current_idx + self.dims['axis_spline_kumaraswamy_cholesky']
            params['axis_sk_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_axis_spline_kumaraswamy, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_decoupled_spline_tnormal_copula > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            end_idx = current_idx + self.dims['decoupled_spline_tnormal_marginals']
            dec_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], m1_params + 2 * tail_dim)
            params['dec_stn_m1_logits'] = clamp(dec_params[..., :m1_params], min=-12.0, max=12.0)
            params['dec_stn_tn_locs'] = clamp_prob(torch.sigmoid(dec_params[..., m1_params:m1_params + tail_dim]))
            params['dec_stn_log_tn_scales'] = clamp_log(dec_params[..., m1_params + tail_dim:m1_params + 2 * tail_dim], low=-8.0, high=5.0)
            current_idx = end_idx
            end_idx = current_idx + self.dims['decoupled_spline_tnormal_cholesky']
            params['dec_stn_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_decoupled_spline_tnormal_copula, self.cholesky_params_dim)
            current_idx = end_idx
        if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            end_idx = current_idx + self.dims['decoupled_spline_kumaraswamy_marginals']
            dec_params = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], m1_params + 2 * tail_dim)
            params['dec_sk_m1_logits'] = clamp(dec_params[..., :m1_params], min=-12.0, max=12.0)
            params['dec_sk_log_k_a'] = clamp_kumaraswamy(dec_params[..., m1_params:m1_params + tail_dim])
            params['dec_sk_log_k_b'] = clamp_kumaraswamy(dec_params[..., m1_params + tail_dim:m1_params + 2 * tail_dim])
            current_idx = end_idx
            end_idx = current_idx + self.dims['decoupled_spline_kumaraswamy_cholesky']
            params['dec_sk_cholesky_params'] = latent_vector[..., current_idx:end_idx].view(*latent_vector.shape[:-1], self.num_corr_decoupled_spline_kumaraswamy_copula, self.cholesky_params_dim)
            current_idx = end_idx
        assert current_idx == self.latent_dim, 'Dimension mismatch during un-flattening!'
        return params

    @torch.no_grad()
    def sample(self, num_samples, num_posterior_draws=1):
        """
        Draws samples from the learned population distribution.
        """
        if not self.is_trained:
            raise RuntimeError('Model must be trained first.')
        if num_posterior_draws != 1:
            raise NotImplementedError('This sampling method is implemented for num_posterior_draws=1.')
        latent_samples = self.flow_dist.sample(torch.Size([1]))
        params = self._unflatten_latent_vector(latent_samples)
        weights = torch.softmax(params['weights_logits'].squeeze(0), dim=-1)
        cat_dist = dist.Categorical(weights)
        component_indices = cat_dist.sample(torch.Size([num_samples]))
        all_samples_norm = torch.zeros(num_samples, self.data_dim, device=self.device)
        current_k = 0
        if self.num_corr_kumaraswamy > 0:
            all_cholesky_params = params['k_cholesky_params'].squeeze(0)
            all_log_a = params['log_k_a'].squeeze(0)
            all_log_b = params['log_k_b'].squeeze(0)
            for k in range(self.num_corr_kumaraswamy):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    a_k = torch.exp(all_log_a[k]) + 0.0001
                    b_k = torch.exp(all_log_b[k]) + 0.0001
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    marginal_dist = SafeKumaraswamy(a_k, b_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_tnormal > 0:
            all_cholesky_params = params['tnormal_cholesky_params'].squeeze(0)
            all_locs = params['tn_locs'].squeeze(0)
            all_log_scales = params['log_tn_scales'].squeeze(0)
            for k in range(self.num_corr_tnormal):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    locs_k = all_locs[k]
                    scales_k = torch.exp(all_log_scales[k]) + 0.0001
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    marginal_dist = TruncatedNormal(locs_k, scales_k, low=0.0001, high=1.0 - 0.0001)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_tpareto > 0:
            all_cholesky_params = params['tpareto_cholesky_params'].squeeze(0)
            all_log_alphas = params['log_tp_alphas'].squeeze(0)
            for k in range(self.num_corr_tpareto):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    alphas_k = torch.exp(all_log_alphas[k]) + 0.0001
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    marginal_dist = TruncatedPareto(alphas_k, low=0.0001, high=1.0)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_logitnormal > 0:
            all_cholesky_params = params['logitnormal_cholesky_params'].squeeze(0)
            all_locs = params['ln_locs'].squeeze(0)
            all_log_scales = params['log_ln_scales'].squeeze(0)
            for k in range(self.num_corr_logitnormal):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    locs_k = all_locs[k]
                    scales_k = torch.exp(all_log_scales[k]) + 0.0001
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    marginal_dist = BoundedLogitNormal(locs_k, scales_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_axis_lntnormal > 0:
            all_cholesky_params = params['axis_lntn_cholesky_params'].squeeze(0)
            all_ln_locs = params['axis_lntn_ln_locs'].squeeze(0)
            all_log_ln_scales = params['axis_lntn_log_ln_scales'].squeeze(0)
            all_tn_locs = params['axis_lntn_tn_locs'].squeeze(0)
            all_log_tn_scales = params['axis_lntn_log_tn_scales'].squeeze(0)
            for k in range(self.num_corr_axis_lntnormal):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    marginal_dist = self._axis_lntnormal_marginal_dist(
                        all_ln_locs[k],
                        all_log_ln_scales[k],
                        all_tn_locs[k],
                        all_log_tn_scales[k],
                    )
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_axis_lnkumaraswamy > 0:
            all_cholesky_params = params['axis_lnk_cholesky_params'].squeeze(0)
            all_ln_locs = params['axis_lnk_ln_locs'].squeeze(0)
            all_log_ln_scales = params['axis_lnk_log_ln_scales'].squeeze(0)
            all_log_k_a = params['axis_lnk_log_k_a'].squeeze(0)
            all_log_k_b = params['axis_lnk_log_k_b'].squeeze(0)
            for k in range(self.num_corr_axis_lnkumaraswamy):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    marginal_dist = self._axis_lnkumaraswamy_marginal_dist(
                        all_ln_locs[k],
                        all_log_ln_scales[k],
                        all_log_k_a[k],
                        all_log_k_b[k],
                    )
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_axis_bernstein_tnormal > 0:
            all_cholesky_params = params['axis_btn_cholesky_params'].squeeze(0)
            all_m1_logits = params['axis_btn_m1_logits'].squeeze(0)
            all_tn_locs = params['axis_btn_tn_locs'].squeeze(0)
            all_log_tn_scales = params['axis_btn_log_tn_scales'].squeeze(0)
            for k in range(self.num_corr_axis_bernstein_tnormal):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    marginal_dist = self._axis_bernstein_tnormal_marginal_dist(
                        all_m1_logits[k],
                        all_tn_locs[k],
                        all_log_tn_scales[k],
                    )
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_axis_bernstein_kumaraswamy > 0:
            all_cholesky_params = params['axis_bk_cholesky_params'].squeeze(0)
            all_m1_logits = params['axis_bk_m1_logits'].squeeze(0)
            all_log_k_a = params['axis_bk_log_k_a'].squeeze(0)
            all_log_k_b = params['axis_bk_log_k_b'].squeeze(0)
            for k in range(self.num_corr_axis_bernstein_kumaraswamy):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    marginal_dist = self._axis_bernstein_kumaraswamy_marginal_dist(
                        all_m1_logits[k],
                        all_log_k_a[k],
                        all_log_k_b[k],
                    )
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_axis_spline_tnormal > 0:
            all_cholesky_params = params['axis_stn_cholesky_params'].squeeze(0)
            all_m1_logits = params['axis_stn_m1_logits'].squeeze(0)
            all_tn_locs = params['axis_stn_tn_locs'].squeeze(0)
            all_log_tn_scales = params['axis_stn_log_tn_scales'].squeeze(0)
            for k in range(self.num_corr_axis_spline_tnormal):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    marginal_dist = self._axis_spline_tnormal_marginal_dist(
                        all_m1_logits[k],
                        all_tn_locs[k],
                        all_log_tn_scales[k],
                    )
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_axis_spline_kumaraswamy > 0:
            all_cholesky_params = params['axis_sk_cholesky_params'].squeeze(0)
            all_m1_logits = params['axis_sk_m1_logits'].squeeze(0)
            all_log_k_a = params['axis_sk_log_k_a'].squeeze(0)
            all_log_k_b = params['axis_sk_log_k_b'].squeeze(0)
            for k in range(self.num_corr_axis_spline_kumaraswamy):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    marginal_dist = self._axis_spline_kumaraswamy_marginal_dist(
                        all_m1_logits[k],
                        all_log_k_a[k],
                        all_log_k_b[k],
                    )
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_decoupled_spline_tnormal_copula > 0:
            all_cholesky_params = params['dec_stn_cholesky_params'].squeeze(0)
            marginal_dist = self._decoupled_spline_tnormal_marginal_dist(
                params['dec_stn_m1_logits'].squeeze(0),
                params['dec_stn_tn_locs'].squeeze(0),
                params['dec_stn_log_tn_scales'].squeeze(0),
            )
            for k in range(self.num_corr_decoupled_spline_tnormal_copula):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
            all_cholesky_params = params['dec_sk_cholesky_params'].squeeze(0)
            marginal_dist = self._decoupled_spline_kumaraswamy_marginal_dist(
                params['dec_sk_m1_logits'].squeeze(0),
                params['dec_sk_log_k_a'].squeeze(0),
                params['dec_sk_log_k_b'].squeeze(0),
            )
            for k in range(self.num_corr_decoupled_spline_kumaraswamy_copula):
                mask = component_indices == current_k
                if mask.any():
                    n_s = mask.sum()
                    L_k = self._build_cholesky_from_unconstrained(all_cholesky_params[k], self.data_dim, self.device)
                    copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_k)
                    y = copula_mvn.sample((n_s,))
                    u = self._standard_normal.cdf(y)
                    all_samples_norm[mask] = marginal_dist.icdf(u)
                current_k += 1
        all_samples = all_samples_norm * (self.max_vals - self.min_vals) + self.min_vals
        return all_samples

    def _pyro_model(self, data_samples):
        device = data_samples[0].device if isinstance(data_samples, list) else data_samples.device
        num_events = len(data_samples) if isinstance(data_samples, list) else data_samples.shape[0]
        num_samples_per_event = data_samples[0].shape[0] if isinstance(data_samples, list) else data_samples.shape[1]
        data_tensor = torch.stack(data_samples) if isinstance(data_samples, list) else data_samples
        samples_phys = data_tensor[..., :3]
        priors = data_tensor[..., 3]
        sample_weights = data_tensor[..., 4] if data_tensor.shape[-1] >= 5 else torch.ones_like(priors)
        data_norm = (samples_phys - self.min_vals) / (self.max_vals - self.min_vals)
        in_support = ((data_norm >= 0) & (data_norm <= 1)).all(dim=-1)
        extra_support = getattr(self, 'event_support_function', None)
        if extra_support is not None:
            in_support = in_support & extra_support(samples_phys)
        if not torch.all(in_support.any(dim=1)):
            raise ValueError('An event has no posterior samples in the population support.')
        data_norm = data_norm.clamp(min=1e-07, max=1.0 - 1e-07)
        zeros = torch.zeros((), device=device)
        ones = torch.ones((), device=device)
        weights_logits = torch.clamp(pyro.sample('weights_logits', dist.Normal(zeros, ones).expand([self.K]).to_event(1)), min=-20.0, max=20.0)
        log_probs_per_component = []
        if self.num_corr_kumaraswamy > 0:
            with pyro.plate('kumaraswamy_components', self.num_corr_kumaraswamy):
                log_k_a_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('log_k_a', dist.Normal(zeros, ones).expand([self.data_dim]).to_event(1)))
                log_k_b_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('log_k_b', dist.Normal(zeros, ones).expand([self.data_dim]).to_event(1)))
                k_L_cholesky_val = pyro.sample('k_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            a = torch.exp(log_k_a_val) + 0.0001
            b = torch.exp(log_k_b_val) + 0.0001
            marginal_dist = SafeKumaraswamy(a, b)
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, k_L_cholesky_val, data_norm))
        if self.num_corr_tnormal > 0:
            with pyro.plate('tnormal_components', self.num_corr_tnormal):
                tn_locs_val = self._clamp_prob_param(pyro.sample('tn_locs', dist.Uniform(zeros, ones).expand([self.data_dim]).to_event(1)))
                log_tn_scales_val = self._clamp_log_param(pyro.sample('log_tn_scales', dist.Normal(zeros, ones).expand([self.data_dim]).to_event(1)), low=-8.0, high=5.0)
                tnormal_L_cholesky_val = pyro.sample('tnormal_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            scales = torch.exp(log_tn_scales_val) + 0.0001
            marginal_dist = TruncatedNormal(tn_locs_val, scales, low=0.0001, high=1.0 - 0.0001)
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, tnormal_L_cholesky_val, data_norm))
        if self.num_corr_tpareto > 0:
            with pyro.plate('tpareto_components', self.num_corr_tpareto):
                log_tp_alphas_val = self._clamp_log_param(pyro.sample('log_tp_alphas', dist.Normal(zeros, ones).expand([self.data_dim]).to_event(1)))
                tpareto_L_cholesky_val = pyro.sample('tpareto_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            alphas = torch.exp(log_tp_alphas_val) + 0.0001
            marginal_dist = TruncatedPareto(alphas, low=0.0001, high=1.0)
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, tpareto_L_cholesky_val, data_norm))
        if self.num_corr_logitnormal > 0:
            with pyro.plate('logitnormal_components', self.num_corr_logitnormal):
                ln_locs_val = torch.clamp(pyro.sample('ln_locs', dist.Normal(zeros, ones).expand([self.data_dim]).to_event(1)), min=-8.0, max=8.0)
                log_ln_scales_val = self._clamp_log_param(pyro.sample('log_ln_scales', dist.Normal(zeros, ones).expand([self.data_dim]).to_event(1)), low=-8.0, high=5.0)
                logitnormal_L_cholesky_val = pyro.sample('logitnormal_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            ln_scales = torch.exp(log_ln_scales_val) + 0.0001
            marginal_dist = BoundedLogitNormal(ln_locs_val, ln_scales)
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, logitnormal_L_cholesky_val, data_norm))
        if self.num_corr_axis_lntnormal > 0:
            tail_dim = self.data_dim - 1
            with pyro.plate('axis_lntnormal_components', self.num_corr_axis_lntnormal):
                axis_lntn_ln_locs_val = torch.clamp(pyro.sample('axis_lntn_ln_locs', dist.Normal(zeros, ones)), min=-8.0, max=8.0)
                axis_lntn_log_ln_scales_val = self._clamp_log_param(pyro.sample('axis_lntn_log_ln_scales', dist.Normal(zeros, ones)), low=-8.0, high=5.0)
                axis_lntn_tn_locs_val = self._clamp_prob_param(pyro.sample('axis_lntn_tn_locs', dist.Uniform(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_lntn_log_tn_scales_val = self._clamp_log_param(pyro.sample('axis_lntn_log_tn_scales', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)), low=-8.0, high=5.0)
                axis_lntn_L_cholesky_val = pyro.sample('axis_lntn_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            marginal_dist = self._axis_lntnormal_marginal_dist(
                axis_lntn_ln_locs_val,
                axis_lntn_log_ln_scales_val,
                axis_lntn_tn_locs_val,
                axis_lntn_log_tn_scales_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, axis_lntn_L_cholesky_val, data_norm))
        if self.num_corr_axis_lnkumaraswamy > 0:
            tail_dim = self.data_dim - 1
            with pyro.plate('axis_lnkumaraswamy_components', self.num_corr_axis_lnkumaraswamy):
                axis_lnk_ln_locs_val = torch.clamp(pyro.sample('axis_lnk_ln_locs', dist.Normal(zeros, ones)), min=-8.0, max=8.0)
                axis_lnk_log_ln_scales_val = self._clamp_log_param(pyro.sample('axis_lnk_log_ln_scales', dist.Normal(zeros, ones)), low=-8.0, high=5.0)
                axis_lnk_log_k_a_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('axis_lnk_log_k_a', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_lnk_log_k_b_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('axis_lnk_log_k_b', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_lnk_L_cholesky_val = pyro.sample('axis_lnk_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            marginal_dist = self._axis_lnkumaraswamy_marginal_dist(
                axis_lnk_ln_locs_val,
                axis_lnk_log_ln_scales_val,
                axis_lnk_log_k_a_val,
                axis_lnk_log_k_b_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, axis_lnk_L_cholesky_val, data_norm))
        m1_roughness_logits = {}
        if self.num_corr_axis_bernstein_tnormal > 0:
            tail_dim = self.data_dim - 1
            basis = self.axis_bernstein_num_basis
            with pyro.plate('axis_bernstein_tnormal_components', self.num_corr_axis_bernstein_tnormal):
                axis_btn_m1_logits_val = torch.clamp(pyro.sample('axis_btn_m1_logits', dist.Normal(zeros, ones).expand([basis]).to_event(1)), min=-12.0, max=12.0)
                axis_btn_tn_locs_val = self._clamp_prob_param(pyro.sample('axis_btn_tn_locs', dist.Uniform(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_btn_log_tn_scales_val = self._clamp_log_param(pyro.sample('axis_btn_log_tn_scales', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)), low=-8.0, high=5.0)
                axis_btn_L_cholesky_val = pyro.sample('axis_btn_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            m1_roughness_logits['axis_btn_m1_logits'] = axis_btn_m1_logits_val
            marginal_dist = self._axis_bernstein_tnormal_marginal_dist(
                axis_btn_m1_logits_val,
                axis_btn_tn_locs_val,
                axis_btn_log_tn_scales_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, axis_btn_L_cholesky_val, data_norm))
        if self.num_corr_axis_bernstein_kumaraswamy > 0:
            tail_dim = self.data_dim - 1
            basis = self.axis_bernstein_num_basis
            with pyro.plate('axis_bernstein_kumaraswamy_components', self.num_corr_axis_bernstein_kumaraswamy):
                axis_bk_m1_logits_val = torch.clamp(pyro.sample('axis_bk_m1_logits', dist.Normal(zeros, ones).expand([basis]).to_event(1)), min=-12.0, max=12.0)
                axis_bk_log_k_a_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('axis_bk_log_k_a', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_bk_log_k_b_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('axis_bk_log_k_b', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_bk_L_cholesky_val = pyro.sample('axis_bk_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            m1_roughness_logits['axis_bk_m1_logits'] = axis_bk_m1_logits_val
            marginal_dist = self._axis_bernstein_kumaraswamy_marginal_dist(
                axis_bk_m1_logits_val,
                axis_bk_log_k_a_val,
                axis_bk_log_k_b_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, axis_bk_L_cholesky_val, data_norm))
        if self.num_corr_axis_spline_tnormal > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            with pyro.plate('axis_spline_tnormal_components', self.num_corr_axis_spline_tnormal):
                axis_stn_m1_logits_val = torch.clamp(pyro.sample('axis_stn_m1_logits', dist.Normal(zeros, ones).expand([m1_params]).to_event(1)), min=-12.0, max=12.0)
                axis_stn_tn_locs_val = self._clamp_prob_param(pyro.sample('axis_stn_tn_locs', dist.Uniform(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_stn_log_tn_scales_val = self._clamp_log_param(pyro.sample('axis_stn_log_tn_scales', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)), low=-8.0, high=5.0)
                axis_stn_L_cholesky_val = pyro.sample('axis_stn_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            m1_roughness_logits['axis_stn_m1_logits'] = axis_stn_m1_logits_val
            marginal_dist = self._axis_spline_tnormal_marginal_dist(
                axis_stn_m1_logits_val,
                axis_stn_tn_locs_val,
                axis_stn_log_tn_scales_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, axis_stn_L_cholesky_val, data_norm))
        if self.num_corr_axis_spline_kumaraswamy > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            with pyro.plate('axis_spline_kumaraswamy_components', self.num_corr_axis_spline_kumaraswamy):
                axis_sk_m1_logits_val = torch.clamp(pyro.sample('axis_sk_m1_logits', dist.Normal(zeros, ones).expand([m1_params]).to_event(1)), min=-12.0, max=12.0)
                axis_sk_log_k_a_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('axis_sk_log_k_a', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_sk_log_k_b_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('axis_sk_log_k_b', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
                axis_sk_L_cholesky_val = pyro.sample('axis_sk_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            m1_roughness_logits['axis_sk_m1_logits'] = axis_sk_m1_logits_val
            marginal_dist = self._axis_spline_kumaraswamy_marginal_dist(
                axis_sk_m1_logits_val,
                axis_sk_log_k_a_val,
                axis_sk_log_k_b_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, axis_sk_L_cholesky_val, data_norm))
        if self.num_corr_decoupled_spline_tnormal_copula > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            dec_stn_m1_logits_val = torch.clamp(pyro.sample('dec_stn_m1_logits', dist.Normal(zeros, ones).expand([m1_params]).to_event(1)), min=-12.0, max=12.0)
            dec_stn_tn_locs_val = self._clamp_prob_param(pyro.sample('dec_stn_tn_locs', dist.Uniform(zeros, ones).expand([tail_dim]).to_event(1)))
            dec_stn_log_tn_scales_val = self._clamp_log_param(pyro.sample('dec_stn_log_tn_scales', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)), low=-8.0, high=5.0)
            m1_roughness_logits['dec_stn_m1_logits'] = dec_stn_m1_logits_val
            with pyro.plate('decoupled_spline_tnormal_copula_components', self.num_corr_decoupled_spline_tnormal_copula):
                dec_stn_L_cholesky_val = pyro.sample('dec_stn_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            marginal_dist = self._decoupled_spline_tnormal_marginal_dist(
                dec_stn_m1_logits_val,
                dec_stn_tn_locs_val,
                dec_stn_log_tn_scales_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, dec_stn_L_cholesky_val, data_norm))
        if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
            tail_dim = self.data_dim - 1
            m1_params = self.m1_spline_num_params
            dec_sk_m1_logits_val = torch.clamp(pyro.sample('dec_sk_m1_logits', dist.Normal(zeros, ones).expand([m1_params]).to_event(1)), min=-12.0, max=12.0)
            dec_sk_log_k_a_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('dec_sk_log_k_a', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
            dec_sk_log_k_b_val = self._maybe_clamp_kumaraswamy_log_param(pyro.sample('dec_sk_log_k_b', dist.Normal(zeros, ones).expand([tail_dim]).to_event(1)))
            m1_roughness_logits['dec_sk_m1_logits'] = dec_sk_m1_logits_val
            with pyro.plate('decoupled_spline_kumaraswamy_copula_components', self.num_corr_decoupled_spline_kumaraswamy_copula):
                dec_sk_L_cholesky_val = pyro.sample('dec_sk_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=torch.tensor(1.5, device=device)))
            marginal_dist = self._decoupled_spline_kumaraswamy_marginal_dist(
                dec_sk_m1_logits_val,
                dec_sk_log_k_a_val,
                dec_sk_log_k_b_val,
            )
            log_probs_per_component.append(self._copula_log_prob(marginal_dist, dec_sk_L_cholesky_val, data_norm))
        log_probs_all = torch.cat(log_probs_per_component, dim=2)
        log_weights = torch.log_softmax(weights_logits, dim=-1)
        log_prob_mix_per_sample = torch.logsumexp(log_weights + log_probs_all, dim=2)
        log_prob_reweighted = log_prob_mix_per_sample - torch.log(priors.clamp(min=1e-100))
        self._ensure_finite('log_prob_reweighted', log_prob_reweighted)
        log_event_sample_weights = torch.log(sample_weights.clamp(min=1e-100))
        self._ensure_finite('log_event_sample_weights', log_event_sample_weights)
        log_weighted_event_integrand = log_prob_reweighted + log_event_sample_weights
        self._ensure_finite('log_weighted_event_integrand', log_weighted_event_integrand)
        log_weighted_event_integrand = torch.where(
            in_support, log_weighted_event_integrand, -torch.inf)
        weights_per_sample = torch.exp(log_weighted_event_integrand - torch.max(log_weighted_event_integrand, dim=1, keepdim=True).values)
        neff_per_event = compute_neff(weights_per_sample)
        min_neff_obs = torch.min(neff_per_event)
        self._capture_model_diagnostics(selection_mode='none', min_neff_obs=min_neff_obs)
        NEFF_OBS_THR = float(self.obs_neff_threshold)
        if NEFF_OBS_THR > 0.0 and min_neff_obs < NEFF_OBS_THR:
            pyro.factor('neff_obs_penalty', torch.tensor(-10000000000.0, device=device))
        obs_neff_penalty_applied = bool(NEFF_OBS_THR > 0.0 and self._safe_float(min_neff_obs) is not None and self._safe_float(min_neff_obs) < NEFF_OBS_THR)
        obs_neff_penalty_value = -10000000000.0 if obs_neff_penalty_applied else 0.0
        log_prob_per_event = torch.logsumexp(log_weighted_event_integrand, dim=1) - torch.logsumexp(log_event_sample_weights, dim=1)
        q_monotonicity_penalty_value = 0.0
        q_monotonicity_min_diff = None
        q_monotonicity_negative_diff_count = 0
        selection_neff_penalty_applied = False
        selection_neff_penalty_value = 0.0
        selection_neff_threshold = None
        if self.correct_selection_effects:
            current_params = {'weights_logits': weights_logits}
            if self.num_corr_kumaraswamy > 0:
                current_params['log_k_a'] = log_k_a_val
                current_params['log_k_b'] = log_k_b_val
                current_params['k_L_cholesky'] = k_L_cholesky_val
            if self.num_corr_tnormal > 0:
                current_params['tn_locs'] = tn_locs_val
                current_params['log_tn_scales'] = log_tn_scales_val
                current_params['tnormal_L_cholesky'] = tnormal_L_cholesky_val
            if self.num_corr_tpareto > 0:
                current_params['log_tp_alphas'] = log_tp_alphas_val
                current_params['tpareto_L_cholesky'] = tpareto_L_cholesky_val
            if self.num_corr_logitnormal > 0:
                current_params['ln_locs'] = ln_locs_val
                current_params['log_ln_scales'] = log_ln_scales_val
                current_params['logitnormal_L_cholesky'] = logitnormal_L_cholesky_val
            if self.num_corr_axis_lntnormal > 0:
                current_params['axis_lntn_ln_locs'] = axis_lntn_ln_locs_val
                current_params['axis_lntn_log_ln_scales'] = axis_lntn_log_ln_scales_val
                current_params['axis_lntn_tn_locs'] = axis_lntn_tn_locs_val
                current_params['axis_lntn_log_tn_scales'] = axis_lntn_log_tn_scales_val
                current_params['axis_lntn_L_cholesky'] = axis_lntn_L_cholesky_val
            if self.num_corr_axis_lnkumaraswamy > 0:
                current_params['axis_lnk_ln_locs'] = axis_lnk_ln_locs_val
                current_params['axis_lnk_log_ln_scales'] = axis_lnk_log_ln_scales_val
                current_params['axis_lnk_log_k_a'] = axis_lnk_log_k_a_val
                current_params['axis_lnk_log_k_b'] = axis_lnk_log_k_b_val
                current_params['axis_lnk_L_cholesky'] = axis_lnk_L_cholesky_val
            if self.num_corr_axis_bernstein_tnormal > 0:
                current_params['axis_btn_m1_logits'] = axis_btn_m1_logits_val
                current_params['axis_btn_tn_locs'] = axis_btn_tn_locs_val
                current_params['axis_btn_log_tn_scales'] = axis_btn_log_tn_scales_val
                current_params['axis_btn_L_cholesky'] = axis_btn_L_cholesky_val
            if self.num_corr_axis_bernstein_kumaraswamy > 0:
                current_params['axis_bk_m1_logits'] = axis_bk_m1_logits_val
                current_params['axis_bk_log_k_a'] = axis_bk_log_k_a_val
                current_params['axis_bk_log_k_b'] = axis_bk_log_k_b_val
                current_params['axis_bk_L_cholesky'] = axis_bk_L_cholesky_val
            if self.num_corr_axis_spline_tnormal > 0:
                current_params['axis_stn_m1_logits'] = axis_stn_m1_logits_val
                current_params['axis_stn_tn_locs'] = axis_stn_tn_locs_val
                current_params['axis_stn_log_tn_scales'] = axis_stn_log_tn_scales_val
                current_params['axis_stn_L_cholesky'] = axis_stn_L_cholesky_val
            if self.num_corr_axis_spline_kumaraswamy > 0:
                current_params['axis_sk_m1_logits'] = axis_sk_m1_logits_val
                current_params['axis_sk_log_k_a'] = axis_sk_log_k_a_val
                current_params['axis_sk_log_k_b'] = axis_sk_log_k_b_val
                current_params['axis_sk_L_cholesky'] = axis_sk_L_cholesky_val
            if self.num_corr_decoupled_spline_tnormal_copula > 0:
                current_params['dec_stn_m1_logits'] = dec_stn_m1_logits_val
                current_params['dec_stn_tn_locs'] = dec_stn_tn_locs_val
                current_params['dec_stn_log_tn_scales'] = dec_stn_log_tn_scales_val
                current_params['dec_stn_L_cholesky'] = dec_stn_L_cholesky_val
            if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
                current_params['dec_sk_m1_logits'] = dec_sk_m1_logits_val
                current_params['dec_sk_log_k_a'] = dec_sk_log_k_a_val
                current_params['dec_sk_log_k_b'] = dec_sk_log_k_b_val
                current_params['dec_sk_L_cholesky'] = dec_sk_L_cholesky_val
            if self.injection_data is not None and self.selection_function is None:
                inj_log_m1 = self.injection_data['log_m1']
                inj_z = self.injection_data['z']
                inj_q = self.injection_data['q']
                inj_points = torch.stack([inj_log_m1, inj_z, inj_q], dim=1)
                inj_points_norm = (inj_points - self.min_vals) / (self.max_vals - self.min_vals)
                log_selection_fraction = self._log_selection_fraction_from_params(current_params, inj_points_norm)
                self._ensure_finite('log_selection_fraction(model)', log_selection_fraction)
                weights_inj = torch.exp(self._last_selection_log_weights - torch.max(self._last_selection_log_weights))
                neff_sel = compute_neff(weights_inj)
                self._capture_model_diagnostics(
                    selection_mode='injection',
                    min_neff_obs=min_neff_obs,
                    selection_fraction=torch.exp(log_selection_fraction),
                    neff_sel=neff_sel,
                )
                NEFF_SEL_THR = num_events * float(self.selection_neff_multiplier)
                selection_neff_threshold = NEFF_SEL_THR
                if NEFF_SEL_THR > 0.0 and neff_sel < NEFF_SEL_THR:
                    pyro.factor('neff_sel_penalty', torch.tensor(-10000000000.0, device=device))
                selection_neff_penalty_applied = bool(NEFF_SEL_THR > 0.0 and self._safe_float(neff_sel) is not None and self._safe_float(neff_sel) < NEFF_SEL_THR)
                selection_neff_penalty_value = -10000000000.0 if selection_neff_penalty_applied else 0.0
            elif self.selection_function is not None:
                selection_fraction = self._selection_fraction_for_params(current_params)
                self._ensure_finite('selection_fraction(model_mc)', selection_fraction)
                log_selection_fraction = torch.log(selection_fraction.clamp(min=1e-12))
                self._capture_model_diagnostics(
                    selection_mode='selection_function',
                    min_neff_obs=min_neff_obs,
                    selection_fraction=selection_fraction,
                    neff_sel=None,
                )
            else:
                raise ValueError('Selection effects requested, but neither injection_data nor selection_function is available.')
            '\n            # 7. (可选) 不可观测质量惩罚 - 方案 C (解决归一化逃逸问题)            \n            # 参数:\n            #   - sigma_log_eff: 允许的对数偏离标准差\n            #                    σ=1.0 意味着 ε 在 [det_eff/e, det_eff*e] 内单侧 1-sigma\n            #   - strength: 先验强度 (相当于有效样本数)\n\n            log_det_eff = self.injection_data[\'log_det_eff\']  # 预计算的 log(det_eff)\n\n            # 计算 log(ε / det_eff) = log(ε) - log(det_eff)\n            log_ratio = log_selection_fraction - log_det_eff\n\n            \n            margin = 1.0    # 允许 ε 最低到 det_eff / e ≈ det_eff * 0.37\n            strength = 1e6  # 惩罚强度\n            \n            # 当 log_ratio < -margin 时 (即 ε < det_eff * exp(-margin))，惩罚开始\n            one_sided_penalty = strength * torch.clamp(log_ratio + margin, max=0.0)\n            \n            pyro.factor("selection_efficiency_prior", one_sided_penalty)\n            '
            if float(self.q_monotonicity_lambda) > 0.0:
                lambda_mono = float(self.q_monotonicity_lambda)
                num_q_points = int(self.q_monotonicity_num_points)
                q_grid_norm = torch.linspace(0.1, 0.99, num_q_points, device=device)
                m1_fixed_norm = torch.full((num_q_points,), 0.5, device=device)
                z_fixed_norm = torch.full((num_q_points,), 0.3, device=device)
                mono_test_points = torch.stack([m1_fixed_norm, z_fixed_norm, q_grid_norm], dim=1)
                mono_log_pdf = self._log_pdf_phys_from_params(current_params, mono_test_points)
                mono_diffs = mono_log_pdf[1:] - mono_log_pdf[:-1]
                mono_penalty = lambda_mono * torch.sum(torch.relu(-mono_diffs))
                pyro.factor('q_monotonicity_penalty', -mono_penalty)
                q_monotonicity_penalty_value = self._safe_float(mono_penalty) or 0.0
                q_monotonicity_min_diff = self._safe_float(torch.min(mono_diffs))
                q_monotonicity_negative_diff_count = int(torch.sum(mono_diffs.detach() < 0.0).item())
            self._apply_m1_roughness_factor(m1_roughness_logits)
            sum_log_prob = torch.sum(log_prob_per_event)
            final_log_prob = sum_log_prob - num_events * log_selection_fraction
            if self.diagnostic_objective_terms_enabled:
                self._latest_model_diagnostics.update(
                    {
                        'num_events': int(num_events),
                        'num_samples_per_event': int(num_samples_per_event),
                        'event_loglike_sum': self._safe_float(sum_log_prob),
                        'event_loglike_mean': self._safe_float(torch.mean(log_prob_per_event)),
                        'event_loglike_min': self._safe_float(torch.min(log_prob_per_event)),
                        'event_loglike_max': self._safe_float(torch.max(log_prob_per_event)),
                        'log_selection_fraction': self._safe_float(log_selection_fraction),
                        'selection_objective_term': self._safe_float(-num_events * log_selection_fraction),
                        'obs_log_prob_factor': self._safe_float(final_log_prob),
                        'obs_neff_threshold': float(NEFF_OBS_THR),
                        'obs_neff_penalty_applied': bool(obs_neff_penalty_applied),
                        'obs_neff_penalty_value': float(obs_neff_penalty_value),
                        'selection_neff_threshold': self._safe_float(selection_neff_threshold),
                        'selection_neff_penalty_applied': bool(selection_neff_penalty_applied),
                        'selection_neff_penalty_value': float(selection_neff_penalty_value),
                        'q_monotonicity_lambda': float(self.q_monotonicity_lambda),
                        'q_monotonicity_penalty_value': float(q_monotonicity_penalty_value),
                        'q_monotonicity_factor_value': float(-q_monotonicity_penalty_value),
                        'q_monotonicity_min_diff': q_monotonicity_min_diff,
                        'q_monotonicity_negative_diff_count': int(q_monotonicity_negative_diff_count),
                        'm1_roughness_lambda': self._safe_float(self._latest_model_diagnostics.get('m1_roughness_lambda')),
                        'm1_roughness_penalty_value': self._safe_float(self._latest_model_diagnostics.get('m1_roughness_penalty_value')),
                        'm1_roughness_factor_value': self._safe_float(self._latest_model_diagnostics.get('m1_roughness_factor_value')),
                    }
                )
            pyro.factor('obs_log_prob', final_log_prob)
        else:
            self._apply_m1_roughness_factor(m1_roughness_logits)
            if self.diagnostic_objective_terms_enabled:
                self._latest_model_diagnostics.update(
                    {
                        'num_events': int(num_events),
                        'num_samples_per_event': int(num_samples_per_event),
                        'event_loglike_sum': self._safe_float(torch.sum(log_prob_per_event)),
                        'event_loglike_mean': self._safe_float(torch.mean(log_prob_per_event)),
                        'event_loglike_min': self._safe_float(torch.min(log_prob_per_event)),
                        'event_loglike_max': self._safe_float(torch.max(log_prob_per_event)),
                        'obs_neff_threshold': float(NEFF_OBS_THR),
                        'obs_neff_penalty_applied': bool(obs_neff_penalty_applied),
                        'obs_neff_penalty_value': float(obs_neff_penalty_value),
                        'm1_roughness_lambda': self._safe_float(self._latest_model_diagnostics.get('m1_roughness_lambda')),
                        'm1_roughness_penalty_value': self._safe_float(self._latest_model_diagnostics.get('m1_roughness_penalty_value')),
                        'm1_roughness_factor_value': self._safe_float(self._latest_model_diagnostics.get('m1_roughness_factor_value')),
                    }
                )
            with pyro.plate('data', num_events):
                pyro.factor('obs_log_prob', log_prob_per_event)

    def _copula_cdf_for_log_prob(self, marginal_dist, data_expanded):
        if self.copula_cdf_double and isinstance(marginal_dist, SafeKumaraswamy):
            value = data_expanded.to(torch.float64).clamp(0.0, 1.0)
            a = marginal_dist.concentration1.to(torch.float64)
            b = marginal_dist.concentration2.to(torch.float64)
            log_inner_term = b * torch.log(-torch.expm1(a * torch.log(value)))
            return -torch.expm1(log_inner_term)
        return marginal_dist.cdf(data_expanded)

    def _gaussianize_copula_cdf(self, u):
        eps = float(getattr(self, 'copula_cdf_clamp_eps', 1.0e-7))
        eps = min(max(eps, 1.0e-15), 0.49)
        u = torch.nan_to_num(u, nan=0.5, neginf=eps, posinf=1.0 - eps).clamp(eps, 1.0 - eps)
        self._ensure_finite('copula_cdf', u)
        if u.dtype == torch.float64:
            normal = dist.Normal(
                torch.tensor(0.0, dtype=torch.float64, device=self.device),
                torch.tensor(1.0, dtype=torch.float64, device=self.device),
            )
            y = normal.icdf(u).to(dtype=self.min_vals.dtype)
        else:
            y = self._standard_normal.icdf(u)
        self._ensure_finite('copula_gaussianized_samples', y)
        return y

    def _copula_log_prob(self, marginal_dist, L_cholesky, data_norm):
        data_expanded = data_norm.unsqueeze(2)
        marginal_log_probs = torch.nan_to_num(marginal_dist.log_prob(data_expanded).sum(-1), nan=-1.0e12, neginf=-1.0e12, posinf=1.0e12)
        self._ensure_finite('marginal_log_probs', marginal_log_probs)
        y = self._gaussianize_copula_cdf(self._copula_cdf_for_log_prob(marginal_dist, data_expanded))
        copula_mvn = dist.MultivariateNormal(torch.zeros(self.data_dim, device=self.device), scale_tril=L_cholesky)
        copula_log_prob = torch.nan_to_num(copula_mvn.log_prob(y) - self._standard_normal.log_prob(y).sum(-1), nan=-1.0e12, neginf=-1.0e12, posinf=1.0e12)
        total_log_prob = torch.nan_to_num(marginal_log_probs + copula_log_prob, nan=-1.0e12, neginf=-1.0e12, posinf=1.0e12)
        self._ensure_finite('copula_total_log_prob', total_log_prob)
        return total_log_prob

    def _log_pdf_phys_from_params(self, params, points_norm):
        in_support = ((points_norm >= 0) & (points_norm <= 1)).all(dim=-1)
        points_norm = points_norm.clamp(min=1e-7, max=1.0 - 1e-7)
        points_norm_3d = points_norm.unsqueeze(0)
        log_probs_parts = []
        if self.num_corr_kumaraswamy > 0:
            a = torch.exp(params['log_k_a']) + 0.0001
            b = torch.exp(params['log_k_b']) + 0.0001
            L = params.get('k_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['k_cholesky_params'], self.data_dim, self.device)
            marginal_dist_k = SafeKumaraswamy(a, b)
            log_probs_parts.append(self._copula_log_prob(marginal_dist_k, L, points_norm_3d))
        if self.num_corr_tnormal > 0:
            scales = torch.exp(params['log_tn_scales']) + 0.0001
            L = params.get('tnormal_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['tnormal_cholesky_params'], self.data_dim, self.device)
            marginal_dist_tn = TruncatedNormal(params['tn_locs'], scales, low=0.0001, high=1.0 - 0.0001)
            log_probs_parts.append(self._copula_log_prob(marginal_dist_tn, L, points_norm_3d))
        if self.num_corr_tpareto > 0:
            alphas = torch.exp(params['log_tp_alphas']) + 0.0001
            L = params.get('tpareto_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['tpareto_cholesky_params'], self.data_dim, self.device)
            marginal_dist_tp = TruncatedPareto(alphas, low=0.0001, high=1.0)
            log_probs_parts.append(self._copula_log_prob(marginal_dist_tp, L, points_norm_3d))
        if self.num_corr_logitnormal > 0:
            scales = torch.exp(params['log_ln_scales']) + 0.0001
            L = params.get('logitnormal_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['logitnormal_cholesky_params'], self.data_dim, self.device)
            marginal_dist_ln = BoundedLogitNormal(params['ln_locs'], scales)
            log_probs_parts.append(self._copula_log_prob(marginal_dist_ln, L, points_norm_3d))
        if self.num_corr_axis_lntnormal > 0:
            L = params.get('axis_lntn_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['axis_lntn_cholesky_params'], self.data_dim, self.device)
            marginal_dist_axis_lntn = self._axis_lntnormal_marginal_dist(
                params['axis_lntn_ln_locs'],
                params['axis_lntn_log_ln_scales'],
                params['axis_lntn_tn_locs'],
                params['axis_lntn_log_tn_scales'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_axis_lntn, L, points_norm_3d))
        if self.num_corr_axis_lnkumaraswamy > 0:
            L = params.get('axis_lnk_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['axis_lnk_cholesky_params'], self.data_dim, self.device)
            marginal_dist_axis_lnk = self._axis_lnkumaraswamy_marginal_dist(
                params['axis_lnk_ln_locs'],
                params['axis_lnk_log_ln_scales'],
                params['axis_lnk_log_k_a'],
                params['axis_lnk_log_k_b'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_axis_lnk, L, points_norm_3d))
        if self.num_corr_axis_bernstein_tnormal > 0:
            L = params.get('axis_btn_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['axis_btn_cholesky_params'], self.data_dim, self.device)
            marginal_dist_axis_btn = self._axis_bernstein_tnormal_marginal_dist(
                params['axis_btn_m1_logits'],
                params['axis_btn_tn_locs'],
                params['axis_btn_log_tn_scales'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_axis_btn, L, points_norm_3d))
        if self.num_corr_axis_bernstein_kumaraswamy > 0:
            L = params.get('axis_bk_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['axis_bk_cholesky_params'], self.data_dim, self.device)
            marginal_dist_axis_bk = self._axis_bernstein_kumaraswamy_marginal_dist(
                params['axis_bk_m1_logits'],
                params['axis_bk_log_k_a'],
                params['axis_bk_log_k_b'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_axis_bk, L, points_norm_3d))
        if self.num_corr_axis_spline_tnormal > 0:
            L = params.get('axis_stn_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['axis_stn_cholesky_params'], self.data_dim, self.device)
            marginal_dist_axis_stn = self._axis_spline_tnormal_marginal_dist(
                params['axis_stn_m1_logits'],
                params['axis_stn_tn_locs'],
                params['axis_stn_log_tn_scales'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_axis_stn, L, points_norm_3d))
        if self.num_corr_axis_spline_kumaraswamy > 0:
            L = params.get('axis_sk_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['axis_sk_cholesky_params'], self.data_dim, self.device)
            marginal_dist_axis_sk = self._axis_spline_kumaraswamy_marginal_dist(
                params['axis_sk_m1_logits'],
                params['axis_sk_log_k_a'],
                params['axis_sk_log_k_b'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_axis_sk, L, points_norm_3d))
        if self.num_corr_decoupled_spline_tnormal_copula > 0:
            L = params.get('dec_stn_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['dec_stn_cholesky_params'], self.data_dim, self.device)
            marginal_dist_dec_stn = self._decoupled_spline_tnormal_marginal_dist(
                params['dec_stn_m1_logits'],
                params['dec_stn_tn_locs'],
                params['dec_stn_log_tn_scales'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_dec_stn, L, points_norm_3d))
        if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
            L = params.get('dec_sk_L_cholesky')
            if L is None:
                L = self._build_cholesky_from_unconstrained(params['dec_sk_cholesky_params'], self.data_dim, self.device)
            marginal_dist_dec_sk = self._decoupled_spline_kumaraswamy_marginal_dist(
                params['dec_sk_m1_logits'],
                params['dec_sk_log_k_a'],
                params['dec_sk_log_k_b'],
            )
            log_probs_parts.append(self._copula_log_prob(marginal_dist_dec_sk, L, points_norm_3d))
        log_probs_all = torch.cat(log_probs_parts, dim=2).squeeze(0)
        log_weights = torch.log_softmax(params['weights_logits'], dim=-1)
        log_pdf_norm = torch.logsumexp(log_weights + log_probs_all, dim=1)
        log_vol = torch.sum(torch.log(self.max_vals - self.min_vals))
        return torch.where(in_support, log_pdf_norm - log_vol, -torch.inf)

    def _log_selection_fraction_from_params(self, params, inj_points_norm):
        inj_log_p_draw = self.injection_data['log_p_draw'].to(self.device)
        inj_log_p_det = self.injection_data.get('log_p_det')
        if inj_log_p_det is not None:
            inj_log_p_det = inj_log_p_det.to(self.device)
        log_pdf_inj_phys = self._log_pdf_phys_from_params(params, inj_points_norm)
        log_weights_inj = log_pdf_inj_phys - inj_log_p_draw
        if inj_log_p_det is not None:
            log_weights_inj = log_weights_inj + inj_log_p_det
        self._last_selection_log_weights = log_weights_inj
        return torch.logsumexp(log_weights_inj, dim=0) - np.log(self.injection_data['Ndraw'])

    def _selection_mc_samples_norm(self, num_samples, device):
        mode = str(getattr(self, 'selection_mc_mode', 'random') or 'random')
        if mode == 'random':
            return torch.rand(num_samples, self.data_dim, device=device).clamp(min=1e-06, max=1.0 - 1.0e-06)
        if mode not in {'fixed_sobol', 'sobol_fixed'}:
            raise ValueError(f'Unsupported selection_mc_mode: {mode}')
        cache_key = (int(num_samples), str(device), str(torch.float32), int(getattr(self, 'selection_mc_seed', 12345)))
        cached = self._selection_mc_cache.get(cache_key)
        if cached is None or cached.device != device:
            engine = torch.quasirandom.SobolEngine(
                dimension=self.data_dim,
                scramble=True,
                seed=int(getattr(self, 'selection_mc_seed', 12345)),
            )
            cached = engine.draw(int(num_samples)).to(device=device, dtype=torch.float32).clamp(min=1e-06, max=1.0 - 1.0e-06)
            self._selection_mc_cache[cache_key] = cached
        return cached

    def _pyro_guide(self, data_samples):
        pyro.module('flow_transforms', self.flow_transforms)
        self._current_flow_forward_context = 'rsample'
        latent_vector = self.flow_dist.rsample()
        self._ensure_finite('guide latent_vector', latent_vector)
        self._current_flow_forward_context = 'log_prob'
        guide_log_prob = (self.flow_dist.log_prob(latent_vector)
                          - self._latent_log_abs_det_jacobian(latent_vector))
        self._ensure_finite('guide_log_prob', guide_log_prob)
        if self._current_guide_flow_stats is not None:
            self._current_guide_flow_stats = {
                'latent_vector': _tensor_stats_dict(latent_vector),
                'guide_log_prob': _tensor_stats_dict(guide_log_prob),
            }
        if self._current_flow_numerics_stats is not None and self.diagnostic_flow_numerics_enabled:
            previous_context = self._current_flow_forward_context
            self._current_flow_forward_context = 'inverse_diagnostic'
            self._current_flow_numerics_stats = {
                'inverse_path': self._summarize_flow_inverse_path(latent_vector),
            }
            self._current_flow_forward_context = previous_context
        self._current_flow_forward_context = None
        # Trace_ELBO subtracts guide factors, so this must be +log q.
        pyro.factor('guide_log_prob', guide_log_prob, has_rsample=True)
        params = self._unflatten_latent_vector(latent_vector, apply_numerical_limits=False)
        pyro.sample('weights_logits', dist.Delta(params['weights_logits']).to_event(1))
        if self.num_corr_kumaraswamy > 0:
            with pyro.plate('kumaraswamy_components', self.num_corr_kumaraswamy):
                pyro.sample('log_k_a', dist.Delta(params['log_k_a']).to_event(1))
                pyro.sample('log_k_b', dist.Delta(params['log_k_b']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['k_cholesky_params'], self.data_dim, self.device)
                pyro.sample('k_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_tnormal > 0:
            with pyro.plate('tnormal_components', self.num_corr_tnormal):
                pyro.sample('tn_locs', dist.Delta(params['tn_locs']).to_event(1))
                pyro.sample('log_tn_scales', dist.Delta(params['log_tn_scales']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['tnormal_cholesky_params'], self.data_dim, self.device)
                pyro.sample('tnormal_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_tpareto > 0:
            with pyro.plate('tpareto_components', self.num_corr_tpareto):
                pyro.sample('log_tp_alphas', dist.Delta(params['log_tp_alphas']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['tpareto_cholesky_params'], self.data_dim, self.device)
                pyro.sample('tpareto_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_logitnormal > 0:
            with pyro.plate('logitnormal_components', self.num_corr_logitnormal):
                pyro.sample('ln_locs', dist.Delta(params['ln_locs']).to_event(1))
                pyro.sample('log_ln_scales', dist.Delta(params['log_ln_scales']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['logitnormal_cholesky_params'], self.data_dim, self.device)
                pyro.sample('logitnormal_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_axis_lntnormal > 0:
            with pyro.plate('axis_lntnormal_components', self.num_corr_axis_lntnormal):
                pyro.sample('axis_lntn_ln_locs', dist.Delta(params['axis_lntn_ln_locs']))
                pyro.sample('axis_lntn_log_ln_scales', dist.Delta(params['axis_lntn_log_ln_scales']))
                pyro.sample('axis_lntn_tn_locs', dist.Delta(params['axis_lntn_tn_locs']).to_event(1))
                pyro.sample('axis_lntn_log_tn_scales', dist.Delta(params['axis_lntn_log_tn_scales']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['axis_lntn_cholesky_params'], self.data_dim, self.device)
                pyro.sample('axis_lntn_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_axis_lnkumaraswamy > 0:
            with pyro.plate('axis_lnkumaraswamy_components', self.num_corr_axis_lnkumaraswamy):
                pyro.sample('axis_lnk_ln_locs', dist.Delta(params['axis_lnk_ln_locs']))
                pyro.sample('axis_lnk_log_ln_scales', dist.Delta(params['axis_lnk_log_ln_scales']))
                pyro.sample('axis_lnk_log_k_a', dist.Delta(params['axis_lnk_log_k_a']).to_event(1))
                pyro.sample('axis_lnk_log_k_b', dist.Delta(params['axis_lnk_log_k_b']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['axis_lnk_cholesky_params'], self.data_dim, self.device)
                pyro.sample('axis_lnk_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_axis_bernstein_tnormal > 0:
            with pyro.plate('axis_bernstein_tnormal_components', self.num_corr_axis_bernstein_tnormal):
                pyro.sample('axis_btn_m1_logits', dist.Delta(params['axis_btn_m1_logits']).to_event(1))
                pyro.sample('axis_btn_tn_locs', dist.Delta(params['axis_btn_tn_locs']).to_event(1))
                pyro.sample('axis_btn_log_tn_scales', dist.Delta(params['axis_btn_log_tn_scales']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['axis_btn_cholesky_params'], self.data_dim, self.device)
                pyro.sample('axis_btn_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_axis_bernstein_kumaraswamy > 0:
            with pyro.plate('axis_bernstein_kumaraswamy_components', self.num_corr_axis_bernstein_kumaraswamy):
                pyro.sample('axis_bk_m1_logits', dist.Delta(params['axis_bk_m1_logits']).to_event(1))
                pyro.sample('axis_bk_log_k_a', dist.Delta(params['axis_bk_log_k_a']).to_event(1))
                pyro.sample('axis_bk_log_k_b', dist.Delta(params['axis_bk_log_k_b']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['axis_bk_cholesky_params'], self.data_dim, self.device)
                pyro.sample('axis_bk_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_axis_spline_tnormal > 0:
            with pyro.plate('axis_spline_tnormal_components', self.num_corr_axis_spline_tnormal):
                pyro.sample('axis_stn_m1_logits', dist.Delta(params['axis_stn_m1_logits']).to_event(1))
                pyro.sample('axis_stn_tn_locs', dist.Delta(params['axis_stn_tn_locs']).to_event(1))
                pyro.sample('axis_stn_log_tn_scales', dist.Delta(params['axis_stn_log_tn_scales']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['axis_stn_cholesky_params'], self.data_dim, self.device)
                pyro.sample('axis_stn_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_axis_spline_kumaraswamy > 0:
            with pyro.plate('axis_spline_kumaraswamy_components', self.num_corr_axis_spline_kumaraswamy):
                pyro.sample('axis_sk_m1_logits', dist.Delta(params['axis_sk_m1_logits']).to_event(1))
                pyro.sample('axis_sk_log_k_a', dist.Delta(params['axis_sk_log_k_a']).to_event(1))
                pyro.sample('axis_sk_log_k_b', dist.Delta(params['axis_sk_log_k_b']).to_event(1))
                L = self._build_cholesky_from_unconstrained(params['axis_sk_cholesky_params'], self.data_dim, self.device)
                pyro.sample('axis_sk_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_decoupled_spline_tnormal_copula > 0:
            pyro.sample('dec_stn_m1_logits', dist.Delta(params['dec_stn_m1_logits']).to_event(1))
            pyro.sample('dec_stn_tn_locs', dist.Delta(params['dec_stn_tn_locs']).to_event(1))
            pyro.sample('dec_stn_log_tn_scales', dist.Delta(params['dec_stn_log_tn_scales']).to_event(1))
            with pyro.plate('decoupled_spline_tnormal_copula_components', self.num_corr_decoupled_spline_tnormal_copula):
                L = self._build_cholesky_from_unconstrained(params['dec_stn_cholesky_params'], self.data_dim, self.device)
                pyro.sample('dec_stn_L_cholesky', dist.Delta(L).to_event(2))
        if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
            pyro.sample('dec_sk_m1_logits', dist.Delta(params['dec_sk_m1_logits']).to_event(1))
            pyro.sample('dec_sk_log_k_a', dist.Delta(params['dec_sk_log_k_a']).to_event(1))
            pyro.sample('dec_sk_log_k_b', dist.Delta(params['dec_sk_log_k_b']).to_event(1))
            with pyro.plate('decoupled_spline_kumaraswamy_copula_components', self.num_corr_decoupled_spline_kumaraswamy_copula):
                L = self._build_cholesky_from_unconstrained(params['dec_sk_cholesky_params'], self.data_dim, self.device)
                pyro.sample('dec_sk_L_cholesky', dist.Delta(L).to_event(2))

    def train(self, data_samples, epochs=5000, lr=0.001, param_names=None, correct_selection_effects=False, injection_data=None, selection_function=None, observation_years=None, clip_norm=10.0, warmup_epochs=0):
        """
        带有热身阶段和梯度裁剪功能的训练方法。
        """
        if correct_selection_effects:
            if injection_data is None and selection_function is None:
                raise ValueError("当 correct_selection_effects=True 时，必须提供 'injection_data' 或 'selection_function' 参数。")
            self.injection_data = injection_data
            self.selection_function = selection_function
            if self.injection_data is not None:
                for key in self.injection_data:
                    if isinstance(self.injection_data[key], torch.Tensor):
                        self.injection_data[key] = self.injection_data[key].to(self.device)
                print(f'Injection data loaded on {self.device}')
                print(f"  可用注入样本数: {self.injection_data['N_det']:,}")
                print(f"  总注入数 (Ndraw): {self.injection_data['Ndraw']:,}")
            elif self.selection_function is not None:
                print('Selection-function correction enabled without injection data.')
        print(f'Starting training with correlated compound-mixture model and normalizing-flow guide on device: {self.device}...')
        self.training_diagnostics = {
            'status': 'running',
            'device': self.device,
            'epochs': int(epochs),
            'warmup_epochs': int(warmup_epochs),
            'lr': float(lr),
            'clip_norm': float(clip_norm),
            'flow_param_clip_max': None if self.flow_param_clip_max is None else float(self.flow_param_clip_max),
            'flow_grad_scale': float(self.flow_grad_scale),
            'flow_grad_warmup_epochs': int(self.flow_grad_warmup_epochs),
            'flow_grad_warmup_scale': None if self.flow_grad_warmup_scale is None else float(self.flow_grad_warmup_scale),
            'kumaraswamy_log_param_clip': None if self.kumaraswamy_log_param_clip is None or self.kumaraswamy_log_param_clip is False else [float(self.kumaraswamy_log_param_clip[0]), float(self.kumaraswamy_log_param_clip[1])],
            'reset_flow_optimizer_state_on_sanitize': bool(self.reset_flow_optimizer_state_on_sanitize),
            'phases': [],
            'failure': None,
        }
        self._latest_model_diagnostics = {}
        if param_names:
            self.param_names = param_names
        data_samples_on_device = [d.to(self.device) for d in data_samples] if isinstance(data_samples, list) else data_samples.to(self.device)
        all_points_full = torch.cat(data_samples_on_device, dim=0) if isinstance(data_samples_on_device, list) else data_samples_on_device.reshape(-1, data_samples_on_device.shape[-1])
        all_points = all_points_full[:, :3]
        if self.provided_bounds is None:
            print('未检测到预设物理边界。正在从数据推断边界 (Data-driven bounds)...')
            self.min_vals, self.max_vals = (all_points.min(dim=0).values, all_points.max(dim=0).values)
            range_vals = self.max_vals - self.min_vals
            range_vals[range_vals == 0] = 1.0
            self.min_vals -= range_vals * 0.01
            self.max_vals += range_vals * 0.01
            print(f'  Inferred Min: {self.min_vals.cpu().numpy()}')
            print(f'  Inferred Max: {self.max_vals.cpu().numpy()}')
        else:
            is_below = all_points < self.min_vals
            is_above = all_points > self.max_vals
            if is_below.any() or is_above.any():
                n_out = (is_below | is_above).any(dim=1).sum().item()
                print(f'{n_out} / {all_points.shape[0]} posterior samples lie outside population support and contribute zero density.')
        pyro.clear_param_store()
        self._apply_flow_initialization()
        self._configure_flow_gradient_hooks()
        scheduler_config = {'optimizer': torch.optim.Adam, 'optim_args': {'lr': lr}, 'mode': 'min', 'patience': 150, 'factor': 0.5, 'verbose': True}
        clip_args_config = {'clip_norm': clip_norm}
        scheduler = pyro.optim.ReduceLROnPlateau(scheduler_config, clip_args=clip_args_config)
        svi = SVI(self._pyro_model, self._pyro_guide, scheduler, loss=Trace_ELBO())
        self.training_diagnostics['initial_flow_state'] = self._summarize_flow_state()
        def run_phase(num_epochs, phase_name):
            phase_diag = {
                'phase': phase_name,
                'num_epochs': int(num_epochs),
                'losses': [],
                'selection_fraction_history': [],
                'min_neff_obs_history': [],
                'selection_neff_history': [],
                'flow_state_history': [],
                'flow_grad_history': [],
                'status': 'running',
            }
            self.training_diagnostics['phases'].append(phase_diag)
            pbar = trange(num_epochs, disable=not sys.stderr.isatty())
            prev_flow_state = self._clone_flow_param_state()
            for epoch in pbar:
                self._current_training_phase = phase_name
                self._current_training_epoch = int(epoch)
                self._current_flow_grad_stats = {}
                self._current_flow_head_forward_stats = {} if self.diagnostic_flow_head_enabled else None
                self._current_guide_flow_stats = {}
                self._current_flow_numerics_stats = {} if self.diagnostic_flow_numerics_enabled else None
                trace_optimizer_capture = None
                if self.diagnostic_trace_optimizer_enabled and self._trace_epoch_enabled(phase_name, epoch):
                    trace_optimizer_capture = self._capture_trace_optimizer_pre_step(
                        scheduler,
                        data_samples_on_device,
                        phase_name,
                        epoch,
                        clip_norm,
                    )
                try:
                    loss = svi.step(data_samples_on_device)
                except Exception as exc:
                    self._update_failure_diagnostics(f'{phase_name}:svi_step:epoch_{epoch}', exc)
                    raise
                if not np.isfinite(loss):
                    exc = NumericalStabilityError(f'Non-finite ELBO loss during {phase_name} at epoch {epoch}: {loss}')
                    self._update_failure_diagnostics(f'{phase_name}:loss:epoch_{epoch}', exc)
                    raise exc
                phase_diag['losses'].append(float(loss))
                latest_diag = dict(self._latest_model_diagnostics) if self._latest_model_diagnostics else {}
                phase_diag['selection_fraction_history'].append(self._safe_float(latest_diag.get('selection_fraction')))
                phase_diag['min_neff_obs_history'].append(self._safe_float(latest_diag.get('min_neff_obs')))
                phase_diag['selection_neff_history'].append(self._safe_float(latest_diag.get('selection_neff')))
                if trace_optimizer_capture is not None:
                    trace_entry, trace_pre_flow_state = trace_optimizer_capture
                    trace_entry.update(
                        {
                            'post_loss': float(loss),
                            'post_flow_hash': self._hash_flow_state(),
                            'post_param_store_hash': self._hash_param_store_state(),
                            'post_optimizer_hash': self._hash_optimizer_state(scheduler),
                            'post_model_diagnostics': latest_diag,
                            'post_update_summary': self._summarize_flow_state(
                                previous_state=trace_pre_flow_state,
                                max_params=int(self.diagnostic_max_flow_params),
                            ),
                            'post_optimizer_summary': self._summarize_flow_optimizer_state(
                                scheduler,
                                previous_state=trace_pre_flow_state,
                                max_params=int(self.diagnostic_max_flow_params),
                            ),
                        }
                    )
                    if getattr(self, 'diagnostic_torch_internals_enabled', False):
                        trace_entry['adam_reconstruction_summary'] = self._summarize_adam_reconstruction(
                            scheduler,
                            previous_state=trace_pre_flow_state,
                            max_params=int(self.diagnostic_max_flow_params),
                        )
                        trace_entry['post_grad_lifecycle_summary'] = self._summarize_flow_grad_lifecycle(
                            scheduler,
                            stage='post_step',
                            max_params=int(self.diagnostic_max_flow_params),
                        )
                    phase_diag.setdefault('trace_optimizer_history', []).append(trace_entry)
                if self.diagnostic_objective_terms_enabled:
                    objective_keys = [
                        'num_events',
                        'num_samples_per_event',
                        'event_loglike_sum',
                        'event_loglike_mean',
                        'event_loglike_min',
                        'event_loglike_max',
                        'log_selection_fraction',
                        'selection_objective_term',
                        'obs_log_prob_factor',
                        'obs_neff_threshold',
                        'obs_neff_penalty_applied',
                        'obs_neff_penalty_value',
                        'selection_neff_threshold',
                        'selection_neff_penalty_applied',
                        'selection_neff_penalty_value',
                        'q_monotonicity_lambda',
                        'q_monotonicity_penalty_value',
                        'q_monotonicity_factor_value',
                        'q_monotonicity_min_diff',
                        'q_monotonicity_negative_diff_count',
                        'm1_roughness_lambda',
                        'm1_roughness_penalty_value',
                        'm1_roughness_factor_value',
                    ]
                    phase_diag.setdefault('objective_terms_history', []).append(
                        {
                            'epoch': int(epoch),
                            **{key: latest_diag.get(key) for key in objective_keys if key in latest_diag},
                        }
                    )
                if self.diagnostic_flow_head_enabled:
                    phase_diag.setdefault('flow_head_forward_history', []).append(
                        {
                            'epoch': int(epoch),
                            'summary': list((self._current_flow_head_forward_stats or {}).get('all', {}).values()),
                            'by_context': {
                                key: list(value.values())
                                for key, value in (self._current_flow_head_forward_stats or {}).items()
                                if key != 'all'
                            },
                        }
                    )
                phase_diag.setdefault('guide_flow_history', []).append(
                    {
                        'epoch': int(epoch),
                        'summary': dict(self._current_guide_flow_stats or {}),
                    }
                )
                if self.diagnostic_flow_numerics_enabled:
                    phase_diag.setdefault('flow_numerics_history', []).append(
                        {
                            'epoch': int(epoch),
                            'summary': dict(self._current_flow_numerics_stats or {}),
                        }
                    )
                flow_grad_summary = self._summarize_flow_grad_stats(
                    self._current_flow_grad_stats,
                    max_params=int(self.diagnostic_max_flow_params),
                )
                phase_diag['flow_grad_history'].append(
                    {
                        'epoch': int(epoch),
                        'scale_applied': float(self._current_flow_grad_scale()),
                        'summary': flow_grad_summary,
                    }
                )
                flow_state_pre_sanitize = self._summarize_flow_state(
                    previous_state=prev_flow_state,
                    max_params=int(self.diagnostic_max_flow_params),
                )
                optimizer_state_pre_sanitize = None
                if self.diagnostic_optimizer_state_enabled:
                    optimizer_state_pre_sanitize = self._summarize_flow_optimizer_state(
                        scheduler,
                        previous_state=prev_flow_state,
                        max_params=int(self.diagnostic_max_flow_params),
                    )
                    phase_diag.setdefault('optimizer_state_history', []).append(
                        {
                            'epoch': int(epoch),
                            'pre_sanitize': optimizer_state_pre_sanitize,
                        }
                    )
                sanitized = self._sanitize_param_store()
                flow_entry = {
                    'epoch': int(epoch),
                    'pre_sanitize': flow_state_pre_sanitize,
                    'sanitized': bool(bool(sanitized)),
                }
                if sanitized:
                    optimizer_state_reset = self._reset_flow_optimizer_state_for_sanitized_params(
                        scheduler,
                        sanitized.keys(),
                    )
                    phase_diag.setdefault('sanitized_epochs', []).append(
                        {
                            'epoch': int(epoch),
                            'num_params': int(len(sanitized)),
                            'param_names': list(sanitized.keys()),
                            'optimizer_state_reset': optimizer_state_reset,
                        }
                    )
                    flow_entry['post_sanitize'] = self._summarize_flow_state(
                        previous_state=prev_flow_state,
                        max_params=int(self.diagnostic_max_flow_params),
                    )
                    if self.diagnostic_optimizer_state_enabled:
                        phase_diag['optimizer_state_history'][-1]['post_sanitize'] = self._summarize_flow_optimizer_state(
                            scheduler,
                            previous_state=prev_flow_state,
                            max_params=int(self.diagnostic_max_flow_params),
                        )
                phase_diag['flow_state_history'].append(flow_entry)
                prev_flow_state = self._clone_flow_param_state()
                desc = 'Warm-up' if phase_name == 'warmup' else 'Main'
                pbar.set_description(f'{desc} ELBO Loss: {loss:.4f}')
                if epoch > 0 and epoch % 100 == 0:
                    scheduler.step(loss)
            phase_diag['status'] = 'completed'
            if phase_diag['losses']:
                phase_diag['final_loss'] = phase_diag['losses'][-1]
                phase_diag['min_loss'] = min(phase_diag['losses'])
                phase_diag['max_loss'] = max(phase_diag['losses'])
                finite_selection = [x for x in phase_diag['selection_fraction_history'] if x is not None]
                finite_neff_obs = [x for x in phase_diag['min_neff_obs_history'] if x is not None]
                finite_neff_sel = [x for x in phase_diag['selection_neff_history'] if x is not None]
                if finite_selection:
                    phase_diag['selection_fraction_summary'] = {
                        'first': float(finite_selection[0]),
                        'last': float(finite_selection[-1]),
                        'min': float(min(finite_selection)),
                        'max': float(max(finite_selection)),
                    }
                if finite_neff_obs:
                    phase_diag['min_neff_obs_summary'] = {
                        'first': float(finite_neff_obs[0]),
                        'last': float(finite_neff_obs[-1]),
                        'min': float(min(finite_neff_obs)),
                        'max': float(max(finite_neff_obs)),
                    }
                if finite_neff_sel:
                    phase_diag['selection_neff_summary'] = {
                        'first': float(finite_neff_sel[0]),
                        'last': float(finite_neff_sel[-1]),
                        'min': float(min(finite_neff_sel)),
                        'max': float(max(finite_neff_sel)),
                    }
                if phase_diag['flow_state_history']:
                    phase_diag['initial_flow_state'] = phase_diag['flow_state_history'][0]['pre_sanitize']
                    phase_diag['final_flow_state'] = phase_diag['flow_state_history'][-1].get('post_sanitize', phase_diag['flow_state_history'][-1]['pre_sanitize'])
                    phase_diag['first_sanitized_epoch'] = next((entry['epoch'] for entry in phase_diag['flow_state_history'] if entry.get('sanitized')), None)
                    phase_diag['num_sanitized_epochs'] = sum(1 for entry in phase_diag['flow_state_history'] if entry.get('sanitized'))
                    phase_diag['flow_delta_l2_max'] = float(max(entry['pre_sanitize'].get('delta_l2_norm', 0.0) for entry in phase_diag['flow_state_history']))
                    phase_diag['flow_delta_abs_max'] = float(max(entry['pre_sanitize'].get('delta_max_abs', 0.0) for entry in phase_diag['flow_state_history']))
                if phase_diag['flow_grad_history']:
                    phase_diag['flow_grad_max_abs'] = float(max(entry['summary'].get('max_abs', 0.0) for entry in phase_diag['flow_grad_history']))
                    phase_diag['flow_grad_l2_max'] = float(max(entry['summary'].get('l2_norm_total', 0.0) for entry in phase_diag['flow_grad_history']))
                    phase_diag['first_large_flow_grad_epoch'] = next(
                        (entry['epoch'] for entry in phase_diag['flow_grad_history'] if entry['summary'].get('max_abs', 0.0) > 100.0),
                        None,
                    )
        original_correct_selection_effects = correct_selection_effects
        original_selection_function = self.selection_function
        if warmup_epochs > 0 and original_correct_selection_effects:
            print(f'\n--- [Phase 1/2] Starting WARM-UP ({warmup_epochs} epochs without selection effects) ---')
            self.correct_selection_effects = False
            self.selection_function = None
            self.observation_years = None
            run_phase(warmup_epochs, 'warmup')
            print('--- Warm-up complete! ---')
        main_epochs = epochs - warmup_epochs
        if main_epochs <= 0:
            print('\nWarning: `epochs` is less than or equal to `warmup_epochs`. Skipping main training.')
            self.is_trained = True
            return
        self.correct_selection_effects = original_correct_selection_effects
        self.selection_function = original_selection_function
        if self.correct_selection_effects:
            if (self.injection_data is None and self.selection_function is None) or observation_years is None:
                raise ValueError("当 correct_selection_effects=True 时，必须提供 ('injection_data' 或 'selection_function') 和 'observation_years' 参数。")
            self.observation_years = observation_years
            phase_str = '[Phase 2/2]' if warmup_epochs > 0 else '[Single Phase]'
            print(f'\n--- {phase_str} Starting MAIN TRAINING ({main_epochs} epochs with selection effects, clip_norm={clip_norm}) ---')
        else:
            print(f'\n--- [Single Phase] Starting training ({main_epochs} epochs without selection effects) ---')
        run_phase(main_epochs, 'main')
        self.is_trained = True
        self.training_diagnostics['status'] = 'completed'
        self.training_diagnostics['posterior_parameter_summary'] = self.summarize_posterior_parameters()
        self.training_diagnostics['final_flow_state'] = self._summarize_flow_state()
        self._current_training_phase = None
        self._current_training_epoch = None
        self._current_flow_grad_stats = None
        self._current_flow_head_forward_stats = None
        self._current_guide_flow_stats = None
        self._current_flow_numerics_stats = None
        self._current_flow_forward_context = None
        print('Shape training complete!')

    def _calculate_posterior_mean_selection_fraction(self, num_posterior_samples=100):
        """
        计算选择效应分数 <Λ_det> 的后验均值。
        """
        print('Calculating posterior mean selection fraction...')
        with torch.no_grad():
            latent_samples = self.flow_dist.sample(torch.Size([num_posterior_samples]))
            selection_fractions = []
            use_injections = self.injection_data is not None and self.selection_function is None
            if use_injections:
                inj_log_m1 = self.injection_data['log_m1'].to(self.device)
                inj_z = self.injection_data['z'].to(self.device)
                inj_q = self.injection_data['q'].to(self.device)
                inj_points = torch.stack([inj_log_m1, inj_z, inj_q], dim=1)
                inj_points_norm = (inj_points - self.min_vals) / (self.max_vals - self.min_vals)
                inj_points_norm = inj_points_norm.clamp(min=1e-06, max=1.0 - 1e-06)
            for i in trange(num_posterior_samples, desc='Averaging selection fraction'):
                params = self._unflatten_latent_vector(latent_samples[i])
                if use_injections:
                    log_selection_fraction = self._log_selection_fraction_from_params(params, inj_points_norm)
                    self._ensure_finite('log_selection_fraction', log_selection_fraction)
                    selection_fractions.append(torch.exp(log_selection_fraction))
                    continue
                else:
                    mc_samples_norm = self._selection_mc_samples_norm(self.lambda_det_mc_samples, self.device)
                    samples_norm_expanded = mc_samples_norm.unsqueeze(0)
                mc_log_probs_per_comp = []
                if self.num_corr_kumaraswamy > 0:
                    a = torch.exp(params['log_k_a']) + 0.0001
                    b = torch.exp(params['log_k_b']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['k_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = SafeKumaraswamy(a, b)
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_tnormal > 0:
                    locs = params['tn_locs']
                    scales = torch.exp(params['log_tn_scales']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['tnormal_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = TruncatedNormal(locs, scales, low=0.0001, high=1.0 - 0.0001)
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_tpareto > 0:
                    alphas = torch.exp(params['log_tp_alphas']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['tpareto_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = TruncatedPareto(alphas, low=0.0001, high=1.0)
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_logitnormal > 0:
                    locs = params['ln_locs']
                    scales = torch.exp(params['log_ln_scales']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['logitnormal_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = BoundedLogitNormal(locs, scales)
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_axis_lntnormal > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_lntn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_lntnormal_marginal_dist(
                        params['axis_lntn_ln_locs'],
                        params['axis_lntn_log_ln_scales'],
                        params['axis_lntn_tn_locs'],
                        params['axis_lntn_log_tn_scales'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_axis_lnkumaraswamy > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_lnk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_lnkumaraswamy_marginal_dist(
                        params['axis_lnk_ln_locs'],
                        params['axis_lnk_log_ln_scales'],
                        params['axis_lnk_log_k_a'],
                        params['axis_lnk_log_k_b'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_axis_bernstein_tnormal > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_btn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_bernstein_tnormal_marginal_dist(
                        params['axis_btn_m1_logits'],
                        params['axis_btn_tn_locs'],
                        params['axis_btn_log_tn_scales'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_axis_bernstein_kumaraswamy > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_bk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_bernstein_kumaraswamy_marginal_dist(
                        params['axis_bk_m1_logits'],
                        params['axis_bk_log_k_a'],
                        params['axis_bk_log_k_b'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_axis_spline_tnormal > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_stn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_spline_tnormal_marginal_dist(
                        params['axis_stn_m1_logits'],
                        params['axis_stn_tn_locs'],
                        params['axis_stn_log_tn_scales'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_axis_spline_kumaraswamy > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_sk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_spline_kumaraswamy_marginal_dist(
                        params['axis_sk_m1_logits'],
                        params['axis_sk_log_k_a'],
                        params['axis_sk_log_k_b'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_decoupled_spline_tnormal_copula > 0:
                    L = self._build_cholesky_from_unconstrained(params['dec_stn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._decoupled_spline_tnormal_marginal_dist(
                        params['dec_stn_m1_logits'],
                        params['dec_stn_tn_locs'],
                        params['dec_stn_log_tn_scales'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
                    L = self._build_cholesky_from_unconstrained(params['dec_sk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._decoupled_spline_kumaraswamy_marginal_dist(
                        params['dec_sk_m1_logits'],
                        params['dec_sk_log_k_a'],
                        params['dec_sk_log_k_b'],
                    )
                    mc_log_probs_per_comp.append(self._copula_log_prob(marginal_dist, L, samples_norm_expanded))
                log_probs_mc_all = torch.cat(mc_log_probs_per_comp, dim=2).squeeze(0)
                log_weights = torch.log_softmax(params['weights_logits'], dim=-1)
                if not use_injections:
                    pdf_mc = torch.exp(torch.logsumexp(log_weights + log_probs_mc_all, dim=1))
                    mc_samples_unnorm = mc_samples_norm * (self.max_vals - self.min_vals) + self.min_vals
                    p_det_mc = self.selection_function(mc_samples_unnorm)
                    selection_fraction = torch.mean(pdf_mc * p_det_mc)
                    self._ensure_finite('selection_fraction_mc', selection_fraction)
                    selection_fractions.append(selection_fraction)
        mean_selection_fraction = torch.mean(torch.stack(selection_fractions))
        self._ensure_finite('posterior_mean_selection_fraction', mean_selection_fraction)
        return mean_selection_fraction

    def _population_selection_fraction(self, params):
        if self.K != self.num_corr_kumaraswamy + self.num_corr_tnormal:
            raise ValueError('Population quadrature currently supports Kumaraswamy and normal components.')
        uniform = self._selection_mc_samples_norm(self.lambda_det_mc_samples, self.device)
        normal = torch.special.ndtri(uniform)
        fractions = []
        for family, count, chol_key, raw_key in (
            ('kumaraswamy', self.num_corr_kumaraswamy, 'k_L_cholesky', 'k_cholesky_params'),
            ('normal', self.num_corr_tnormal, 'tnormal_L_cholesky', 'tnormal_cholesky_params'),
        ):
            if count == 0:
                continue
            cholesky = params.get(chol_key)
            if cholesky is None:
                cholesky = self._build_cholesky_from_unconstrained(params[raw_key], self.data_dim, self.device)
            for index in range(count):
                cdf = torch.special.ndtr(normal @ cholesky[index].T).clamp(1e-7, 1 - 1e-7)
                if family == 'kumaraswamy':
                    marginal = SafeKumaraswamy(params['log_k_a'][index].exp() + .0001,
                                               params['log_k_b'][index].exp() + .0001)
                else:
                    marginal = TruncatedNormal(params['tn_locs'][index],
                                                params['log_tn_scales'][index].exp() + .0001,
                                                low=.0001, high=.9999)
                points = marginal.icdf(cdf) * (self.max_vals - self.min_vals) + self.min_vals
                fractions.append(self.selection_function(points).mean())
        return torch.dot(torch.softmax(params['weights_logits'], dim=-1), torch.stack(fractions))

    def _selection_fraction_for_params(self, params):
        if not self.correct_selection_effects:
            return torch.ones((), device=self.device)
        if getattr(self, 'selection_integration', 'uniform') == 'population_qmc':
            return self._population_selection_fraction(params)
        if self.injection_data is not None and self.selection_function is None:
            points = torch.stack([self.injection_data[k].to(self.device)
                                  for k in ('log_m1', 'z', 'q')], dim=1)
            points_norm = (points - self.min_vals) / (self.max_vals - self.min_vals)
            return self._log_selection_fraction_from_params(params, points_norm).exp()
        points_norm = self._selection_mc_samples_norm(self.lambda_det_mc_samples, self.device)
        points = points_norm * (self.max_vals - self.min_vals) + self.min_vals
        log_volume = torch.log(self.max_vals - self.min_vals).sum()
        density = (self._log_pdf_phys_from_params(params, points_norm) + log_volume).exp()
        return (density * self.selection_function(points)).mean()

    def infer_rate(self, num_observed_events, num_mcmc_samples=2000, warmup_steps=500):
        """Reconstruct the rate from the same scale-invariant-prior posterior.

        ``num_mcmc_samples`` is retained as the draw count for old callers.
        No MCMC or warmup is needed for the conditional Gamma distribution.
        """
        if not self.is_trained:
            raise RuntimeError('Must train the shape model first by calling train().')
        if self.observation_years is None:
            raise ValueError('`observation_years` was not set. Cannot infer rate.')
        with torch.no_grad():
            self.rate_shape_latent_samples = self.flow_dist.sample((int(num_mcmc_samples),))
            self.rate_selection_fractions = torch.stack([
                self._selection_fraction_for_params(self._unflatten_latent_vector(latent))
                for latent in self.rate_shape_latent_samples])
            draws = conditional_rate_draws(num_observed_events, self.observation_years,
                                           self.rate_selection_fractions)
        self.yearly_rate_posterior_samples = draws['rate']
        self.expected_intrinsic_count_samples = draws['intrinsic_mean']
        self.expected_detected_count_samples = draws['detected_mean']
        self.replicated_detected_count_samples = draws['replicated_detected_count']
        self.posterior_mean_selection_fraction = float(self.rate_selection_fractions.mean())
        self.is_rate_inferred = True
        self.rate_diagnostics = {
            'status': 'completed',
            'method': 'conditional_gamma',
            'prior': 'scale_invariant',
            'shape_rate_draws_paired': True,
            'num_observed_events': int(num_observed_events),
            'observation_years': float(self.observation_years),
            'selection_fraction_mean': float(self.posterior_mean_selection_fraction),
            'num_draws': int(num_mcmc_samples),
            'rate_mean': float(draws['rate'].mean()),
            'rate_std': float(draws['rate'].std()),
        }
        print('Conditional Gamma rate reconstruction complete.')

    def visualize(self, mode='rate_density', q_slices=None, elev=None, azim=None):
        if not self.is_trained:
            raise RuntimeError('Model must be trained before visualizing.')
        if mode == 'rate_density' and (not self.is_rate_inferred):
            raise RuntimeError('Rate has not been inferred for rate_density mode. Call infer_rate() first.')
        rate_type = 'Intrinsic' if self.correct_selection_effects else 'Detectable'
        print(f'Generating visualization for 3D distribution as 2D slices (FINAL FIX)...')
        if q_slices is None:
            min_q, max_q = (self.min_vals[2].item(), self.max_vals[2].item())
            q_slices = np.linspace(min_q + 0.05, max_q - 0.05, 3)
        fig, axes = plt.subplots(1, len(q_slices), figsize=(6 * len(q_slices), 5), subplot_kw={'projection': '3d'})
        if len(q_slices) == 1:
            axes = [axes]
        min_v, max_v = (self.min_vals.cpu(), self.max_vals.cpu())
        logM_grid = np.linspace(min_v[0], max_v[0], 70)
        z_grid = np.linspace(min_v[1], max_v[1], 70)
        logM_mesh, z_mesh = np.meshgrid(logM_grid, z_grid, indexing='ij')
        surf = None
        for i, q_val in enumerate(q_slices):
            ax = axes[i]
            print(f'Processing slice at q = {q_val:.3f}...')
            q_mesh = np.full_like(logM_mesh, q_val)
            grid_points_numpy = np.stack([logM_mesh.ravel(), z_mesh.ravel(), q_mesh.ravel()], axis=-1)
            grid_points_tensor = torch.tensor(grid_points_numpy, dtype=torch.float32)
            with torch.no_grad():
                mean_val, _, _, _ = self.predict(grid_points_tensor, mode=mode)
            mean_val_reshaped = mean_val.cpu().numpy().reshape(logM_mesh.shape)
            surf = ax.plot_surface(z_mesh, logM_mesh, mean_val_reshaped, cmap='viridis', rstride=1, cstride=1, alpha=0.9, linewidth=0, antialiased=False)
            title = f'{rate_type} Rate Density\nat q = {q_val:.2f}'
            z_label = 'Rate Density [yr⁻¹ ...]' if mode == 'rate_density' else 'PDF'
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(self.param_names[1])
            ax.set_ylabel(self.param_names[0])
            ax.set_zlabel(z_label)
            if elev is not None and azim is not None:
                ax.view_init(elev=elev, azim=azim)
        if surf:
            fig.colorbar(surf, ax=axes, shrink=0.5, aspect=10, pad=0.1)
        plt.tight_layout()
        plt.show()

    def predict(self, grid_points, num_samples=500, mode='rate_density'):
        if not self.is_trained:
            raise RuntimeError('The model has not been trained yet for shape.')
        if mode == 'rate_density' and (not self.is_rate_inferred):
            raise RuntimeError('Rate has not been inferred. Call infer_rate() first.')
        grid_points = grid_points.to(self.device)
        grid_norm = (grid_points - self.min_vals) / (self.max_vals - self.min_vals)
        in_support = ((grid_norm >= 0) & (grid_norm <= 1)).all(dim=-1)
        grid_norm = grid_norm.clamp(min=1e-7, max=1.0 - 1e-7)
        jacobian_det = torch.prod(self.max_vals - self.min_vals)
        all_results = []
        with torch.no_grad():
            latent_samples = self.flow_dist.sample(torch.Size([num_samples]))
            for i in trange(num_samples, desc=f'Calculating {mode} from posterior'):
                params = self._unflatten_latent_vector(latent_samples[i])
                grid_norm_expanded = grid_norm.view(1, -1, self.data_dim)
                log_probs_per_component = []
                if self.num_corr_kumaraswamy > 0:
                    a = torch.exp(params['log_k_a']) + 0.0001
                    b = torch.exp(params['log_k_b']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['k_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = SafeKumaraswamy(a, b)
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_tnormal > 0:
                    locs = params['tn_locs']
                    scales = torch.exp(params['log_tn_scales']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['tnormal_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = TruncatedNormal(locs, scales, low=0.0001, high=1.0 - 0.0001)
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_tpareto > 0:
                    alphas = torch.exp(params['log_tp_alphas']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['tpareto_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = TruncatedPareto(alphas, low=0.0001, high=1.0)
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_logitnormal > 0:
                    locs = params['ln_locs']
                    scales = torch.exp(params['log_ln_scales']) + 0.0001
                    L = self._build_cholesky_from_unconstrained(params['logitnormal_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = BoundedLogitNormal(locs, scales)
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_axis_lntnormal > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_lntn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_lntnormal_marginal_dist(
                        params['axis_lntn_ln_locs'],
                        params['axis_lntn_log_ln_scales'],
                        params['axis_lntn_tn_locs'],
                        params['axis_lntn_log_tn_scales'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_axis_lnkumaraswamy > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_lnk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_lnkumaraswamy_marginal_dist(
                        params['axis_lnk_ln_locs'],
                        params['axis_lnk_log_ln_scales'],
                        params['axis_lnk_log_k_a'],
                        params['axis_lnk_log_k_b'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_axis_bernstein_tnormal > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_btn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_bernstein_tnormal_marginal_dist(
                        params['axis_btn_m1_logits'],
                        params['axis_btn_tn_locs'],
                        params['axis_btn_log_tn_scales'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_axis_bernstein_kumaraswamy > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_bk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_bernstein_kumaraswamy_marginal_dist(
                        params['axis_bk_m1_logits'],
                        params['axis_bk_log_k_a'],
                        params['axis_bk_log_k_b'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_axis_spline_tnormal > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_stn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_spline_tnormal_marginal_dist(
                        params['axis_stn_m1_logits'],
                        params['axis_stn_tn_locs'],
                        params['axis_stn_log_tn_scales'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_axis_spline_kumaraswamy > 0:
                    L = self._build_cholesky_from_unconstrained(params['axis_sk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._axis_spline_kumaraswamy_marginal_dist(
                        params['axis_sk_m1_logits'],
                        params['axis_sk_log_k_a'],
                        params['axis_sk_log_k_b'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_decoupled_spline_tnormal_copula > 0:
                    L = self._build_cholesky_from_unconstrained(params['dec_stn_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._decoupled_spline_tnormal_marginal_dist(
                        params['dec_stn_m1_logits'],
                        params['dec_stn_tn_locs'],
                        params['dec_stn_log_tn_scales'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                if self.num_corr_decoupled_spline_kumaraswamy_copula > 0:
                    L = self._build_cholesky_from_unconstrained(params['dec_sk_cholesky_params'], self.data_dim, self.device)
                    marginal_dist = self._decoupled_spline_kumaraswamy_marginal_dist(
                        params['dec_sk_m1_logits'],
                        params['dec_sk_log_k_a'],
                        params['dec_sk_log_k_b'],
                    )
                    log_probs_per_component.append(self._copula_log_prob(marginal_dist, L, grid_norm_expanded))
                log_probs_all = torch.cat(log_probs_per_component, dim=2).squeeze(0)
                weights = torch.softmax(params['weights_logits'], dim=-1)
                total_prob_norm = torch.exp(torch.logsumexp(torch.log(weights) + log_probs_all, dim=1))
                pdf_values = total_prob_norm / jacobian_det
                pdf_values = torch.where(in_support, pdf_values, 0.0)
                if mode == 'pdf':
                    all_results.append(pdf_values)
                elif mode == 'rate_density':
                    alpha = self._selection_fraction_for_params(params)
                    lambda_yr_sample = conditional_rate_draws(
                        self.rate_diagnostics['num_observed_events'], self.observation_years,
                        alpha)['rate']
                    rate_density_values = lambda_yr_sample * pdf_values
                    all_results.append(rate_density_values)
        results_tensor = torch.stack(all_results, dim=0)
        mean_val = results_tensor.mean(dim=0)
        std_val = results_tensor.std(dim=0)
        lower_bound = torch.quantile(results_tensor, 0.05, dim=0)
        upper_bound = torch.quantile(results_tensor, 0.95, dim=0)
        return (mean_val, std_val, lower_bound, upper_bound)

def save_trained_model(estimator, filepath):
    """
    完全保存已训练的 BayesianMixtureEstimator_NF_Correlated 模型。
    """
    if not estimator.is_trained:
        raise RuntimeError('模型尚未训练，无法保存。')
    all_transforms = estimator.flow_transforms.parts
    permutation_indices = []
    num_affine_coupling = 0
    for transform in all_transforms:
        if isinstance(transform, T.Permute):
            permutation_indices.append(transform.permutation.cpu())
        elif isinstance(transform, T.AffineCoupling):
            num_affine_coupling += 1
    print(f'  检测到 {num_affine_coupling} 个 AffineCoupling 层')
    print(f'  检测到 {len(permutation_indices)} 个 Permute 层')
    original_state_dict = estimator.flow_transforms.state_dict()
    module_indices = sorted(set((k.split('.')[0] for k in original_state_dict.keys())))
    print(f'  原始模型的模块索引: {module_indices}')
    checkpoint = {'flow_transforms_state_dict': original_state_dict, 'original_module_indices': module_indices, 'permutation_indices': permutation_indices, 'pyro_param_store': pyro.get_param_store().get_state(), 'config': {'data_dim': estimator.data_dim, 'num_corr_kumaraswamy': estimator.num_corr_kumaraswamy, 'num_corr_tnormal': estimator.num_corr_tnormal, 'num_corr_tpareto': estimator.num_corr_tpareto, 'num_corr_logitnormal': getattr(estimator, 'num_corr_logitnormal', 0), 'num_corr_axis_lntnormal': getattr(estimator, 'num_corr_axis_lntnormal', 0), 'num_corr_axis_lnkumaraswamy': getattr(estimator, 'num_corr_axis_lnkumaraswamy', 0), 'num_corr_axis_bernstein_tnormal': getattr(estimator, 'num_corr_axis_bernstein_tnormal', 0), 'num_corr_axis_bernstein_kumaraswamy': getattr(estimator, 'num_corr_axis_bernstein_kumaraswamy', 0), 'axis_bernstein_degree': getattr(estimator, 'axis_bernstein_degree', 8), 'num_corr_axis_spline_tnormal': getattr(estimator, 'num_corr_axis_spline_tnormal', 0), 'num_corr_axis_spline_kumaraswamy': getattr(estimator, 'num_corr_axis_spline_kumaraswamy', 0), 'axis_spline_num_bins': getattr(estimator, 'axis_spline_num_bins', 12), 'num_corr_decoupled_spline_tnormal_copula': getattr(estimator, 'num_corr_decoupled_spline_tnormal_copula', 0), 'num_corr_decoupled_spline_kumaraswamy_copula': getattr(estimator, 'num_corr_decoupled_spline_kumaraswamy_copula', 0), 'm1_roughness_lambda': getattr(estimator, 'm1_roughness_lambda', 0.0), 'm1_basis_coordinate': getattr(estimator, 'm1_basis_coordinate', 'log10'), 'm1_basis_logit_epsilon': getattr(estimator, 'm1_basis_logit_epsilon', 0.02), 'm1_basis_type': getattr(estimator, 'm1_basis_type', 'single'), 'm1_two_resolution_global_bins': getattr(estimator, 'm1_two_resolution_global_bins', 5), 'm1_two_resolution_local_bins': getattr(estimator, 'm1_two_resolution_local_bins', 16), 'm1_two_resolution_local_min_solar_mass': getattr(estimator, 'm1_two_resolution_local_min_solar_mass', 5.0), 'm1_two_resolution_local_max_solar_mass': getattr(estimator, 'm1_two_resolution_local_max_solar_mass', 50.0), 'latent_dim': estimator.latent_dim, 'hidden_dim_flow': 128, 'num_flow_transforms': num_affine_coupling}, 'min_vals': estimator.min_vals.cpu(), 'max_vals': estimator.max_vals.cpu(), 'provided_bounds': estimator.provided_bounds, 'is_trained': estimator.is_trained, 'is_rate_inferred': estimator.is_rate_inferred, 'param_names': estimator.param_names, 'correct_selection_effects': estimator.correct_selection_effects, 'observation_years': estimator.observation_years, 'yearly_rate_posterior_samples': estimator.yearly_rate_posterior_samples.cpu() if estimator.is_rate_inferred else None, 'training_diagnostics': estimator.training_diagnostics, 'rate_diagnostics': estimator.rate_diagnostics}
    torch.save(checkpoint, filepath)
    print(f'Model checkpoint saved to: {filepath}')

def load_trained_model(filepath, device=None):
    """
    加载已保存的 BayesianMixtureEstimator_NF_Correlated 模型。
    最终修复版：自动映射 state_dict 键名。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint['config']
    permutation_indices = checkpoint['permutation_indices']
    original_module_indices = checkpoint.get('original_module_indices', None)
    print(f'正在从 {filepath} 加载模型...')
    print(f"  - {config['num_flow_transforms']} 个 AffineCoupling 层")
    print(f'  - {len(permutation_indices)} 个 Permute 层')
    if original_module_indices:
        print(f'  - 原始模块索引: {original_module_indices}')
    estimator = object.__new__(BayesianMixtureEstimator_NF_Correlated)
    estimator.data_dim = config['data_dim']
    estimator.num_corr_kumaraswamy = config['num_corr_kumaraswamy']
    estimator.num_corr_tnormal = config['num_corr_tnormal']
    estimator.num_corr_tpareto = config['num_corr_tpareto']
    estimator.num_corr_logitnormal = config.get('num_corr_logitnormal', 0)
    estimator.num_corr_axis_lntnormal = config.get('num_corr_axis_lntnormal', 0)
    estimator.num_corr_axis_lnkumaraswamy = config.get('num_corr_axis_lnkumaraswamy', 0)
    estimator.num_corr_axis_bernstein_tnormal = config.get('num_corr_axis_bernstein_tnormal', 0)
    estimator.num_corr_axis_bernstein_kumaraswamy = config.get('num_corr_axis_bernstein_kumaraswamy', 0)
    estimator.axis_bernstein_degree = int(config.get('axis_bernstein_degree', 8))
    estimator.axis_bernstein_num_basis = estimator.axis_bernstein_degree + 1
    estimator.num_corr_axis_spline_tnormal = config.get('num_corr_axis_spline_tnormal', 0)
    estimator.num_corr_axis_spline_kumaraswamy = config.get('num_corr_axis_spline_kumaraswamy', 0)
    estimator.axis_spline_num_bins = int(config.get('axis_spline_num_bins', 12))
    estimator.axis_spline_num_knots = estimator.axis_spline_num_bins + 1
    estimator.num_corr_decoupled_spline_tnormal_copula = int(config.get('num_corr_decoupled_spline_tnormal_copula', 0))
    estimator.num_corr_decoupled_spline_kumaraswamy_copula = int(config.get('num_corr_decoupled_spline_kumaraswamy_copula', 0))
    estimator.m1_roughness_lambda = float(config.get('m1_roughness_lambda', 0.0))
    estimator.m1_basis_coordinate = str(config.get('m1_basis_coordinate', 'log10'))
    estimator.m1_basis_logit_epsilon = float(config.get('m1_basis_logit_epsilon', 0.02))
    estimator.m1_basis_type = str(config.get('m1_basis_type', 'single')).lower()
    estimator.m1_two_resolution_global_bins = int(config.get('m1_two_resolution_global_bins', 5))
    estimator.m1_two_resolution_local_bins = int(config.get('m1_two_resolution_local_bins', 16))
    estimator.m1_two_resolution_local_min_solar_mass = float(config.get('m1_two_resolution_local_min_solar_mass', 5.0))
    estimator.m1_two_resolution_local_max_solar_mass = float(config.get('m1_two_resolution_local_max_solar_mass', 50.0))
    estimator.m1_two_resolution_global_knots = estimator.m1_two_resolution_global_bins + 1
    estimator.m1_two_resolution_local_knots = estimator.m1_two_resolution_local_bins + 1
    estimator.m1_spline_num_params = (
        estimator.axis_spline_num_knots
        if estimator.m1_basis_type == 'single'
        else estimator.m1_two_resolution_global_knots + estimator.m1_two_resolution_local_knots + 1
    )
    estimator.K = config['num_corr_kumaraswamy'] + config['num_corr_tnormal'] + config['num_corr_tpareto'] + estimator.num_corr_logitnormal + estimator.num_corr_axis_lntnormal + estimator.num_corr_axis_lnkumaraswamy + estimator.num_corr_axis_bernstein_tnormal + estimator.num_corr_axis_bernstein_kumaraswamy + estimator.num_corr_axis_spline_tnormal + estimator.num_corr_axis_spline_kumaraswamy + estimator.num_corr_decoupled_spline_tnormal_copula + estimator.num_corr_decoupled_spline_kumaraswamy_copula
    estimator.latent_dim = config['latent_dim']
    estimator.device = device
    estimator.injection_data = None
    estimator.provided_bounds = checkpoint['provided_bounds']
    estimator.min_vals = checkpoint['min_vals'].to(device)
    estimator.max_vals = checkpoint['max_vals'].to(device)
    estimator.is_trained = checkpoint['is_trained']
    estimator.is_rate_inferred = checkpoint['is_rate_inferred']
    estimator.param_names = checkpoint['param_names']
    estimator.correct_selection_effects = checkpoint['correct_selection_effects']
    estimator.observation_years = checkpoint['observation_years']
    estimator.selection_function = None
    estimator.lambda_det_mc_samples = 40000
    estimator.posterior_mean_selection_fraction = None
    estimator.training_diagnostics = checkpoint.get('training_diagnostics', {'status': 'unknown', 'phases': [], 'failure': None})
    estimator.rate_diagnostics = checkpoint.get('rate_diagnostics', {'status': 'unknown'})
    estimator.obs_neff_threshold = 10.0
    estimator.selection_neff_multiplier = 4.0
    estimator.q_monotonicity_lambda = 50.0
    estimator.q_monotonicity_num_points = 25
    estimator.selection_mc_mode = 'random'
    estimator.selection_mc_seed = 12345
    estimator._selection_mc_cache = {}
    estimator.copula_cdf_clamp_eps = 1.0e-7
    estimator.copula_cdf_double = False
    estimator.flow_param_clip_max = None
    estimator.flow_init_scale = None
    estimator.flow_grad_scale = 1.0
    estimator.flow_grad_warmup_epochs = 0
    estimator.flow_grad_warmup_scale = None
    estimator.kumaraswamy_log_param_clip = (-8.0, 8.0)
    estimator.reset_flow_optimizer_state_on_sanitize = False
    estimator._flow_grad_hooks = []
    estimator._current_training_phase = None
    estimator._current_training_epoch = None
    estimator._current_flow_grad_stats = None
    estimator._current_flow_head_forward_stats = None
    estimator._current_guide_flow_stats = None
    estimator._current_flow_numerics_stats = None
    estimator._current_flow_forward_context = None
    estimator.diagnostic_flow_head_enabled = False
    estimator.diagnostic_flow_numerics_enabled = False
    estimator.diagnostic_optimizer_state_enabled = False
    estimator.diagnostic_objective_terms_enabled = False
    estimator.diagnostic_trace_optimizer_enabled = False
    estimator.diagnostic_trace_epochs_by_phase = {}
    estimator.diagnostic_torch_internals_enabled = False
    estimator.diagnostic_saved_tensor_max_records = 80
    estimator.diagnostic_max_flow_params = 6
    estimator.cholesky_params_dim = estimator.data_dim * (estimator.data_dim - 1) // 2
    estimator.dims = {
        'weights': estimator.K,
        'k_marginals': estimator.num_corr_kumaraswamy * estimator.data_dim * 2,
        'k_cholesky': estimator.num_corr_kumaraswamy * estimator.cholesky_params_dim,
        'tnormal_marginals': estimator.num_corr_tnormal * estimator.data_dim * 2,
        'tnormal_cholesky': estimator.num_corr_tnormal * estimator.cholesky_params_dim,
        'tpareto_marginals': estimator.num_corr_tpareto * estimator.data_dim,
        'tpareto_cholesky': estimator.num_corr_tpareto * estimator.cholesky_params_dim,
        'logitnormal_marginals': estimator.num_corr_logitnormal * estimator.data_dim * 2,
        'logitnormal_cholesky': estimator.num_corr_logitnormal * estimator.cholesky_params_dim,
        'axis_lntnormal_marginals': estimator.num_corr_axis_lntnormal * (2 + 2 * (estimator.data_dim - 1)),
        'axis_lntnormal_cholesky': estimator.num_corr_axis_lntnormal * estimator.cholesky_params_dim,
        'axis_lnkumaraswamy_marginals': estimator.num_corr_axis_lnkumaraswamy * (2 + 2 * (estimator.data_dim - 1)),
        'axis_lnkumaraswamy_cholesky': estimator.num_corr_axis_lnkumaraswamy * estimator.cholesky_params_dim,
        'axis_bernstein_tnormal_marginals': estimator.num_corr_axis_bernstein_tnormal * (estimator.axis_bernstein_num_basis + 2 * (estimator.data_dim - 1)),
        'axis_bernstein_tnormal_cholesky': estimator.num_corr_axis_bernstein_tnormal * estimator.cholesky_params_dim,
        'axis_bernstein_kumaraswamy_marginals': estimator.num_corr_axis_bernstein_kumaraswamy * (estimator.axis_bernstein_num_basis + 2 * (estimator.data_dim - 1)),
        'axis_bernstein_kumaraswamy_cholesky': estimator.num_corr_axis_bernstein_kumaraswamy * estimator.cholesky_params_dim,
        'axis_spline_tnormal_marginals': estimator.num_corr_axis_spline_tnormal * (estimator.m1_spline_num_params + 2 * (estimator.data_dim - 1)),
        'axis_spline_tnormal_cholesky': estimator.num_corr_axis_spline_tnormal * estimator.cholesky_params_dim,
        'axis_spline_kumaraswamy_marginals': estimator.num_corr_axis_spline_kumaraswamy * (estimator.m1_spline_num_params + 2 * (estimator.data_dim - 1)),
        'axis_spline_kumaraswamy_cholesky': estimator.num_corr_axis_spline_kumaraswamy * estimator.cholesky_params_dim,
        'decoupled_spline_tnormal_marginals': (estimator.m1_spline_num_params + 2 * (estimator.data_dim - 1)) if estimator.num_corr_decoupled_spline_tnormal_copula > 0 else 0,
        'decoupled_spline_tnormal_cholesky': estimator.num_corr_decoupled_spline_tnormal_copula * estimator.cholesky_params_dim,
        'decoupled_spline_kumaraswamy_marginals': (estimator.m1_spline_num_params + 2 * (estimator.data_dim - 1)) if estimator.num_corr_decoupled_spline_kumaraswamy_copula > 0 else 0,
        'decoupled_spline_kumaraswamy_cholesky': estimator.num_corr_decoupled_spline_kumaraswamy_copula * estimator.cholesky_params_dim,
    }
    estimator.base_dist = dist.Normal(torch.zeros(estimator.latent_dim, device=device), torch.ones(estimator.latent_dim, device=device))
    hidden_dim_flow = config.get('hidden_dim_flow', 128)
    num_flow_transforms = config.get('num_flow_transforms', 8)
    transforms = []
    for i in range(num_flow_transforms):
        split_dim = estimator.latent_dim // 2
        param_net_output_dim = estimator.latent_dim - split_dim
        param_net = CouplingNet(input_dim=split_dim, output_dim=param_net_output_dim, hidden_dim=hidden_dim_flow)
        transforms.append(T.AffineCoupling(split_dim, param_net))
        transforms.append(T.Permute(permutation_indices[i].to(device)))
    estimator.flow_transforms = T.ComposeTransformModule(transforms).to(device)
    saved_state_dict = checkpoint['flow_transforms_state_dict']
    new_state_dict = estimator.flow_transforms.state_dict()
    new_module_indices = sorted(set((k.split('.')[0] for k in new_state_dict.keys())), key=lambda x: int(x))
    print(f'  - 新模型模块索引: {new_module_indices}')
    if original_module_indices is None:
        original_module_indices = sorted(set((k.split('.')[0] for k in saved_state_dict.keys())), key=lambda x: int(x))
    if len(original_module_indices) != len(new_module_indices):
        raise ValueError(f'模块数量不匹配: 原始={len(original_module_indices)}, 新={len(new_module_indices)}')
    index_mapping = dict(zip(original_module_indices, new_module_indices))
    print(f'  - 索引映射: {index_mapping}')
    remapped_state_dict = {}
    for old_key, value in saved_state_dict.items():
        parts = old_key.split('.', 1)
        old_idx = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        new_idx = index_mapping.get(old_idx, old_idx)
        new_key = f'{new_idx}.{rest}' if rest else new_idx
        remapped_state_dict[new_key] = value
    estimator.flow_transforms.load_state_dict(remapped_state_dict)
    print('  state_dict keys remapped and loaded successfully')
    estimator.flow_dist = dist.TransformedDistribution(estimator.base_dist, estimator.flow_transforms)
    estimator._standard_normal = dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    pyro.clear_param_store()
    pyro.get_param_store().set_state(checkpoint['pyro_param_store'])
    if checkpoint['yearly_rate_posterior_samples'] is not None:
        estimator.yearly_rate_posterior_samples = checkpoint['yearly_rate_posterior_samples'].to(device)
    print(f'Model loaded on {device}')
    return estimator


CorrelatedPopulationEstimator = BayesianMixtureEstimator_NF_Correlated
save_checkpoint = save_trained_model
load_checkpoint = load_trained_model

__all__ = [
    'CorrelatedPopulationEstimator',
    'BayesianMixtureEstimator_NF_Correlated',
    'CouplingNet',
    'NumericalStabilityError',
    'TruncatedNormal',
    'TruncatedPareto',
    'BoundedLogitNormal',
    'BoundedBernsteinMixture',
    'BoundedMonotoneSpline',
    'BoundedTwoResolutionSpline',
    'SafeKumaraswamy',
    'compute_neff',
    'save_checkpoint',
    'load_checkpoint',
    'save_trained_model',
    'load_trained_model',
]

