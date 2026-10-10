"""Kumaraswamy copulas with logarithmic coordinates and probability tails.

All calculations use float64. Coordinates are never moved into a smaller
interval. Paired log coordinates retain transport results that round to an
endpoint when represented as an ordinary floating-point coordinate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp
import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints

LOG_TWO = math.log(2.)
LOG_36 = math.log(36.)
LOG_745 = math.log(745.)


def log1mexp(value):
    """log(1-exp(value)), value <= 0, without subtractive cancellation."""
    value = value.double()
    low = value < -LOG_TWO
    left = torch.where(low, value, -1.)
    right = torch.where(low, -1., value)
    return torch.where(low, torch.log1p(-left.exp()), torch.log(-right.expm1()))


def _log_cdf_from_log_hazard(value):
    # For H < exp(-36), log(1-exp(-H)) = log(H) to double precision.
    small, large = value < -36., value > LOG_745
    safe = torch.where(small | large, 0., value)
    return torch.where(small, value, torch.where(large, -0., log1mexp(-safe.exp())))


def _log_negative_log_cdf(value):
    """log(-log(1-exp(-exp(value)))) without either tail underflowing."""
    large = value > LOG_36
    safe = torch.where(large, 0., value)
    tail = torch.where(large, value, 0.)
    return torch.where(large, -tail.exp(), torch.log(-_log_cdf_from_log_hazard(safe)))


def _log_negative_log_coordinate(log_x, log_one_minus_x):
    lower = log_x <= log_one_minus_x
    left = torch.where(lower, log_x, -1.)
    right = torch.where(lower, -1., log_one_minus_x)
    small = right < -36.
    safe = torch.where(small, -1., right)
    upper_result = torch.where(small, right, torch.log(-log1mexp(safe)))
    return torch.where(lower, torch.log(-left), upper_result)


def _coordinate_from_logs(log_x, log_one_minus_x):
    return torch.where(log_x <= log_one_minus_x, log_x.exp(), -log_one_minus_x.expm1())


class _LowerNormalQuantile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probability):
        # Ordinary ndtri is accurate until exponentiating its argument loses
        # range. Newton iterations invert log_ndtr in the remaining tail.
        tail = (log_probability < -600.) & torch.isfinite(log_probability)
        central = torch.where(tail, -1., log_probability)
        result = torch.special.ndtri(central.exp())
        target = log_probability[tail]
        value = -torch.sqrt(-target) * math.sqrt(2.)
        for _ in range(6):
            current = torch.special.log_ndtr(value)
            slope = math.sqrt(2 / math.pi) / torch.special.erfcx(-value / math.sqrt(2.))
            value = value - (current - target) / slope
        result = result.masked_scatter(tail, value)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        value, = ctx.saved_tensors
        derivative = math.sqrt(math.pi / 2) * torch.special.erfcx(-value / math.sqrt(2.))
        return grad_output * derivative


def normal_quantile_from_log_tails(log_cdf, log_survival):
    """Normal quantile using the smaller tail, with differentiable log input.

    A zero log tail is valid when its opposite is finite; an ordinary CDF
    rounded to one need not lose its quantile. Exact endpoints return infinity.
    """
    log_cdf, log_survival = torch.broadcast_tensors(log_cdf.double(), log_survival.double())
    lower = log_cdf <= log_survival
    result = _LowerNormalQuantile.apply(torch.where(lower, log_cdf, log_survival))
    return torch.where(lower, result, -result)


class KumaraswamyMarginal(TorchDistribution):
    """Normalized bounded marginal with explicit logarithmic tail methods."""

    arg_constraints = {'concentration1': constraints.positive, 'concentration2': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration2, validate_args=None):
        reference = concentration1 if torch.is_tensor(concentration1) else concentration2
        device = reference.device if torch.is_tensor(reference) else None
        a = (concentration1.double() if torch.is_tensor(concentration1)
             else torch.tensor(concentration1, dtype=torch.float64, device=device))
        b = (concentration2.double() if torch.is_tensor(concentration2)
             else torch.tensor(concentration2, dtype=torch.float64, device=device))
        a, b = torch.broadcast_tensors(a, b)
        self.concentration1, self.concentration2 = a, b
        super().__init__(a.shape, validate_args=validate_args)

    def _terms_from_log_coordinates(self, log_x, log_one_minus_x):
        log_x, log_one_minus_x = log_x.double(), log_one_minus_x.double()
        log_power_hazard = (self.concentration1.log()
                            + _log_negative_log_coordinate(log_x, log_one_minus_x))
        log_inner = _log_cdf_from_log_hazard(log_power_hazard)
        log_hazard = self.concentration2.log() + _log_negative_log_cdf(log_power_hazard)
        return log_inner, log_hazard

    def log_prob_from_log_coordinates(self, log_x, log_one_minus_x):
        inner, _ = self._terms_from_log_coordinates(log_x, log_one_minus_x)
        a, b = self.concentration1, self.concentration2
        return a.log() + b.log() + (a - 1) * log_x + (b - 1) * inner

    def log_tails_from_log_coordinates(self, log_x, log_one_minus_x):
        _, hazard = self._terms_from_log_coordinates(log_x, log_one_minus_x)
        return _log_cdf_from_log_hazard(hazard), -hazard.exp()

    def log_tails(self, value):
        x = value.double()
        inside = (x > 0) & (x < 1)
        safe = torch.where(inside, x, .5)
        cdf, survival = self.log_tails_from_log_coordinates(safe.log(), torch.log1p(-safe))
        cdf = torch.where(x <= 0, -torch.inf, torch.where(x >= 1, 0., cdf))
        survival = torch.where(x <= 0, 0., torch.where(x >= 1, -torch.inf, survival))
        return torch.where(x.isnan(), x, cdf), torch.where(x.isnan(), x, survival)

    def log_prob(self, value):
        x = value.double()
        safe = torch.where((x > 0) & (x < 1), x, .5)
        result = self.log_prob_from_log_coordinates(safe.log(), torch.log1p(-safe))
        a, b = self.concentration1, self.concentration2
        lower = torch.where(a == 1, b.log(), torch.where(a < 1, torch.inf, -torch.inf))
        upper = torch.where(b == 1, a.log(), torch.where(b < 1, torch.inf, -torch.inf))
        result = torch.where(x == 0, lower, torch.where(x == 1, upper, result))
        result = torch.where((x < 0) | (x > 1), -torch.inf, result)
        return torch.where(x.isnan(), x, result)

    def cdf(self, value):
        return _coordinate_from_logs(*self.log_tails(value))

    def icdf_log_coordinates(self, log_cdf, log_survival):
        log_hazard = _log_negative_log_coordinate(log_survival.double(), log_cdf.double())
        log_x_hazard = (_log_negative_log_cdf(log_hazard - self.concentration2.log())
                        - self.concentration1.log())
        return -log_x_hazard.exp(), _log_cdf_from_log_hazard(log_x_hazard)

    def icdf(self, value):
        p = value.double()
        safe = torch.where((p > 0) & (p < 1), p, .5)
        coordinates = self.icdf_log_coordinates(safe.log(), torch.log1p(-safe))
        result = _coordinate_from_logs(*coordinates)
        result = torch.where(p == 0, 0., torch.where(p == 1, 1., result))
        return torch.where((p < 0) | (p > 1) | p.isnan(), torch.nan, result)

    def rsample(self, sample_shape=torch.Size()):
        uniform = torch.rand(self._extended_shape(sample_shape), dtype=torch.float64,
                             device=self.concentration1.device)
        return self.icdf(uniform)


def gaussian_copula_log_prob(log_cdf, log_survival, cholesky):
    normal = normal_quantile_from_log_tails(log_cdf, log_survival)
    cholesky = cholesky.double()
    whitened = torch.linalg.solve_triangular(cholesky, normal.unsqueeze(-1), upper=False).squeeze(-1)
    return (-.5 * (whitened.square() - normal.square()).sum(-1)
            - cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1))


def kumaraswamy_mixture_log_prob(parameters, points):
    """Unit-box density; its measure-zero boundary is assigned density zero."""
    points = points.double()
    inside = ((points > 0) & (points < 1)).all(-1)
    safe = torch.where(inside.unsqueeze(-1), points, .5).unsqueeze(-2)
    a = parameters['log_k_a'].double().exp() + .0001
    b = parameters['log_k_b'].double().exp() + .0001
    marginal = KumaraswamyMarginal(a, b, validate_args=False)
    cdf, survival = marginal.log_tails(safe)
    component = marginal.log_prob(safe).sum(-1) + gaussian_copula_log_prob(
        cdf, survival, parameters['k_L_cholesky'])
    result = torch.logsumexp(component + parameters['weights_logits'].double().log_softmax(-1), -1)
    return torch.where(points.isnan().any(-1), torch.nan, torch.where(inside, result, -torch.inf))


def kumaraswamy_component_transport(parameters, normal, return_log_coordinates=False):
    """Transport common independent normals through every mixture component."""
    gaussian = torch.einsum('...j,kij->...ki', normal.double(), parameters['k_L_cholesky'].double())
    marginal = KumaraswamyMarginal(parameters['log_k_a'].double().exp() + .0001,
                                 parameters['log_k_b'].double().exp() + .0001, validate_args=False)
    coordinates = marginal.icdf_log_coordinates(torch.special.log_ndtr(gaussian),
                                                torch.special.log_ndtr(-gaussian))
    return coordinates if return_log_coordinates else _coordinate_from_logs(*coordinates)


def _numpy_log1mexp(value):
    low = value < -LOG_TWO
    return np.where(low, np.log1p(-np.exp(np.where(low, value, -1.))),
                    np.log(-np.expm1(np.where(low, -1., value))))


def _numpy_log_cdf_hazard(value):
    small, large = value < -36., value > LOG_745
    safe = np.where(small | large, 0., value)
    return np.where(small, value, np.where(large, -0., _numpy_log1mexp(-np.exp(safe))))


def _numpy_log_negative_log_cdf(value):
    large = value > LOG_36
    safe = np.where(large, 0., value)
    return np.where(large, -np.exp(np.where(large, value, 0.)), np.log(-_numpy_log_cdf_hazard(safe)))


def numpy_kumaraswamy_mixture_log_prob(parameters, points):
    """Independent NumPy/SciPy evaluator of the same unclipped unit-box law."""
    points = np.asarray(points, dtype=np.float64)
    inside = ((points > 0) & (points < 1)).all(-1)
    log_x = np.log(np.where(inside[..., None], points, .5))
    a = np.exp(np.asarray(parameters['log_k_a'], dtype=np.float64)) + .0001
    b = np.exp(np.asarray(parameters['log_k_b'], dtype=np.float64)) + .0001
    logits = np.asarray(parameters['weights_logits'], dtype=np.float64)
    weights = logits - logsumexp(logits)
    result = np.full(points.shape[:-1], -np.inf)
    for index in range(len(a)):
        power_hazard = np.log(a[index]) + np.log(-log_x)
        inner = _numpy_log_cdf_hazard(power_hazard)
        hazard = np.log(b[index]) + _numpy_log_negative_log_cdf(power_hazard)
        cdf, survival = _numpy_log_cdf_hazard(hazard), -np.exp(hazard)
        lower = cdf <= survival
        normal = ndtri_exp(np.where(lower, cdf, survival))
        normal = np.where(lower, normal, -normal)
        chol = np.asarray(parameters['k_L_cholesky'][index], dtype=np.float64)
        whitened = np.linalg.solve(chol, normal.reshape(-1, normal.shape[-1]).T).T.reshape(normal.shape)
        copula = -.5 * (whitened**2 - normal**2).sum(-1) - np.log(np.diag(chol)).sum()
        marginal = (np.log(a[index]) + np.log(b[index]) + (a[index] - 1) * log_x
                    + (b[index] - 1) * inner).sum(-1)
        result = np.logaddexp(result, weights[index] + marginal + copula)
    return np.where(np.isnan(points).any(-1), np.nan, np.where(inside, result, -np.inf))


def numpy_kumaraswamy_component_transport(parameters, normal, return_log_coordinates=False):
    gaussian = np.einsum('...j,kij->...ki', np.asarray(normal, dtype=np.float64), parameters['k_L_cholesky'])
    cdf, survival = log_ndtr(gaussian), log_ndtr(-gaussian)
    lower = survival <= cdf
    right = np.where(lower, -1., cdf)
    small = right < -36.
    upper = np.where(small, right, np.log(-_numpy_log1mexp(np.where(small, -1., right))))
    log_hazard = np.where(lower, np.log(-np.where(lower, survival, -1.)), upper)
    a = np.exp(np.asarray(parameters['log_k_a'], dtype=np.float64)) + .0001
    b = np.exp(np.asarray(parameters['log_k_b'], dtype=np.float64)) + .0001
    log_x_hazard = _numpy_log_negative_log_cdf(log_hazard - np.log(b)) - np.log(a)
    log_x, log_one_minus_x = -np.exp(log_x_hazard), _numpy_log_cdf_hazard(log_x_hazard)
    if return_log_coordinates:
        return log_x, log_one_minus_x
    return np.where(log_x <= log_one_minus_x, np.exp(log_x), -np.expm1(log_one_minus_x))
