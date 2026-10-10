"""Normalized marginals used by the corrected population likelihood.

Log coordinates are relative to each marginal's support. This keeps both
distances to the support boundary available during copula transport.
"""
from __future__ import annotations

import math

import torch
from torch.nn.functional import softplus

from .bounded_density import (
    _coordinate_from_logs, log1mexp, normal_quantile_from_log_tails,
)


def _inverse_gradient(marginal, coordinates, log_cdf, log_survival):
    # Attach the implicit inverse derivative to an analytically computed root.
    lx, lr = coordinates
    left_coordinate = lx <= lr
    left_probability = log_cdf <= log_survival
    coordinate = torch.where(left_coordinate, lx, lr)
    observed = marginal.log_tails_from_log_coordinates(lx, lr)
    observed = torch.where(left_probability, *observed)
    target = torch.where(left_probability, log_cdf, log_survival)
    log_slope = (marginal.log_prob_from_log_coordinates(lx, lr)
                 + math.log(marginal.high - marginal.low)
                 + coordinate - observed)
    slope = log_slope.exp() * torch.where(left_coordinate == left_probability, 1., -1.)
    corrected = coordinate + (target - observed) / slope.detach()
    opposite = log1mexp(corrected)
    return (torch.where(left_coordinate, corrected, opposite),
            torch.where(left_coordinate, opposite, corrected))


class _BoundedMarginal:
    low = 0.
    high = 1.

    def _coordinates(self, value):
        value = value.double()
        inside = (value > self.low) & (value < self.high)
        safe = torch.where(inside, value, (self.low + self.high) / 2)
        width = self.high - self.low
        return inside, ((safe - self.low).log() - math.log(width),
                        (self.high - safe).log() - math.log(width))

    def log_prob(self, value):
        inside, coordinates = self._coordinates(value)
        result = self.log_prob_from_log_coordinates(*coordinates)
        # A finite density convention at the endpoints is immaterial to the law.
        return torch.where(value.isnan(), torch.nan, torch.where(inside, result, -torch.inf))

    def log_tails(self, value):
        _, coordinates = self._coordinates(value)
        cdf, sf = self.log_tails_from_log_coordinates(*coordinates)
        cdf = torch.where(value <= self.low, -torch.inf, torch.where(value >= self.high, 0., cdf))
        sf = torch.where(value <= self.low, 0., torch.where(value >= self.high, -torch.inf, sf))
        return torch.where(value.isnan(), torch.nan, cdf), torch.where(value.isnan(), torch.nan, sf)

    def cdf(self, value):
        return _coordinate_from_logs(*self.log_tails(value))

    def icdf(self, probability):
        p = probability.double()
        safe = torch.where((p > 0) & (p < 1), p, .5)
        lx, lr = self.icdf_log_coordinates(safe.log(), torch.log1p(-safe))
        width = self.high - self.low
        result = torch.where(lx <= lr, self.low + width * lx.exp(),
                             self.high - width * lr.exp())
        result = torch.where(p == 0, self.low, torch.where(p == 1, self.high, result))
        return torch.where((p < 0) | (p > 1) | p.isnan(), torch.nan, result)


def _normal_interval(start, log_width):
    """Log integral of the standard normal on [start, start+exp(log_width)]."""
    width = log_width.exp()
    middle = start + width / 2
    small = width * (1 + middle.abs()) < 1.e-3
    # The midpoint expansion avoids subtraction when the interval is tiny,
    # including widths that cannot be represented as ordinary doubles.
    w = torch.where(small, width, 0.)
    m = torch.where(small, middle, 0.)
    mw2, w2 = (m * w).square(), w.square()
    correction = (mw2 - w2) / 24 + (mw2.square() - 6 * mw2 * w2 + 3 * w2.square()) / 1920
    local = -.5 * m.square() - .5 * math.log(2 * math.pi) + log_width + torch.log1p(correction)
    lo = torch.where(small, -1., start)
    hi = torch.where(small, 1., start + width)
    # Reflect positive intervals before subtracting CDFs.
    a = torch.where(lo >= 0, -hi, lo)
    b = torch.where(lo >= 0, -lo, hi)
    log_a, log_b = torch.special.log_ndtr(a), torch.special.log_ndtr(b)
    regular = log_b + log1mexp(log_a - log_b)
    return torch.where(small, local, regular)


class TruncatedNormalMarginal(_BoundedMarginal):
    def __init__(self, loc, scale, low=.0001, high=.9999):
        self.loc, self.scale = torch.broadcast_tensors(loc.double(), scale.double())
        self.low, self.high = float(low), float(high)
        self.a = (self.low - self.loc) / self.scale
        self.b = (self.high - self.loc) / self.scale
        self.log_width_std = math.log(self.high - self.low) - self.scale.log()
        self.log_normalizer = _normal_interval(self.a, self.log_width_std)

    def log_prob_from_log_coordinates(self, lx, lr):
        width = self.high - self.low
        x = torch.where(lx <= lr, self.low + width * lx.exp(), self.high - width * lr.exp())
        return (-.5 * ((x - self.loc) / self.scale).square()
                - .5 * math.log(2 * math.pi) - self.scale.log() - self.log_normalizer)

    def log_tails_from_log_coordinates(self, lx, lr):
        cdf = _normal_interval(self.a, self.log_width_std + lx) - self.log_normalizer
        sf = _normal_interval(-self.b, self.log_width_std + lr) - self.log_normalizer
        normalization = torch.logaddexp(cdf, sf)
        return cdf - normalization, sf - normalization

    def icdf_log_coordinates(self, log_cdf, log_survival):
        with torch.no_grad():
            base_cdf = torch.logaddexp(torch.special.log_ndtr(self.a), log_cdf + self.log_normalizer)
            base_sf = torch.logaddexp(torch.special.log_ndtr(-self.b), log_survival + self.log_normalizer)
            z = normal_quantile_from_log_tails(base_cdf, base_sf)
            x = self.loc + self.scale * z
            width = self.high - self.low
            left = log_cdf <= log_survival
            distance = torch.where(left, x - self.low, self.high - x) / width
            target = torch.where(left, log_cdf, log_survival)
            boundary_z = torch.where(left, self.a, self.b)
            boundary_log_density = (-.5 * boundary_z.square() - .5 * math.log(2 * math.pi)
                                    + self.log_width_std - self.log_normalizer)
            asymptotic = target - boundary_log_density
            good = (distance > 0) & (distance < 1)
            t = torch.where(good, torch.where(good, distance, .5).log(), asymptotic)
            t = torch.minimum(t, torch.full_like(t, -1.e-15))
            for _ in range(8):
                other = log1mexp(t)
                lx, lr = torch.where(left, t, other), torch.where(left, other, t)
                tails = self.log_tails_from_log_coordinates(lx, lr)
                observed = torch.where(left, *tails)
                slope = (self.log_prob_from_log_coordinates(lx, lr) + math.log(width) + t - observed).exp()
                candidate = t + (target - observed) / slope
                t = torch.where(candidate < 0, candidate, t / 2)
            other = log1mexp(t)
            coordinates = torch.where(left, t, other), torch.where(left, other, t)
        return _inverse_gradient(self, coordinates, log_cdf, log_survival)


def _gather(values, indices):
    shape = torch.broadcast_shapes(values.shape[:-1], indices.shape)
    return values.expand(*shape, values.shape[-1]).gather(-1, indices.expand(shape).unsqueeze(-1)).squeeze(-1)


def _complement_log_at_knot(value):
    safe = torch.where(value < 0, value, -1.)
    return torch.where(value < 0, log1mexp(safe), -torch.inf)


class PiecewiseLinearMarginal(_BoundedMarginal):
    """Positive linear segments; left/right heights may jump at internal edges."""

    def __init__(self, edges, log_left, log_right):
        self.edges = edges.double()
        self.width = self.edges.diff()
        if self.edges[0] != 0 or self.edges[-1] != 1 or not (self.width > 0).all():
            raise ValueError('Piecewise edges must increase strictly from zero to one.')
        area = self.width.log() + torch.logaddexp(log_left.double(), log_right.double()) - math.log(2)
        norm = torch.logsumexp(area, -1, keepdim=True)
        self.log_left, self.log_right, self.log_area = log_left - norm, log_right - norm, area - norm
        zero = torch.full_like(self.log_area[..., :1], -torch.inf)
        self.before = torch.cat((zero, self.log_area.logcumsumexp(-1)[..., :-1]), -1)
        self.after = torch.cat((self.log_area.flip(-1).logcumsumexp(-1).flip(-1)[..., 1:], zero), -1)

    @classmethod
    def from_logits(cls, logits):
        heights = (softplus(logits.double()) + .0001).log()
        edges = torch.linspace(0, 1, heights.shape[-1], device=logits.device, dtype=torch.float64)
        return cls(edges, heights[..., :-1], heights[..., 1:])

    def _segments(self, lx, lr):
        x = _coordinate_from_logs(lx, lr)
        index = torch.searchsorted(self.edges[1:-1], x.contiguous(), right=True)
        lo, hi, width = self.edges[index], self.edges[index + 1], self.width[index]
        t = (x - lo) / width
        first, last = index == 0, index == len(self.width) - 1
        log_t = torch.where(first, lx - width.log(), torch.where(t > 0, torch.where(t > 0, t, 1.).log(), -torch.inf))
        log_r = torch.where(last, lr - width.log(), torch.where(t < 1, torch.where(t < 1, 1 - t, 1.).log(), -torch.inf))
        log_t = torch.where(last, _complement_log_at_knot(torch.where(last, log_r, -1.)), log_t)
        log_r = torch.where(first, _complement_log_at_knot(torch.where(first, log_t, -1.)), log_r)
        t = torch.where(first, log_t.exp(), torch.where(last, -log_r.expm1(), t))
        r = torch.where(last, log_r.exp(), torch.where(first, -log_t.expm1(), 1 - t))
        return index, width, log_t, log_r, t, r

    def log_prob_from_log_coordinates(self, lx, lr):
        index, _, lt, rt, t, r = self._segments(lx, lr)
        h0, h1 = _gather(self.log_left, index), _gather(self.log_right, index)
        # Ordinary interpolation preserves coordinate gradients at internal knots.
        scale = torch.maximum(h0, h1)
        return scale + (torch.exp(h0 - scale) * r + torch.exp(h1 - scale) * t).log()

    @staticmethod
    def _integral(before, h0, h1, log_width, log_t, t, boundary):
        safe_log_t = torch.where(boundary, log_t, -1.)
        partial = log_width + safe_log_t + torch.logaddexp(h0 + torch.log1p(-safe_log_t.exp() / 2), h1 + safe_log_t - math.log(2))
        scale = torch.maximum(before, log_width + torch.maximum(h0, h1))
        t = torch.where(boundary, .5, t)
        central = scale + (torch.exp(before - scale)
                           + torch.exp(log_width + h0 - scale) * (t - t.square() / 2)
                           + torch.exp(log_width + h1 - scale) * t.square() / 2).log()
        return torch.where(boundary, partial, central)

    def log_tails_from_log_coordinates(self, lx, lr):
        index, width, lt, rt, t, r = self._segments(lx, lr)
        h0, h1 = _gather(self.log_left, index), _gather(self.log_right, index)
        cdf = self._integral(_gather(self.before, index), h0, h1, width.log(), lt, t, index == 0)
        sf = self._integral(_gather(self.after, index), h1, h0, width.log(), rt, r, index == len(self.width) - 1)
        norm = torch.logaddexp(cdf, sf)
        return cdf - norm, sf - norm

    def icdf_log_coordinates(self, log_cdf, log_survival):
        with torch.no_grad():
            left = log_cdf <= log_survival
            target = torch.where(left, log_cdf, log_survival)
            # Select segments from the smaller probability tail.
            cumulative = self.log_area.logcumsumexp(-1)
            reverse = self.log_area.flip(-1).logcumsumexp(-1)
            index_l = (log_cdf.unsqueeze(-1) > cumulative).sum(-1).clamp(max=len(self.width) - 1)
            index_r = (log_survival.unsqueeze(-1) > reverse).sum(-1).clamp(max=len(self.width) - 1)
            index = torch.where(left, index_l, len(self.width) - 1 - index_r)
            before = torch.where(left, _gather(self.before, index), _gather(self.after, index))
            area = _gather(self.log_area, index)
            lv = target + log1mexp(before - target) - area
            lv = torch.minimum(lv, torch.zeros_like(lv))
            rv = log1mexp(lv)
            h0, h1 = _gather(self.log_left, index), _gather(self.log_right, index)
            h0, h1 = torch.where(left, h0, h1), torch.where(left, h1, h0)
            root = .5 * torch.logaddexp(2 * h0 + rv, 2 * h1 + lv)
            lt = lv + torch.logaddexp(h0, h1) - torch.logaddexp(h0, root)
            rt = rv + torch.logaddexp(h0, h1) - torch.logaddexp(h1, root)
            lt, rt = torch.where(left, lt, rt), torch.where(left, rt, lt)
            lx = torch.logaddexp(self.edges[index].log(), self.width[index].log() + lt)
            lr = torch.logaddexp(torch.log1p(-self.edges[index + 1]), self.width[index].log() + rt)
        return _inverse_gradient(self, (lx, lr), log_cdf, log_survival)


class AxisMarginals:
    def __init__(self, marginals):
        self.marginals = marginals
        self.low = [getattr(m, 'low', 0.) for m in marginals]
        self.high = [getattr(m, 'high', 1.) for m in marginals]

    def log_prob(self, value):
        return torch.stack([m.log_prob(value[..., i]) for i, m in enumerate(self.marginals)], -1)

    def log_tails(self, value):
        tails = [m.log_tails(value[..., i]) for i, m in enumerate(self.marginals)]
        return tuple(torch.stack([v[j] for v in tails], -1) for j in (0, 1))

    def icdf_log_coordinates(self, cdf, sf):
        coordinates = [m.icdf_log_coordinates(cdf[..., i], sf[..., i]) for i, m in enumerate(self.marginals)]
        return tuple(torch.stack([v[j] for v in coordinates], -1) for j in (0, 1))


def smooth_mass_marginal(coefficients, low, high, length_scale):
    from .smooth_mass import SmoothLogMassDensity
    # Reuse the historical Gaussian log-height prior, not its clipped density.
    legacy = SmoothLogMassDensity(coefficients.double(), low, high, length_scale)
    heights = legacy.logits
    edges = torch.linspace(0, 1, heights.shape[-1], dtype=torch.float64, device=heights.device)
    return PiecewiseLinearMarginal(edges, heights[..., :-1], heights[..., 1:])


def two_resolution_mass_marginal(global_logits, local_logits, weight_logit, low, high):
    if not 0 < low < high < 1:
        raise ValueError('The local mass interval must lie strictly inside the global support.')
    global_density = PiecewiseLinearMarginal.from_logits(global_logits)
    local_density = PiecewiseLinearMarginal.from_logits(local_logits)
    edges = torch.cat((global_density.edges, low + (high - low) * local_density.edges)).unique(sorted=True)
    centers = (edges[:-1] + edges[1:]) / 2

    def heights(density, local):
        points = (edges - low) / (high - low) if local else edges
        midpoint = (centers - low) / (high - low) if local else centers
        index = torch.searchsorted(density.edges[1:-1], midpoint.contiguous())
        lo, width = density.edges[index], density.width[index]
        t0, t1 = (points[:-1] - lo) / width, (points[1:] - lo) / width
        h0, h1 = density.log_left[..., index], density.log_right[..., index]
        active = (centers >= low) & (centers <= high) if local else torch.ones_like(centers, dtype=torch.bool)
        results = []
        for t in (t0, t1):
            t = torch.where(active, t, .5).clamp(0, 1)
            scale = torch.maximum(h0, h1)
            value = scale + ((1 - t) * (h0 - scale).exp() + t * (h1 - scale).exp()).log()
            results.append(torch.where(active, value - (math.log(high - low) if local else 0), -torch.inf))
        return results

    global_heights, local_heights = heights(global_density, False), heights(local_density, True)
    local_weight, global_weight = -softplus(-weight_logit.double()), -softplus(weight_logit.double())
    combined = [torch.logaddexp(g + global_weight.unsqueeze(-1), l + local_weight.unsqueeze(-1))
                for g, l in zip(global_heights, local_heights)]
    return PiecewiseLinearMarginal(edges, *combined)
