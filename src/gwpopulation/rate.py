"""Rate reconstruction consistent with the rate-marginalized shape likelihood."""
from __future__ import annotations

import torch


def conditional_rate_draws(n_observed, observation_years, selection_fractions):
    """Draw R | Lambda, data under p(R) proportional to 1/R.

    A Gamma(N, 1) draw is the expected detected count. Dividing each draw by
    its own T * alpha preserves the dependence of R on population shape.
    """
    if int(n_observed) != n_observed or n_observed <= 0:
        raise ValueError('The scale-invariant rate prior requires N > 0.')
    if not observation_years > 0:
        raise ValueError('Observation time must be positive.')
    alpha = torch.as_tensor(selection_fractions)
    if not alpha.is_floating_point():
        alpha = alpha.to(torch.float64)
    if not torch.all(torch.isfinite(alpha) & (alpha > 0) & (alpha <= 1)):
        raise ValueError('Detection efficiencies must lie in (0, 1].')
    detected_mean = torch.distributions.Gamma(
        torch.full_like(alpha, float(n_observed)), torch.ones_like(alpha)).sample()
    return {
        'rate': detected_mean / (float(observation_years) * alpha),
        'intrinsic_mean': detected_mean / alpha,
        'detected_mean': detected_mean,
        'replicated_detected_count': torch.poisson(detected_mean),
    }
