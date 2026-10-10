"""Scrambled integration of the controlled selected-event posterior."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc, truncnorm


def selected_normal_mass(x, observed, sigma):
    """Integral on [0, x] of (.05 + .95*t**2) N(t; observed, sigma)."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(observed) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Finite observation and positive measurement scale required')
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError('Integration endpoints must lie in the unit interval')
    a, b = -observed/sigma, (x-observed)/sigma
    probability = ndtr(-a)-ndtr(-b) if a >= 0 else ndtr(b)-ndtr(a)
    phi_a = np.exp(-.5*a*a)/np.sqrt(2*np.pi)
    phi_b = np.exp(-.5*b*b)/np.sqrt(2*np.pi)
    second = ((observed**2+sigma**2)*probability
              + 2*observed*sigma*(phi_a-phi_b)
              + sigma**2*(a*phi_a-b*phi_b))
    return .05*probability + .95*second


def selected_normal_ppf(probability, observed, sigma):
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must lie in the unit interval')
    mass = selected_normal_mass(1., observed, sigma)
    if not np.isfinite(mass) or mass <= np.finfo(float).tiny:
        raise ValueError('Selected event normalization is outside the supported numerical range')
    lower, upper = np.zeros_like(probability), np.ones_like(probability)
    for _ in range(52):
        middle = (lower+upper)/2
        below = selected_normal_mass(middle, observed, sigma) < probability*mass
        lower, upper = np.where(below, middle, lower), np.where(below, upper, middle)
    result = (lower+upper)/2
    return np.where(probability == 0, 0., np.where(probability == 1, 1., result))


def selected_event_cloud(observed, sigma, power, seed):
    """Map every point of one independent scrambled Sobol net per event."""
    observed, sigma = np.asarray(observed, dtype=float), np.asarray(sigma, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 3 or not np.isfinite(observed).all():
        raise ValueError('Finite three-coordinate observations required')
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Three positive measurement scales required')
    if not isinstance(power, (int, np.integer)) or not 1 <= power <= 20:
        raise ValueError('A bounded positive Sobol power is required')
    event_seeds = np.asarray([child.generate_state(1)[0]
        for child in np.random.SeedSequence(seed).spawn(len(observed))], dtype=np.uint32)
    data = np.ones((len(observed), 2**power, 4))
    for event, y in enumerate(observed):
        uniform = qmc.Sobol(3, scramble=True, seed=int(event_seeds[event])).random_base2(power)
        data[event, :, 0] = selected_normal_ppf(uniform[:, 0], y[0], sigma[0])
        data[event, :, 1:3] = truncnorm.ppf(uniform[:, 1:3], -y[1:]/sigma[1:],
            (1-y[1:])/sigma[1:], loc=y[1:], scale=sigma[1:])
    if not np.isfinite(data).all() or np.any((data[..., :3] < 0) | (data[..., :3] > 1)):
        raise ValueError('Invalid transformed integration point')
    return {'data': data, 'observed': observed, 'sigma': sigma, 'event_seeds': event_seeds}
