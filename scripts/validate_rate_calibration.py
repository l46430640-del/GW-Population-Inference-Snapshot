"""Known-efficiency reference calibration under independent Poisson surveys."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import gamma, nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = np.random.default_rng(711093)
    alpha, repeats = .5, 10000
    rows = []
    for mean_intrinsic in (40., 100., 400., 1000.):
        count = rng.poisson(mean_intrinsic * alpha, repeats)
        future = rng.poisson(mean_intrinsic * alpha, repeats)
        valid = count > 0
        observed = count[valid]
        lower, upper = gamma.ppf(.05, observed) / alpha, gamma.ppf(.95, observed) / alpha
        pred_lower, pred_upper = nbinom.ppf(.05, observed, .5), nbinom.ppf(.95, observed, .5)
        coverage = np.mean((lower <= mean_intrinsic) & (mean_intrinsic <= upper))
        predictive_coverage = np.mean((pred_lower <= future[valid]) & (future[valid] <= pred_upper))
        n = np.arange(1, int(poisson.ppf(1 - 1e-12, mean_intrinsic * alpha)) + 1)
        accept = ((gamma.ppf(.05, n) <= mean_intrinsic * alpha)
                  & (mean_intrinsic * alpha <= gamma.ppf(.95, n)))
        exact_coverage = np.sum(poisson.pmf(n, mean_intrinsic * alpha) * accept)
        exact_coverage /= 1 - np.exp(-mean_intrinsic * alpha)
        rows.append({'intrinsic_mean': mean_intrinsic, 'detected_mean': mean_intrinsic * alpha,
                     'catalogs': repeats, 'zero_count_catalogs': int((~valid).sum()),
                     'intrinsic_mean_coverage_90': float(coverage),
                     'exact_poisson_coverage_90': float(exact_coverage),
                     'coverage_mc_standard_error': float(np.sqrt(coverage * (1 - coverage) / valid.sum())),
                     'independent_predictive_count_coverage_90': float(predictive_coverage)})
    result = {'seed': 711093, 'efficiency': alpha, 'observation_years': 1,
              'prior': 'p(R) proportional to 1/R, posterior proper for N>0',
              'scope': 'known efficiency; does not establish population-shape or joint-rate coverage',
              'rows': rows}
    output = ROOT / 'results/analysis/rate_calibration.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
