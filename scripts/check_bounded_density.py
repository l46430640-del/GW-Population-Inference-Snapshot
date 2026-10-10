"""Record lightweight validation of the unclipped density kernels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.integrate import quad
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gwpopulation.bounded_density import (
    KumaraswamyMarginal, kumaraswamy_mixture_log_prob, log1mexp,
    normal_quantile_from_log_tails, numpy_kumaraswamy_mixture_log_prob,
)


def evaluate():
    torch.set_num_threads(1)
    start = time.perf_counter()
    marginal = KumaraswamyMarginal(.02, 1.)

    def integrand(t):
        log_x = torch.tensor(-t, dtype=torch.float64)
        return float((marginal.log_prob_from_log_coordinates(log_x, log1mexp(log_x))-t).exp())

    normalization, error = quad(integrand, 0., np.inf, epsabs=2e-10)
    parameters = {
        'log_k_a': torch.tensor([[.02, 1., 1.]], dtype=torch.float64).sub(.0001).log(),
        'log_k_b': torch.ones(1, 3, dtype=torch.float64).sub(.0001).log(),
        'weights_logits': torch.zeros(1, dtype=torch.float64),
        'k_L_cholesky': torch.eye(3, dtype=torch.float64)[None],
    }
    density = kumaraswamy_mixture_log_prob(parameters, torch.tensor([[1e-10, .5, .5]], dtype=torch.float64)).exp().item()
    gaussian = torch.tensor([-1000., -40., -8., -1., 0., 1., 8., 40., 1000.], dtype=torch.float64)
    tails = torch.special.log_ndtr(gaussian), torch.special.log_ndtr(-gaussian)
    log_tail_error, normal_error = [], []
    for a, b in ((.0001, 1.), (1., .0001), (.02, 5.), (100., .01)):
        distribution = KumaraswamyMarginal(a, b)
        recovered = distribution.log_tails_from_log_coordinates(*distribution.icdf_log_coordinates(*tails))
        log_tail_error.append(max(float((new-old).abs().max()) for new, old in zip(recovered, tails)))
        normal_error.append(float((normal_quantile_from_log_tails(*recovered)-gaussian).abs().max()))

    correlation = torch.tensor([[1., .7, -.2], [.7, 1., .15], [-.2, .15, 1.]], dtype=torch.float64)
    parameters['k_L_cholesky'] = torch.linalg.cholesky(correlation)[None]
    parameters['log_k_a'] = torch.tensor([[20., .02, .4]], dtype=torch.float64).sub(.0001).log()
    parameters['log_k_b'] = torch.tensor([[.2, 50., .02]], dtype=torch.float64).sub(.0001).log()
    points = np.column_stack([np.geomspace(1e-300, .9, 64), np.linspace(.1, 1-1e-15, 64), np.full(64, .7)])
    actual = kumaraswamy_mixture_log_prob(parameters, torch.tensor(points)).numpy()
    expected = numpy_kumaraswamy_mixture_log_prob({name: item.numpy() for name, item in parameters.items()}, points)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=5e-10)
    if abs(normalization-1) > 1e-8 or max(normal_error) > 1e-8:
        raise AssertionError('Kernel normalization or tail roundtrip check failed')
    return {
        'counterexample_shape': [.02, 1.],
        'normalization': normalization, 'quadrature_error_estimate': error,
        'density_at_1e_minus_10': density,
        'maximum_absolute_log_tail_roundtrip_error': max(log_tail_error),
        'maximum_normal_roundtrip_error': max(normal_error),
        'correlated_numpy_torch_maximum_log_density_difference': float(np.max(abs(actual-expected))),
        'correlated_comparison_points': len(points),
        'seconds': time.perf_counter()-start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/analysis/bounded_density_validation.json')
    args = parser.parse_args()
    output = ROOT/args.output
    if output.exists():
        raise FileExistsError('Preserve the record; choose a new --output path.')
    sources = ('src/gwpopulation/bounded_density.py', 'scripts/check_bounded_density.py',
               'tests/test_bounded_density.py', 'src/gwpopulation/core.py')
    report = {
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
        'threads': 1, 'inference_rerun': False, 'adopted_estimator_migrated': False,
        'scope': 'Kumaraswamy and Gaussian-copula kernels only; adopted estimator, other marginals and event/selection integration remain to be migrated and checked.',
        'checks': evaluate(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii', newline='\n')
    print(json.dumps(report['checks'], indent=2))


if __name__ == '__main__':
    main()
