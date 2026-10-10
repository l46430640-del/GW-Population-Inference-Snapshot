"""Record bounded end-to-end checks of the corrected population estimator."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
import scipy
from scipy.integrate import quad
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gwpopulation.stable_model import CorrectedPopulationEstimator


def evaluate():
    torch.set_num_threads(1)
    start = time.perf_counter()
    families = {}
    for family in ('kumaraswamy', 'tnormal', 'smooth', 'two_resolution'):
        pyro.clear_param_store()
        pyro.set_rng_seed(120)
        mass_basis = family in ('smooth', 'two_resolution')
        options = dict(data_dim=3, num_corr_kumaraswamy=3 if family == 'kumaraswamy' else 0,
                       num_corr_tnormal=3 if family == 'tnormal' else 0, num_corr_tpareto=0,
                       num_corr_axis_spline_kumaraswamy=3 if mass_basis else 0,
                       hidden_dim_flow=8, num_flow_transforms=1, device='cpu',
                       physical_bounds=([math.log10(3), 0., 0.], [math.log10(150), 2.3, 1.]),
                       axis_spline_num_bins=16, m1_basis_type=family if mass_basis else 'single')
        with contextlib.redirect_stdout(io.StringIO()):
            model = CorrectedPopulationEstimator(**options)
        latent = torch.zeros(model.latent_dim, dtype=torch.float64, requires_grad=True)
        params = model._unflatten_latent_vector(latent)
        normal = torch.tensor([[-40., .3, 8.], [-8., -.2, .5], [.1, .5, 1.], [8., -8., 40.]], dtype=torch.float64)
        marginal, chol = next(model._components(params))
        tails = torch.special.log_ndtr(normal[:, None]), torch.special.log_ndtr(-normal[:, None])
        coordinates = marginal.icdf_log_coordinates(*tails)
        if mass_basis:
            recovered = [m.log_tails_from_log_coordinates(coordinates[0][..., i], coordinates[1][..., i])
                         for i, m in enumerate(marginal.marginals)]
            recovered = tuple(torch.stack([v[j] for v in recovered], -1) for j in (0, 1))
        else:
            recovered = marginal.log_tails_from_log_coordinates(*coordinates)
        tail_error = max(float((v - t).detach().abs().max()) for v, t in zip(recovered, tails))
        assert tail_error < 2e-8
        points = torch.tensor([[[.2, .3, .4], [.6, .7, .8]], [[.1, .5, .8], [.7, .6, .5]]], dtype=torch.float64)
        physical = model.min_vals + (model.max_vals - model.min_vals) * points
        events = torch.cat((physical, torch.ones(2, 2, 1, dtype=torch.float64)), -1)
        log_likelihood = model.event_log_likelihoods(params, events).sum()
        gradient = torch.autograd.grad(log_likelihood, latent)[0]
        assert torch.isfinite(gradient).all()
        model.correct_selection_effects = True
        model.selection_function = lambda x: .2 + .5 * x[:, 1] / 2.3
        model.lambda_det_mc_samples = 64
        alpha = model._selection_fraction_for_params(params)
        assert abs(float(alpha) - .45) < .005
        svi = SVI(model._pyro_model, model._pyro_guide, pyro.optim.Adam({'lr': 1e-4}), loss=Trace_ELBO())
        losses = [svi.step(events) for _ in range(2)]
        assert np.isfinite(losses).all()
        result = {'components': 3, 'log_tail_roundtrip_max_abs_error': tail_error,
                  'event_log_likelihood': float(log_likelihood.detach()),
                  'event_gradient_finite': True, 'selection_fraction': float(alpha.detach()),
                  'selection_points': 64, 'svi_smoke_steps': 2, 'svi_smoke_losses': losses}
        if mass_basis:
            mass = marginal.marginals[0]
            area = sum(quad(lambda x: float(mass.log_prob(torch.tensor(x, dtype=torch.float64))[0].exp().detach()),
                            float(a), float(b), epsabs=1e-11)[0] for a, b in zip(mass.edges[:-1], mass.edges[1:]))
            result['mass_normalization'] = area
            assert abs(area - 1) < 2e-10
        if family == 'kumaraswamy':
            expected = model.numpy_log_pdf_from_params(params, points)
            actual = model._log_pdf_phys_from_params(params, points)
            result['numpy_log_density_max_abs_difference'] = float(np.max(np.abs(actual.detach().numpy()-expected)))
            assert result['numpy_log_density_max_abs_difference'] < 2e-10
        families[family] = result
    pyro.clear_param_store()
    return {'families': families, 'seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/analysis/stable_population_validation.json')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve the existing record; choose a new --output path.')
    sources = ('src/gwpopulation/core.py', 'src/gwpopulation/smooth_mass.py', 'src/gwpopulation/rate.py',
               'src/gwpopulation/bounded_density.py', 'src/gwpopulation/stable_marginals.py',
               'src/gwpopulation/stable_model.py', 'scripts/check_stable_population.py',
               'tests/test_stable_population.py')
    record = {
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'pyro': pyro.__version__},
        'density_version': CorrectedPopulationEstimator.density_version,
        'threads': 1, 'production_inference_rerun': False, 'saved_posterior_impact_assessed': False,
        'entrypoint': 'gwpopulation.stable_model.CorrectedPopulationEstimator',
        'scope': 'Corrected marginal, copula, event, selection and transport paths for adopted families; tiny SVI smoke checks are not posterior evidence. Historical runners and frozen results are retained unchanged.',
        'checks': evaluate(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='ascii', newline='\n')
    print(json.dumps(record['checks'], indent=2))


if __name__ == '__main__':
    main()
