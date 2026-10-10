"""Corrected likelihood for the component families adopted in the analysis.

The historical core remains executable for paired impact measurements. This
class reuses its latent coordinates, priors, and normalizing-flow guide, but
owns every density, likelihood, prediction, and population transport path.
"""
from __future__ import annotations

import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .bounded_density import (
    KumaraswamyMarginal, gaussian_copula_log_prob, numpy_kumaraswamy_mixture_log_prob,
)
from .core import BayesianMixtureEstimator_NF_Correlated
from .rate import conditional_rate_draws
from .stable_marginals import (
    AxisMarginals, PiecewiseLinearMarginal, TruncatedNormalMarginal,
    smooth_mass_marginal, two_resolution_mass_marginal,
)


class CorrectedPopulationEstimator(BayesianMixtureEstimator_NF_Correlated):
    density_version = 'log_tail_v1'
    _families = (
        ('kumaraswamy', 'k', 'log_k_a', 'log_k_b'),
        ('tnormal', 'tnormal', 'tn_locs', 'log_tn_scales'),
        ('axis_spline_tnormal', 'axis_stn', 'axis_stn_tn_locs', 'axis_stn_log_tn_scales'),
        ('axis_spline_kumaraswamy', 'axis_sk', 'axis_sk_log_k_a', 'axis_sk_log_k_b'),
    )

    def __init__(self, *args, independent_components=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.independent_components = bool(independent_components)
        supported = {'num_corr_' + family for family, *_ in self._families}
        unsupported = [key for key, value in vars(self).items()
                       if key.startswith('num_corr_') and key not in supported and value]
        if unsupported:
            raise ValueError('Unmigrated families: ' + ', '.join(unsupported))
        if self.data_dim != 3 or self.provided_bounds is None:
            raise ValueError('The analysis estimator requires three fixed physical bounds.')
        if self.m1_basis_coordinate != 'log10' or self.m1_roughness_lambda:
            raise ValueError('The analysis uses a log-mass basis without a roughness factor.')
        self.min_vals, self.max_vals = [torch.as_tensor(v, dtype=torch.float64, device=self.device)
                                        for v in self.provided_bounds]
        self.obs_neff_threshold = self.selection_neff_multiplier = self.q_monotonicity_lambda = 0.
        self.kumaraswamy_log_param_clip = None
        self.selection_mc_mode = 'fixed_sobol'
        self.selection_integration = 'population_qmc'
        self.injection_data = None
        self._stable_selection_cache = {}

    def _check_protocol(self):
        if (self.obs_neff_threshold or self.selection_neff_multiplier or self.q_monotonicity_lambda
                or self.m1_roughness_lambda or self.kumaraswamy_log_param_clip is not None):
            raise ValueError('The corrected analysis protocol has no ESS, shape, or Kumaraswamy clipping factors.')

    def _m1_spline_dist(self, logits):
        logits = logits.double()
        if self.m1_basis_type == 'smooth':
            return smooth_mass_marginal(logits, self.min_vals[0], self.max_vals[0], self.m1_smooth_length)
        if self.m1_basis_type == 'single':
            return PiecewiseLinearMarginal.from_logits(logits)
        if self.m1_basis_type != 'two_resolution':
            raise ValueError('Unknown mass basis: ' + self.m1_basis_type)
        ng, nl = self.m1_two_resolution_global_knots, self.m1_two_resolution_local_knots
        width = float(self.max_vals[0] - self.min_vals[0])
        low = (math.log10(self.m1_two_resolution_local_min_solar_mass) - float(self.min_vals[0])) / width
        high = (math.log10(self.m1_two_resolution_local_max_solar_mass) - float(self.min_vals[0])) / width
        return two_resolution_mass_marginal(logits[..., :ng], logits[..., ng:ng + nl],
                                            logits[..., -1], low, high)

    def _axis_spline_kumaraswamy_marginal_dist(self, logits, log_a, log_b):
        return AxisMarginals([self._m1_spline_dist(logits)] + [
            KumaraswamyMarginal(log_a[..., i].double().exp() + .0001,
                               log_b[..., i].double().exp() + .0001, validate_args=False)
            for i in range(self.data_dim - 1)])

    def _axis_spline_tnormal_marginal_dist(self, logits, loc, log_scale):
        return AxisMarginals([self._m1_spline_dist(logits)] + [
            TruncatedNormalMarginal(loc[..., i], log_scale[..., i].double().exp() + .0001)
            for i in range(self.data_dim - 1)])

    def _components(self, params):
        for family, prefix, first_key, second_key in self._families:
            if not getattr(self, 'num_corr_' + family):
                continue
            first, second = params[first_key].double(), params[second_key].double()
            if family.endswith('tnormal'):
                first = self._clamp_prob_param(first)
                second = self._clamp_log_param(second, -8., 5.)
            if family.startswith('axis_spline'):
                logits = params[prefix + '_m1_logits'].double().clamp(-12., 12.)
                factory = (self._axis_spline_tnormal_marginal_dist if family.endswith('tnormal')
                           else self._axis_spline_kumaraswamy_marginal_dist)
                marginal = factory(logits, first, second)
            elif family == 'kumaraswamy':
                marginal = KumaraswamyMarginal(first.exp() + .0001, second.exp() + .0001, validate_args=False)
            else:
                marginal = TruncatedNormalMarginal(first, second.exp() + .0001)
            cholesky = params.get(prefix + '_L_cholesky')
            if cholesky is None:
                cholesky = self._build_cholesky_from_unconstrained(
                    params[prefix + '_cholesky_params'].double(), self.data_dim, self.device)
            yield marginal, cholesky.double()

    @staticmethod
    def _marginal_bounds(marginal, reference):
        return [torch.as_tensor(getattr(marginal, name, default), dtype=torch.float64, device=reference.device)
                for name, default in (('low', 0.), ('high', 1.))]

    def _copula_log_prob(self, marginal_dist, L_cholesky, data_norm):
        data = data_norm.double().unsqueeze(-2)
        low, high = self._marginal_bounds(marginal_dist, data)
        inside = ((data > low) & (data < high)).all(-1)
        safe = torch.where(inside.unsqueeze(-1), data, (low + high) / 2)
        tails = marginal_dist.log_tails(safe)
        value = marginal_dist.log_prob(safe).sum(-1)
        if not self.independent_components:
            value = value + gaussian_copula_log_prob(*tails, L_cholesky)
        return torch.where(data.isnan().any(-1), torch.nan, torch.where(inside, value, -torch.inf))

    def _log_pdf_phys_from_params(self, params, points_norm):
        self._check_protocol()
        components = [self._copula_log_prob(m, chol, points_norm) for m, chol in self._components(params)]
        weights = params['weights_logits'].double().clamp(-20., 20.).log_softmax(-1)
        values = torch.cat(components, -1) + weights
        possible = (values != -torch.inf).any(-1)
        safe = torch.where(possible.unsqueeze(-1), values, torch.zeros_like(values))
        result = torch.logsumexp(safe, -1) - (self.max_vals - self.min_vals).log().sum()
        return torch.where(possible, result, -torch.inf)

    def _sample_prior_parameters(self):
        zero = torch.zeros((), dtype=torch.float64, device=self.device)
        one = torch.ones_like(zero)
        params = {'weights_logits': pyro.sample('weights_logits', dist.Normal(zero, one).expand([self.K]).to_event(1))}
        for family, prefix, first, second in self._families:
            count = getattr(self, 'num_corr_' + family)
            if not count:
                continue
            dim = self.data_dim - int(family.startswith('axis_spline'))
            with pyro.plate(family + '_components', count):
                if family.startswith('axis_spline'):
                    key = prefix + '_m1_logits'
                    params[key] = pyro.sample(key, dist.Normal(zero, one).expand([self.m1_spline_num_params]).to_event(1))
                first_prior = dist.Uniform(zero, one) if family.endswith('tnormal') else dist.Normal(zero, one)
                params[first] = pyro.sample(first, first_prior.expand([dim]).to_event(1))
                params[second] = pyro.sample(second, dist.Normal(zero, one).expand([dim]).to_event(1))
                params[prefix + '_L_cholesky'] = pyro.sample(
                    prefix + '_L_cholesky', dist.LKJCholesky(self.data_dim, concentration=one * 1.5))
        return params

    def numpy_log_pdf_from_params(self, params, points_norm):
        """Independent NumPy density for the pure-Kumaraswamy external tests."""
        self._check_protocol()
        if self.num_corr_kumaraswamy != self.K:
            raise ValueError('The external NumPy evaluator supports pure Kumaraswamy mixtures.')
        _, chol = next(self._components(params))
        if self.independent_components:
            chol = torch.eye(self.data_dim, dtype=chol.dtype, device=chol.device).expand_as(chol)
        mapped = {'weights_logits': params['weights_logits'].double().clamp(-20., 20.),
                  'log_k_a': params['log_k_a'].double(), 'log_k_b': params['log_k_b'].double(),
                  'k_L_cholesky': chol}
        mapped = {key: value.detach().cpu().numpy() for key, value in mapped.items()}
        if torch.is_tensor(points_norm):
            points_norm = points_norm.detach().cpu().numpy()
        volume = np.log((self.max_vals - self.min_vals).detach().cpu().numpy()).sum()
        return numpy_kumaraswamy_mixture_log_prob(mapped, points_norm) - volume

    def event_log_likelihoods(self, params, data_samples):
        data = torch.stack(data_samples) if isinstance(data_samples, list) else data_samples
        data = data.to(device=self.device, dtype=torch.float64)
        if data.ndim != 3 or data.shape[-1] not in (4, 5):
            raise ValueError('Events must have shape (event, sample, coordinate/prior[/weight]).')
        points, prior = data[..., :3], data[..., 3]
        weights = data[..., 4] if data.shape[-1] == 5 else torch.ones_like(prior)
        if not (torch.isfinite(data).all() and (prior > 0).all() and (weights >= 0).all()
                and (weights.sum(-1) > 0).all()):
            raise ValueError('Finite event values, positive PE priors, and nonnegative sample weights are required.')
        points_norm = (points - self.min_vals) / (self.max_vals - self.min_vals)
        log_density = self._log_pdf_phys_from_params(params, points_norm)
        extra_support = getattr(self, 'event_support_function', None)
        if extra_support is not None:
            log_density = torch.where(extra_support(points), log_density, -torch.inf)
        log_weights = weights.log()
        integrand = log_density - prior.log() + log_weights
        possible = (integrand != -torch.inf).any(-1)
        total = torch.logsumexp(torch.where(possible.unsqueeze(-1), integrand, torch.zeros_like(integrand)), -1)
        total = torch.where(possible, total, -torch.inf)
        # An impossible event gives a zero likelihood, rather than a fabricated floor.
        finite = torch.isfinite(total)
        safe = torch.where(finite.unsqueeze(-1), integrand, torch.zeros_like(integrand))
        ess = (2 * torch.logsumexp(safe, -1) - torch.logsumexp(2 * safe, -1)).exp()
        self._last_event_ess = torch.where(finite, ess, 0.)
        return total - torch.logsumexp(log_weights, -1)

    def _log_alpha(self, params):
        if not self.correct_selection_effects:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        if self.injection_data is not None and self.selection_function is None:
            points = torch.stack([self.injection_data[k].to(self.device).double()
                                  for k in ('log_m1', 'z', 'q')], -1)
            norm = (points - self.min_vals) / (self.max_vals - self.min_vals)
            return self._log_selection_fraction_from_params(params, norm)
        if self.selection_function is None:
            raise ValueError('Selection correction requires injections or a selection function.')
        return self._selection_fraction_for_params(params).log()

    def _log_selection_fraction_from_params(self, params, inj_points_norm):
        injection = self.injection_data
        log_draw = injection['log_p_draw'].to(self.device).double()
        if not torch.isfinite(log_draw).all() or injection['Ndraw'] <= 0:
            raise ValueError('Injection draw densities must be finite and Ndraw positive.')
        value = self._log_pdf_phys_from_params(params, inj_points_norm) - log_draw
        if injection.get('log_p_det') is not None:
            log_det = injection['log_p_det'].to(self.device).double()
            if torch.isnan(log_det).any() or (log_det > 0).any():
                raise ValueError('Injection detection log probabilities must be nonpositive.')
            value = value + log_det
        self._last_selection_log_weights = value
        possible = (value != -torch.inf).any()
        safe = torch.where(possible, value, torch.zeros_like(value))
        result = torch.logsumexp(safe, 0) - math.log(injection['Ndraw'])
        return torch.where(possible, result, -torch.inf)

    def _pyro_model(self, data_samples):
        self._check_protocol()
        params = self._sample_prior_parameters()
        events = self.event_log_likelihoods(params, data_samples)
        log_alpha = self._log_alpha(params)
        objective = events.sum() - len(events) * log_alpha
        self._latest_model_diagnostics = {
            'density_version': self.density_version,
            'min_neff_obs': float(self._last_event_ess.detach().min()),
            'event_loglike_sum': float(events.detach().sum()),
            'log_selection_fraction': float(log_alpha.detach()),
            'obs_log_prob_factor': float(objective.detach()),
        }
        pyro.factor('obs_log_prob', objective)

    def component_transport(self, params, normal, return_log_coordinates=False):
        """Common normal points transported through all components (N,K,D)."""
        results, log_coordinates = [], []
        for marginal, chol in self._components(params):
            if self.independent_components:
                chol = torch.eye(self.data_dim, dtype=chol.dtype, device=chol.device).expand_as(chol)
            gaussian = torch.einsum('...j,kij->...ki', normal.double(), chol)
            lx, lr = marginal.icdf_log_coordinates(torch.special.log_ndtr(gaussian), torch.special.log_ndtr(-gaussian))
            low, high = self._marginal_bounds(marginal, gaussian)
            results.append(torch.where(lx <= lr, low + (high - low) * lx.exp(), high - (high - low) * lr.exp()))
            log_coordinates.append((lx, lr))
        if return_log_coordinates:
            # Coordinates are relative to each component's support, including
            # the historical [.0001,.9999] truncated-normal support.
            return tuple(torch.cat([v[i] for v in log_coordinates], -2) for i in (0, 1))
        return torch.cat(results, -2)

    def _selection_mc_samples_norm(self, num_samples, device):
        if self.selection_mc_mode == 'random':
            uniform = torch.rand(num_samples, self.data_dim, dtype=torch.float64, device=device)
        elif self.selection_mc_mode in ('fixed_sobol', 'sobol_fixed'):
            key = (num_samples, str(device), self.selection_mc_seed)
            if key not in self._stable_selection_cache:
                engine = torch.quasirandom.SobolEngine(self.data_dim, scramble=True, seed=self.selection_mc_seed)
                self._stable_selection_cache[key] = engine.draw(num_samples, dtype=torch.float64).to(device)
            uniform = self._stable_selection_cache[key]
        else:
            raise ValueError('Unknown integration sequence: ' + self.selection_mc_mode)
        if not ((uniform > 0) & (uniform < 1)).all():
            raise ValueError('Integration sequence contains an exact endpoint; use a different seed.')
        return uniform

    def _population_selection_fraction(self, params):
        uniform = self._selection_mc_samples_norm(self.lambda_det_mc_samples, self.device)
        points = self.component_transport(params, torch.special.ndtri(uniform))
        points = self.min_vals + (self.max_vals - self.min_vals) * points
        detection = self.selection_function(points.reshape(-1, self.data_dim)).reshape(points.shape[:-1])
        if not (torch.isfinite(detection).all() and (detection >= 0).all() and (detection <= 1).all()):
            raise ValueError('Selection function must return probabilities in [0,1].')
        weights = params['weights_logits'].double().clamp(-20, 20).softmax(-1)
        return torch.dot(weights, detection.mean(0))

    def _selection_fraction_for_params(self, params):
        if not self.correct_selection_effects:
            return torch.ones((), dtype=torch.float64, device=self.device)
        if self.injection_data is not None and self.selection_function is None:
            return self._log_alpha(params).exp()
        if self.selection_integration != 'population_qmc':
            raise ValueError('The corrected selection-function path uses population transport quadrature.')
        return self._population_selection_fraction(params)

    def _calculate_posterior_mean_selection_fraction(self, num_posterior_samples=100):
        with torch.no_grad():
            values = [self._selection_fraction_for_params(self._unflatten_latent_vector(latent))
                      for latent in self.flow_dist.sample((num_posterior_samples,))]
        self.posterior_mean_selection_fraction = float(torch.stack(values).mean())
        return self.posterior_mean_selection_fraction

    def sample(self, num_samples, num_posterior_draws=1):
        if not self.is_trained:
            raise RuntimeError('Model must be trained first.')
        if num_posterior_draws != 1:
            raise NotImplementedError('Use one posterior draw per sampling call.')
        with torch.no_grad():
            params = self._unflatten_latent_vector(self.flow_dist.sample())
            labels = dist.Categorical(logits=params['weights_logits'].double().clamp(-20, 20)).sample((num_samples,))
            normal = torch.randn(num_samples, self.data_dim, dtype=torch.float64, device=self.device)
            points = self.component_transport(params, normal)[torch.arange(num_samples, device=self.device), labels]
        return self.min_vals + (self.max_vals - self.min_vals) * points

    def predict(self, grid_points, num_samples=500, mode='rate_density'):
        if not self.is_trained or (mode == 'rate_density' and not self.is_rate_inferred):
            raise RuntimeError('Fit the shape and, for a rate density, infer its rate first.')
        if mode not in ('pdf', 'rate_density'):
            raise ValueError('Prediction mode must be pdf or rate_density.')
        norm = (grid_points.to(self.device).double() - self.min_vals) / (self.max_vals - self.min_vals)
        results = []
        with torch.no_grad():
            for latent in self.flow_dist.sample((num_samples,)):
                params = self._unflatten_latent_vector(latent)
                value = self._log_pdf_phys_from_params(params, norm).exp()
                if mode == 'rate_density':
                    value = value * conditional_rate_draws(self.rate_diagnostics['num_observed_events'],
                                                          self.observation_years,
                                                          self._selection_fraction_for_params(params))['rate']
                results.append(value)
        values = torch.stack(results)
        return values.mean(0), values.std(0), values.quantile(.05, dim=0), values.quantile(.95, dim=0)
