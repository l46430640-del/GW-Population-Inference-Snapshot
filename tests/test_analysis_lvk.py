import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_lvk_population import coordinates, spin_density
from lvk_injections import covered_mass_pairs, spin_measure_jacobian


def test_physical_bbh_coordinates_and_jacobian():
    x = np.array([np.log10(25.), .3, .7])

    def physical(theta):
        m1 = 10**theta[0]
        return np.array([m1, 3 + (m1 - 3) * theta[2], theta[1]])

    source = physical(x)
    recovered, jacobian = coordinates(source[0:1], source[1:2], source[2:3])
    np.testing.assert_allclose(recovered[0], x)
    columns = []
    for i in range(3):
        dx = np.eye(3)[i] * 1e-5
        columns.append((physical(x + dx) - physical(x - dx)) / 2e-5)
    np.testing.assert_allclose(abs(np.linalg.det(np.column_stack(columns))), jacobian[0], rtol=1e-8)


def test_fixed_spin_density_is_normalized_in_declared_coordinates():
    for mode in ('uniform', 'low'):
        # Integrate both magnitudes; the two tilt cosines supply a factor four.
        value = quad(lambda a: float(spin_density(np.array([a]), np.array([.4]), mode)[0]), 0, .99)[0]
        reference = float(spin_density(np.array([.4]), np.array([.4]), mode)[0])
        np.testing.assert_allclose(4 * value**2 / reference, 1., rtol=1e-8)
        assert spin_density(np.array([.995]), np.array([.4]), mode)[0] == 0


def test_cartesian_spin_measure_conversion():
    a1,a2 = .3,.8
    # A uniform isotropic magnitude law has Cartesian density 1/(4*pi*a^2).
    cartesian = 1/(4*np.pi*a1**2)*1/(4*np.pi*a2**2)
    np.testing.assert_allclose(cartesian*spin_measure_jacobian(a1,a2),.25)
def test_two_resolution_mass_inverse_matches_cdf():
    import torch
    from gwpopulation.core import BoundedTwoResolutionSpline
    from summarize_lvk_marginals import piecewise_mass_icdf
    generator = torch.Generator().manual_seed(821)
    distribution = BoundedTwoResolutionSpline(torch.randn(3,6,generator=generator,dtype=torch.float64),
                                               torch.randn(3,17,generator=generator,dtype=torch.float64),
                                               torch.tensor([-.7,0.,1.2],dtype=torch.float64),.15,.72)
    probabilities = np.random.default_rng(18).uniform(1e-6,1-1e-6,(1000,3))
    points = piecewise_mass_icdf(distribution,probabilities)
    recovered = distribution.cdf(torch.tensor(points)).numpy()
    np.testing.assert_allclose(recovered,probabilities,rtol=1e-10,atol=1e-10)


def test_joint_injection_domain_and_conditioned_normalization():
    import torch
    from scipy.special import ndtri
    from scipy.stats import qmc
    from gwpopulation.core import BayesianMixtureEstimator_NF_Correlated
    from summarize_lvk_marginals import conditioned_marginal_bins
    np.testing.assert_array_equal(covered_mass_pairs(np.array([99,100,101,200]),
                                                   np.array([3,3,9,10])),[True,True,False,True])
    model = BayesianMixtureEstimator_NF_Correlated(3,1,0,0,hidden_dim_flow=16,
        num_flow_transforms=2,physical_bounds=([np.log10(3),0,0],[np.log10(300),2.3,1]),device='cpu')
    latent = np.zeros(model.latent_dim)
    latent[1:7] = np.log(.9999)
    normal = ndtri(qmc.Sobol(3,scramble=True,seed=837).random_base2(19))
    density,change,probability,error = conditioned_marginal_bins(model,latent,normal)
    low,high = float(model.min_vals[0]),float(model.max_vals[0])
    expected = 1-quad(lambda ell:7/(10**ell-3)/(high-low),2,high,epsabs=1e-12)[0]
    np.testing.assert_allclose(probability,expected,atol=5e-5)
    np.testing.assert_allclose(density.sum(1)*np.array([high-low,2.3,1])/96,1,atol=1e-14)
    assert np.all(density>=0)


def test_conditional_population_normalizer_cancels_from_shape_likelihood():
    normalizations = np.array([.65,.9,.999])
    events = np.array([[.2,.3],[.5,.6],[.8,.9]])
    alpha = np.array([.01,.02,.03])
    base = np.log(events).sum(1)-2*np.log(alpha)
    conditioned = np.log(events/normalizations[:,None]).sum(1)-2*np.log(alpha/normalizations)
    np.testing.assert_allclose(base,conditioned,atol=1e-14)

