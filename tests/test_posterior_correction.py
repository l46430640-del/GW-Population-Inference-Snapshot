import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pyro
import pytest
import torch
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correct_selected_posterior import (
    compare_features, draw_proposal, importance_weights, latent_log_target,
    latent_sites, proposal_log_prob, selected_model, swap_components,
)


@pytest.fixture
def model():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    pyro.clear_param_store()
    pyro.set_rng_seed(10071)
    yield selected_model()
    pyro.clear_param_store()
    torch.set_default_dtype(previous)


def tiny_data():
    return torch.tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]],
                         [[.6, .5, .7, 1.], [.7, .6, .8, 1.]]])


def test_permutation_is_an_involution_and_preserves_target(model):
    latent = torch.linspace(-.6, .6, model.latent_dim)
    swapped = swap_components(model, latent)
    torch.testing.assert_close(swap_components(model, swapped), latent, atol=0, rtol=0)
    for key, value in latent_sites(model, latent).items():
        torch.testing.assert_close(latent_sites(model, swapped)[key], value.flip(0))
    torch.testing.assert_close(latent_log_target(model, latent, tiny_data()),
                               latent_log_target(model, swapped, tiny_data()))


def test_latent_density_ratio_matches_original_pyro_guide(model):
    data = tiny_data()
    with torch.no_grad():
        guide = pyro.poutine.trace(model._pyro_guide).get_trace(data)
        replay = pyro.poutine.trace(pyro.poutine.replay(model._pyro_model, trace=guide)).get_trace(data)
        # Reconstruct the sampled flow coordinates without changing their measure.
        sites = {name: guide.nodes[name]['value'] for name in (
            'weights_logits', 'log_k_a', 'log_k_b', 'k_L_cholesky')}
        from torch.distributions.transforms import CorrCholeskyTransform
        latent = torch.cat([
            sites['weights_logits'].reshape(-1),
            torch.stack([sites['log_k_a'], sites['log_k_b']], -1).reshape(-1),
            CorrCholeskyTransform().inv(sites['k_L_cholesky']).reshape(-1)])
        actual = latent_log_target(model, latent, data) - model.flow_dist.log_prob(latent)
        expected = replay.log_prob_sum() - guide.log_prob_sum()
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)


def test_proposal_averages_density_and_is_label_symmetric(model):
    other = selected_model()
    points, _, _ = draw_proposal([model, other], 24, 1921)
    swapped = swap_components(model, points)
    with torch.no_grad():
        terms = torch.stack([m.flow_dist.log_prob(p) for m in (model, other) for p in (points, swapped)])
    expected = torch.logsumexp(terms, 0) - np.log(4)
    torch.testing.assert_close(proposal_log_prob([model, other], points), expected)
    torch.testing.assert_close(proposal_log_prob([model, other], swapped), expected)
    again, _, _ = draw_proposal([model, other], 24, 1921)
    torch.testing.assert_close(again, points, atol=0, rtol=0)


def test_weights_are_stable_under_log_normalizer_shifts():
    values = np.random.default_rng(771).normal(size=1000)
    actual, diagnostic = importance_weights(values)
    shifted, _ = importance_weights(values + 10000)
    for name in actual:
        assert actual[name].sum() == pytest.approx(1)
        np.testing.assert_allclose(actual[name], shifted[name], rtol=1e-9)
        assert 1 <= diagnostic[name]['ess'] <= len(values)
    np.testing.assert_allclose(actual['raw'], np.exp(values - logsumexp(values)))
    for bad in ([0.] * 10, [np.inf] * 20, [[0.] * 20]):
        with pytest.raises(ValueError):
            importance_weights(bad)


def test_weighted_comparison_matches_repeated_empirical_sample():
    values = np.arange(4.)[:, None]
    counts = np.array([1, 3, 2, 4])
    reference = np.repeat(values, counts, axis=0)
    comparison = compare_features(reference, values, ['x'], counts)
    assert comparison['mean_wasserstein_in_nuts_sd'] == pytest.approx(0)
    assert comparison['maximum_absolute_mean_offset_in_nuts_sd'] == pytest.approx(0)
    assert comparison['features'][0]['sd_ratio'] == pytest.approx(1)
    with pytest.raises(ValueError):
        compare_features(reference, values, ['x'], np.array([1., -1., 1., 1.]))


def test_correction_recovers_known_normal_moments():
    points = np.random.default_rng(981).normal(.5, 1., 40000)
    log_ratio = -.5 * points**2 + .5 * (points - .5)**2
    weights, diagnostic = importance_weights(log_ratio)
    for w in weights.values():
        mean = np.dot(w, points)
        variance = np.dot(w, (points - mean)**2)
        assert abs(mean) < .025
        assert abs(variance - 1) < .035
    assert diagnostic['pareto_k'] < .7


@pytest.mark.parametrize('seed', [731001, 731002])
def test_retained_products_match_weights_sources_and_original_target(model, seed):
    root = Path(__file__).resolve().parents[1]
    path = root / f'results/analysis/importance_correction/repeat_{seed}.json'
    report = json.loads(path.read_text())
    product_path = path.with_suffix('.npz')
    assert hashlib.sha256(product_path.read_bytes()).hexdigest() == report['draw_product_sha256']
    for name, expected in report['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    with np.load(product_path) as product:
        weights, diagnostic = importance_weights(product['log_target'] - product['log_proposal'])
        for name, values in weights.items():
            np.testing.assert_allclose(product[f'{name}_weights'], values, atol=1e-14, rtol=1e-12)
        assert diagnostic == report['weight_diagnostics']
        assert product['features'].shape == (8192, 12)
        data = torch.tensor(np.load(root / report['configuration']['source'] / 'train_catalog.npz')['data'])
        models = [model, selected_model()]
        for index, proposal in enumerate(models):
            checkpoint = torch.load(root / report['configuration']['source'] / f'vi_start_{index}.pt',
                                    weights_only=True, map_location='cpu')
            proposal.flow_transforms.load_state_dict(checkpoint['state'])
        indices = [0, 4096, 8191]
        latent = torch.tensor(product['latent'][indices])
        np.testing.assert_allclose(proposal_log_prob(models, latent), product['log_proposal'][indices], atol=1e-10)
        target = [float(latent_log_target(model, point, data)) for point in latent]
        np.testing.assert_allclose(target, product['log_target'][indices], atol=1e-10)
