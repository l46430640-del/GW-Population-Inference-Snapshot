import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correct_selected_posterior import swap_components
from diagnose_lisa_correction import default_dtype, eligibility, lisa_model, target_value
from correct_selected_posterior import importance_weights


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration():
    return {'detector': 'lisa', 'family': 'kumaraswamy', 'components': 2,
            'catalog_seed': 823010, 'morphology': 'bimodal', 'quadrature': 128,
            'integration': 'population_qmc'}


def data(dtype):
    return torch.tensor([[[.2, .3, .4, 1.], [.3, .4, .5, 1.]],
                         [[.6, .5, .7, 1.], [.7, .6, .8, 1.]]], dtype=dtype)


def test_lisa_adapter_preserves_sobol_nodes_and_float32_target():
    from validate_population_morphologies import build_model
    adapted = lisa_model(configuration(), torch.float32)
    with default_dtype(torch.float32):
        original = build_model('kumaraswamy', 2, 128, 823011)
    original.selection_integration = 'population_qmc'
    original.selection_function = adapted.selection_function
    raw = torch.linspace(-.3, .4, adapted.latent_dim, dtype=torch.float32)
    assert target_value(original, raw, data(torch.float32)) == target_value(adapted, raw, data(torch.float32))
    double = lisa_model(configuration())
    torch.testing.assert_close(double._selection_mc_samples_norm(128, 'cpu'),
                               original._selection_mc_samples_norm(128, 'cpu').double(), atol=0, rtol=0)
    assert abs(target_value(double, raw.double(), data(torch.float64))
               - target_value(original, raw, data(torch.float32))) < 1e-4


def test_lisa_target_is_label_symmetric_and_uses_lisa_selection():
    model = lisa_model(configuration())
    raw = torch.linspace(-.3, .4, model.latent_dim, dtype=torch.float64)
    first = target_value(model, raw, data(torch.float64))
    second = target_value(model, swap_components(model, raw), data(torch.float64))
    assert first == pytest.approx(second, abs=1e-10)
    model.selection_function = lambda points: torch.full((len(points),), .5, dtype=points.dtype)
    assert abs(target_value(model, raw, data(torch.float64)) - first) > .01


def test_default_dtype_restored_on_error():
    old = torch.get_default_dtype()
    with pytest.raises(RuntimeError):
        with default_dtype(torch.float64):
            assert torch.get_default_dtype() == torch.float64
            raise RuntimeError('check cleanup')
    assert torch.get_default_dtype() == old


def test_failed_weights_or_numerics_cannot_accept_corrected_summaries():
    config = {'maximum_pareto_k': .7, 'minimum_effective_samples': 400,
              'maximum_refined_log_target_sd': .3}
    diagnostic = {'pareto_k': .5, 'pareto_k_limit': .7, 'raw': {'ess': 500}, 'psis': {'ess': 600}}
    numerical = {'uniform': {'total_change_sd': .1}, 'psis': {'total_change_sd': .2}}
    assert eligibility(diagnostic, numerical, True, config)['eligible_for_corrected_summaries']
    for key, value in (('pareto_k', .8), ('raw', {'ess': 20})):
        changed = dict(diagnostic, **{key: value})
        assert not eligibility(changed, numerical, True, config)['eligible_for_corrected_summaries']
    assert not eligibility(diagnostic, numerical, False, config)['eligible_for_corrected_summaries']
    assert not eligibility(diagnostic, {'psis': {'total_change_sd': .4}}, True, config)['eligible_for_corrected_summaries']


@pytest.mark.parametrize('seed', [732001, 732002])
def test_retained_lisa_diagnosis_reproduces_weights_target_and_decision(seed):
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/analysis/lisa_correction'
    path = output / f'repeat_{seed}.json'
    report = json.loads(path.read_text())
    for name, expected in report['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == report['draw_product_sha256']
    product = np.load(path.with_suffix('.npz'))
    weights, diagnostic = importance_weights(product['log_target'] - product['log_proposal'])
    assert diagnostic == report['weight_diagnostics']
    for name, values in weights.items():
        np.testing.assert_allclose(product[name + '_weights'], values, atol=1e-14)
    assert report['decision'] == eligibility(diagnostic, report['refinement'], True, report['configuration'])
    assert product['latent'].shape == (4096, 20)
    for name in ('uniform', 'psis', 'largest_raw_weights'):
        selected = product['refinement_' + name + '_indices']
        offsets = np.searchsorted(product['refinement_indices'], selected)
        changes = product['refinement_event_change'][offsets] + product['refinement_selection_change'][offsets]
        assert changes.std() == pytest.approx(report['refinement'][name]['total_change_sd'])
    source = root / report['configuration']['sources'][0]
    config = json.loads((source / 'summary.json').read_text())['configuration']
    model = lisa_model(config)
    observations = torch.tensor(np.load(source / 'catalog.npz')['data'], dtype=torch.float64)
    for index in (0, int(weights['raw'].argmax())):
        latent = torch.tensor(product['latent'][index], dtype=torch.float64)
        assert target_value(model, latent, observations) == pytest.approx(product['log_target'][index], abs=1e-10)
