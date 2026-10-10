"""Initialize a guide using detected event summaries, without injected truth."""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.special import logit, ndtri
from scipy.stats import rankdata
import torch
from torch.distributions.transforms import CorrCholeskyTransform


def initialize_from_events(model, data, seed):
    smooth = (model.K == model.num_corr_axis_spline_kumaraswamy
              and model.m1_basis_type == 'smooth')
    if not (model.K == model.num_corr_kumaraswamy or model.K == model.num_corr_tnormal or smooth):
        raise ValueError('Event initialization supports homogeneous Kumaraswamy or normal mixtures')
    points = data[...,:3].mean(1).numpy().astype(np.float64)
    points = (points-model.min_vals.numpy())/(model.max_vals-model.min_vals).numpy()
    points = np.clip(points,1e-5,1-1e-5)
    if model.K == 1:
        labels = np.zeros(len(points),dtype=int)
    else:
        scale = np.maximum(points.std(0),.05)
        _,labels = kmeans2(points/scale,model.K,iter=50,minit='++',seed=seed)
    weights, marginals, correlations = [], [], []
    for k in range(model.K):
        selected = points[labels==k]
        if len(selected)<4:
            raise ValueError('Initialization produced a component with fewer than four events')
        weights.append(len(selected)/len(points))
        marginal = []
        for axis in range(3):
            x = selected[:,axis]
            if smooth and axis == 0:
                mass = model._m1_spline_dist(torch.zeros(model.m1_spline_num_params, dtype=torch.float64))
                knots = np.linspace(0, 1, model.m1_spline_num_params)
                bandwidth = max(.04, 1.06 * x.std() * len(x)**(-.2))
                density = np.exp(-.5 * ((knots[:, None] - x) / bandwidth)**2).mean(1) / (np.sqrt(2*np.pi) * bandwidth)
                target = np.log(density + .001)
                target -= target.mean()
                factor = mass.log_height_factor.numpy()
                coefficients = factor.T @ np.linalg.solve(factor @ factor.T + .25*np.eye(len(knots)), target)
                marginal.append(coefficients)
            elif model.num_corr_kumaraswamy or smooth:
                def objective(values):
                    a,b = np.exp(values)
                    return -np.mean(np.log(a)+np.log(b)+(a-1)*np.log(x)
                                    +(b-1)*np.log(-np.expm1(a*np.log(x))))
                result = minimize(objective,np.log([2.,2.]),method='L-BFGS-B',bounds=[(-3,4)]*2)
                if not result.success:
                    raise RuntimeError(result.message)
                marginal.append(result.x)
            else:
                marginal.append([logit(x.mean()),np.log(max(x.std(),.025))])
        if smooth:
            tail = np.asarray(marginal[1:])
            marginals.append(np.r_[marginal[0], tail[:, 0], tail[:, 1]])
        else:
            marginals.append(marginal)
        gaussian = ndtri(np.column_stack([(rankdata(selected[:,axis])-.5)/len(selected) for axis in range(3)]))
        correlation = .8*np.corrcoef(gaussian.T)+.2*np.eye(3)
        chol = torch.tensor(np.linalg.cholesky(correlation))
        correlations.append(CorrCholeskyTransform().inv(chol).numpy())
    logits = np.log(weights)
    logits -= logits.mean()
    location = torch.tensor(np.r_[logits,np.array(marginals).ravel(),np.array(correlations).ravel()],dtype=torch.float32)
    if len(location)!=model.latent_dim:
        raise ValueError('Unsupported latent layout for affine initialization')
    couplings = [transform for transform in model.flow_transforms.parts if hasattr(transform,'nn')]
    with torch.no_grad():
        for coupling in couplings:
            coupling.nn.net[-1].weight.zero_()
            coupling.nn.net[-1].bias.zero_()
        half = len(location)//2
        outputs = len(location) - half
        first_mean = torch.cat([torch.zeros(half), location[half:]])
        second_shift = location.flip(0)[half:] - .5 * first_mean.flip(0)[half:]
        for coupling,center in zip(couplings[:2],(location[half:],second_shift)):
            coupling.nn.net[-1].bias[:outputs].copy_(center)
            coupling.nn.net[-1].bias[outputs:].fill_(float(np.arctanh(np.log(.5))))
    scales = np.full(len(location), .5)
    if len(location) % 2:
        scales[half] = .25
    return {'method':'clustering observed posterior means; marginal fitting for initialization only',
            'seed':seed,'component_event_counts':[int(np.sum(labels==k)) for k in range(model.K)],
            'initial_latent_location':location.tolist(),
            'initial_latent_scale':scales.tolist() if len(location) % 2 else .5}
