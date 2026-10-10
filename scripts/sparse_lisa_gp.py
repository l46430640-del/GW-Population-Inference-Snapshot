"""Exact likelihood evaluation on occupied event bins, retaining every GP bin."""
from __future__ import annotations

import numpy as np
import torch

from run_external_gp_selected import InitializedGPPotential


class SparseEventGPPotential(InitializedGPPotential):
    def __init__(self, bins, log_event_weights, selection, settings, jitter=1e-8):
        super().__init__(bins, log_event_weights, selection, settings, jitter)
        weights = np.asarray(log_event_weights)
        occupied = np.isfinite(weights)
        counts = occupied.sum(1)
        width = int(counts.max())
        indices = np.zeros((len(weights), width), dtype=np.int64)
        values = np.full(indices.shape, -np.inf)
        for event in range(len(weights)):
            index = np.flatnonzero(occupied[event])
            indices[event, :len(index)] = index
            values[event, :len(index)] = weights[event, index]
        self.event_indices = torch.tensor(indices)
        self.sparse_log_event_weights = torch.tensor(values, dtype=self.log_event_weights.dtype)
        self.occupied_counts = counts

    def __call__(self, coordinate):
        field = self.field(coordinate)
        events = torch.logsumexp(self.sparse_log_event_weights+field[self.event_indices], -1).sum()
        selected = torch.logsumexp(self.log_selection+field, -1)
        return -self.log_prior(coordinate)-events+self.log_event_weights.shape[0]*selected
