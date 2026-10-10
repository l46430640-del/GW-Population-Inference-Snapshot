"""Run the unchanged GP sampler with bounded-memory Pyro diagnostics."""
from __future__ import annotations

from run_external_gp_pilot import ROOT, PROCESS_STARTED

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import pyro
from pyro.ops import stats
import torch

from correct_selected_posterior import sha256
import run_external_gp_selected as production
from run_lisa_production import write_json


def cumulative_minimum(values):
    """Exact prefix minima for finite diagnostic arrays, with linear storage."""
    if not torch.isfinite(values).all():
        raise ValueError("Finite autocorrelation diagnostics are required")
    return torch.cummin(values, dim=0).values


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    config = json.loads((ROOT / args.config).read_text())
    adapter = config["diagnostic_adapter"]
    if adapter["source"] != Path(__file__).relative_to(ROOT).as_posix() or sha256(Path(__file__)) != adapter["sha256"]:
        raise ValueError("The diagnostic adapter differs from its declared source")
    original = stats._cummin
    original_hash = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if pyro.__version__ != adapter["pyro_version"] or original_hash != adapter["original_cummin_sha256"]:
        raise ValueError("The installed Pyro diagnostic differs from the verified implementation")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    record = {"adapter": adapter, "torch_version": torch.__version__, "scope": "post-sampling cumulative-minimum diagnostic"}
    record_path = output / "diagnostic_adapter.json"
    if record_path.exists():
        if json.loads(record_path.read_text()) != record:
            raise ValueError("Existing diagnostic adapter record differs")
    else:
        write_json(record_path, record)
    # This helper is called only after sampling. Preserve the sampler and
    # every diagnostic, replacing the quadratic temporary with torch.cummin.
    stats._cummin = cumulative_minimum
    try:
        production.PROCESS_STARTED = PROCESS_STARTED
        production.main()
    finally:
        stats._cummin = original


if __name__ == "__main__":
    main()
