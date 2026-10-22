#!/usr/bin/env python3
"""Dependency and formula checks for reliability-calibrated EVI."""

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
for path in (
    SCRIPT_DIR,
    SCRIPT_DIR.parent / "care-evidential-conflict-router-v1",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reliability_calibrated_core import (  # noqa: E402
    continuous_stage1_risk,
    make_ranking_pairs,
    weighted_pairwise_rank_loss,
)


def main() -> None:
    labels = np.asarray([1, 1, 1, 0])
    probability = np.asarray([0.95, 0.52, 0.08, 0.90])
    risk = continuous_stage1_risk(labels, probability)
    assert np.allclose(risk, [0.05, 0.48, 0.92, 0.90])
    left, right, weight = make_ranking_pairs(risk, seed=7, max_pairs=32)
    uncertainty = torch.as_tensor(risk, dtype=torch.float32)
    good = weighted_pairwise_rank_loss(
        uncertainty,
        torch.as_tensor(left),
        torch.as_tensor(right),
        torch.as_tensor(weight),
    )
    bad = weighted_pairwise_rank_loss(
        1.0 - uncertainty,
        torch.as_tensor(left),
        torch.as_tensor(right),
        torch.as_tensor(weight),
    )
    assert good.item() < bad.item()
    print("reliability-calibrated evidential router self-test passed")


if __name__ == "__main__":
    main()
