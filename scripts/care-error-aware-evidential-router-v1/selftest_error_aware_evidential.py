#!/usr/bin/env python3
"""Formula checks for routing-aware evidential calibration."""

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCY_DIR = SCRIPT_DIR.parent / "care-evidential-conflict-router-v1"
for path in (SCRIPT_DIR, DEPENDENCY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from routing_aware_evidential_core import (  # noqa: E402
    balanced_route_bce,
    torch_conflict_fusion,
)


def main() -> None:
    agreement = torch.tensor([[[10.0, 0.0]] * 4])
    conflict = torch.tensor([[
        [10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]
    ]])
    agree_u = torch_conflict_fusion(agreement, 1.0)["uncertainty"]
    conflict_u = torch_conflict_fusion(conflict, 1.0)["uncertainty"]
    assert conflict_u.item() > agree_u.item()
    score = torch.tensor([0.2, 0.8, 0.3, 0.7])
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.isfinite(balanced_route_bce(score, target))
    print("error-aware evidential router self-test passed")


if __name__ == "__main__":
    main()
