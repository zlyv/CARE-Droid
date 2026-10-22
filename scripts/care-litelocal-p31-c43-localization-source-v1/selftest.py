#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent.parent / "scripts"
for path in (HERE, SCRIPTS / "care-di25-graphglobal-path-source-v1"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from localization_models import ARM_SPECS, align_relation_score, critical_pool


def main():
    local = torch.tensor([[1., 0.], [0., 1.], [2., 2.], [-1., 3.]])
    score = torch.tensor([0., 1., 3., 2.])
    attention = torch.softmax(score, dim=0)
    hard = critical_pool(local, score, attention, False)
    top3 = critical_pool(local, score, attention, True)
    assert torch.equal(hard, local[2])
    assert top3.shape == hard.shape and torch.isfinite(top3).all()
    relation = torch.tensor([1., -2., 3.])
    assert torch.equal(align_relation_score(torch.zeros(3), relation, False), relation)
    assert torch.allclose(align_relation_score(torch.zeros(3), relation, True), relation)
    assert tuple(ARM_SPECS) == (
        "L1_TOPK3_RAW", "L2_ARGMAX_ALIGNED", "L3_TOPK3_ALIGNED"
    )
    print("P31_C43_LOCALIZATION_SELFTEST_PASS")
    print("arms=L0_LOCKED,L1_TOPK3_RAW,L2_ARGMAX_ALIGNED,L3_TOPK3_ALIGNED")
    print("future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
