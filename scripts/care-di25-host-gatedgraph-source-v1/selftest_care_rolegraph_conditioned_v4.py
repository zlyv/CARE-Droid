#!/usr/bin/env python3
"""Dependency, S7 mapping, and forward-pass self-test for CARE v4."""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import torch

import run_care_rolegraph_conditioned_dualstream_v4 as v4


def main() -> None:
    rng = np.random.RandomState(20260901)
    s7 = rng.normal(size=(4, v4.S7_DIM)).astype(np.float32)
    s7[:, 120:120 + v4.PAIR_DIM] = np.abs(
        s7[:, 120:120 + v4.PAIR_DIM]
    )
    node, graph_global, pair, inter = v4.decompose_s7(s7)
    assert node.shape == (4, 31, 3)
    assert graph_global.shape == (4, 27)
    assert pair.shape == (4, 31, 31)
    assert inter.shape == (4, 31, 31)
    assert np.allclose(pair, np.swapaxes(pair, 1, 2))
    assert np.allclose(np.diagonal(pair, axis1=1, axis2=2), 0.0)
    pair_list = list(combinations(range(31), 2))
    for index, (left, right) in enumerate(pair_list):
        expected = s7[0, 120 + index]
        assert pair[0, left, right] == expected
        assert pair[0, right, left] == expected

    scaler = v4.fit_role_graph_scaler(s7[:3])
    transformed = v4.transform_role_graphs(s7, scaler)
    assert transformed["node"].shape == (4, 31, 3)
    assert transformed["edge"].shape == (4, 31, 31, 3)
    assert transformed["global"].shape == (4, 27)

    args = SimpleNamespace(
        graph_dim=32,
        graph_heads=4,
        graph_layers=2,
        dropout=0.0,
        hidden_dim=48,
        embedding_dim=24,
        max_delta=1.5,
    )
    one_graph = {
        key: torch.as_tensor(value[0], dtype=torch.float32)
        for key, value in transformed.items()
    }
    regions = torch.randn(11, v4.AUGMENTED_REGION_DIM)
    regions[:, -4:] = torch.abs(regions[:, -4:])
    for variant in v4.MODEL_VARIANTS:
        model = v4.RoleGraphConditionedEvidence(24, variant, args)
        logit, diagnostic = model(
            torch.randn(24), torch.tensor(0.2), one_graph, regions
        )
        assert logit.shape == ()
        assert diagnostic.shape == (len(v4.DIAGNOSTIC_NAMES),)
        assert bool(torch.isfinite(logit))
        assert bool(torch.isfinite(diagnostic).all())
        print(variant, "forward OK")

    print("SELFTEST_OK")
    print("s7_dim:", v4.S7_DIM)
    print("node_shape: 31x3")
    print("graph_global_dim:", v4.GRAPH_GLOBAL_DIM)
    print("pair_dim:", v4.PAIR_DIM)
    print("inter_dim:", v4.INTER_DIM)
    print("future_loaded: no")


if __name__ == "__main__":
    main()
