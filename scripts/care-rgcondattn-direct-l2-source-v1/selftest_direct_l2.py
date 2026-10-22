#!/usr/bin/env python3
"""Verify that Direct-L2 is invariant to first-layer inputs."""

from types import SimpleNamespace

import torch

from run_care_rgcondattn_direct_l2_source_v1 import (
    DIAGNOSTIC_NAMES,
    DirectRGCondAttn,
    v4,
)


def main() -> None:
    args = SimpleNamespace(
        graph_dim=32,
        embedding_dim=24,
        hidden_dim=48,
        dropout=0.20,
        graph_heads=4,
        graph_layers=2,
    )
    torch.manual_seed(7)
    model = DirectRGCondAttn(24, "RGCondAttn", args).eval()
    graph = {
        "node": torch.randn(v4.ROLE_COUNT, v4.ROLE_NODE_WIDTH),
        "edge": torch.randn(v4.ROLE_COUNT, v4.ROLE_COUNT, 3),
        "global": torch.randn(v4.GRAPH_GLOBAL_DIM),
    }
    regions = torch.randn(5, v4.AUGMENTED_REGION_DIM)
    regions[:, -1] = torch.rand(5)
    with torch.no_grad():
        first, diagnostic = model(
            torch.zeros(24), torch.tensor(-9.0), graph, regions
        )
        second, _ = model(
            torch.full((24,), 999.0), torch.tensor(9.0), graph, regions
        )
    if not torch.equal(first, second):
        raise RuntimeError("Direct-L2 unexpectedly depends on first-layer inputs")
    if diagnostic.numel() != len(DIAGNOSTIC_NAMES):
        raise RuntimeError("diagnostic width mismatch")
    print("DIRECT_L2_SELFTEST_OK")
    print("first_layer_input_invariant: True")
    print("diagnostic_width:", diagnostic.numel())
    print("parameters:", sum(value.numel() for value in model.parameters()))


if __name__ == "__main__":
    main()
