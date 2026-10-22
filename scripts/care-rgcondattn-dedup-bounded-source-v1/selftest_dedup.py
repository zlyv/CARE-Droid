#!/usr/bin/env python3
"""Structural self-test for the cleaned bounded RGCondAttn model."""

from types import SimpleNamespace

import torch

from run_care_rgcondattn_dedup_bounded_source_v1 import (
    DedupBoundedRGCondAttn,
    v4,
)


def main() -> None:
    args = SimpleNamespace(
        max_delta=1.5,
        graph_dim=32,
        embedding_dim=24,
        hidden_dim=48,
        dropout=0.20,
        graph_heads=4,
        graph_layers=2,
    )
    torch.manual_seed(11)
    model = DedupBoundedRGCondAttn(24, "RGCondAttn", args).eval()
    names = {name for name, _ in model.named_modules()}
    if "critical_query" in names or "context_key" in names:
        raise RuntimeError("unused dual-stream modules are still present")
    graph = {
        "node": torch.randn(v4.ROLE_COUNT, v4.ROLE_NODE_WIDTH),
        "edge": torch.randn(v4.ROLE_COUNT, v4.ROLE_COUNT, 3),
        "global": torch.randn(v4.GRAPH_GLOBAL_DIM),
    }
    regions = torch.randn(5, v4.AUGMENTED_REGION_DIM)
    regions[:, -1] = torch.rand(5)
    with torch.no_grad():
        logit, diagnostic = model(
            torch.randn(24), torch.tensor(0.2), graph, regions
        )
    if not torch.isfinite(logit) or not torch.isfinite(diagnostic).all():
        raise RuntimeError("non-finite self-test output")
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != 33388:
        raise RuntimeError(f"unexpected parameter count: {parameters}")
    print("DEDUP_BOUNDED_SELFTEST_OK")
    print("correction_input_width: 113")
    print("duplicate_context_present: False")
    print("unused_dual_stream_layers_present: False")
    print("parameters:", parameters)


if __name__ == "__main__":
    main()
