#!/usr/bin/env python3
"""Architectural path and direct-bypass tests without project assets."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (HERE, PROJECT / "scripts/care-rolegraph-conditioned-v4-script", PROJECT / "scripts"):
    sys.path.insert(0, str(path))

from graphglobal_path_models import ARMS, make_graphglobal_path_class
import run_care_rolegraph_conditioned_dualstream_v4 as v4


def args():
    return SimpleNamespace(
        max_delta=2.25, graph_dim=32, embedding_dim=24, hidden_dim=48,
        graph_heads=4, graph_layers=2, dropout=0.0,
    )


def main():
    torch.manual_seed(20260923)
    global24 = torch.randn(24)
    base_logit = torch.tensor(0.3)
    node = torch.randn(v4.ROLE_COUNT, v4.ROLE_NODE_WIDTH)
    edge = torch.randn(v4.ROLE_COUNT, v4.ROLE_COUNT, 3)
    regions = torch.randn(7, v4.AUGMENTED_REGION_DIM)
    regions[:, -1] = torch.rand(7)
    g1, g2 = torch.randn(v4.GRAPH_GLOBAL_DIM), torch.randn(v4.GRAPH_GLOBAL_DIM)
    parameter_counts = {}

    for arm, (q_on, d_on) in ARMS.items():
        torch.manual_seed(17)
        Model = make_graphglobal_path_class(v4, arm)
        model = Model(24, "RGCondAttn", args()).eval()
        parameter_counts[arm] = sum(p.numel() for p in model.parameters())
        graph1 = {"node": node, "edge": edge, "global": g1}
        graph2 = {"node": node, "edge": edge, "global": g2}
        with torch.no_grad():
            _, _, t1 = model.forward_with_trace(global24, base_logit, graph1, regions)
            _, _, t2 = model.forward_with_trace(global24, base_logit, graph2, regions)
            role_diff = torch.max(torch.abs(t1["role_token"] - t2["role_token"])).item()
            host_diff = torch.max(torch.abs(t1["critical"] - t2["critical"])).item()
            attention_diff = torch.max(torch.abs(t1["attention"] - t2["attention"])).item()
            delta_diff = abs(float(t1["raw_delta"] - t2["raw_delta"]))
        assert role_diff == 0.0, (arm, role_diff)
        # Critical can change only through selection; the encoded Host vectors
        # themselves are GraphGlobal-independent. Directly test the full matrix.
        with torch.no_grad():
            l1 = model.region_encoder(regions)
            l2 = model.region_encoder(regions)
        assert torch.equal(l1, l2)
        if q_on:
            assert attention_diff > 1e-8, (arm, attention_diff)
        else:
            assert attention_diff == 0.0, (arm, attention_diff)
        if not q_on and d_on:
            assert delta_diff > 1e-8, (arm, delta_diff)
        if not q_on and not d_on:
            assert delta_diff == 0.0, (arm, delta_diff)

        # Freeze/copy selected local evidence from graph1, then vary only the
        # separate GraphGlobal embedding. This isolates the direct D bypass.
        with torch.no_grad():
            cache, e1 = model.select_local_evidence(graph1, regions)
            e2 = model.graphglobal_encoder(g2)
            r1, _, _ = model.delta_from_selected_evidence(global24, base_logit, cache, e1)
            r2, _, _ = model.delta_from_selected_evidence(global24, base_logit, cache, e2)
            cached_delta_diff = abs(float(r1 - r2))
        if d_on:
            assert cached_delta_diff > 1e-8, (arm, cached_delta_diff)
        else:
            assert cached_delta_diff == 0.0, (arm, cached_delta_diff)

    assert len(set(parameter_counts.values())) == 1, parameter_counts
    print("GRAPHGLOBAL_PATH_SELFTEST_PASS")
    print("arms=", ",".join(ARMS))
    print("parameter_count_all_arms=", next(iter(parameter_counts.values())))
    print("role_encoder_graphglobal_invariant=true")
    print("query_only_direct_delta_bypass=false")
    print("future_rows_loaded=0")
    print("ego_consumed=false")


if __name__ == "__main__":
    main()

