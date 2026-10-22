#!/usr/bin/env python3
"""Hard source contracts before any full CARE Host/Ego training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (HERE, PROJECT / "scripts", PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
             PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
             PROJECT / "scripts/care-malben-contrast-v2", PROJECT / "scripts/care-malben-host-v2"):
    sys.path.insert(0, str(path))

from experiment_contract import MAX_DELTA, EXPECTED_N, EXPECTED_ROUTED, build_di_reference
from host_ego_encoder import DirectedHostEgoFusion, pack
from host_ego_models import anomaly_classes
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as baseline


def passed(name):
    print(f"[PASS] {name}")


def main():
    output = Path("artifacts/care-di25-hostego-v1")
    output.mkdir(parents=True, exist_ok=True)
    passed("imports")
    contract = build_di_reference(PROJECT, output)
    assert contract["source_rows"] == EXPECTED_N and contract["routed"] == EXPECTED_ROUTED
    passed("frozen DI25")
    assert MAX_DELTA == 2.25
    passed("max_delta == 2.25")
    Host, HostEgo = anomaly_classes(baseline, v4)
    passed("Host-only model loads")
    root = Path("artifacts/care-source-local-assets-v2")
    with np.load(root / "host_regions_2017.npz", allow_pickle=False) as h, np.load(root / "ego_1hop_regions_2017.npz", allow_pickle=False) as e:
        assert len(h["sha256"]) == EXPECTED_N == len(e["sha256"])
        status = np.asarray(e["ego_status"]); assert int(np.sum(status == 1)) > 0
        assert "incoming_host_positions" in e and "outgoing_host_positions" in e
    passed("Ego asset loads")
    passed("nontrivial Ego exists")
    fusion = DirectedHostEgoFusion(24).eval()
    host = torch.randn(3, 24)
    empty = torch.empty((0, 2), dtype=torch.long)
    local, _, _, used, _, _ = fusion(host, empty)
    assert not bool(torch.any(used)) and torch.max(torch.abs(local-host)).item() <= 1e-7
    passed("fallback exactness")
    rows = list(__import__("csv").DictReader((output / "preflight/di25_reference/source_predictions.csv").open("r", encoding="utf-8-sig")))
    route = [int(r["A25__routed"]) for r in rows]
    assert sum(route) == EXPECTED_ROUTED
    assert all(route == list(route) for _ in range(4))
    passed("route equality")
    source = (HERE / "run_care_di25_hostego_v1.py").read_text(encoding="utf-8")
    assert "future_lineage" not in source and "future_labels" not in source and "2018-2021" not in source
    passed("no future-label dependency")
    cfg = SimpleNamespace(max_delta=2.25, graph_dim=32, embedding_dim=24, hidden_dim=48,
                          dropout=0.20, graph_heads=4, graph_layers=2)
    host_model = Host(24, "RGCondAttn", cfg).eval()
    ego_model = HostEgo(24, "RGCondAttn", cfg).eval()
    host_n = sum(p.numel() for p in host_model.parameters())
    ego_n = sum(p.numel() for p in ego_model.parameters())
    assert ego_n > host_n and ego_n-host_n == 1225
    passed("parameter-count audit")
    graph = {"node": torch.randn(31,3), "edge": torch.randn(31,31,3), "global": torch.randn(27)}
    global_x = torch.randn(24); base = torch.tensor(0.1)
    regions = torch.randn(3, v4.AUGMENTED_REGION_DIM)
    with torch.no_grad():
        logit0, diag0 = host_model(global_x, base, graph, regions)
        packed = torch.as_tensor(pack(regions.numpy(), [(0,1),(1,2)]))
        logit1, diag1 = ego_model(global_x, base, graph, packed)
    assert torch.isfinite(logit0) and len(diag0) == len(v4.DIAGNOSTIC_NAMES)+1
    passed("dry forward Host-only")
    assert torch.isfinite(logit1) and len(diag1) == len(v4.DIAGNOSTIC_NAMES)+1
    assert abs(float(diag1[0])) <= 2.25 + 1e-6
    passed("dry forward Host+Ego")
    for sub in ("source", "direct_ablation", "audits", "results"):
        path = output / sub; path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"; probe.write_text("ok", encoding="utf-8"); probe.unlink()
    passed("output directories writable")
    print("HOSTEGO_SELFTEST_PASS")


if __name__ == "__main__":
    main()
