#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from common import OUTPUT, assets, dump_json, file_record, load_attention, parse_formal, read_endpoint_rows


def main() -> None:
    found = assets(); rows = parse_formal(found["formal"]); wanted = {row["sha256"] for row in rows}
    if [sum(row["fold"] == fold for row in rows) for fold in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fixed fold sizes changed")
    if sum(row["routed"] for row in rows) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: DI25 route budget changed")
    missing_graphs = [sid for sid in sorted(wanted) if not (found["graphs"] / f"{sid}.json.gz").is_file()]
    if missing_graphs:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: CMW graph cache coverage {799-len(missing_graphs)}/799 missing_first={missing_graphs[:10]}")
    endpoints = read_endpoint_rows(found["endpoint"], wanted); attention = load_attention(found["attention"])
    if set(endpoints) != wanted or not wanted.issubset(attention):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint={len(endpoints)}/799 attention={len(wanted & set(attention))}/799")
    manifest = [file_record(found[name], name) for name in ("formal", "endpoint", "attention")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": manifest})
    protocol = {"name": "CARE-Droid Sensitive Consequence Slice v1", "source_year": 2017, "source_rows": 799,
        "future_rows_loaded": 0, "stage1": "P31-RF", "router": "RF-anchored MeanOpposition DI25 Top25%",
        "localization": "frozen A0 Role C43-RQ attention", "max_delta": 2.25,
        "research_stage": "label-free coverage then side-conditional direction sufficiency; no end-to-end correction",
        "target_representation": "Host-to-coarse-effect shortest-path projection, depth<=4, no entry/root requirement",
        "effect_groups": ["TRANSFER", "PERSIST", "EXECUTE", "VISIBLE", "SENSITIVE_ACTION"],
        "arms": ["REAL", "LOCAL_ONLY", "ONE_HOP", "EFFECT_SHUFFLE", "EDGE_SHUFFLE", "COUNT_ONLY", "PATH_ONLY"],
        "model": "fixed StandardScaler + balanced L2 LogisticRegression C=0.25",
        "selection": "2017 fixed 5-fold OOF only", "future_policy": "not loaded and not authorized by this source package",
        "boundary": "reachability is not data flow; external API identity is used only for frozen coarse effect mapping"}
    dump_json(OUTPUT / "protocol.json", protocol)
    digest = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    (OUTPUT / "protocol_hash.txt").write_text(digest + "\n", encoding="utf-8")
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799, "route": 200,
        "graph_rows": 799, "endpoint_rows": len(endpoints), "attention_rows": len(wanted & set(attention)),
        "graph_dir": str(found["graphs"]), "future_rows_loaded": 0})
    print("SCS_PREFLIGHT_PASS source_rows=799 route=200 graph_rows=799 future_rows_loaded=0", flush=True)


if __name__ == "__main__":
    main()
