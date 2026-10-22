#!/usr/bin/env python3
from __future__ import annotations

from transforms import apply, apply_record, endpoint_nodes


def payload():
    return {"node_ids": [10, 11, 12, 13], "node_role_ids": [[], [], [2], []],
            "node_base_features": [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,1], [0,0,0,0,0]],
            "edge_index": [[0,1,3], [2,3,1]], "edge_features": [[1]*7,[2]*7,[3]*7], "edge_type_ids": ["a","b","c"]}


def signature(value):
    ids = value["node_ids"]; endpoints = endpoint_nodes(value)
    edges = {(ids[s], ids[t]) for s,t in zip(*value["edge_index"])}
    return set(ids), endpoints, edges


def main():
    p = payload(); q, audit = apply(p, "a"*64, "T0_ORDER", 0)
    assert len(q["node_ids"]) == 4 and len(q["edge_index"][0]) == 3 and audit["applicable"]
    w, a = apply(p, "b"*64, "T1_WRAPPER_SENSITIVE", 0)
    assert len(w["node_ids"]) == 5 and len(w["edge_index"][0]) == 4 and a["replaced_edges"] == 1
    d, a = apply(p, "c"*64, "T2_DEAD_SUBGRAPH", 0); assert len(d["node_ids"]) >= 6
    n, a = apply(p, "d"*64, "T3_DEAD_EDGE_NEAR_HOST", 0); assert len(n["node_ids"]) == 5
    f, a = apply(p, "e"*64, "N1_WRAPPER_FAR", 0); assert a["applicable"] and len(f["node_ids"]) > 4
    empty={"node_ids":[],"node_role_ids":[],"node_base_features":[]}
    for name in ("T0_ORDER","T1_WRAPPER_SENSITIVE","T2_DEAD_SUBGRAPH","T3_DEAD_EDGE_NEAR_HOST","N1_WRAPPER_FAR"):
        value,audit=apply(empty,"f"*64,name,0)
        assert value["edge_index"]==[[],[]] and not audit["applicable"]
    nested, audit = apply_record({"sha256": "g"*64, "status": "OK", "payload": p}, "g"*64, "T2_DEAD_SUBGRAPH", 0)
    assert nested["sha256"] == "g"*64 and nested["status"] == "OK"
    assert audit["record_container"] == "payload" and audit["applicable"]
    assert len(nested["payload"]["node_ids"]) > len(p["node_ids"])
    print("C43_EQUIVALENCE_PROXY_SELFTEST_PASS")


if __name__ == "__main__": main()
