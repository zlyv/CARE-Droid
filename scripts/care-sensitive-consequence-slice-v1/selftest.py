#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import stable_id
from effect_slice import extract, extract_variant


def method(signature, tags=()):
    cls, rest = signature.split("->", 1)
    return {"class": cls, "name": rest.split("(", 1)[0], "descriptor": "()v", "signature": signature,
            "node_id": stable_id(signature), "code_present": True, "external_tags": list(tags)}


def main() -> None:
    methods = [method("lapp/a;->host()v"), method("lapp/a;->wrapper()v"),
               method("lapp/net;->send()v", ("NETWORK",)), method("lapp/x;->unrelated()v")]
    graph = {"sha256": "0" * 64, "methods": methods, "edges": [[0, 1, 0, 1], [1, 2, 0, 2]],
             "stats": {"method_count": 4, "edge_count": 2}}
    hosts = [{"node_id": methods[0]["node_id"], "roles": [3]}]
    real, rows, audit = extract_variant(graph, hosts, np.asarray([1.0]), 0, graph["sha256"], "REAL")
    assert rows[0]["groups"]["TRANSFER"]["distance"] == 2
    assert audit["slice_node_count"] == 3 and abs(real["slice_reduction"] - .25) < 1e-9
    local, _, _ = extract_variant(graph, hosts, np.asarray([1.0]), 0, graph["sha256"], "LOCAL_ONLY")
    assert local["attn_TRANSFER_reach"] == 0
    empty_payload = {"node_ids": [], "node_role_ids": [], "node_base_features": [], "edge_index": [[], []]}
    empty = extract(graph, empty_payload, {"attention": np.asarray([1.0]), "critical": 0})
    assert empty["audit"]["host_count"] == 0 and empty["audit"]["empty_host_payload"] == 1
    assert empty["variants"]["REAL"]["attn_any_effect"] == 0
    assert stable_id(" LAPP/A; -> host()V ") == stable_id("lapp/a;->host()v")
    print("SCS_SELFTEST_PASS entry_free=true shortest_slice=true empty_host_padding=true matched_nulls=true", flush=True)


if __name__ == "__main__":
    main()
