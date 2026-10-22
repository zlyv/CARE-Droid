#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import stable_id
from fullgraph import classify_host, multi_source_distances, parse_method_reference, shortest_path
from run_pipeline import analyze_graph


def main() -> None:
    reference = parse_method_reference("v0, v1, Lcom/example/A;->work(Ljava/lang/String;)V")
    assert reference == "lcom/example/a;->work(ljava/lang/string;)v"
    graph = {0: [1], 1: [2], 2: [3], 4: [2]}
    user = multi_source_distances(graph, {0})
    autonomous = multi_source_distances(graph, {4})
    assert shortest_path(graph, {0}, 3) == [0, 1, 2, 3]
    assert classify_host(1, user, {}, {})[0] == "USER_ONLY"
    assert classify_host(2, user, autonomous, {})[0] == "BOTH"
    assert classify_host(99, user, autonomous, {})[0] == "UNREACHABLE"
    root_signature = "ltest/click;->onclick(landroid/view/view;)v"
    ordinary_signature = "ltest/a;->helper()v"
    host_signature = "ltest/a;->host()v"
    endpoint_signature = "landroid/location/locationmanager;->getlastknownlocation(ljava/lang/string;)landroid/location/location;"
    synthetic = {
        "classes": {
            "ltest/click;": {"name": "ltest/click;", "super": "ljava/lang/object;", "interfaces": ["landroid/view/view$onclicklistener;"]},
            "ltest/a;": {"name": "ltest/a;", "super": "ljava/lang/object;", "interfaces": []},
        },
        "methods": [
            {"class": "ltest/click;", "name": "onclick", "descriptor": "(landroid/view/view;)v", "signature": root_signature, "node_id": stable_id(root_signature), "code_present": True},
            {"class": "ltest/a;", "name": "helper", "descriptor": "()v", "signature": ordinary_signature, "node_id": stable_id(ordinary_signature), "code_present": True},
            {"class": "ltest/a;", "name": "host", "descriptor": "()v", "signature": host_signature, "node_id": stable_id(host_signature), "code_present": True},
        ],
        "edges": [[0, 1], [1, 2]],
        "stats": {"invoke_parse_rate": 1.0, "isolated_method_count": 0},
    }
    endpoint = {
        "node_ids": [stable_id(host_signature), stable_id(endpoint_signature)],
        "node_role_ids": [[], [13]],
        "node_base_features": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
        "edge_index": [[0], [1]],
    }
    metric, host_rows, examples = analyze_graph("0" * 64, 1, synthetic, endpoint, {"attention": np.asarray([1.0]), "critical": 0})
    assert host_rows[0]["trigger_type"] == "USER_ONLY"
    assert metric["host_exact_alignment"] == 1.0
    assert metric["critical_directional_assignment"] == 1
    print("CARE_FULLGRAPH_TRCBE_SELFTEST_PASS tests=invoke_reference,direction,multihop,classification")


if __name__ == "__main__":
    main()
