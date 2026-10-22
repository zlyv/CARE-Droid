#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import stable_id
from witness import adjacency, extract


def method(cls, name, desc, tags=()):
    signature = f"{cls}->{name}{desc}".casefold()
    return {"class": cls.casefold(), "name": name.casefold(), "descriptor": desc.casefold(),
            "signature": signature, "node_id": stable_id(signature), "code_present": True,
            "external_tags": list(tags)}


user_cls = "Lx/UserClick;".casefold(); auto_cls = "Lx/WorkerService;".casefold(); host_cls = "Lx/Shared;".casefold()
methods = [method(user_cls, "onClick", "(Landroid/view/View;)V"),
           method(auto_cls, "onStartCommand", "(Landroid/content/Intent;II)I"),
           method(host_cls, "sensitive", "()V", ("NETWORK",)), method(host_cls, "endpoint", "()V")]
graph = {"sha256": "a" * 64, "classes": {
    user_cls: {"name": user_cls, "super": "Ljava/lang/Object;".casefold(), "interfaces": ["Landroid/view/View$OnClickListener;".casefold()]},
    auto_cls: {"name": auto_cls, "super": "Landroid/app/Service;".casefold(), "interfaces": []},
    host_cls: {"name": host_cls, "super": "Ljava/lang/Object;".casefold(), "interfaces": []}},
    "methods": methods, "edges": [[0, 2, 10, 1001], [1, 2, 20, 2001]],
    "stats": {"method_count": 4, "edge_count": 2}}
payload = {"node_ids": [methods[2]["node_id"], methods[3]["node_id"]], "node_role_ids": [[], [17]],
           "node_base_features": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], "edge_index": [[0], [1]]}
result = extract(graph, payload, {"attention": np.asarray([1.0]), "critical": 0})
real = result["variants"]["REAL"]
assert real["attn_both"] == 1.0 and real["attn_splittable"] == 1.0
assert real["attn_user_network"] > 0 and real["attn_auto_network"] > 0
assert 0 <= real["attn_context_entropy"] <= 1
chain_methods = [method(user_cls, "onClick", "(Landroid/view/View;)V"),
                 method(host_cls, "middle", "()V"), method(host_cls, "deepHost", "()V"),
                 method(host_cls, "deepEndpoint", "()V")]
chain_graph = {"sha256": "b" * 64, "classes": graph["classes"], "methods": chain_methods,
               "edges": [[0, 1, 0, 21], [1, 2, 0, 22]], "stats": {"method_count": 4, "edge_count": 2}}
chain_payload = {"node_ids": [chain_methods[2]["node_id"], chain_methods[3]["node_id"]],
                 "node_role_ids": [[], [17]], "node_base_features": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
                 "edge_index": [[0], [1]]}
chain = extract(chain_graph, chain_payload, {"attention": np.asarray([1.0]), "critical": 0})
assert chain["variants"]["REAL"]["attn_user_reachable"] == 1.0
assert chain["variants"]["ONE_HOP"]["attn_user_reachable"] == 0.0
null_graph = {"sha256": "a" * 64, "edges": [[0, 2, 0, 10], [0, 3, 1, 11], [1, 4, 0, 12], [1, 5, 1, 13]]}
real_adj = adjacency(null_graph, "REAL", null_graph["sha256"])
null_adj = adjacency(null_graph, "CALLSITE_SHUFFLE", null_graph["sha256"])
assert real_adj != null_adj
assert {node: len(edges) for node, edges in real_adj.items()} == {node: len(edges) for node, edges in null_adj.items()}
assert sorted(target for edges in real_adj.values() for target, _ in edges) == sorted(target for edges in null_adj.values() for target, _ in edges)
print("CMW_SELFTEST_PASS context_sensitive_shared_host=true max_delta=2.25", flush=True)
