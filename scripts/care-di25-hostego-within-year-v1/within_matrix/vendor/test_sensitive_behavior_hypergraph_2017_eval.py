import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sensitive_behavior_hypergraph_2017_eval.py"
SPEC = importlib.util.spec_from_file_location("sensitive_behavior_hypergraph", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _synthetic_payload():
    # host 0 directly invokes role-1 and role-2 endpoints and reaches host 1
    # through ordinary method 2.  Host 1 directly invokes a role-3 endpoint.
    node_count = 7
    base = [[0.0] * 15 for _ in range(node_count)]
    for endpoint in (4, 5, 6):
        base[endpoint][4] = 1.0
    for node in range(node_count):
        base[node][5] = 1.0
    roles = [[], [], [], [], [1], [2], [3]]
    edges = [(0, 4), (0, 5), (0, 2), (2, 1), (1, 6)]
    return {
        "node_ids": [f"node-{index}" for index in range(node_count)],
        "node_role_ids": roles,
        "node_base_features": base,
        "edge_index": [
            [source for source, _ in edges],
            [target for _, target in edges],
        ],
    }


def test_hypergraph_collapses_ordinary_methods_and_keeps_high_order_roles():
    vector, audit = MODULE.sensitive_behavior_hypergraph_vector(_synthetic_payload())
    assert vector.shape == (MODULE.HYPERGRAPH_DIMENSION,)
    assert audit["selected_sensitive_endpoint_count"] == 3
    assert audit["behavior_host_method_count"] == 2
    assert audit["shared_caller_hyperedge_count"] == 1
    assert audit["compressed_reachability_hyperedge_count"] == 1
    assert audit["compressed_reachability_high_order_count"] == 1
    assert not audit["path_state_cap_hit"]
