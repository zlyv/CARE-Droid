import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_method_spatial_organization_2017_eval.py"
SPEC = importlib.util.spec_from_file_location("method_spatial", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payload(host_roles):
    # Nodes 0..H-1 are callers; remaining nodes are endpoint role vertices.
    host_count = len(host_roles)
    endpoint_roles = sorted({role for roles in host_roles for role in roles})
    role_to_node = {
        role: host_count + index for index, role in enumerate(endpoint_roles)
    }
    node_count = host_count + len(endpoint_roles)
    base = np.zeros((node_count, 15), dtype=float)
    base[:host_count, 5] = 1.0
    base[host_count:, 4] = 1.0
    base[host_count:, 7] = 1.0
    role_rows = [[] for _ in range(node_count)]
    for role, node in role_to_node.items():
        role_rows[node] = [role]
    source, target = [], []
    for host, roles in enumerate(host_roles):
        for role in roles:
            source.append(host)
            target.append(role_to_node[role])
    return {
        "node_ids": [str(index) for index in range(node_count)],
        "node_base_features": base.tolist(),
        "node_role_ids": role_rows,
        "edge_index": [source, target],
    }


def test_fixed_dimensions_and_empty_payload():
    vector, audit = MODULE.method_spatial_organization_vector(payload([]))
    assert vector.shape == (MODULE.ORGANIZATION_DIMENSION,)
    assert np.isfinite(vector).all()
    assert audit["observed_behavior_host_method_count"] == 0


def test_same_role_inventory_different_method_organization_differs():
    concentrated = payload([{0, 1}, {0, 1}])
    dispersed = payload([{0}, {1}])
    left, left_audit = MODULE.method_spatial_organization_vector(concentrated)
    right, right_audit = MODULE.method_spatial_organization_vector(dispersed)
    assert not np.allclose(left, right)
    assert left_audit["multi_role_host_method_count"] == 2
    assert right_audit["multi_role_host_method_count"] == 0


def test_high_order_colocation_is_recorded():
    vector, audit = MODULE.method_spatial_organization_vector(
        payload([{0, 1, 2}, {0}, {1}, {2}])
    )
    assert audit["high_order_host_method_count"] == 1
    assert audit["triple_pattern_count"] == 1
    triple_start = MODULE.DISTRIBUTION_DIMENSION + MODULE.PAIR_DIMENSION
    assert vector[triple_start:].sum() > 0.0
