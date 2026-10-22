import importlib.util
from pathlib import Path

import numpy as np

from tasa_droid.p3.a3_graph_model import CachedGraph


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_entry_increment_ablation.py"
SPEC = importlib.util.spec_from_file_location("entry_increment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_role_stratified_layout_is_exact():
    layout = MODULE.entry_layout()
    assert int(layout["node_width"][0]) == 36
    assert int(layout["global_width"][0]) == 113
    assert int(layout["role_width"][0]) == 6
    assert int(layout["full_width"][0]) == 299
    assert len(layout["entry_presence"]) == 3
    assert len(layout["sensitive_role_support"]) == 124
    assert len(layout["sensitive_role_context"]) == 124
    assert len(layout["role_node_entry_overlap"]) == 31
    groups = [
        set(layout[name]) for name in (
            "entry_presence",
            "entry_global_structure",
            "sensitive_role_support",
            "sensitive_role_context",
            "role_node_entry_overlap",
        )
    ]
    assert sum(len(group) for group in groups) == 299
    assert len(set().union(*groups)) == 299


def test_entry_role_reachability_records_role_and_distance():
    x = np.zeros((3, 36), dtype=float)
    x[0, 3] = 1.0
    x[2, 4 + 7] = 1.0
    graph = CachedGraph(
        x=x,
        source=np.asarray([0, 1]),
        target=np.asarray([1, 2]),
        edge_features=np.zeros((2, 3)),
    )
    values = MODULE.entry_role_reachability_features(graph).reshape(31, 4)
    assert values[7, 0] == 1.0
    assert np.isclose(values[7, 1], 2.0 / 5.0)
    assert values[7, 2] > 0.0
    assert values[6].sum() == 0.0


def test_transition_names():
    assert MODULE.transition_name(1, 0, 1) == "CORRECTED"
    assert MODULE.transition_name(0, 0, 1) == "REGRESSED"
    assert MODULE.transition_name(1, 0, 0) == "STILL_WRONG"
    assert MODULE.transition_name(0, 0, 0) == "STILL_CORRECT"
