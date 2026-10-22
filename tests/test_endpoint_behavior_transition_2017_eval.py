import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_endpoint_behavior_transition_2017_eval.py"
)
for value in (str(SCRIPT.parent), str(SCRIPT.parents[1] / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)
SPEC = importlib.util.spec_from_file_location("endpoint_behavior_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EndpointBehaviorTransitionTest(unittest.TestCase):
    def test_name_free_two_hop_behavior_vector(self):
        payload = {
            "node_ids": [11, 12, 13],
            "node_role_ids": [[], [], [2]],
            "node_base_features": [
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0.5],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0.5],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1.0],
            ],
            "edge_index": [[0, 1], [1, 2]],
            "edge_features": [[0] * 7, [0] * 7],
            "edge_type_ids": [1, 1],
        }
        vector, audit = MODULE.endpoint_behavior_vector(payload)
        self.assertEqual(vector.shape, (MODULE.BEHAVIOR_DIMENSION,))
        # Both one-hop and two-hop callers are unmapped normal functions.
        self.assertGreater(vector[MODULE.NORMAL_ROLE], 0.0)
        self.assertGreater(
            vector[MODULE.ROLE_COUNT + MODULE.NORMAL_ROLE], 0.0
        )
        endpoint_hist_offset = len(MODULE.STAGES) * MODULE.ROLE_COUNT
        self.assertGreater(vector[endpoint_hist_offset + 2], 0.0)
        self.assertEqual(audit["sensitive_endpoint_count"], 1)
        self.assertEqual(audit["entry_unreachable_endpoint_count"], 1)

    def test_threshold_and_group_count_are_fixed(self):
        self.assertEqual(MODULE.THRESHOLD, 0.5)
        self.assertEqual(MODULE.ENTRY_GRAPH_WEIGHT, 0.5)
        self.assertEqual(len(MODULE.STAGES), 3)
        self.assertEqual(len(MODULE.PAIR_FAMILIES), 2)

    def test_collapses_normal_functions_between_sensitive_endpoints(self):
        payload = {
            "node_ids": [21, 22, 23, 24],
            "node_role_ids": [[4], [], [], [17]],
            "node_base_features": [
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1.0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0.5],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0.5],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1.0],
            ],
            "edge_index": [[0, 1, 2], [1, 2, 3]],
            "edge_features": [[0] * 7 for _ in range(3)],
            "edge_type_ids": [1, 1, 1],
        }
        vector, audit = MODULE.endpoint_behavior_vector(payload)
        pair_offset = (
            len(MODULE.STAGES) * MODULE.ROLE_COUNT
            + MODULE.ROLE_COUNT
            + MODULE.ROLE_COUNT * MODULE.ROLE_COUNT
        )
        pair_index = pair_offset + 4 * MODULE.ROLE_COUNT + 17
        self.assertGreater(vector[pair_index], 0.0)
        self.assertEqual(audit["collapsed_sensitive_path_count"], 1)
        self.assertFalse(audit["collapsed_path_state_cap_hit"])


if __name__ == "__main__":
    unittest.main()
