#!/usr/bin/env python3
"""Unit tests for conservative Host-method S1/S2/S3 extraction and privacy rules."""
from __future__ import annotations

import json
import unittest

from common import stable_id
from semantic_core import analyze_method, categorize_string


class SemanticTests(unittest.TestCase):
    def test_s1_categories_do_not_return_literal(self):
        secret = "https://private.example/path?q=secret"
        self.assertEqual(categorize_string(secret), "NETWORK_ENDPOINT")
        self.assertNotIn(secret, json.dumps(categorize_string(secret)))

    def test_direct_and_copy_flow(self):
        a = "Lx/A;->read()Ljava/lang/String;"
        b = "Lx/B;->send(Ljava/lang/String;)V"
        rows = [
            {"offset": 0, "name": "invoke-static", "reference": a, "registers": []},
            {"offset": 2, "name": "move-result-object", "registers": [0]},
            {"offset": 4, "name": "move-object", "registers": [1, 0]},
            {"offset": 6, "name": "invoke-static", "reference": b, "registers": [1]},
        ]
        result = analyze_method(rows, {str(stable_id(a)): [2], str(stable_id(b)): [8]})
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["dependencies"][0]["dependency_type"], "TRACKED_LOCAL_COPY_FLOW")

    def test_branch_is_not_data_flow(self):
        a = "Lx/A;->read()I"; b = "Lx/B;->send(I)V"
        rows = [{"offset": 0, "name": "invoke-static", "reference": a, "registers": []},
                {"offset": 2, "name": "move-result", "registers": [0]},
                {"offset": 4, "name": "if-eqz", "registers": [0], "target": 10},
                {"offset": 6, "name": "invoke-static", "reference": b, "registers": [0]}]
        result = analyze_method(rows, {str(stable_id(a)): [1], str(stable_id(b)): [3]})
        self.assertEqual(result["dependencies"], [])
        self.assertIn("UNKNOWN", result["events"][0]["s2_uses"])

    def test_multirole_endpoint_is_one_event(self):
        a = "Lx/A;->read()I"
        result = analyze_method([{"offset": 0, "name": "invoke-static", "reference": a, "registers": []}],
                                {str(stable_id(a)): [1, 4, 9]})
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["role_ids"], [1, 4, 9])

    def test_unknown_for_orphan_result(self):
        result = analyze_method([{"offset": 0, "name": "move-result-object", "registers": [0]}], {})
        self.assertIn("ORPHAN_MOVE_RESULT", result["unknown_reasons"])


if __name__ == "__main__":
    from common import OUTPUT, atomic_json
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SemanticTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(OUTPUT / "unit_test_results.json", {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "covers": ["finite S1 categories", "direct/copy S3", "CFG reachability excluded",
                   "multi-role invoke deduplication", "UNKNOWN preservation"],
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)

