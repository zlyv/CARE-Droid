#!/usr/bin/env python3
import importlib.util
from pathlib import Path

path = Path(__file__).with_name("run_host_complementarity_audit.py")
spec = importlib.util.spec_from_file_location("audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(graph, critical, attended):
    return {
        "graph_score": graph,
        "critical_hostego_score": critical,
        "attended_hostego_score": attended,
    }


synthetic = [
    row(0.2, 0.8, 0.1),  # Graph missed, Critical-only rescue.
    row(0.3, 0.1, 0.9),  # Graph missed, Attended-only rescue.
    row(0.4, 0.8, 0.9),  # Graph missed, both/union rescue.
    row(0.1, 0.2, 0.3),  # Graph missed, no rescue.
    row(0.9, 0.8, 0.1),  # Graph+ / Critical+.
    row(0.9, 0.1, 0.1),  # Graph+ / Host-.
]

critical = audit.rescue_summary(synthetic, "critical")
attended = audit.rescue_summary(synthetic, "attended")
union = audit.rescue_summary(synthetic, "union")
assert critical["graph_missed_n"] == 4 and critical["rescued_n"] == 2
assert attended["graph_missed_n"] == 4 and attended["rescued_n"] == 2
assert union["graph_missed_n"] == 4 and union["rescued_n"] == 3

table = audit.contingency(synthetic, "critical")
assert table == {
    "graph_positive_host_positive": 1,
    "graph_positive_host_negative": 1,
    "graph_negative_host_positive": 2,
    "graph_negative_host_negative": 2,
}
assert audit.exact_mcnemar(0, 4) == 0.125
assert audit.exact_mcnemar(2, 2) == 1.0

safe_correct = [row(0.1, 0.8, 0.1), row(0.2, 0.1, 0.9), row(0.9, 0.9, 0.9)]
graph_negative = [item for item in safe_correct if not audit.flags(item)["graph"]]
assert len(graph_negative) == 2
assert sum(audit.flags(item)["critical"] for item in graph_negative) == 1
assert sum(audit.flags(item)["attended"] for item in graph_negative) == 1
assert sum(audit.flags(item)["union"] for item in graph_negative) == 2

low, high = audit.clopper_pearson(0, 10)
assert low == 0.0 and 0.30 < high < 0.31
low, high = audit.clopper_pearson(10, 10)
assert 0.69 < low < 0.70 and high == 1.0
print("HOST_COMPLEMENTARITY_SELFTEST_PASS")
