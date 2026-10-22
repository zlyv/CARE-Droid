#!/usr/bin/env python3
import importlib.util
from pathlib import Path

path = Path(__file__).with_name("run_failure_block_audit.py")
spec = importlib.util.spec_from_file_location("audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

assert audit.BLOCKS == (
    "B0_graph_token32",
    "B1_critical_hostego24",
    "B2_attended_hostego24",
    "B3_attention_anomaly_summary8",
)
assert audit.GROUP_ORDER[4] == "A_WRONG_DIRECTION"
assert audit.ITERATIONS == 10000
result = audit.bootstrap_mean_difference([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
assert result["mean_difference"] < 0
assert result["bootstrap_ci_high"] < 0
assert audit.standardized_difference([1, 2, 3], [1, 2, 3]) == 0.0
assert audit.binary_auc([1, 1, 0, 0], [0.8, 0.9, 0.1, 0.2]) == 1.0
assert audit.binary_auc([1, 0], [0.5, 0.5]) == 0.5
print("FAILURE_BLOCK_AUDIT_SELFTEST_PASS")
