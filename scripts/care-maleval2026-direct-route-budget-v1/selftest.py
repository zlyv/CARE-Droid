#!/usr/bin/env python3
import numpy as np
from common import BUDGETS, metrics, top25, top_fraction
from run_stage2 import validate_host_alignment

sha = np.asarray([f"{i:064x}" for i in range(20)])
route = top25(np.arange(20), sha)
assert int(route.sum()) == 5
m = metrics([0, 0, 1, 1], [.1, .9, .2, .8])
assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (1, 1, 1, 1)
benign_only = metrics([0, 0], [.1, .9])
assert np.isnan(benign_only["BA"]) and np.isnan(benign_only["AUC"])
assert benign_only["specificity"] == .5 and np.isnan(benign_only["sensitivity"])
base = np.zeros((2, 74), dtype=np.float32)
base[0, 0] = 1.0
base[1, 1] = 1.0
payload_records = [
    {"host_node": 3, "presence": np.eye(31, dtype=np.float32)[0]},
    {"host_node": 9, "presence": np.eye(31, dtype=np.float32)[1]},
]
model_records = [
    {"host_node": 3, "presence": np.eye(31, dtype=np.float32)[7]},
    {"host_node": 9, "presence": np.eye(31, dtype=np.float32)[4]},
]
assert validate_host_alignment(base, payload_records, model_records, "synthetic") is model_records
ids = np.asarray([f"{i:064x}" for i in range(255)])
score = np.linspace(0.0, 1.0, 255)
routes = {budget: top_fraction(score, ids, budget / 100.0) for budget in BUDGETS}
assert {budget: int(route.sum()) for budget, route in routes.items()} == {25: 64, 30: 76, 40: 102, 50: 128}
assert np.all(routes[25] <= routes[30]) and np.all(routes[30] <= routes[40]) and np.all(routes[40] <= routes[50])
print("MALEVAL2026_DIRECT_ROUTE_BUDGET_SELFTEST_PASS")

