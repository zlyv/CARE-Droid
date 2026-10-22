#!/usr/bin/env python3
import tempfile
from pathlib import Path

import numpy as np

from common import binary_auc, dump, read_csv, write_csv

assert abs(binary_auc([0, 1], [0.1, 0.9]) - 1.0) < 1e-12
assert abs(binary_auc([0, 1], [0.9, 0.1]) - 0.0) < 1e-12
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "rows.csv"
    write_csv(path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert len(read_csv(path)) == 2
    dump(Path(directory) / "value.json", {"ok": True})
print("CARE_INFORMATION_AUDIT_SELFTEST_PASS")

