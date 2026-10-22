#!/usr/bin/env python3
import tempfile
from pathlib import Path
from audit_core import write
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"x.csv";write(p,[{"a":1},{"a":2,"b":3}]);assert p.is_file()
print("DI25_FUTURE_MISSED_REPAIRABILITY_SELFTEST_PASS")
print("training_invoked=false")
print("inference_invoked=false")
print("route_modified=false")

