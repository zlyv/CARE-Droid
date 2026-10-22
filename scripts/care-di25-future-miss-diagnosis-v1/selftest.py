#!/usr/bin/env python3
import csv
import tempfile
from pathlib import Path
import numpy as np
from diagnosis_core import exact_year_top, reason, uncertainty_scores
from run_diagnosis import write

p=np.array([[.9,.9,.9,.9],[.9,.1,.8,.2],[.48,.52,.49,.51],[.1,.1,.1,.1]])
b=np.array([.9,.9,.52,.1])
s=uncertainty_scores(b,p)
assert s["MeanOpposition_DI25"][0] < s["MeanOpposition_DI25"][1]
r,c=exact_year_top(np.arange(8),np.array([2020]*4+[2021]*4),np.array([str(i) for i in range(8)]))
assert r.sum()==2 and c[2020]==3 and c[2021]==7
assert reason(.1,0,.4).startswith("A_")
assert reason(.55,3,.6).startswith("D_")
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/"heterogeneous.csv"
    write(path,[{"year":2020,"reason_A":1},{"year":2021,"reason_B":2}])
    with path.open(encoding="utf-8",newline="") as h:rows=list(csv.DictReader(h))
    assert set(rows[0])=={"year","reason_A","reason_B"} and rows[1]["reason_B"]=="2"
print("DI25_FUTURE_MISS_DIAGNOSIS_SELFTEST_PASS")
print("training_invoked=false")
print("route_modified=false")

