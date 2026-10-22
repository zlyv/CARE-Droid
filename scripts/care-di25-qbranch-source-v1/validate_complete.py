#!/usr/bin/env python3
import csv, sys
from pathlib import Path
p=Path(sys.argv[1])
root=p if p.is_dir() else p.parent
try:
    with (root/"source_predictions.csv").open("r",encoding="utf-8-sig",newline="") as h: rows=list(csv.DictReader(h))
    assert len(rows)==799 and len({r["sha256"].lower() for r in rows})==799
    with (root/"fit_audit.csv").open("r",encoding="utf-8-sig",newline="") as h: fit=list(csv.DictReader(h))
    assert fit and {int(float(r["parameter_n"])) for r in fit}=={33036}
    with (root/"source_summary.csv").open("r",encoding="utf-8-sig",newline="") as h: summary=list(csv.DictReader(h))
    assert summary
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
