#!/usr/bin/env python3
"""Label-free aggregate matching-quality summary; excludes sample identities."""
import csv
import json
import statistics
from pathlib import Path

out = Path("/home/boyuanjie/projects/tasa-droid/outputs/care-2020-fn-tn-dex-review-v2")
with (out / "matched_pairs_restricted.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
result = {"eligible_pairs": len(rows), "pilot_pairs": 8}
for name in ("stage1_probability_difference", "host_count_ratio", "p31_standardized_cosine_distance"):
    for scope, subset in (("all_eligible", rows), ("pilot", rows[:8])):
        values = [float(r[name]) for r in subset]
        result[f"{scope}_{name}"] = {"median": statistics.median(values), "max": max(values)}
(out / "match_quality_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
