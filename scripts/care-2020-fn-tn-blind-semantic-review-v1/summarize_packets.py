#!/usr/bin/env python3
"""Read only the label-free v2 packets; never read the sealed key."""
import json
from collections import Counter
from pathlib import Path

PACKETS = Path("/home/boyuanjie/projects/tasa-droid/outputs/care-2020-fn-tn-dex-review-v2/blind_packets")

for path in sorted(PACKETS.glob("M*.json")):
    item = json.loads(path.read_text(encoding="utf-8"))
    print("\n===", item["case_id"], "match", item["match_quality_without_labels"], "===")
    for side in ("A", "B"):
        data = item["sides"][side]
        roles = Counter(data["role_call_counts"])
        print(side, "dex", data["dex_count"], "methods", data["method_count"],
              "sensitive_methods", data["sensitive_method_count"],
              "roles", roles.most_common(12))
        selected = data["example_sensitive_methods"][:6]
        for row in selected:
            role_set = sorted({s["role"] for s in row["sensitive_calls"]})
            refs = [s["reference"] for s in row["sensitive_calls"][:3]]
            print(" ", row["method"][:130], "calls", row["sensitive_call_total"],
                  "roles", role_set, "refs", refs)
