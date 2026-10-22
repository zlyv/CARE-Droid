#!/usr/bin/env python3
"""Reviewer-facing concise summaries, with no access to the sealed key."""
import json
import sys
from pathlib import Path

PACKETS = Path("/home/boyuanjie/projects/tasa-droid/outputs/care-2020-fn-tn-blind-semantic-review-v1/blind_packets")
for path in sorted(PACKETS.glob("M*.json")):
    if len(sys.argv) > 1 and path.stem not in sys.argv[1:]:
        continue
    item = json.loads(path.read_text(encoding="utf-8"))
    print("\n===", item["case_id"], "===")
    for side in ("A", "B"):
        data = item["sides"][side]
        print(side, "app_method_fraction", round(data["app_package_prefix_method_count"]/max(data["method_count"],1),3),
              "app_sensitive_fraction", round(data["app_package_prefix_role_mapped_invokes"]/max(data["all_role_mapped_invokes"],1),3),
              "app_sensitive_methods", data["app_package_prefix_sensitive_methods"],
              "non_known_sdk_sensitive_methods", data["non_known_sdk_sensitive_methods"])
        for row in data["focused_methods"][:10]:
            print(" ", "APP" if row["app_package_prefix"] else "OTHER",
                  "sdk?" if row["known_sdk_prefix_heuristic"] else "", row["method"][:120],
                  "roles", row["roles"], "branches", row["branch_count"])
