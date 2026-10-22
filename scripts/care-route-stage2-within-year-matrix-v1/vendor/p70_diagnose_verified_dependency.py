#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    root = args.input

    summary_path = root / "result_centered_slice_summary.csv"
    units_path = root / "result_centered_behavior_units.jsonl"

    status = collections.Counter()
    reasons = collections.Counter()
    sink_roles = collections.Counter()
    argument_kinds = collections.Counter()
    pairs = collections.Counter()
    evidence = collections.Counter()

    apk_count = 0
    ok_apks = 0
    total_units = 0

    with summary_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        summaries = list(csv.DictReader(f))

    apk_count = len(summaries)
    ok_apks = sum(
        row.get("status") == "OK"
        for row in summaries
    )

    with units_path.open(
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            wrapper = json.loads(line)

            for unit in wrapper.get("units", []):
                total_units += 1

                status[
                    unit.get(
                        "dependency_status",
                        "<MISSING>",
                    )
                ] += 1

                sink_roles[
                    unit.get(
                        "sink_role",
                        "<MISSING>",
                    )
                ] += 1

                argument_kinds[
                    unit.get(
                        "sink_argument_kind",
                        "<MISSING>",
                    )
                ] += 1

                reason = unit.get(
                    "unresolved_reason",
                    "",
                )

                if reason:
                    reasons[reason] += 1

                sink = unit.get(
                    "sink_role",
                    "",
                )

                for source in (
                    unit.get("source_roles", [])
                    or []
                ):
                    pairs[
                        (source, sink)
                    ] += 1

                evidence.update(
                    unit.get("evidence", [])
                    or []
                )

    print("========================================")
    print("P70 VERIFIED DEPENDENCY DIAGNOSTIC")
    print("========================================")
    print("APK count       :", apk_count)
    print("OK APKs         :", ok_apks)
    print("Total sink units:", total_units)

    print("\n===== STATUS =====")
    for key, value in status.most_common():
        ratio = value / max(1, total_units)
        print(
            f"{key:55s}"
            f"{value:8d}"
            f"  {ratio:8.3%}"
        )

    print("\n===== SINK ROLES =====")
    for key, value in sink_roles.most_common():
        print(
            f"{key:55s} {value:8d}"
        )

    print("\n===== ARGUMENT KIND =====")
    for key, value in argument_kinds.most_common():
        print(
            f"{key:55s} {value:8d}"
        )

    print("\n===== UNRESOLVED REASONS =====")

    unresolved_total = sum(reasons.values())

    for key, value in reasons.most_common():
        ratio = value / max(1, unresolved_total)

        print(
            f"{key:65s}"
            f"{value:8d}"
            f"  {ratio:8.3%}"
        )

    print("\n===== VERIFIED SOURCE -> SINK =====")

    if not pairs:
        print("(none)")
    else:
        for (source, sink), value in pairs.most_common():
            print(
                f"{source:35s}"
                f" -> "
                f"{sink:42s}"
                f"{value:8d}"
            )

    print("\n===== EVIDENCE =====")
    for key, value in evidence.most_common():
        print(
            f"{key:55s} {value:8d}"
        )


if __name__ == "__main__":
    main()
