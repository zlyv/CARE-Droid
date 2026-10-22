#!/usr/bin/env python3
"""Summarize the four completed 2017 OOF arms and enforce route identity."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from graphglobal_path_models import ARMS

ROOT = Path("artifacts/care-di25-graphglobal-path-source-v1")


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    canonical = read(ROOT / "preflight/di25_reference/source_predictions.csv")
    canonical_route = {r["sha256"]: int(float(r["A25__routed"])) for r in canonical}
    metrics, integrity, parameters = [], [], {}
    for arm in ARMS:
        arm_root = ROOT / "arms" / arm
        summary = read(arm_root / "source_summary.csv")
        method = "DI25__Host__Anomaly__GraphGlobalPath__" + arm + "__Bounded"
        match = [r for r in summary if r["method"] == method]
        if len(match) != 1:
            raise RuntimeError(f"Missing unique summary row for {arm}")
        metrics.append({"arm": arm, **match[0]})
        predictions = read(arm_root / "source_predictions.csv")
        if len(predictions) != 799:
            raise RuntimeError(f"{arm}: rows={len(predictions)}")
        for row in predictions:
            sha = row["sha256"].lower()
            observed = int(float(row["A25__routed"]))
            expected = canonical_route.get(sha, -1)
            integrity.append({
                "sha256": sha, "arm": arm,
                "route_frozen_DI25": expected, "route_arm": observed,
                "different": int(observed != expected),
            })
        fit = read(arm_root / "fit_audit.csv")
        values = {int(float(r["parameter_n"])) for r in fit}
        if len(values) != 1:
            raise RuntimeError(f"{arm}: inconsistent parameter counts {values}")
        parameters[arm] = next(iter(values))
    if sum(int(r["different"]) for r in integrity):
        raise RuntimeError("BLOCKED_BY_CONTRACT: an arm did not use frozen DI25")
    if len(set(parameters.values())) != 1:
        raise RuntimeError("BLOCKED_BY_CONTRACT: parameter counts differ " + repr(parameters))

    write(ROOT / "results/metrics.csv", metrics)
    write(ROOT / "results/route_integrity.csv", integrity)
    write(ROOT / "results/parameter_count.csv", [
        {"arm": arm, "parameter_count": n} for arm, n in parameters.items()
    ])
    lines = [
        "CARE-Droid GraphGlobal path 2x2 source ablation v1", "",
        "Strict 2017 five-fold OOF; frozen DI25=200/799; Host-only; no future row; no Ego.",
        "All arms share the same instantiated parameter count and 113-D bounded head.",
        "Q controls direct GraphGlobal -> Host attention; D controls direct GraphGlobal -> delta.",
        "Global24 and Stage-I logit remain correction inputs in all arms.", "",
    ]
    for row in metrics:
        lines.append(
            f"{row['arm']}: BA={float(row['BA']):.6f} AUC={float(row['AUC']):.6f} "
            f"F1={float(row['F1']):.6f} fixed={int(float(row['fixed']))} "
            f"broken={int(float(row['broken']))} net={int(float(row['net']))}"
        )
    lines += ["", "Hard audits:", "route_difference_count=0",
              f"parameter_count_all_arms={next(iter(parameters.values()))}",
              "Historical M0 is an external reference, not a factorial arm: it injects GraphGlobal inside the RoleGraph encoder.",
              "Do not run a future diagnostic until a source-side rule is frozen."]
    (ROOT / "results/REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "results/summary.json").write_text(json.dumps({
        "source_rows": 799, "route_DI": 200, "route_difference_count": 0,
        "parameter_counts": parameters, "future_rows_loaded": 0,
        "ego_consumed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("GRAPHGLOBAL_PATH_SOURCE_SUMMARY_PASS")
    print((ROOT / "results/REPORT.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

