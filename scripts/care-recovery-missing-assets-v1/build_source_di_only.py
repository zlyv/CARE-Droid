#!/usr/bin/env python3
"""Rebuild the frozen 2017 DI table from strict source OOF committee scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


MEMBERS = ("p_lr", "p_rf", "p_xgb", "p_svm")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_top_fraction(scores: np.ndarray, fraction: float) -> np.ndarray:
    n_select = int(np.ceil(len(scores) * fraction))
    order = np.argsort(-scores, kind="mergesort")
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:n_select]] = True
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_csv(args.input)
    required = {
        "sha256", "year", "label", "fold", "p_stage1",
        "p_frozen_bounded_expert", "p_bar", *MEMBERS,
    }
    if not rows or required - set(rows[0]):
        raise RuntimeError("source committee schema mismatch: " + repr(sorted(required - set(rows[0]) if rows else required)))
    ids = [row["sha256"].strip().lower() for row in rows]
    if len(rows) != 799 or len(set(ids)) != 799:
        raise RuntimeError(f"expected 799 unique source rows, got rows={len(rows)} unique={len(set(ids))}")
    if {int(row["year"]) for row in rows} != {2017}:
        raise RuntimeError("future/source scope violation")
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if sorted(int(np.sum(folds == fold)) for fold in sorted(set(folds))) != [159, 160, 160, 160, 160]:
        raise RuntimeError("fixed source folds changed")
    p_stage1 = np.asarray([float(row["p_stage1"]) for row in rows])
    members = np.column_stack([
        np.asarray([float(row[name]) for row in rows]) for name in MEMBERS
    ])
    p_bar = members.mean(axis=1)
    stored_bar = np.asarray([float(row["p_bar"]) for row in rows])
    if not np.allclose(p_bar, stored_bar, atol=1e-10, rtol=0.0):
        raise RuntimeError("stored committee mean changed")
    stage1_decision = p_stage1 >= 0.5
    support = np.where(stage1_decision, p_bar, 1.0 - p_bar)
    u_di = 1.0 - support
    route = np.zeros(len(rows), dtype=bool)
    per_fold = {}
    for fold in sorted(set(folds)):
        index = np.flatnonzero(folds == fold)
        selected = exact_top_fraction(u_di[index], args.budget)
        route[index] = selected
        per_fold[str(int(fold))] = int(np.sum(selected))
    expert = np.asarray([float(row["p_frozen_bounded_expert"]) for row in rows])
    final = np.where(route, expert, p_stage1)

    output_rows = []
    for i, row in enumerate(rows):
        item = dict(row)
        item.update({
            "committee_support_for_stage1": float(support[i]),
            "U_DI": float(u_di[i]),
            "support_lr": float(members[i, 0] if stage1_decision[i] else 1.0 - members[i, 0]),
            "support_rf": float(members[i, 1] if stage1_decision[i] else 1.0 - members[i, 1]),
            "support_xgb": float(members[i, 2] if stage1_decision[i] else 1.0 - members[i, 2]),
            "support_svm": float(members[i, 3] if stage1_decision[i] else 1.0 - members[i, 3]),
            "route_DI": int(route[i]),
            "final_probability_DI": float(final[i]),
        })
        output_rows.append(item)
    write_csv(args.output, output_rows)
    write_csv(args.mirror, output_rows)
    audit = {
        "passed": True,
        "source_n": 799,
        "source_unique_ids": 799,
        "future_rows_loaded": False,
        "formula": "U_DI = 1 - support(mean committee probability, frozen Stage-I decision)",
        "budget": args.budget,
        "routed_total": int(np.sum(route)),
        "routed_by_fold": per_fold,
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "mirror_sha256": sha256(args.mirror),
    }
    if audit["routed_total"] != 200 or sorted(per_fold.values()) != [40, 40, 40, 40, 40]:
        raise RuntimeError("DI25 route count contract failed: " + json.dumps(audit))
    report = args.output.parent / "source_only_recovery_audit.json"
    report.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_DI25_RECOVERY_PASS")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
