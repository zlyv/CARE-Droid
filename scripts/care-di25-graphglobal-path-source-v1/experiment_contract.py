#!/usr/bin/env python3
"""Fail-closed frozen-DI25 source contract."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

EXPECTED_N = 799
EXPECTED_ROUTED = 200
EXPECTED_FOLDS = {1: 160, 2: 160, 3: 160, 4: 160, 5: 159}
EXPECTED_ROUTE_BY_FOLD = {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
MAX_DELTA = 2.25


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def norm(value):
    return str(value).strip().lower()


def validate_di25_rows(rows):
    fields = set(rows[0]) if rows else set()
    id_key = "sha256" if "sha256" in fields else "sample_id" if "sample_id" in fields else None
    reasons = []
    ids = [norm(row.get(id_key, "")) for row in rows] if id_key else []
    if len(rows) != EXPECTED_N:
        reasons.append(f"rows={len(rows)}")
    if len(set(ids)) != EXPECTED_N or "" in ids:
        reasons.append(f"unique_sha={len(set(ids))}")
    required = {"label", "fold", "p_stage1", "route_DI"}
    if not required.issubset(fields):
        reasons.append("missing=" + repr(sorted(required - fields)))
        return reasons
    folds = Counter(int(float(row["fold"])) for row in rows)
    routes = Counter(
        int(float(row["fold"]))
        for row in rows if int(float(row["route_DI"])) == 1
    )
    if dict(sorted(folds.items())) != EXPECTED_FOLDS:
        reasons.append(f"fold_sizes={dict(sorted(folds.items()))}")
    if sum(routes.values()) != EXPECTED_ROUTED:
        reasons.append(f"routed={sum(routes.values())}")
    if dict(sorted(routes.items())) != EXPECTED_ROUTE_BY_FOLD:
        reasons.append(f"route_by_fold={dict(sorted(routes.items()))}")
    if "year" in fields:
        years = {int(float(row["year"])) for row in rows}
        if years != {2017}:
            reasons.append(f"years={sorted(years)}")
    return reasons


def find_di25_source(project: Path):
    candidates = [
        project / "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv",
        project / "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv",
        project / "artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
        project / "artifacts/care-route-stage2-within-year-matrix-v1/preflight/source_sample_scores.csv",
    ]
    rejected = []
    for path in candidates:
        if not path.is_file():
            rejected.append(f"{path}: absent")
            continue
        rows = read_csv(path)
        reasons = validate_di25_rows(rows)
        if not reasons:
            return path, rows
        rejected.append(f"{path}: " + ", ".join(reasons))
    raise RuntimeError(
        "BLOCKED_BY_CONTRACT: no exact 799-row frozen DI25 source:\n- "
        + "\n- ".join(rejected)
    )


def build_di_reference(project: Path, output: Path):
    source, rows = find_di25_source(project)
    result = []
    for row in rows:
        score = next((row[key] for key in (
            "U_DI", "DI_score", "decision_instability"
        ) if key in row and str(row[key]).strip()), 0.0)
        result.append({
            "sha256": norm(row.get("sha256", row.get("sample_id"))),
            "label": int(float(row["label"])),
            "fold": int(float(row["fold"])),
            "Global24__prob": float(row["p_stage1"]),
            "Global24__A25__JointS7_ordinary__prob": float(row["p_stage1"]),
            "A25__score": float(score),
            "A25__routed": int(float(row["route_DI"])),
        })
    target = Path(output) / "preflight/di25_reference/source_predictions.csv"
    write_csv(target, result)
    contract = {
        "source_rows": 799,
        "unique_sha256": 799,
        "year": 2017,
        "fold_sizes": EXPECTED_FOLDS,
        "route_DI": 200,
        "route_DI_per_fold": EXPECTED_ROUTE_BY_FOLD,
        "semantic_route": "frozen DI25",
        "legacy_loader_route_column": "A25__routed",
        "di25_source": str(source.resolve()),
        "max_delta_logit": MAX_DELTA,
        "future_rows_loaded": 0,
    }
    contract_path = Path(output) / "preflight/di25_contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return contract

