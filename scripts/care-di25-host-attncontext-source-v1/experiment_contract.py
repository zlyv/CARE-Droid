#!/usr/bin/env python3
"""Self-contained frozen-DI25 source contract for the input ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

MAX_DELTA = 2.25
EXPECTED_N = 799
EXPECTED_ROUTED = 200


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def norm(value):
    return str(value).strip().lower()


def build_di_reference(project: Path, output: Path) -> dict:
    candidates = [
        project / "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv",
        project / "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv",
        project / "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv",
        project / "artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
        project / "artifacts/care-route-stage2-within-year-matrix-v1/preflight/source_sample_scores.csv",
    ]
    accepted, rows, rejected = None, [], []
    for path in candidates:
        if not path.is_file():
            rejected.append(f"{path}: absent"); continue
        candidate = read_csv(path); fields = set(candidate[0]) if candidate else set()
        id_key = "sha256" if "sha256" in fields else "sample_id" if "sample_id" in fields else None
        ids = [norm(row.get(id_key, "")) for row in candidate] if id_key else []
        reasons = []
        if len(candidate) != EXPECTED_N: reasons.append(f"rows={len(candidate)}")
        if len(set(ids)) != EXPECTED_N or "" in ids: reasons.append(f"unique_ids={len(set(ids))}")
        if not {"label", "fold", "p_stage1", "route_DI"}.issubset(fields): reasons.append("missing columns")
        try: routed = sum(int(float(row["route_DI"])) for row in candidate)
        except Exception: routed = -1
        if routed != EXPECTED_ROUTED: reasons.append(f"routed={routed}")
        if reasons: rejected.append(f"{path}: " + ", ".join(reasons)); continue
        accepted, rows = path, candidate; break
    if accepted is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: no exact frozen DI25 table:\n- " + "\n- ".join(rejected))

    result = []
    for row in rows:
        score = next((row[key] for key in ("U_DI", "DI_score", "decision_instability")
                      if key in row and str(row[key]).strip()), 0.0)
        result.append({
            "sha256": norm(row.get("sha256", row.get("sample_id"))),
            "label": int(float(row["label"])), "fold": int(float(row["fold"])),
            "Global24__prob": float(row["p_stage1"]),
            "Global24__A25__JointS7_ordinary__prob": float(row["p_stage1"]),
            "A25__score": score, "A25__routed": int(float(row["route_DI"])),
        })
    reference = output / "preflight/di25_reference/source_predictions.csv"
    write_csv(reference, result)
    contract = {"source_rows": 799, "unique_sha256": 799, "routed": 200,
                "semantic_route": "frozen DI25", "legacy_loader_route_column": "A25__routed",
                "di25_source": str(accepted.resolve()), "max_delta_logit": 2.25}
    (output / "preflight").mkdir(parents=True, exist_ok=True)
    (output / "preflight/di25_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return contract

