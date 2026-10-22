#!/usr/bin/env python3
"""Frozen source-protocol contracts for the CARE DI25 Host/Ego ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

MAX_DELTA = 2.25
EXPECTED_N = 799
EXPECTED_ROUTED = 200
FOLD_SIZES = [160, 160, 160, 160, 159]


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
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


def norm(value) -> str:
    return str(value).strip().lower()


def resolve_first(paths: list[Path], description: str) -> Path:
    for path in paths:
        if Path(path).is_file():
            return Path(path)
    raise RuntimeError(
        "BLOCKED_BY_CONTRACT: missing " + description + ":\n- "
        + "\n- ".join(str(path) for path in paths)
    )


def build_di_reference(project: Path, output: Path) -> dict:
    """Build the loader-compatible reference solely from frozen DI25 scores.

    Historical method outputs are deliberately excluded here: they are useful
    for optional reproduction audits, but must never become run dependencies.
    """
    di_candidates = [
        project / "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv",
        project / "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv",
        project / "artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
        project / "artifacts/care-route-stage2-within-year-matrix-v1/preflight/source_sample_scores.csv",
    ]
    di_path = None
    di_rows = []
    rejected = []
    for candidate in di_candidates:
        if not candidate.is_file():
            rejected.append(str(candidate) + ": absent")
            continue
        rows = read_csv(candidate)
        fields = set(rows[0]) if rows else set()
        id_key = "sha256" if "sha256" in fields else "sample_id" if "sample_id" in fields else None
        ids = [norm(row.get(id_key, "")) for row in rows] if id_key else []
        reasons = []
        if len(rows) != EXPECTED_N:
            reasons.append(f"rows={len(rows)}")
        if id_key is None or len(set(ids)) != EXPECTED_N or "" in ids:
            reasons.append(f"unique_ids={len(set(ids))}")
        missing = {"label", "fold", "p_stage1", "route_DI"} - fields
        if missing:
            reasons.append("missing=" + repr(sorted(missing)))
        try:
            routed = sum(int(float(row["route_DI"])) for row in rows) if not missing else -1
        except (TypeError, ValueError):
            routed = -1
        if routed != EXPECTED_ROUTED:
            reasons.append(f"routed={routed}")
        if reasons:
            rejected.append(str(candidate) + ": " + ", ".join(reasons))
            continue
        di_path, di_rows = candidate, rows
        break
    if di_path is None:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: no valid frozen source DI25 sample score file:\n- "
            + "\n- ".join(rejected)
        )
    di_by_sha = {norm(row.get("sha256", row.get("sample_id"))): row for row in di_rows}
    if len(di_by_sha) != EXPECTED_N:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort is not exact 799")
    ordered_sha = [norm(row.get("sha256", row.get("sample_id"))) for row in di_rows]
    result = []
    route_n = 0
    for sha in ordered_sha:
        if sha not in di_by_sha or "route_DI" not in di_by_sha[sha]:
            raise RuntimeError("BLOCKED_BY_CONTRACT: DI25 route missing for " + sha)
        source = di_by_sha[sha]
        item = {
            "sha256": sha, "label": int(float(source["label"])),
            "fold": int(float(source["fold"])),
            "Global24__prob": float(source["p_stage1"]),
            # Loader-only comparator, never used by Stage-II training or final
            # routing in this experiment.
            "Global24__A25__JointS7_ordinary__prob": float(source["p_stage1"]),
            "A25__score": float(source.get("U_DI", 0.0)),
        }
        route = int(float(di_by_sha[sha]["route_DI"]))
        route_n += route
        item["A25__routed"] = route
        # The v4 loader requires this legacy field, but it is not used to route.
        # Preserve the frozen DI score when available; otherwise retain the old
        # audit-only score while replacing the route with exact frozen DI25.
        for key in ("DI__score", "DI_score", "decision_instability", "U_DI"):
            if key in di_by_sha[sha] and str(di_by_sha[sha][key]).strip():
                item["A25__score"] = di_by_sha[sha][key]
                break
        result.append(item)
    if route_n != EXPECTED_ROUTED:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: frozen DI25 routed {route_n}, expected 200")
    reference_dir = output / "preflight/di25_reference"
    write_csv(reference_dir / "source_predictions.csv", result)
    contract = {
        "source_rows": len(result), "unique_sha256": len({norm(r["sha256"]) for r in result}),
        "routed": route_n, "routing_rate": route_n / len(result),
        "global24_source": str(di_path.resolve()) + "::p_stage1",
        "di25_source": str(di_path.resolve()),
        "ordinary_probability_compatibility_field_used_by_training": False,
        "legacy_loader_route_column": "A25__routed", "semantic_route": "frozen DI25",
        "max_delta_logit": MAX_DELTA,
    }
    (output / "preflight").mkdir(parents=True, exist_ok=True)
    (output / "preflight/di25_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return contract


def write_static_audits(output: Path) -> None:
    audits = output / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    semantics = {
        "host_definition": "caller method retained in the bounded endpoint-centred graph that directly calls at least one retained mapped sensitive endpoint; roles are the union of those endpoint roles",
        "host_graph_definition": "directed graph induced by retained Host methods inside the endpoint-centred bounded graph",
        "ego_graph_definition": "center Host plus incoming and outgoing adjacent Host methods in that induced directed Host graph",
        "edge_definition": "retained raw call edge after Host-to-Host filtering",
        "direction_available": True,
        "raw_method_nodes_in_ego": False,
        "ordinary_non_host_method_nodes_in_ego": False,
        "path_contraction_used": False,
        "bounded_parent_graph": True,
        "max_reachability_hops_if_known": {"upstream": 2, "downstream": 1},
        "evidence": [
            "care-source-local-assets-v2/extract_source_local_assets_v2.py::reconstruct",
            "care-source-local-assets-v2/README.md",
            "endpoint_centered_view.py::EndpointViewPolicy",
        ],
    }
    leakage = {
        "DI25": "source-only frozen", "Global24": "source protocol",
        "Bounded max_delta": "2.25 frozen", "future labels used in training": False,
        "future labels used in route": False, "future labels used in prototype": False,
        "future labels used in threshold selection": False,
        "future labels used in model selection": False,
        "Host+Ego gate uses label": False, "Host+Ego gate uses year": False,
        "Host+Ego gate uses future domain": False,
    }
    (audits / "ego_semantics_audit.json").write_text(json.dumps(semantics, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (audits / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
