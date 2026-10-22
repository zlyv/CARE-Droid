#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from common import OUTPUT, PROJECT, dump_json, ensure_output, load_json, write_csv


TERMS = {
    "exact_tpl_identity": ["libscout", "libradar", "libid", "libscan", "library_identity"],
    "method_owner": ["first_party", "third_party", "method_ownership", "host_ownership"],
    "app_to_tpl": ["first_to_third", "app_to_library", "app->library", "a\\u2192l"],
    "tpl_to_app": ["third_to_first", "library_to_app", "library->app", "l\\u2192a"],
    "unknown_state": ["unknown", "unresolved"],
    "matched_null": ["owner-shuffle", "ownership_shuffle", "matched deletion", "role-degree"],
}


def scan_tree(root: Path, limit: int = 2500) -> tuple[list[dict], int]:
    hits: list[dict] = []
    files_seen = 0
    if not root.exists():
        return hits, files_seen
    allowed = {".py", ".md", ".json", ".sh", ".txt", ".csv"}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "cache", "checkpoints"}]
        for name in files:
            path = Path(base) / name
            if path.suffix.lower() not in allowed:
                continue
            files_seen += 1
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            matched = [field for field, terms in TERMS.items() if any(term in text for term in terms)]
            if matched:
                hits.append({"path": str(path.resolve()), "matched_dimensions": "|".join(matched)})
                if len(hits) >= limit:
                    return hits, files_seen
    return hits, files_seen


def main() -> None:
    ensure_output()
    roots = [PROJECT / "scripts", PROJECT / "outputs", PROJECT / "artifacts"]
    all_hits, files_seen = [], 0
    for root in roots:
        hits, count = scan_tree(root)
        all_hits.extend(hits)
        files_seen += count

    obcbc = PROJECT / "outputs/care-obcbc-source-v1"
    final = load_json(obcbc / "final_decision.json", {})
    feasibility = load_json(obcbc / "feasibility_decision.json", {})
    owner = load_json(obcbc / "ownership_audit.json", {})
    context = load_json(obcbc / "context_coverage.json", {})
    obcbc_present = bool(final or feasibility or owner or context)

    matrix = [
        {
            "experiment": "care-obcbc-source-v1",
            "exact_tpl_or_version_identity": "NO",
            "method_level_owner": "YES_HEURISTIC",
            "app_to_tpl_boundary": "YES",
            "tpl_to_app_boundary": "YES_APPROXIMATE",
            "unknown_preserved": "YES",
            "independent_owner_precision_validation": "NO",
            "matched_owner_null_completed": "NO_STOPPED_AT_FEASIBILITY",
            "equivalence": "PARTIAL",
            "evidence": "identity-free corpus reuse plus namespace heuristic; Stage-B stopped before labels/null probe",
        },
        {
            "experiment": "Shared commonness / External benign / Provenance6",
            "exact_tpl_or_version_identity": "NOT_ESTABLISHED",
            "method_level_owner": "NOT_ESTABLISHED",
            "app_to_tpl_boundary": "NO",
            "tpl_to_app_boundary": "NO",
            "unknown_preserved": "UNKNOWN",
            "independent_owner_precision_validation": "NO",
            "matched_owner_null_completed": "NO",
            "equivalence": "NOT_EQUIVALENT_UNLESS_ASSET_PROVES_OTHERWISE",
            "evidence": "commonness/function provenance is not a verified TPL identity",
        },
    ]
    fields = list(matrix[0])
    write_csv(OUTPUT / "historical_equivalence_matrix.csv", matrix, fields)
    dump_json(OUTPUT / "existing_experiment_audit.json", {
        "status": "PARTIAL" if obcbc_present else "NOT_FOUND",
        "files_scanned": files_seen,
        "candidate_files": len(all_hits),
        "candidate_hits": all_hits[:500],
        "obcbc_present": obcbc_present,
        "obcbc_final_status": final.get("status") or feasibility.get("status"),
        "obcbc_owner_coverage": owner.get("host_ownership_coverage"),
        "obcbc_critical_owner_coverage": owner.get("critical_host_ownership_coverage"),
        "obcbc_accountable_attention_mass": context.get("accountable_attention_mass"),
        "obcbc_top10pct_boundary_share": context.get("top10pct_apk_boundary_share"),
        "equivalence_reason": "OBCBC already tests heuristic owner/boundary observability, but lacks verified concrete TPL/version identity and independent owner precision; PASEA may only fill that exact gap.",
        "future_rows_loaded": 0,
        "labels_used": False,
    })
    print("PASEA_EXISTING_EXPERIMENT_AUDIT_DONE status=" + ("PARTIAL" if obcbc_present else "NOT_FOUND"))


if __name__ == "__main__":
    main()

