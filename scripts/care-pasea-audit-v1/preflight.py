#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import OUTPUT, PROJECT, dump_json, ensure_output, load_json


def first_existing(paths: list[Path]) -> Path | None:
    return next((path.resolve() for path in paths if path.exists()), None)


def main() -> None:
    ensure_output()
    if not PROJECT.exists():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: project missing: {PROJECT}")
    if OUTPUT == PROJECT or PROJECT not in OUTPUT.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: output outside project or equals project: {OUTPUT}")

    obcbc_root = PROJECT / "outputs/care-obcbc-source-v1"
    obcbc_decision = first_existing([
        obcbc_root / "final_decision.json",
        obcbc_root / "feasibility_decision.json",
    ])
    obcbc_owner = first_existing([obcbc_root / "ownership_audit.json"])
    obcbc_context = first_existing([obcbc_root / "context_coverage.json"])
    direct_predictions = first_existing([
        PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source_predictions.csv",
        PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/predictions.csv",
        PROJECT / "outputs/care-svm-c43-unfixed-audit-v1/routed_error_audit.csv",
    ])
    formal_predictions = first_existing([
        PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
        PROJECT / "artifacts/care-droid-formal-v1/source_predictions.csv",
    ])

    decision_payload = load_json(obcbc_decision, {}) if obcbc_decision else {}
    result = {
        "status": "PASS",
        "project": str(PROJECT),
        "output": str(OUTPUT),
        "future_rows_loaded": 0,
        "labels_used": False,
        "obcbc_decision": str(obcbc_decision) if obcbc_decision else None,
        "obcbc_status": decision_payload.get("status") or decision_payload.get("stage_b_status"),
        "obcbc_ownership_audit": str(obcbc_owner) if obcbc_owner else None,
        "obcbc_context_coverage": str(obcbc_context) if obcbc_context else None,
        "direct_predictions_candidate": str(direct_predictions) if direct_predictions else None,
        "formal_predictions_candidate": str(formal_predictions) if formal_predictions else None,
        "scope": "history equivalence and exact-TPL/owner observability only; no model training or label probe",
    }
    dump_json(OUTPUT / "preflight.json", result)
    print("PASEA_PREFLIGHT_PASS")
    print(result)


if __name__ == "__main__":
    main()

