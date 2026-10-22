#!/usr/bin/env python3
from __future__ import annotations

from common import OUTPUT, dump_json, ensure_output, load_json


def main() -> None:
    ensure_output()
    existing = load_json(OUTPUT / "existing_experiment_audit.json", {})
    inventory = load_json(OUTPUT / "environment_inventory.json", {})
    assignments = inventory.get("assignment_audit", {})
    validation = inventory.get("validation_audit", {})
    explicit = inventory.get("explicit", {})

    precision = validation.get("owner_precision")
    exact_assets = bool(explicit.get("tool_exists") and explicit.get("profile_db_exists") and explicit.get("profile_count", 0) > 0)
    assignment_ready = bool(assignments.get("valid_schema"))
    validation_ready = bool(validation.get("valid_schema") and precision is not None and precision >= 0.90)
    host_ready = bool(assignments.get("sensitive_host_owner_coverage", 0) >= 0.50)
    attention_ready = bool(assignments.get("attention_owner_coverage", 0) >= 0.50)

    criteria = {
        "prior_experiment_is_not_fully_equivalent": existing.get("status") != "FOUND_EQUIVALENT",
        "explicit_tpl_tool_and_profile_db": exact_assets,
        "imported_method_or_host_assignments": assignment_ready,
        "independent_high_confidence_owner_precision>=0.90": validation_ready,
        "sensitive_host_owner_coverage>=0.50": host_ready,
        "attention_owner_coverage>=0.50": attention_ready,
    }
    ready = all(criteria.values())
    if ready:
        status = "PASEA_OBSERVABILITY_CONFIRMED"
        next_action = "authorize a separate frozen-C43 mechanism-audit package; no model training yet"
        reason = "Verified concrete TPL identity, independent owner precision and owner coverage all passed the frozen observability gate."
    elif exact_assets or assignment_ready or validation.get("present"):
        status = "PASEA_CONDITIONALLY_FEASIBLE"
        next_action = "verify a concrete TPL/profile candidate and export owner assignments plus independent validation using the documented schema"
        reason = "Candidate tooling/assets exist, but verified identity assignments and/or independent owner precision are incomplete."
    else:
        status = "PASEA_OBSERVABILITY_INSUFFICIENT"
        next_action = "stop PASEA unless a verified TPL signature database and independent owner-validation set are supplied"
        reason = "The only existing owner experiment is heuristic OBCBC and no verified concrete TPL identity asset was found."

    decision = {
        "status": status,
        "reason": reason,
        "next_action": next_action,
        "existing_experiment": existing.get("status"),
        "criteria": criteria,
        "observed": {
            "owner_precision": precision,
            "sensitive_host_owner_coverage": assignments.get("sensitive_host_owner_coverage"),
            "critical_host_owner_coverage": assignments.get("critical_host_owner_coverage"),
            "attention_owner_coverage": assignments.get("attention_owner_coverage"),
            "candidate_assets": inventory.get("candidate_count", 0),
        },
        "mechanism_probe_run": False,
        "model_training_run": False,
        "future_rows_loaded": 0,
        "labels_used": False,
    }
    dump_json(OUTPUT / "feasibility_decision.json", decision)
    dump_json(OUTPUT / "final_decision.json", decision)

    lines = [
        "# CARE-Droid PASEA feasibility and observability audit v1",
        "",
        f"Final status: **{status}**.",
        "",
        "PASEA is an audit hypothesis, not a trained model. This run never reads labels, never trains C43, and never loads future rows.",
        "",
        "## Historical equivalence",
        "",
        f"- Existing experiment status: {existing.get('status')}",
        f"- OBCBC final status: {existing.get('obcbc_final_status')}",
        f"- OBCBC Host owner coverage: {existing.get('obcbc_owner_coverage')}",
        f"- OBCBC accountable attention mass: {existing.get('obcbc_accountable_attention_mass')}",
        "- OBCBC is partial only: it uses identity-free corpus reuse/namespace heuristics, not verified concrete TPL/version identity.",
        "",
        "## Frozen observability criteria",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in criteria.items())
    lines += [
        "",
        "## Decision boundary",
        "",
        reason,
        "",
        f"Next action: {next_action}.",
        "",
        "No owner deletion, fixed/broken label analysis, mechanism model or future diagnostic is authorized unless observability is confirmed.",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASEA_AUDIT_FINAL_STATUS={status}")
    print(f"PASEA_AUDIT_NEXT_ACTION={next_action}")


if __name__ == "__main__":
    main()

