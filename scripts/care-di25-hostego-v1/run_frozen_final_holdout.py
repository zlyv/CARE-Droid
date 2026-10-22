#!/usr/bin/env python3
"""Fail-closed entry point for a future, genuinely untouched 2022--2024 holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_final_method import FINAL_FROZEN_METHOD, ROOT, blocked, contract


READINESS = ROOT / "final_holdout/final_holdout_readiness.json"
LEAKAGE = ROOT / "final_audits/final_leakage_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-untouched-final-evaluation", action="store_true")
    parser.add_argument("--provenance", type=Path, help="Independent JSON confirming 2022-2024 were not used in current design")
    parser.add_argument("--execute-frozen-backend", action="store_true", help="Reserved for when all audited assets and the deployment backend exist")
    args = parser.parse_args()
    if not args.confirm_untouched_final_evaluation:
        blocked("explicit --confirm-untouched-final-evaluation is required")
    frozen = contract()
    guards = {
        "method": frozen["method_name"] == FINAL_FROZEN_METHOD,
        "router": frozen["router"] == "DI25",
        "budget": float(frozen["router_budget"]) == .25,
        "max_delta": float(frozen["max_delta"]) == 2.25,
        "HostEgo_enabled": frozen["hostego_parameter_count"] == 34613,
        "evidence": "Anomaly" in FINAL_FROZEN_METHOD,
        "decision_head": "Bounded" in FINAL_FROZEN_METHOD,
    }
    if not all(guards.values()):
        blocked(f"final-method guard failed: {guards}")
    if not READINESS.is_file():
        blocked(f"readiness audit missing: {READINESS}")
    ready = json.loads(READINESS.read_text(encoding="utf-8"))
    if not ready or not all(row.get("ready_for_frozen_inference") is True for row in ready):
        missing = {row.get("year"): row.get("missing", []) for row in ready}
        blocked(f"2022-2024 assets are not ready: {missing}")
    if args.provenance is None or not args.provenance.is_file():
        blocked("an explicit provenance JSON is required before calling 2022-2024 untouched")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance.get("2022_2024_used_in_current_design") is not False:
        blocked("provenance must explicitly confirm 2022_2024_used_in_current_design=false")
    if not args.execute_frozen_backend:
        blocked("all gates passed, but --execute-frozen-backend was not supplied")
    # This package deliberately does not invent a holdout backend before the
    # real assets and lineage are available. Adding that backend is an
    # implementation task, not permission to tune the frozen method.
    blocked("frozen holdout deployment backend is not yet implemented because audited 2022-2024 assets are unavailable")


if __name__ == "__main__":
    main()
