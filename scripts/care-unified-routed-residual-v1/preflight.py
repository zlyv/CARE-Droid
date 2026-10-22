#!/usr/bin/env python3
from common import FORMAL, MEMBER, OUTPUT, asset_record, dump_json, norm, read_csv


def main():
    for path in (MEMBER, FORMAL):
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: missing input " + str(path))
    member = read_csv(MEMBER); formal = read_csv(FORMAL)
    required_member = {"sha256", "label", "fold", "p_P31__LR", "p_P31__RF", "p_P31__XGB", "p_P31__SVM"}
    required_formal = {"sha256", "label", "fold", "Global24__prob", "A25__routed",
                       "P31RF__RFRoute__A0_ROLE_ONLY__Bounded__prob"}
    if len(member) != 799 or not required_member.issubset(member[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: member OOF schema/cohort changed")
    if len(formal) != 799 or not required_formal.issubset(formal[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal A0 schema/cohort changed")
    if {norm(r["sha256"]) for r in member} != {norm(r["sha256"]) for r in formal}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: input SHA cohorts differ")
    assets = [asset_record(MEMBER, "strict source member OOF probabilities"),
              asset_record(FORMAL, "formal RF-DI25 A0-C43 OOF predictions")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", assets)
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799,
        "fixed_route_expected": 200, "future_rows_loaded": 0, "parameter_learning": False,
        "primary_formula": "clip(c43_delta + median(other_member_logits)-rf_logit, -2.25, 2.25)"})
    print("UNIFIED_ROUTED_RESIDUAL_PREFLIGHT_PASS source=799 route_expected=200 future_rows_loaded=0")


if __name__ == "__main__":
    main()
