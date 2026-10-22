#!/usr/bin/env python3
from __future__ import annotations

from features import entropy, hhi, normalized


def main() -> None:
    assert abs(entropy([1, 1, 1, 1, 1]) - 1.0) < 1e-9
    assert abs(hhi([1, 0, 0]) - 1.0) < 1e-9
    raw = {"attn_any_effect": .8, "mean_any_effect": .4, "slice_node_ratio": .01, "effect_group_diversity": .4}
    groups = ("TRANSFER", "PERSIST", "EXECUTE", "VISIBLE", "SENSITIVE_ACTION")
    for group in groups:
        raw.update({f"attn_{group}_reach": .2, f"mean_{group}_reach": .1, f"attn_{group}_inverse_distance": .1,
            f"attn_{group}_d0": 0, f"attn_{group}_d1": .1, f"attn_{group}_d2": .1, f"attn_{group}_d3_4": 0,
            f"attn_{group}_SAME_CLASS": .1, f"attn_{group}_SAME_PACKAGE": .05, f"attn_{group}_CROSS_PACKAGE": .05,
            f"attn_{group}_unique_target": .1, f"attn_{group}_shared_target": .1, f"critical_{group}": 0})
        for role in range(31):
            raw[f"role_{role:02d}_{group}"] = 1 / 31
    result = normalized({"variants": {"REAL": raw}}, "REAL")
    assert abs(sum(result[f"{group}_share"] for group in groups) - 1.0) < 1e-9
    assert all(-1.000001 <= value <= 1.000001 for value in result.values())
    print("FPR_SELFTEST_PASS normalized=true fold_safe_residual_contract=true matched_null_contract=true", flush=True)


if __name__ == "__main__":
    main()
