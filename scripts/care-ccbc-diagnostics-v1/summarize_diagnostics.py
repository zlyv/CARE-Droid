#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT, dump_json, read_csv


def number(row, key):
    value = row.get(key, "nan")
    try: return float(value)
    except Exception: return float("nan")


def find(rows, protocol, scope, arm):
    matches = [row for row in rows if row.get("protocol") == protocol and str(row.get("scope")) == str(scope) and row.get("arm") == arm and not str(row.get("fold", "")).strip()]
    if len(matches) != 1: raise RuntimeError(f"BLOCKED_BY_CONTRACT: summary lookup {protocol}/{scope}/{arm} matches={len(matches)}")
    return matches[0]


def direction(rows, protocol, scope, variant="real"):
    matches = [row for row in rows if row.get("protocol") == protocol and str(row.get("scope", "pooled")) == str(scope) and row.get("variant") == variant]
    if len(matches) != 1: raise RuntimeError(f"BLOCKED_BY_CONTRACT: direction lookup {protocol}/{scope}/{variant} matches={len(matches)}")
    return matches[0]


def main():
    frozen_metrics = read_csv(OUTPUT / "frozen_metrics.csv"); within_metrics = read_csv(OUTPUT / "within_metrics.csv")
    frozen_direction = read_csv(OUTPUT / "frozen_direction.csv"); within_direction = read_csv(OUTPUT / "within_direction.csv")
    mapping = json.loads((OUTPUT / "future_component_mapping_audit.json").read_text(encoding="utf-8"))
    frozen_a0 = find(frozen_metrics, "frozen_2017_to_future", "pooled", "A0_FORMAL_DI25_C43")
    frozen_a3 = find(frozen_metrics, "frozen_2017_to_future", "pooled", "A3_CCBC_ROUTE_CCBC_DIRECTION")
    frozen_null = find(frozen_metrics, "frozen_2017_to_future", "pooled", "A4_CONTRACT_SHUFFLE")
    within_a0 = find(within_metrics, "independent_same_year_5fold", "pooled", "A0_FORMAL_DI25_C43")
    within_a3 = find(within_metrics, "independent_same_year_5fold", "pooled", "A3_CCBC_ROUTE_CCBC_DIRECTION")
    frozen_real_direction = direction(frozen_direction, "frozen_2017_to_future", "pooled")
    within_real_direction = direction(within_direction, "independent_same_year_5fold", "pooled")
    transfer_supported = (
        number(frozen_a3, "net") > number(frozen_a0, "net")
        and number(frozen_a3, "broken") <= number(frozen_a0, "broken")
        and number(frozen_a3, "net") > number(frozen_null, "net")
        and number(frozen_real_direction, "FP_vs_TP_AUC") > .5
        and number(frozen_real_direction, "FN_vs_TN_AUC") > .5
    )
    final = {
        "source_conclusion": "PARTIALLY_SUPPORTED",
        "source_integration_recommendation": False,
        "reason": "source A3 net=-5 versus formal A0 net=+4; future cannot reverse source selection",
        "frozen_future_transfer_supported": bool(transfer_supported),
        "future_mapping": mapping,
        "frozen_pooled": {"A0": frozen_a0, "A3": frozen_a3, "contract_shuffle": frozen_null, "direction": frozen_real_direction},
        "same_year_pooled": {"A0": within_a0, "A3": within_a3, "direction": within_real_direction},
        "future_used_for_selection": False,
        "same_year_is_temporal_generalization": False,
        "untouched_temporal_generalization": "NOT_VALIDATED; requires 2022+ or independent external data",
    }
    dump_json(OUTPUT / "final_diagnostic_summary.json", final)
    lines = [
        "CARE-Droid CCBC frozen future and same-year diagnostics v1", "",
        "Source conclusion remains PARTIALLY_SUPPORTED and CCBC is not selected for the formal method.",
        "2018-2021 are post-selection diagnostics and cannot change the source decision.", "",
        f"Future mapping: entry_ratio={mapping['entry_mapping_ratio']:.6f} host_assignment_ratio={mapping['host_assignment_ratio']:.6f}",
        f"FROZEN A0: BA={number(frozen_a0,'BA'):.6f} fixed={int(number(frozen_a0,'fixed'))} broken={int(number(frozen_a0,'broken'))} net={int(number(frozen_a0,'net'))}",
        f"FROZEN A3: BA={number(frozen_a3,'BA'):.6f} fixed={int(number(frozen_a3,'fixed'))} broken={int(number(frozen_a3,'broken'))} net={int(number(frozen_a3,'net'))}",
        f"FROZEN direction: FN-vs-TN AUC={number(frozen_real_direction,'FN_vs_TN_AUC'):.6f} FP-vs-TP AUC={number(frozen_real_direction,'FP_vs_TP_AUC'):.6f}",
        f"FROZEN real-transfer-supported={transfer_supported}", "",
        f"SAME-YEAR A0: BA={number(within_a0,'BA'):.6f} fixed={int(number(within_a0,'fixed'))} broken={int(number(within_a0,'broken'))} net={int(number(within_a0,'net'))}",
        f"SAME-YEAR A3: BA={number(within_a3,'BA'):.6f} fixed={int(number(within_a3,'fixed'))} broken={int(number(within_a3,'broken'))} net={int(number(within_a3,'net'))}",
        f"SAME-YEAR direction: FN-vs-TN AUC={number(within_real_direction,'FN_vs_TN_AUC'):.6f} FP-vs-TP AUC={number(within_real_direction,'FP_vs_TP_AUC'):.6f}",
        "Same-year uses labels inside each corresponding year and is not temporal generalization.",
        "Untouched temporal generalization remains unvalidated.", "",
        "CCBC_DIAGNOSTICS_DONE",
    ]
    (OUTPUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__": main()
