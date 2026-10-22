#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from common import OUTPUT, PROJECT, dump_json, file_record, sha256_file
from diagnostic_assets import (
    FORMAL_ROOT, SOURCE_OUTPUT, a0_region_cache_path, endpoint_cache_path,
    frozen_formal_rows, future_fold_rows, future_p31_path, source_protocol_check,
    within_formal_rows,
)


def main():
    if os.environ.get("CCBC_FUTURE_DIAGNOSTIC_ACK") != "1":
        raise RuntimeError("BLOCKED_BY_CONTRACT: set CCBC_FUTURE_DIAGNOSTIC_ACK=1 after source protocol freeze")
    source = source_protocol_check(); frozen = frozen_formal_rows(); within = within_formal_rows(); folds = future_fold_rows()
    assets = [
        (SOURCE_OUTPUT / "protocol.json", "frozen CCBC source protocol"),
        (SOURCE_OUTPUT / "protocol_hash.txt", "frozen CCBC source protocol hash"),
        (SOURCE_OUTPUT / "final_summary.json", "frozen CCBC source result"),
        (SOURCE_OUTPUT / "cache/component_scope_2017.jsonl.gz", "source component scope"),
        (SOURCE_OUTPUT / "cache/formal_attention.npz", "source formal attention"),
        (FORMAL_ROOT / "frozen/future_predictions_label_free.csv", "formal frozen future prediction"),
        (FORMAL_ROOT / "within_2018_2019/predictions.csv", "formal same-year prediction"),
        (FORMAL_ROOT / "within_2020_2021/predictions.csv", "formal same-year prediction"),
        (future_p31_path(), "future P31"), (endpoint_cache_path(), "endpoint cache"),
        (a0_region_cache_path(), "formal A0 2017-2021 Host cache"),
    ]
    records = [file_record(path, purpose) for path, purpose in assets]
    dump_json(OUTPUT / "diagnostic_asset_manifest_before.json", {"status": "CAPTURED", "assets": records})
    protocol = {
        "name": "CARE_CCBC_FROZEN_AND_SAME_YEAR_DIAGNOSTICS_V1",
        "source_protocol_hash": source["protocol_hash"],
        "source_conclusion": "PARTIALLY_SUPPORTED",
        "source_facts": {"FN_vs_TN_AUC": 0.480609, "FP_vs_TP_AUC": 0.726524,
                         "A0_net": 4, "A3_net": -5},
        "frozen_future": "train/freeze on 2017; diagnose 2018-2021 without retuning",
        "same_year": "independent fixed five-fold OOF inside each year; not temporal generalization",
        "stage1": "calibrated P31-RF only", "router": "RF-anchored MeanOpposition plus frozen CCBC rule, Top25%",
        "stage2_baseline": "A0 Role C43-RQ", "max_delta": 2.25,
        "ccbc_features": "unchanged from source protocol", "minimum_support": 3,
        "nulls": ["contract_shuffle", "role_edge_shuffle", "count_only", "declaration_only", "origin_only"],
        "future_used_for_selection": False, "future_may_not_change_source_conclusion": True,
        "source_rows": 799, "future_rows": len(frozen), "same_year_rows": len(within), "fold_rows": len(folds),
    }
    dump_json(OUTPUT / "diagnostic_protocol.json", protocol)
    (OUTPUT / "diagnostic_protocol_hash.txt").write_text(sha256_file(OUTPUT / "diagnostic_protocol.json") + "\n", encoding="utf-8")
    print("CCBC_DIAGNOSTIC_PREFLIGHT_PASS")
    print(json.dumps({"source_protocol_hash": source["protocol_hash"], "future_rows": len(frozen),
                      "same_year_rows": len(within), "future_used_for_selection": False}, indent=2))


if __name__ == "__main__": main()
