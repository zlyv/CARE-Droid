#!/usr/bin/env python3
"""Hard 2017-only preflight. It performs no model training."""
from __future__ import annotations

import json
from pathlib import Path

from source_contract import (
    ACTUALLY_CONSUMED_FOR_PREFLIGHT, ALLOWED_READ_PATHS, FORBIDDEN_BASENAMES,
    HOST_NPZ, OUTPUT, REGIONS_NPZ, blocked, write_json,
)
from source_loader import load_source_contract
from probe_protocol import ARMS


def main() -> None:
    # The allowed list is closed: no runtime option can inject a future path.
    forbidden_allowed = [str(path) for path in ALLOWED_READ_PATHS if path.name in FORBIDDEN_BASENAMES]
    if forbidden_allowed:
        raise blocked("forbidden file entered read whitelist: " + repr(forbidden_allowed))
    if REGIONS_NPZ in ACTUALLY_CONSUMED_FOR_PREFLIGHT:
        raise blocked("generic regions_2017 must not be consumed when dedicated Host NPZ exists")

    data = load_source_contract()
    if data.future_rows_loaded != 0 or data.ego_consumed is not False:
        raise blocked("source-only/no-Ego runtime assertion failed")
    if len(data.ids) != 799 or int(data.route.sum()) != 200:
        raise blocked("loaded source cohort differs from canonical 799/200")

    noego = {
        "passed": True,
        "ego_consumed": False,
        "ego_zero_filled": False,
        "ego_encoder_instantiated": False,
        "ego_gate_instantiated": False,
        "host_npz": str(HOST_NPZ),
        "host_npz_available_keys": data.host_available_keys,
        "host_npz_consumed_keys": data.host_consumed_keys,
        "generic_regions_npz_consumed": False,
        "ego_1hop_npz_consumed": False,
        "representation_contract": "S2=true Host-only Anomaly 88-D; graph32+critical_host24+attended_host24+summary8",
    }
    protocol = {
        "passed": True,
        "scope": "strict source-only 2017",
        "source_rows": 799, "year": 2017, "route_DI": 200,
        "future_rows_loaded": 0, "ego_consumed": False,
        "outer_fold_source": "canonical source_router_inputs.csv fold column",
        "outer_validation_stage1": "frozen p_stage1 from canonical router",
        "outer_train_stage1": "four-fold inner-OOF Global24 fitted entirely within outer-train",
        "formal_training_started": False,
        "arms": ARMS,
        "read_whitelist": [str(path) for path in ALLOWED_READ_PATHS],
        "preflight_files_actually_read": [str(path) for path in ACTUALLY_CONSUMED_FOR_PREFLIGHT],
        "output_root": str(OUTPUT),
    }
    audit = {
        **data.router_audit,
        "future_rows_loaded": 0, "ego_consumed": False,
        "features_rows": len(data.feature_rows),
        "p72_cohort_aligned": True,
        "endpoint_cache_cohort_aligned": True,
        "host_model_bag_rows": len(data.host_bags),
        "real_host_total": int(data.real_host_count.sum()),
        "formal_training_started": False,
    }
    write_json(OUTPUT / "preflight_audit.json", audit)
    write_json(OUTPUT / "noego_contract.json", noego)
    write_json(OUTPUT / "protocol.json", protocol)

    print("SOURCE_NEEDGATE_PREFLIGHT_PASS")
    print("source_rows = 799")
    print("year = 2017")
    print("route_DI = 200")
    print("future_rows_loaded = 0")
    print("ego_consumed = false")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        if str(error).startswith("BLOCKED_BY_CONTRACT"):
            print(str(error), flush=True)
            raise SystemExit(2)
        raise
