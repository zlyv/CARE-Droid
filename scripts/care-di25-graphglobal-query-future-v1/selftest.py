#!/usr/bin/env python3
"""Dependency-light structural test; no artifact is opened."""
from pathlib import Path
import json

def main():
    root=Path(__file__).resolve().parent
    contract=json.loads((root/"freeze_contract.json").read_text(encoding="utf-8"))
    assert contract["candidate"]=="Q_QUERY_ONLY"
    assert contract["control"]=="N0_NO_GRAPHGLOBAL"
    assert contract["architecture"]["ego_consumed"] is False
    assert contract["architecture"]["graphglobal_to_attention"] is True
    assert contract["architecture"]["graphglobal_to_delta_direct"] is False
    assert contract["architecture"]["parameter_count"]==33036
    required={
        "run_frozen_future_arm.py":["matrix.load_common", "make_graphglobal_path_class", "parameter count changed", "future_labels_loaded_by_model"],
        "preflight.py":["future_rows_loaded=0", "ego_consumed=false", "route_DI"],
        "prepare_future_inter.py":["matrix.load_common", "FUTURE_INTER_CACHE_PASS", "future_labels_loaded"],
        "summarize_future.py":["future_label_file_posthoc_only", "diagnostic-only", "exact_p"],
        "run_on_server.sh":["CUDA_VISIBLE_DEVICES=\"$gpu\"", "run_arm N0_NO_GRAPHGLOBAL 0", "run_arm Q_QUERY_ONLY 1", "wait"],
    }
    for name,markers in required.items():
        text=(root/name).read_text(encoding="utf-8")
        for marker in markers: assert marker in text,(name,marker)
    shell=(root/"run_on_server.sh").read_text(encoding="utf-8")
    assert 'local arm="$1" gpu="$2"' not in shell
    assert 'local arm="$1"\n  local gpu="$2"\n  local output="$ROOT/arms/$arm"' in shell
    print("GRAPHGLOBAL_QUERY_FUTURE_SELFTEST_PASS")
    print("candidate=Q_QUERY_ONLY control=N0_NO_GRAPHGLOBAL")
    print("future_rows_loaded=0")
    print("ego_consumed=false")

if __name__=="__main__": main()
