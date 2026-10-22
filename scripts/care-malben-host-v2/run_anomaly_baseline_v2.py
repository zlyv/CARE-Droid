#!/usr/bin/env python3
"""Run the frozen 33,388-parameter Host-only anomaly baseline at DeltaMax=2.25."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT / "care-rgcondattn-dedup-bounded-source-v1",
             ROOT / "care-rolegraph-conditioned-v4-script", ROOT):
    sys.path.insert(0, str(path))

import run_care_rgcondattn_dedup_bounded_source_v1 as baseline
from source_label_contract import install_source_label_contract


def main():
    install_source_label_contract(baseline.v4)
    baseline.main()
    output = Path("artifacts/care-malben-host-v2/source/anomaly")
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    path = output / "architecture.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({
        "experiment": "Frozen Host-only AnomalyBaseline for MalBen v2 control",
        "evaluation": "strict fixed 2017 outer-fold OOF only",
        "future_data_loaded": False,
        "host_only": True,
        "ego_cache_used_by_model": False,
        "bounded_correction": 2.25,
        "bounded_correction_logit_max_delta": 2.25,
        "parameter_count": 33388,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
