#!/usr/bin/env python3
from __future__ import annotations
from common import *


def main():
    inputs=((FROZEN_STAGE2,"frozen SVM+C43 predictions"),(WITHIN_STAGE2,"same-year SVM+C43 predictions"),
            (FROZEN_MEMBERS,"frozen calibrated member probabilities"),(WITHIN_MEMBERS,"same-year calibrated member probabilities"),
            (SOURCE_AUDIT/"source_decision.json","frozen 2017 selector decision"))
    missing=[str(path) for path,_ in inputs if not path.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs\n- " + "\n- ".join(missing))
    source=json.loads((SOURCE_AUDIT/"source_decision.json").read_text(encoding="utf-8"))
    if source.get("status")!="SVM_CONSENSUS_GATE_NOT_SUPPORTED":
        raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected source decision; this diagnostic must not alter source selection")
    manifest=[file_record(path,purpose) for path,purpose in inputs]
    OUTPUT.mkdir(parents=True,exist_ok=True);dump(OUTPUT/"baseline_asset_manifest_before.json",manifest)
    dump(OUTPUT/"protocol.json",{"source_decision":"SVM_CONSENSUS_GATE_NOT_SUPPORTED",
        "rules_frozen_before_future":["S1_ALL_C43","S2_MAJORITY_2_OF_3","S3_UNANIMOUS_3_OF_3"],
        "frozen_future":"train 2017, test 2018-2021; diagnostic only",
        "same_year":"independent five-fold OOF per year; not temporal generalization",
        "future_used_for_selection":False,"thresholds_changed_after_future":False,"seed":SEED})
    print("SVM_CONSENSUS_DIAGNOSTICS_PREFLIGHT_PASS source_rule_frozen=true")


if __name__=="__main__":main()
