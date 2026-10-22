#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path("artifacts/care-role-conditioned-api-v1")
def main():
    checks={"A0_final_implementation":Path("scripts/care-p31rf-c43-haig-host-ablation-source-v1/run_arm.py").exists(),"nested_P31_RF":Path("scripts/care-p31-stage1-rf-selection-source-v1/nested_stage1.py").exists(),"compact_background":Path("scripts/care-di25-compact43-relationquery-source-v1/compact_background.py").exists(),"endpoint_cache":Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz").exists()}
    decision="FEASIBLE" if all(checks.values()) else "BLOCKED";(ROOT/"feasibility_report.json").write_text(json.dumps({"decision":decision,"checks":checks,"requires_apk_reanalysis":False,"requires_future":False},indent=2)+"\n");print("ROLE_CONDITIONED_API_FEASIBILITY:",decision)
    if decision!="FEASIBLE":raise RuntimeError("BLOCKED_BY_CONTRACT: missing implementation assets")
if __name__=="__main__":main()
