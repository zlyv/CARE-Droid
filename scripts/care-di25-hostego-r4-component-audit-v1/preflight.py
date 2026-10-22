#!/usr/bin/env python3
from pathlib import Path
import json
root=Path("artifacts/care-di25-hostego-dmd-bounded-v1/feature_sufficiency_audit")
required=[root/"pre_delta_feature_schema.json",root/"preprocessing_reconstruction_audit.csv",root/"NO_STAGE2_RETRAINING_AUDIT.json"]
missing=[str(p) for p in required if not p.is_file()]
for y in (2018,2019,2020,2021):
 for f in range(1,6):
  for s in (900000,900001,900002):
   for split in ("train","validation"):
    p=root/"cache"/str(y)/f"fold_{f}"/f"seed_{s}"/f"{split}_r4.npz"
    if not p.is_file():missing.append(str(p))
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing frozen R4 inputs:\n- "+"\n- ".join(missing))
audit=json.loads((root/"NO_STAGE2_RETRAINING_AUDIT.json").read_text(encoding="utf-8"))
if audit.get("stage2_neural_training_invoked") is not False or audit.get("preprocessing_reconstruction_passed") is not True:raise RuntimeError("BLOCKED_BY_CONTRACT: V2 audit did not pass")
print("R4_COMPONENT_AUDIT_PREFLIGHT_PASS")
