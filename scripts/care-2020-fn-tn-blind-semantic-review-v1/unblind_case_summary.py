#!/usr/bin/env python3
"""Post-freeze case labels only; no APK SHA, path or model-selection output."""
import csv
import hashlib
import json
from pathlib import Path

root = Path("/home/boyuanjie/projects/tasa-droid/outputs")
out = root / "care-2020-fn-tn-blind-semantic-review-v1"
freeze = json.loads((out / "blind_review_freeze.json").read_text(encoding="utf-8"))
assert hashlib.sha256((out / "blind_review_frozen.csv").read_bytes()).hexdigest() == freeze["review_form_sha256"]
with (root / "care-2020-fn-tn-dex-review-v2/sealed_key_restricted.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
for case in (f"M{i:02d}" for i in range(1, 9)):
    labels = {r["side"]: r["diagnostic_kind"] for r in rows if r["case_id"] == case}
    item = json.loads((out / "blind_packets" / (case + ".json")).read_text(encoding="utf-8"))
    mono = [side for side in ("A", "B") if any("MonoPackageManager" in r["method"] for r in item["sides"][side]["focused_methods"])]
    print(case, "FN_side=" + next(side for side in labels if labels[side] == "UNFIXED_FN"),
          "Mono_side=" + (mono[0] if mono else "none"))
