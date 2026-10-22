import csv
from pathlib import Path

root = Path("/home/boyuanjie/projects/tasa-droid/outputs")
with (root / "care-2020-fn-tn-dex-review-v1/sealed_key_restricted.csv").open(newline="") as f:
    scanned = {r["sha256"] for r in csv.DictReader(f)}
with (root / "care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv").open(newline="") as f:
    pairs = list(csv.DictReader(f))
for k in (8, 12, 22):
    chosen = pairs[:k]
    overlap = sum(r[key] in scanned for r in chosen for key in ("fn_sha256", "tn_sha256"))
    print(f"top{k}_overlap={overlap}/{2*k}")
