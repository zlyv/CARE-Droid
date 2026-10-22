#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

CAP = ROOT / "instrumented_m0/routed_internal_capture.npz"
ROWS = ROOT / "analysis/routed_mechanism_rows.csv"
OUT = ROOT / "analysis"

d = pd.read_csv(ROWS)
d["sha256"] = d["sha256"].astype(str).str.lower()

with np.load(CAP, allow_pickle=False) as z:
    ids = z["sha256"].astype(str)
    raw = np.asarray(z["raw_delta"], float)
    diagnostic = np.asarray(z["diagnostic"], float)
    fused = np.asarray(z["fused"], float)

assert raw.shape == (3, 497)
assert fused.shape == (3, 497, 113)

d = d.set_index("sha256").loc[ids].reset_index()

# bounded_delta is diagnostic column 0
bounded = diagnostic[:, :, 0]

y = d["label"].to_numpy(int)

# Correct correction direction:
# malware/FN => positive
# benign/FP  => negative
truth_sign = np.where(y == 1, 1.0, -1.0)

alignment = bounded * truth_sign[None, :]

# Each seed independently chooses the correct direction?
seed_right = alignment > 0

d["seed_correct_direction_count"] = seed_right.sum(axis=0)
d["seed_wrong_direction_count"] = 3 - d["seed_correct_direction_count"]

# Do all three seeds give same correction sign?
signs = np.sign(bounded)

d["three_seed_same_sign"] = (
    (signs[0] == signs[1]) &
    (signs[1] == signs[2])
).astype(int)

d["all3_direction_correct"] = (
    d["seed_correct_direction_count"] == 3
).astype(int)

d["all3_direction_wrong"] = (
    d["seed_correct_direction_count"] == 0
).astype(int)

d["bounded_delta_seed_sd"] = np.std(
    bounded,
    axis=0
)

d["raw_delta_seed_sd_direct"] = np.std(
    raw,
    axis=0
)

# -------------------------------------------------------
# Representation dispersion across three seeds
# -------------------------------------------------------

blocks = {
    "graph": (25, 57),
    "critical": (57, 81),
    "attended": (81, 105),
    "summary": (105, 113),
}

for name, (a, b) in blocks.items():

    x = fused[:, :, a:b]

    center = np.mean(
        x,
        axis=0,
        keepdims=True
    )

    # Average L2 deviation of a seed representation
    # from the 3-seed centroid.
    dispersion = np.mean(
        np.linalg.norm(
            x - center,
            axis=2
        ),
        axis=0
    )

    d[f"{name}_seed_dispersion"] = dispersion


print()
print("=" * 100)
print("DIRECTION CONSENSUS BY COHORT")
print("=" * 100)

table = (
    d.groupby("cohort")[
        [
            "all3_direction_correct",
            "all3_direction_wrong",
            "three_seed_same_sign",
            "seed_correct_direction_count",
            "bounded_delta_seed_sd",
        ]
    ]
    .agg(["count", "mean"])
)

print(table.to_string())


print()
print("=" * 100)
print("UNFIXED FAILURE x SEED DIRECTION COUNT")
print("=" * 100)

u = d[d["cohort"] == "Unfixed"]

print(
    pd.crosstab(
        u["failure_type"],
        u["seed_correct_direction_count"],
        margins=True
    ).to_string()
)


print()
print("=" * 100)
print("WRONG DIRECTION BY YEAR")
print("=" * 100)

w = d[d["failure_type"] == "WRONG_DIRECTION"]

print(
    w.groupby("year")[
        [
            "all3_direction_wrong",
            "three_seed_same_sign",
            "bounded_delta_seed_sd",
            "graph_seed_dispersion",
            "critical_seed_dispersion",
            "attended_seed_dispersion",
        ]
    ]
    .agg(["count", "mean"])
    .to_string()
)

d.to_csv(
    OUT / "direction_consensus_rows.csv",
    index=False
)

print()
print(
    "saved:",
    OUT / "direction_consensus_rows.csv"
)
