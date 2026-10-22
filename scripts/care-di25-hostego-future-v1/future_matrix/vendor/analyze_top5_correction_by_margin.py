#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import Counter


INPUT = Path(
    "artifacts/top5-upstream-refinement-a025/"
    "compact_error_diagnostic.csv"
)

BINS = [
    (0.00, 0.05, "0.00-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.30, "0.20-0.30"),
    (0.30, float("inf"), ">=0.30"),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def bin_name(margin):
    for lo, hi, name in BINS:
        if lo <= margin < hi:
            return name
    raise RuntimeError(margin)


rows = []

with INPUT.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        r["label"] = int(r["label"])
        r["compact_probability"] = float(r["compact_probability"])
        r["refined_probability"] = float(r["refined_probability"])
        r["compact_wrong_margin"] = float(r["compact_wrong_margin"])
        r["true_direction_shift"] = float(r["true_direction_shift"])
        r["top5_present_count"] = int(r["top5_present_count"])
        r["top5_context_count"] = int(r["top5_context_count"])
        r["margin_bin"] = bin_name(r["compact_wrong_margin"])
        rows.append(r)

print("Compact-error APKs =", len(rows))
print()

print(
    "margin_bin    N   corr still "
    "corr_rate toward_truth mean_shift med_shift "
    "mean_ctx benign malware"
)

for _, _, name in BINS:
    rr = [r for r in rows if r["margin_bin"] == name]

    if not rr:
        continue

    corrected = sum(
        r["transition"] == "CORRECTED"
        for r in rr
    )

    still_wrong = sum(
        r["transition"] == "STILL_WRONG"
        for r in rr
    )

    toward = sum(
        r["true_direction_shift"] > 0
        for r in rr
    )

    shifts = [
        r["true_direction_shift"]
        for r in rr
    ]

    contexts = [
        r["top5_context_count"]
        for r in rr
    ]

    benign = sum(r["label"] == 0 for r in rr)
    malware = sum(r["label"] == 1 for r in rr)

    print(
        f"{name:11s} "
        f"{len(rr):4d} "
        f"{corrected:4d} "
        f"{still_wrong:5d} "
        f"{corrected/len(rr):8.2%} "
        f"{toward/len(rr):11.2%} "
        f"{mean(shifts):+10.4f} "
        f"{median(shifts):+9.4f} "
        f"{mean(contexts):8.2f} "
        f"{benign:6d} "
        f"{malware:7d}"
    )


print()
print("=== WITH ACTUAL TOP5 CALLER CONTEXT ONLY ===")

context_rows = [
    r for r in rows
    if r["top5_context_count"] > 0
]

for _, _, name in BINS:
    rr = [
        r for r in context_rows
        if r["margin_bin"] == name
    ]

    if not rr:
        continue

    corrected = sum(
        r["transition"] == "CORRECTED"
        for r in rr
    )

    toward = sum(
        r["true_direction_shift"] > 0
        for r in rr
    )

    shifts = [
        r["true_direction_shift"]
        for r in rr
    ]

    print(
        f"{name:11s} "
        f"N={len(rr):4d} "
        f"corrected={corrected:3d} "
        f"rate={corrected/len(rr):7.2%} "
        f"toward_truth={toward/len(rr):7.2%} "
        f"mean_shift={mean(shifts):+.4f}"
    )


print()
print("=== LABEL x MARGIN ===")

for label, label_name in (
    (0, "BENIGN_FP"),
    (1, "MALWARE_FN"),
):
    print()
    print(label_name)

    for _, _, name in BINS:
        rr = [
            r for r in rows
            if r["label"] == label
            and r["margin_bin"] == name
        ]

        if not rr:
            continue

        corrected = sum(
            r["transition"] == "CORRECTED"
            for r in rr
        )

        toward = sum(
            r["true_direction_shift"] > 0
            for r in rr
        )

        shifts = [
            r["true_direction_shift"]
            for r in rr
        ]

        print(
            f"  {name:11s} "
            f"N={len(rr):4d} "
            f"corrected={corrected:3d} "
            f"rate={corrected/len(rr):7.2%} "
            f"toward={toward/len(rr):7.2%} "
            f"shift={mean(shifts):+.4f}"
        )


print()
print("=== HIGH-CONFIDENCE WRONG CASES ===")

hard = sorted(
    [
        r for r in rows
        if r["compact_wrong_margin"] >= 0.20
    ],
    key=lambda r: r["compact_wrong_margin"],
    reverse=True,
)

direction = Counter(
    "TOWARD_TRUTH"
    if r["true_direction_shift"] > 0
    else "AWAY_FROM_TRUTH"
    for r in hard
)

print("margin >= 0.20:", len(hard))
print("toward truth:", direction["TOWARD_TRUTH"])
print("away truth  :", direction["AWAY_FROM_TRUTH"])

print()
print(
    "Top 20 most confident Compact errors:"
)

for r in hard[:20]:
    label_name = (
        "BENIGN_FP"
        if r["label"] == 0
        else "MALWARE_FN"
    )

    print(
        f"{r['year']} "
        f"{label_name:10s} "
        f"margin={r['compact_wrong_margin']:.3f} "
        f"C={r['compact_probability']:.3f} "
        f"R={r['refined_probability']:.3f} "
        f"shift={r['true_direction_shift']:+.3f} "
        f"ctx={r['top5_context_count']} "
        f"{r['transition']} "
        f"{r['sha256'][:16]}"
    )
