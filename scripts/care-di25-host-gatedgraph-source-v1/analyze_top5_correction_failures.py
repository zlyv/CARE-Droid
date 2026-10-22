#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

PAIRED = Path(
    "artifacts/top5-upstream-refinement-a025/"
    "top5_upstream_paired_apk.csv"
)

COMPACT = Path(
    "artifacts/p1-features/features.csv"
)

UPSTREAM = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

OUTPUT = Path(
    "artifacts/top5-upstream-refinement-a025/"
    "compact_error_diagnostic.csv"
)


def key(x):
    return str(x).strip().casefold()


def fnum(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    return statistics.median(xs) if xs else 0.0


# -------------------------------------------------------
# Load Compact
# -------------------------------------------------------
compact = {}

with COMPACT.open(
    newline="",
    encoding="utf-8",
) as f:
    for r in csv.DictReader(f):
        compact[key(r["sha256"])] = r


# -------------------------------------------------------
# Load upstream
# -------------------------------------------------------
upstream = {}

with gzip.open(
    UPSTREAM,
    "rt",
    newline="",
    encoding="utf-8",
) as f:
    for r in csv.DictReader(f):
        upstream[key(r["sha256"])] = r


# -------------------------------------------------------
# Load paired predictions
# -------------------------------------------------------
rows = []

with PAIRED.open(
    newline="",
    encoding="utf-8",
) as f:
    for r in csv.DictReader(f):
        sha = key(r["sha256"])

        if sha not in compact or sha not in upstream:
            raise RuntimeError(
                f"missing features for {sha}"
            )

        y = int(r["label"])
        pc = float(r["compact_probability"])
        pr = float(r["refined_probability"])

        present_roles = []
        context_roles = []

        for role in TOP5:
            present = (
                fnum(
                    compact[sha].get(
                        f"role_node__{role}",
                        0,
                    )
                )
                > 0
            )

            if not present:
                continue

            present_roles.append(role)

            caller = fnum(
                upstream[sha].get(
                    f"up_role__{role}__caller_log",
                    0,
                )
            )

            if caller > 0:
                context_roles.append(role)

        # Positive means refined probability moved
        # toward the true class.
        true_direction_shift = (
            (2 * y - 1) * (pr - pc)
        )

        # For a Compact-wrong APK this is positive.
        # Larger = Compact was more confidently wrong.
        compact_wrong_margin = (
            (1 - 2 * y) * (pc - 0.5)
        )

        if not present_roles:
            availability = "TOP5_ABSENT"
        elif not context_roles:
            availability = "TOP5_PRESENT_NO_CONTEXT"
        else:
            availability = "TOP5_PRESENT_WITH_CONTEXT"

        row = {
            "sha256": sha,
            "year": r["year"],
            "label": y,
            "transition": r["transition"],
            "compact_probability": pc,
            "refined_probability": pr,
            "probability_delta": pr - pc,
            "true_direction_shift": true_direction_shift,
            "compact_wrong_margin": compact_wrong_margin,
            "top5_present_count": len(present_roles),
            "top5_context_count": len(context_roles),
            "availability": availability,
            "present_roles": "|".join(present_roles),
            "context_roles": "|".join(context_roles),
        }

        rows.append(row)


# -------------------------------------------------------
# Restrict to APKs Compact originally got wrong.
# -------------------------------------------------------
wrong = [
    r
    for r in rows
    if r["transition"]
    in {"CORRECTED", "STILL_WRONG"}
]

corrected = [
    r for r in wrong
    if r["transition"] == "CORRECTED"
]

still_wrong = [
    r for r in wrong
    if r["transition"] == "STILL_WRONG"
]


print("======================================")
print("COMPACT ERROR COHORT")
print("======================================")
print("Compact wrong total =", len(wrong))
print("CORRECTED           =", len(corrected))
print("STILL_WRONG         =", len(still_wrong))

if wrong:
    print(
        "overall correction rate =",
        f"{len(corrected)/len(wrong):.2%}"
    )


# -------------------------------------------------------
# Availability
# -------------------------------------------------------
print()
print("======================================")
print("BY TOP5 / CALLER AVAILABILITY")
print("======================================")

for name in (
    "TOP5_ABSENT",
    "TOP5_PRESENT_NO_CONTEXT",
    "TOP5_PRESENT_WITH_CONTEXT",
):
    rr = [
        r for r in wrong
        if r["availability"] == name
    ]

    c = sum(
        r["transition"] == "CORRECTED"
        for r in rr
    )

    sw = sum(
        r["transition"] == "STILL_WRONG"
        for r in rr
    )

    rate = c / len(rr) if rr else 0

    print(
        f"{name:28s} "
        f"N={len(rr):4d} "
        f"corrected={c:3d} "
        f"still_wrong={sw:4d} "
        f"rate={rate:7.2%}"
    )


# -------------------------------------------------------
# Label-conditioned
# -------------------------------------------------------
print()
print("======================================")
print("LABEL-CONDITIONED COMPACT ERRORS")
print("======================================")

for label, label_name in (
    (0, "BENIGN_FP"),
    (1, "MALWARE_FN"),
):
    rr = [
        r for r in wrong
        if r["label"] == label
    ]

    c = sum(
        r["transition"] == "CORRECTED"
        for r in rr
    )

    print(
        f"{label_name:12s} "
        f"N={len(rr):4d} "
        f"corrected={c:3d} "
        f"still_wrong={len(rr)-c:4d} "
        f"rate={c/len(rr) if rr else 0:7.2%}"
    )


# -------------------------------------------------------
# Number of available contexts
# -------------------------------------------------------
print()
print("======================================")
print("BY NUMBER OF TOP5 CALLER CONTEXTS")
print("======================================")

for n in range(0, 6):
    rr = [
        r for r in wrong
        if r["top5_context_count"] == n
    ]

    if not rr:
        continue

    c = sum(
        r["transition"] == "CORRECTED"
        for r in rr
    )

    print(
        f"context_count={n} "
        f"N={len(rr):4d} "
        f"corrected={c:3d} "
        f"still_wrong={len(rr)-c:4d} "
        f"rate={c/len(rr):7.2%}"
    )


# -------------------------------------------------------
# Probability mechanics
# -------------------------------------------------------
print()
print("======================================")
print("PROBABILITY MECHANICS")
print("======================================")

for transition in (
    "CORRECTED",
    "STILL_WRONG",
):
    rr = [
        r for r in wrong
        if r["transition"] == transition
    ]

    shifts = [
        r["true_direction_shift"]
        for r in rr
    ]

    margins = [
        r["compact_wrong_margin"]
        for r in rr
    ]

    toward_truth = sum(
        x > 0 for x in shifts
    )

    print()
    print(transition, "N=", len(rr))

    print(
        "Compact wrong margin "
        f"mean={mean(margins):.4f} "
        f"median={median(margins):.4f}"
    )

    print(
        "true-direction shift "
        f"mean={mean(shifts):+.4f} "
        f"median={median(shifts):+.4f}"
    )

    print(
        "moved toward truth =",
        f"{toward_truth}/{len(rr)} "
        f"({toward_truth/len(rr) if rr else 0:.2%})"
    )


# -------------------------------------------------------
# Per-role diagnostic
# Role cohorts overlap -- NOT independent contribution.
# -------------------------------------------------------
print()
print("======================================")
print("PER ROLE AMONG COMPACT ERRORS")
print("======================================")

for role in TOP5:
    present = [
        r
        for r in wrong
        if role in r["present_roles"].split("|")
    ]

    context = [
        r
        for r in present
        if role in r["context_roles"].split("|")
    ]

    c = sum(
        r["transition"] == "CORRECTED"
        for r in present
    )

    print(
        f"{role:32s} "
        f"present={len(present):4d} "
        f"context={len(context):4d} "
        f"corrected={c:3d} "
        f"corr_rate={c/len(present) if present else 0:7.2%}"
    )


# -------------------------------------------------------
# Per-year diagnostic
# -------------------------------------------------------
print()
print("======================================")
print("PER YEAR COMPACT ERRORS")
print("======================================")

for year in (
    "2018",
    "2019",
    "2020",
    "2021",
):
    rr = [
        r
        for r in wrong
        if str(r["year"]) == year
    ]

    c = sum(
        r["transition"] == "CORRECTED"
        for r in rr
    )

    print(
        f"{year} "
        f"N={len(rr):4d} "
        f"corrected={c:3d} "
        f"still_wrong={len(rr)-c:4d} "
        f"rate={c/len(rr) if rr else 0:7.2%}"
    )


# -------------------------------------------------------
# Save error cohort for later case studies.
# -------------------------------------------------------
with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(wrong[0]),
    )
    writer.writeheader()
    writer.writerows(wrong)

print()
print("saved:", OUTPUT)
