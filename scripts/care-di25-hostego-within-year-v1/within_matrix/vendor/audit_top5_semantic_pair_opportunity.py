#!/usr/bin/env python3

import csv
import gzip
from itertools import combinations
from collections import Counter
from pathlib import Path


TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

FULL = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

TEST20 = Path(
    "artifacts/top5-semantic-org-test20/"
    "sensitive_upstream_features.csv.gz"
)


def k(x):
    return str(x).strip().casefold()


def f(x):
    try:
        return float(x or 0)
    except Exception:
        return 0.0


# ============================================================
# Existing all-role cache.
# Used ONLY as an availability/opportunity diagnostic.
# ============================================================

full = {}

with gzip.open(
    FULL,
    "rt",
    newline="",
    encoding="utf-8",
) as fh:
    for r in csv.DictReader(fh):
        if r.get("status") != "OK":
            continue

        full[k(r["sha256"])] = r


def context_roles(r):
    return {
        role
        for role in TOP5
        if f(
            r.get(
                f"up_role__{role}__caller_log",
                0,
            )
        ) > 0
    }


# ============================================================
# A. 2017 source-year opportunity.
# ============================================================

source = [
    r
    for r in full.values()
    if str(r.get("metadata_year", "")).strip() == "2017"
]

print("========================================")
print("2017 TOP5 CALLER-CONTEXT OPPORTUNITY")
print("========================================")
print("2017 OK APKs =", len(source))

dist = Counter(
    len(context_roles(r))
    for r in source
)

for n in range(6):
    count = dist[n]
    print(
        f"context_role_count={n} "
        f"N={count:4d} "
        f"rate={count/len(source) if source else 0:7.2%}"
    )

multi = sum(
    len(context_roles(r)) >= 2
    for r in source
)

triple = sum(
    len(context_roles(r)) >= 3
    for r in source
)

print()
print(
    "APK with >=2 Top5 caller contexts =",
    multi,
    f"({multi/len(source) if source else 0:.2%})",
)

print(
    "APK with >=3 Top5 caller contexts =",
    triple,
    f"({triple/len(source) if source else 0:.2%})",
)


print()
print("========================================")
print("2017 PAIR OPPORTUNITY")
print("========================================")

pair_opp = {}

for a, b in combinations(TOP5, 2):
    n = sum(
        a in context_roles(r)
        and b in context_roles(r)
        for r in source
    )

    pair_opp[(a, b)] = n

    print(
        f"{a:30s} + "
        f"{b:30s} "
        f"N={n:4d} "
        f"rate={n/len(source) if source else 0:7.2%}"
    )


# ============================================================
# B. For the EXACT same test20 APKs:
#
# opportunity:
#   both roles have caller context in old full cache
#
# realization:
#   new Top5-only semantic extractor finds >=1 exact
#   shared internal caller.
#
# This separates sample sparsity from definition sparsity.
# ============================================================

with gzip.open(
    TEST20,
    "rt",
    newline="",
    encoding="utf-8",
) as fh:
    test20 = [
        r
        for r in csv.DictReader(fh)
        if r.get("status") == "OK"
    ]


print()
print("========================================")
print("TEST20: OPPORTUNITY vs EXACT SHARED CALLER")
print("========================================")
print("test20 OK =", len(test20))

total_opportunity = 0
total_realized = 0

for a, b in combinations(TOP5, 2):
    opportunity = 0
    realized = 0

    feature = (
        f"sem_pair__{a}__{b}"
        "__shared_caller_log"
    )

    for new_r in test20:
        sha = k(new_r["sha256"])
        old_r = full.get(sha)

        if old_r is None:
            continue

        cr = context_roles(old_r)

        if a in cr and b in cr:
            opportunity += 1

        if f(new_r.get(feature, 0)) > 0:
            realized += 1

    total_opportunity += opportunity
    total_realized += realized

    print(
        f"{a:30s} + "
        f"{b:30s} "
        f"opportunity={opportunity:2d} "
        f"exact_shared={realized:2d}"
    )


print()
print("TOTAL pair opportunities =", total_opportunity)
print("TOTAL exact shared       =", total_realized)

if total_opportunity:
    print(
        "exact/opportunity ratio =",
        f"{total_realized/total_opportunity:.2%}"
    )


# ============================================================
# C. Same-APK context-role distribution for test20.
# ============================================================

print()
print("========================================")
print("TEST20 CONTEXT-ROLE COUNTS")
print("========================================")

test_dist = Counter()

for new_r in test20:
    old_r = full.get(k(new_r["sha256"]))

    if old_r is None:
        continue

    test_dist[len(context_roles(old_r))] += 1

for n in range(6):
    print(
        f"context_role_count={n} "
        f"N={test_dist[n]}"
    )
