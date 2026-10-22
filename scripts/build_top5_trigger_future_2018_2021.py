#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SPLIT_FILE = (
    ROOT
    / "artifacts/p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

LINEAGE_FILE = (
    ROOT
    / "artifacts/clean-local-context-future/"
    "lineage_2018_2021.csv"
)

COMPACT_FILE = (
    ROOT
    / "artifacts/p1-features/"
    "features.csv"
)

OUTPUT = (
    ROOT
    / "artifacts/top5-trigger-future-2018-2021"
)

YEARS = (2018, 2019, 2020, 2021)

TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def norm_sha(x):
    return str(x).strip().casefold()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def sha_field(rows):
    if not rows:
        raise RuntimeError("empty CSV")

    for name in (
        "sha256",
        "sha",
        "apk_sha256",
    ):
        if name in rows[0]:
            return name

    raise RuntimeError(
        "cannot find SHA field: "
        + str(list(rows[0]))
    )


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

splits = read_csv(SPLIT_FILE)
lineage = read_csv(LINEAGE_FILE)
compact = read_csv(COMPACT_FILE)

lsha = sha_field(lineage)
csha = sha_field(compact)

lineage_by_sha = {
    norm_sha(r[lsha]): r
    for r in lineage
}

compact_by_sha = {
    norm_sha(r[csha]): r
    for r in compact
}

print("future lineage rows =", len(lineage))
print("compact rows        =", len(compact))

print("\nlineage columns:")
print(list(lineage[0].keys()))

path_candidates = [
    k
    for k in lineage[0]
    if (
        "path" in k.lower()
        or k.lower() in {
            "apk",
            "apk_file",
            "file",
        }
    )
]

print("\npossible APK path fields:")
print(path_candidates)

summary_rows = []

all_triggered = []
all_missing_compact = []
all_missing_lineage = []

for year in YEARS:

    split_name = (
        f"train_fixed_2017__test_{year}"
    )

    test_rows = [
        r
        for r in splits
        if (
            r.get("split_name") == split_name
            and r.get("partition") == "test"
            and int(r.get("year", 0)) == year
        )
    ]

    # 去重
    target = {}

    for r in test_rows:
        sha = norm_sha(r["sha256"])
        target[sha] = r

    triggered = []
    missing_compact = []
    missing_lineage = []
    hit_counter = Counter()
    hit_count_counter = Counter()

    benign_trigger = 0
    malware_trigger = 0

    for sha, split_row in target.items():

        c = compact_by_sha.get(sha)
        l = lineage_by_sha.get(sha)

        if c is None:
            missing_compact.append({
                "sha256": sha,
                "year": year,
                "label": split_row.get("label", ""),
            })
            continue

        if l is None:
            missing_lineage.append({
                "sha256": sha,
                "year": year,
                "label": split_row.get("label", ""),
            })
            continue

        hits = []

        for role in TOP5:
            name = "role_node__" + role

            if safe_float(
                c.get(name, 0.0)
            ) > 0.0:
                hits.append(role)
                hit_counter[role] += 1

        hit_count_counter[
            len(hits)
        ] += 1

        if not hits:
            continue

        out = dict(l)

        # 强制规范关键字段
        out["sha256"] = sha
        out["year"] = str(year)
        out["label"] = split_row.get(
            "label",
            out.get("label", ""),
        )
        out["top5_trigger"] = "1"
        out["top5_hit_count"] = str(
            len(hits)
        )
        out["top5_hit_roles"] = "|".join(
            hits
        )

        triggered.append(out)

        label = str(
            split_row.get(
                "label",
                "",
            )
        ).strip().lower()

        if label in {
            "benign",
            "0",
            "0.0",
        }:
            benign_trigger += 1
        else:
            malware_trigger += 1

    # 保持原 lineage 字段，再附加3个路由字段
    fields = list(lineage[0].keys())

    for extra in (
        "top5_trigger",
        "top5_hit_count",
        "top5_hit_roles",
    ):
        if extra not in fields:
            fields.append(extra)

    out_file = (
        OUTPUT
        / f"lineage_{year}_top5_trigger.csv"
    )

    write_csv(
        out_file,
        triggered,
        fields,
    )

    write_csv(
        OUTPUT
        / f"missing_compact_{year}.csv",
        missing_compact,
        ["sha256", "year", "label"],
    )

    write_csv(
        OUTPUT
        / f"missing_lineage_{year}.csv",
        missing_lineage,
        ["sha256", "year", "label"],
    )

    all_triggered.extend(triggered)
    all_missing_compact.extend(
        missing_compact
    )
    all_missing_lineage.extend(
        missing_lineage
    )

    print()
    print("=" * 72)
    print("YEAR", year)
    print("=" * 72)

    print(
        "future test APKs =",
        len(target),
    )

    print(
        "Top5 triggered   =",
        len(triggered),
        f"({len(triggered)/max(1,len(target))*100:.2f}%)",
    )

    print(
        "benign triggered =",
        benign_trigger,
    )

    print(
        "malware triggered=",
        malware_trigger,
    )

    print(
        "missing compact  =",
        len(missing_compact),
    )

    print(
        "missing lineage  =",
        len(missing_lineage),
    )

    print("\nhit count:")
    for k in sorted(
        hit_count_counter
    ):
        print(
            k,
            hit_count_counter[k],
        )

    print("\nrole hits:")
    for role in TOP5:
        print(
            role,
            hit_counter[role],
        )

    summary_rows.append({
        "year": year,
        "test_apk_count": len(target),
        "triggered_count": len(triggered),
        "triggered_ratio": (
            len(triggered)
            / max(1, len(target))
        ),
        "benign_triggered": benign_trigger,
        "malware_triggered": malware_trigger,
        "missing_compact": len(
            missing_compact
        ),
        "missing_lineage": len(
            missing_lineage
        ),
    })


write_csv(
    OUTPUT / "summary.csv",
    summary_rows,
    list(summary_rows[0].keys()),
)

print()
print("=" * 72)
print("TOTAL")
print("=" * 72)

print(
    "triggered total =",
    len(all_triggered),
)

print(
    "missing compact =",
    len(all_missing_compact),
)

print(
    "missing lineage =",
    len(all_missing_lineage),
)

print(
    "saved to =",
    OUTPUT,
)
