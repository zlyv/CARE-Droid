#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gz_csv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def q(values, quantile):
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "artifacts/frozen-semantic-2018-2021/predictions.csv"
        ),
    )
    parser.add_argument(
        "--semantic-future",
        type=Path,
        default=Path(
            "artifacts/clean-local-context-future/"
            "semantic_relation_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/frozen-semantic-2018-2021/"
            "semantic-near-miss-analysis"
        ),
    )
    args = parser.parse_args()

    semantic_rows = read_gz_csv(args.semantic_future)
    support_names = sorted(
        name
        for name in semantic_rows[0]
        if name.startswith("semrel_support__")
    )
    support_names = [
        name
        for name in support_names
        if name.split("__")[1] != name.split("__")[2]
    ]

    semantic_by_sha = {}
    for row in semantic_rows:
        sha = row["sha256"].strip().casefold()
        pair_count = 0
        support_sum = 0.0
        anchors = set()
        for name in support_names:
            value = float(row[name])
            support_sum += value
            if value > 0.0:
                pair_count += 1
                anchors.add(name.split("__")[1])
        semantic_by_sha[sha] = {
            "raw_nonzero": int(pair_count > 0),
            "pair_count": pair_count,
            "support_sum": support_sum,
            "active_anchor_count": len(anchors),
        }

    prediction_rows = read_csv(args.predictions)
    grouped = defaultdict(list)
    labels = {}
    for row in prediction_rows:
        variant = row["variant"]
        if variant not in ("compact_only", "compact_plus_support150"):
            continue
        key = (row["split_name"], variant, row["sha256"].strip().casefold())
        grouped[key].append(float(row["probability"]))
        labels[(row["split_name"], row["sha256"].strip().casefold())] = int(
            row["label"]
        )

    probabilities = {
        key: float(np.mean(values))
        for key, values in grouped.items()
    }

    records = []
    for split_name, sha in sorted(labels):
        if (split_name, "compact_only", sha) not in probabilities:
            continue
        if (split_name, "compact_plus_support150", sha) not in probabilities:
            continue
        if sha not in semantic_by_sha:
            continue
        if not semantic_by_sha[sha]["raw_nonzero"]:
            continue

        year_match = re.search(r"test_(20\d{2})$", split_name)
        year = int(year_match.group(1)) if year_match else split_name
        label = labels[(split_name, sha)]
        compact_probability = probabilities[(split_name, "compact_only", sha)]
        semantic_probability = probabilities[
            (split_name, "compact_plus_support150", sha)
        ]
        probability_delta = semantic_probability - compact_probability
        signed_delta = (2 * label - 1) * probability_delta
        compact_prediction = int(compact_probability >= 0.5)
        semantic_prediction = int(semantic_probability >= 0.5)
        compact_correct = int(compact_prediction == label)
        semantic_correct = int(semantic_prediction == label)

        if not compact_correct and semantic_correct:
            category = "corrected"
        elif compact_correct and not semantic_correct:
            category = "degraded"
        elif not compact_correct and signed_delta > 1e-15:
            category = "wrong_toward_truth_but_no_flip"
        elif not compact_correct:
            category = "wrong_away_or_flat"
        elif signed_delta > 1e-15:
            category = "correct_toward_truth"
        elif signed_delta < -1e-15:
            category = "correct_away_from_truth"
        else:
            category = "correct_flat"

        if compact_correct:
            required_delta = 0.0
        elif label == 1:
            required_delta = max(0.0, 0.5 - compact_probability)
        else:
            required_delta = max(0.0, compact_probability - 0.5)

        record = {
            "year": year,
            "split_name": split_name,
            "sha256": sha,
            "label": label,
            "compact_probability": compact_probability,
            "semantic_probability": semantic_probability,
            "probability_delta": probability_delta,
            "signed_delta": signed_delta,
            "required_delta_to_flip": required_delta,
            "compact_prediction": compact_prediction,
            "semantic_prediction": semantic_prediction,
            "compact_correct": compact_correct,
            "semantic_correct": semantic_correct,
            "category": category,
            **semantic_by_sha[sha],
        }
        records.append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "records.csv", records)

    print("======================================")
    print("Semantic nonzero near-miss analysis")
    print("======================================")
    print("nonzero samples analyzed =", len(records))

    print()
    print("===== Overall categories =====")
    counts = Counter(row["category"] for row in records)
    for category, count in counts.most_common():
        print(f"{category:<34} {count:4d}")

    print()
    print("===== Categories by year =====")
    by_year = defaultdict(Counter)
    for row in records:
        by_year[row["year"]][row["category"]] += 1
    for year in sorted(by_year):
        print("---", year, "---")
        for category, count in by_year[year].most_common():
            print(f"{category:<34} {count:4d}")

    print()
    print("===== Wrong Compact predictions =====")
    wrong = [row for row in records if not row["compact_correct"]]
    for category in (
        "corrected",
        "wrong_toward_truth_but_no_flip",
        "wrong_away_or_flat",
    ):
        current = [row for row in wrong if row["category"] == category]
        if not current:
            continue
        print(
            category,
            "n=", len(current),
            "mean_compact_p=", f"{np.mean([r['compact_probability'] for r in current]):.6f}",
            "median_required=", f"{q([r['required_delta_to_flip'] for r in current], 0.5):.6f}",
            "mean_signed_delta=", f"{np.mean([r['signed_delta'] for r in current]):+.6f}",
            "mean_pairs=", f"{np.mean([r['pair_count'] for r in current]):.3f}",
            "mean_anchors=", f"{np.mean([r['active_anchor_count'] for r in current]):.3f}",
        )

    print()
    print("===== Wrong-toward samples by Compact margin =====")
    toward = [
        row
        for row in wrong
        if row["category"] == "wrong_toward_truth_but_no_flip"
    ]
    for upper in (0.05, 0.10, 0.20, 1.0):
        current = [
            row
            for row in toward
            if row["required_delta_to_flip"] <= upper
        ]
        print(
            "required_delta <=",
            upper,
            "n=",
            len(current),
            "of toward=",
            len(toward),
        )

    print()
    print("===== Nonzero semantic strength =====")
    for category in sorted(counts):
        current = [row for row in records if row["category"] == category]
        print(
            category,
            "n=", len(current),
            "pair_q50=", f"{q([r['pair_count'] for r in current], 0.5):.3f}",
            "pair_q90=", f"{q([r['pair_count'] for r in current], 0.9):.3f}",
            "anchor_q50=", f"{q([r['active_anchor_count'] for r in current], 0.5):.3f}",
            "signed_q50=", f"{q([r['signed_delta'] for r in current], 0.5):+.6f}",
        )

    summary_rows = []
    for year in sorted(by_year):
        for category in sorted(by_year[year]):
            current = [
                row
                for row in records
                if row["year"] == year and row["category"] == category
            ]
            summary_rows.append({
                "year": year,
                "category": category,
                "n": len(current),
                "mean_compact_probability": float(
                    np.mean([row["compact_probability"] for row in current])
                ),
                "mean_probability_delta": float(
                    np.mean([row["probability_delta"] for row in current])
                ),
                "mean_signed_delta": float(
                    np.mean([row["signed_delta"] for row in current])
                ),
                "median_required_delta_to_flip": q(
                    [row["required_delta_to_flip"] for row in current], 0.5
                ),
                "mean_pair_count": float(
                    np.mean([row["pair_count"] for row in current])
                ),
                "mean_active_anchor_count": float(
                    np.mean([row["active_anchor_count"] for row in current])
                ),
            })
    write_csv(args.output / "summary.csv", summary_rows)
    print()
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
