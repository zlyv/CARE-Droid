#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

from common import discover_assets, norm, read_csv

ARMS = ("LR", "RF", "XGB", "SVM")


def main() -> None:
    assets = discover_assets(); temporal = read_csv(assets["temporal"])
    if len(temporal) != 2788:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: temporal rows={len(temporal)} expected=2788")
    years = Counter(int(float(row["year"])) for row in temporal)
    if years != Counter({2017: 799, 2018: 793, 2019: 399, 2020: 397, 2021: 400}):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: temporal cohort changed {dict(years)}")
    for key, expected in (("source_family", 799 * 5), ("future_family", 1989 * 5)):
        rows = read_csv(assets[key])
        counts = Counter(norm(row.get("arm")) for row in rows)
        if len(rows) != expected or any(counts[norm(arm)] != expected // 5 for arm in (*ARMS, "MEAN4")):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: {key} long-form predictions changed rows={len(rows)} counts={dict(counts)}")
    neighbors = read_csv(assets["neighbors"])
    primary = [row for row in neighbors if norm(row.get("representation")) == "hidden" and norm(row.get("distance")) == "cosine"]
    if len(primary) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: hidden-cosine routed rows={len(primary)} expected=200")
    print("STATIC_IDENTIFIABILITY_CEILING_PREFLIGHT_PASS")
    print({"rows": len(temporal), "years": dict(sorted(years.items())), "routed_hidden_rows": len(primary),
           "assets": {key: str(path) for key, path in assets.items()}})


if __name__ == "__main__": main()
