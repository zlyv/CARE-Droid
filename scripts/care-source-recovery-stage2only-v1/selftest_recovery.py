#!/usr/bin/env python3
"""Verify immutable bundled recovery assets before a long run."""

import csv
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    with (root / "assets" / "source_reference_799.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 799
    assert len({row["sha256"].strip().lower() for row in rows}) == 799
    assert sum(int(row["label"]) == 0 for row in rows) == 400
    assert sum(int(row["label"]) == 1 for row in rows) == 399
    data = np.load(root / "assets" / "multiscale_features.npz", allow_pickle=False)
    assert "sha256" in data.files
    assert len(data["sha256"]) >= 799
    print("Recovery asset self-test passed")


if __name__ == "__main__":
    main()
