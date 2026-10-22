#!/usr/bin/env python3
"""Correct upstream dedup metadata that historically hard-coded 1.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta", type=float, required=True)
    args = parser.parse_args()
    path = args.output / "architecture.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["bounded_correction"] = float(args.max_delta)
    data["requested_max_delta_logit"] = float(args.max_delta)
    data["metadata_corrected_by"] = "care-di-budget-bound-selection-v1"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "REQUESTED_MAX_DELTA.txt").write_text(
        str(float(args.max_delta)) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

