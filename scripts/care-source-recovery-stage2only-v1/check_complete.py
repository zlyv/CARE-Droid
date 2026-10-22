#!/usr/bin/env python3
"""Small completion checks used by the resumable recovery runner."""

import argparse
import csv
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("csv", "feature_csv", "jsonl_gz"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int, default=799)
    args = parser.parse_args()
    if not args.path.is_file():
        raise SystemExit(1)
    if args.kind in {"csv", "feature_csv"}:
        with args.path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if args.kind == "feature_csv":
            rows = [row for row in rows if str(row.get("status", "")).strip() == "OK"]
        shas = {str(row.get("sha256", row.get("sample_id", ""))).strip().lower() for row in rows}
        good = len(rows) == args.expected and len(shas) == args.expected and "" not in shas
    else:
        shas = set()
        count = 0
        with gzip.open(args.path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sha = str(row.get("sha256", row.get("sample_id", ""))).strip().lower()
                if sha:
                    shas.add(sha)
                count += 1
        good = count == args.expected and len(shas) == args.expected
    raise SystemExit(0 if good else 1)


if __name__ == "__main__":
    main()
