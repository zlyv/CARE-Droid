#!/usr/bin/env python3
"""Minimal executable contract for the source-lineage repair gate."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    script = Path(__file__).with_name("audit_repair_source_features.py")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lineage = root / "lineage.csv"
        features = root / "features.csv"
        failures = root / "failures.csv"
        output = root / "out"
        ids = ["a" * 64, "b" * 64, "c" * 64]
        write(lineage, ["sha256"], [{"sha256": value} for value in ids])
        rows = [
            {"sha256": ids[0], "status": "OK", "mapped_api_node_count": "1", "external_node_count": "10", "sensitive_edge_count": "2"},
            {"sha256": ids[1], "status": "OK", "mapped_api_node_count": "2", "external_node_count": "20", "sensitive_edge_count": "3"},
            {"sha256": ids[1], "status": "OK", "mapped_api_node_count": "2", "external_node_count": "20", "sensitive_edge_count": "3"},
            {"sha256": ids[2], "status": "OK", "mapped_api_node_count": "3", "external_node_count": "30", "sensitive_edge_count": "4"},
        ]
        fields = list(rows[0])
        write(features, fields, rows)
        write(failures, ["sha256", "status"], [])
        subprocess.run([
            sys.executable, str(script), "--lineage", str(lineage),
            "--features", str(features), "--failures", str(failures),
            "--output", str(output), "--expected", "3",
            "--repair-identical-duplicates",
        ], check=True, capture_output=True, text=True)
        with features.open("r", encoding="utf-8", newline="") as handle:
            repaired = list(csv.DictReader(handle))
        assert len(repaired) == 3
        report = (output / "source_lineage_integrity_audit.txt").read_text(encoding="utf-8")
        assert "duplicate_extra_rows_before_repair: 1" in report
        assert "mapped_api_node_total: 6" in report
        assert "gate_passed: true" in report
    print("source lineage integrity audit self-test passed")


if __name__ == "__main__":
    main()
