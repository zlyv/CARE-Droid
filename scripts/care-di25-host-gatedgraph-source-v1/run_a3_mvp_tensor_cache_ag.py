#!/usr/bin/env python3
"""Build a low-disk A3-MVP tensor cache.

The Androguard/NetworkX graph and raw method signatures exist only while one
APK is processed.  Successful cache rows contain bounded hashed node IDs,
role IDs, edge indexes, and compact numeric features.  They never contain a
raw DEX, a serialized call graph, or a method-signature list.

This cache is an input artifact, not a trained model.  D/T/one-minus-F are
intentionally not written here: those values must be computed separately from
the outer training partition for each split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_a3_mvp_structure_dry_run_ag import (  # noqa: E402
    analyze_one,
    _suppress_loguru,
)
from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402
from tasa_droid.p2.graph_representation import GraphPolicy  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_successful_shas(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A killed process can leave a final partial line.  It is
                    # ignored; the next run will retry that APK.
                    continue
                if row.get("status") == "OK" and row.get("sha256"):
                    result.add(str(row["sha256"]).casefold())
    except (OSError, EOFError):
        return set()
    return result


def append_jsonl_gzip(path: Path, row: dict) -> None:
    with gzip.open(path, "at", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def append_failure(path: Path, row: dict) -> None:
    fields = [
        "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
        "metadata_label", "status", "error_type", "error_message",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="A3-MVP low-disk tensor cache")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--edge-semantics", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K-out", type=int, default=8)
    parser.add_argument("--P-cap", type=int, default=8)
    parser.add_argument("--E-cap", type=int, default=2000)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    _suppress_loguru()
    suppress_logs()
    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(args.edge_semantics)
    rules = list(mapping.get("rules", []))
    roles = list(mapping.get("roles", []))
    edge_rules = list(edge_mapping.get("edge_rules", []))

    excluded = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {
            row.get("sha256", "").casefold()
            for row in read_csv(args.failure_csv)
            if row.get("sha256")
        }
    rows = [
        row for row in read_csv(args.lineage)
        if row.get("sha256", "").casefold() not in excluded
    ]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "a3_mvp_tensor_cache.jsonl.gz"
    failure_path = output / "a3_mvp_tensor_failures.csv"
    already_done = existing_successful_shas(cache_path) if args.resume else set()
    rows = [row for row in rows if row.get("sha256", "").casefold() not in already_done]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no APK selected after exclusion/resume filtering")

    policy = GraphPolicy(
        H=max(0, args.H),
        K_out=max(1, args.K_out),
        P_cap=max(1, args.P_cap),
        E_cap=max(1, args.E_cap),
        max_states=max(100, args.max_states),
    )
    results = Counter()
    failure_count = 0
    print(
        f"A3-MVP tensor cache started: {len(rows)} APKs "
        f"(resume_skipped={len(already_done)})",
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        free_gb = shutil.disk_usage(output).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"disk guard stopped run: free_gb={free_gb:.3f} "
                f"< min_free_gb={args.min_free_gb:.3f}"
            )
        result = analyze_one(
            row, rules, priority, edge_rules, roles, policy, include_payload=True
        )
        results[result["status"]] += 1
        if result["status"] == "OK":
            append_jsonl_gzip(cache_path, result)
        else:
            failure_count += 1
            append_failure(failure_path, result)
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(
                f"processed {index}/{len(rows)} "
                f"ok={results['OK']} failed={results['FAILED']}",
                flush=True,
            )
        gc.collect()

    cache_sha = sha256_file(cache_path) if cache_path.exists() else None
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_A3_MVP_BOUNDED_TENSOR_CACHE",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "requested_apk_count": len(rows),
        "resume_skipped_success_count": len(already_done),
        "processed_apk_count": sum(results.values()),
        "status_counts": dict(results),
        "excluded_static_failure_count": len(excluded),
        "policy": {
            "H": policy.H,
            "K_out": policy.K_out,
            "P_cap": policy.P_cap,
            "E_cap": policy.E_cap,
            "max_states": policy.max_states,
        },
        "reliability_components": ["D", "T", "one_minus_F"],
        "reliability_status": "NOT_WRITTEN_TRAINING_FOLD_ONLY",
        "cache_sha256": cache_sha,
        "outputs": {
            "cache": str(cache_path),
            "failures": str(failure_path),
        },
        "blocking_reasons": [
            "training_fold_reliability_components_not_generated",
            "formal_training_config_not_resolved",
        ],
        "notes": [
            "raw DEX, raw graphs, and raw method signatures are never persisted",
            "cache rows contain only bounded hashed IDs and compact numeric arrays",
            "D/T/one-minus-F must be computed from the corresponding training fold",
            "this is an input cache stage, not model training",
        ],
    }
    (output / "a3_mvp_tensor_cache_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
