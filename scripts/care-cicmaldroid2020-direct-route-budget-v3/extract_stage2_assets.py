#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

from common import OUTPUT, PROJECT, dump_json, read_csv, read_json


def run(command):
    print("CIC_STAGE2_COMMAND", " ".join(str(x) for x in command), flush=True)
    environment = dict(os.environ)
    project_src = str((PROJECT / "src").resolve())
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = project_src + (os.pathsep + existing if existing else "")
    print("CIC_STAGE2_PYTHONPATH", environment["PYTHONPATH"], flush=True)
    subprocess.run([str(x) for x in command], check=True, env=environment)


def compact_success_ids(path):
    if not path.is_file(): return set()
    return {str(row.get("sha256", "")).strip().casefold() for row in read_csv(path)
            if row.get("status") == "OK" and row.get("sha256")}


def endpoint_success_ids(path):
    result = set()
    if not path.is_file(): return result
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                if row.get("status") == "OK" and row.get("sha256"):
                    result.add(str(row["sha256"]).strip().casefold())
    except (OSError, EOFError):
        # Preserve complete rows recovered before a damaged gzip tail.  A
        # partial cache is resumed unless the frozen coverage gate is met.
        pass
    return result


def main():
    builders = read_json(OUTPUT / "stage2_builders.json")
    lineage = OUTPUT / "routed_manifest_label_free.csv"
    routed = read_csv(lineage)
    expected = len(routed)
    if expected <= 0:
        raise RuntimeError("BLOCKED_BY_CONTRACT: empty Top50 routed cohort")
    minimum_complete = max(1, int(round(expected * .75)))
    compact_output = OUTPUT / "cache/compact"
    endpoint_output = OUTPUT / "cache/endpoint"
    compact_path = compact_output / "features.csv"
    endpoint_path = endpoint_output / "endpoint_centered_tensor_cache.jsonl.gz"
    compact_ids = compact_success_ids(compact_path)
    if len(compact_ids) == len(routed):
        print(f"CIC_COMPACT_RESUME_COMPLETE ok={len(compact_ids)}", flush=True)
    else:
        run([sys.executable, builders["compact_builder"], "--lineage", lineage,
             "--mapping", builders["mapping"], "--priority", builders["priority"],
             "--output", compact_output, "--progress-every", "5", "--min-free-gb", "1"])
        compact_ids = compact_success_ids(compact_path)

    endpoint_ids = endpoint_success_ids(endpoint_path)
    endpoint_return_tolerated = False
    if len(endpoint_ids) == len(routed):
        print(f"CIC_ENDPOINT_RESUME_COMPLETE ok={len(endpoint_ids)}", flush=True)
    else:
        try:
            run([sys.executable, builders["endpoint_builder"], "--lineage", lineage,
                 "--mapping", builders["mapping"], "--priority", builders["priority"],
                 "--edge-semantics", builders["edge_semantics"], "--output", endpoint_output,
                 "--limit", "0", "--resume", "--upstream-hops", "2", "--downstream-hops", "1",
                 "--K-in", "8", "--K-out", "4", "--endpoint-cap", "256", "--node-cap", "2048",
                 "--edge-cap", "4096", "--reachability-max-states", "100000",
                 "--min-free-gb", "1", "--progress-every", "2"])
        except subprocess.CalledProcessError:
            endpoint_ids = endpoint_success_ids(endpoint_path)
            overlap = compact_ids & endpoint_ids
            if len(overlap) < minimum_complete:
                raise
            endpoint_return_tolerated = True
            print(f"CIC_ENDPOINT_NONZERO_TOLERATED endpoint_ok={len(endpoint_ids)} overlap={len(overlap)}", flush=True)
        endpoint_ids = endpoint_success_ids(endpoint_path)

    overlap = compact_ids & endpoint_ids
    if len(overlap) < minimum_complete:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: insufficient Top50 Stage-II asset intersection compact={len(compact_ids)} endpoint={len(endpoint_ids)} overlap={len(overlap)}")
    dump_json(OUTPUT / "stage2_asset_audit.json", {"status": "PASS", "routed": expected,
        "compact_ok": len(compact_ids), "endpoint_ok": len(endpoint_ids), "complete_intersection": len(overlap),
        "endpoint_nonzero_return_tolerated": endpoint_return_tolerated,
        "coverage_gate": f"at least 75% ({minimum_complete}/{expected}) Top50 APKs complete",
        "fallback_rule": "budget-routed APK without complete assets retains Stage-I probability"})
    print(f"CIC_STAGE2_ASSETS_DONE compact_ok={len(compact_ids)} endpoint_ok={len(endpoint_ids)} overlap={len(overlap)}", flush=True)


if __name__ == "__main__": main()

