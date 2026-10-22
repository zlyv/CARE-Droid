#!/usr/bin/env python3
"""Static source-2017 argument-effect experiment with resumable APK extraction."""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from argument_core import SCHEMA
from argument_io import (NAME, protocol, read_json, read_csv, write_json, write_csv,
                         source_rows, fold_representations, verify_inputs, status, digest)
from argument_extract import run_scans, scan_one, validate_cache


def prepare(project, reference, lineage, out, resume):
    if out == project or out in project.parents or out == reference or reference in out.parents:
        raise ValueError("choose a separate, new experiment output directory")
    config, paths = protocol(project, reference, lineage, out)
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "androguard", "loguru"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    missing = [name for name, value in versions.items() if value == "NOT_INSTALLED"]
    if missing:
        raise RuntimeError("missing dependencies " + repr(missing) + "; use requirements.txt in an isolated environment")
    version = re.match(r"^4\.(\d+)", versions["androguard"])
    if version is None or int(version.group(1)) < 1:
        raise RuntimeError("this decoder requires Androguard >=4.1,<5; no automatic package upgrades are performed")
    config["dependency_versions"] = versions
    from argument_io import signature
    config.pop("contract_sha256")
    config["contract_sha256"] = signature(config)
    if out.exists():
        if not resume or not (out / "protocol.json").is_file():
            raise RuntimeError("output exists; preserve it and use --resume only for this exact contract")
        if read_json(out / "protocol.json") != json.loads(json.dumps(config)):
            raise RuntimeError("resume input/code/config/dependency fingerprints differ")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "protocol.json", config)
    return config, paths


def feasibility(reference, out, config):
    labels, routes = source_rows(reference)
    _, folds = fold_representations(reference, labels)
    records, rows = {}, []
    for sid in sorted(labels):
        path = out / f"cache/{sid}.json.gz"
        if not path.exists():
            raise RuntimeError("missing scan; exclusions and invented zero features are forbidden: " + sid)
        record = validate_cache(path, sid, reference / f"features/frozen_views/{sid}.json.gz", config["contract_sha256"])
        records[sid] = record
        summary = record["summary"]
        rows.append({"sha256": sid, "fold": folds[sid], "events": summary["events"],
                     "resolved": summary["resolved"], "classified": summary["classified"],
                     "hosts": len(record["hosts"]),
                     "ambiguous_hosts": sum(h["status"] == "AMBIGUOUS_HOST_ID" for h in record["hosts"]),
                     "unrecovered_targets": sum(h.get("unrecovered_targets", h["expected_targets"]) for h in record["hosts"]),
                     "routed_any_seed": int(any(str(routes[seed][sid]["routed"]).lower() in {"1", "true"} for seed in routes))})
    resolved = [r for r in rows if r["resolved"] > 0]
    routed = [r for r in resolved if r["routed_any_seed"]]
    total = sum(r["events"] for r in rows)
    fraction = sum(r["resolved"] for r in rows) / max(total, 1)
    limits = config["feasibility"]
    passed = (len(resolved) >= limits["minimum_resolved_apks"]
              and len(routed) >= limits["minimum_routed_resolved_apks"]
              and fraction >= limits["minimum_resolved_event_fraction"]
              and all(sum(r["fold"] == fold for r in resolved) >= 3 for fold in ("S1", "S2", "S3", "S4", "S5")))
    result = {"status": "INPUT_READY_FOR_FIXED_PROBE" if passed else "INSUFFICIENT_STATIC_INPUT",
              "apks": len(rows), "apks_with_command_events": sum(r["events"] > 0 for r in rows),
              "resolved_apks": len(resolved), "routed_resolved_apks": len(routed),
              "resolved_event_fraction": fraction, "command_events": total,
              "ambiguous_hosts": sum(r["ambiguous_hosts"] for r in rows),
              "unrecovered_targets": sum(r["unrecovered_targets"] for r in rows),
              "gate_uses_labels": False, "thresholds": limits}
    write_csv(out / "reports/extraction_coverage.csv", rows)
    write_json(out / "reports/feasibility.json", result, replace=True)
    return result


def finish(out, config, decision):
    verify_inputs(out)
    files = sorted((out / "cache").glob("*.json.gz"))
    files += sorted((out / "models").glob("*/*/*.json"))
    files += sorted((out / "reports").glob("*.csv"))
    write_json(out / "integrity/outputs_sha256.json", [digest(p) for p in files], replace=True)
    write_json(out / "integrity/after.json", {
        "status": "PASS", "verified_inputs": len(config["assets"]),
        "scope": "named source assets and code; APK contents verified before/after each uncached scan",
        "cached_apks_not_rehashed_on_resume": True,
    }, replace=True)
    lines = ["# Static API argument-effect experiment", "",
             "Decision: " + decision["status"], "",
             "Only source 2017 was read. No APK was installed or executed.",
             "Command categories describe static forms, not binary identity, runtime behavior or maliciousness.",
             "Frozen W0 and router were not retrained. This is conditional source evidence, not independent final validation.", ""]
    if (out / "reports/source_metrics.csv").exists():
        metrics = read_csv(out / "reports/source_metrics.csv")
        lines += ["| Arm | Mean end-to-end BA | Mean Net Fixed |", "|---|---:|---:|"]
        arms = sorted({r["arm"] for r in metrics})
        for arm in arms:
            part = [r for r in metrics if r["arm"] == arm and r["scope"] == "end_to_end"]
            if part:
                lines.append("| " + arm + " | " + f"{sum(float(r['BA']) for r in part)/len(part):.6f}" + " | "
                             + f"{sum(float(r['Net_Fixed']) for r in part)/len(part):.3f}" + " |")
    else:
        lines.append("The fixed readout probe was not run because the preregistered input gate did not pass.")
    target = out / "reports/final_report.md"
    text = "\n".join(lines) + "\n"
    # This file belongs to the current experiment; no formal report is replaced.
    target.write_text(text, encoding="utf-8")
    write_json(out / "reports/final_decision.json", decision, replace=True)
    status(out, "COMPLETE", "complete", decision=decision["status"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "smoke", "extract", "feasibility", "probe", "all", "scan-one"))
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--reference", type=Path)
    p.add_argument("--lineage", type=Path, default=Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv"))
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--smoke-count", type=int, default=2)
    p.add_argument("--sha")
    p.add_argument("--apk", type=Path)
    p.add_argument("--cache-output", type=Path)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    project = a.project.resolve()
    reference = (a.reference or project / "artifacts/care_api_manifest_fusion_2017_future_v1").resolve()
    out = (a.output or project / ("artifacts/" + NAME)).resolve()
    if not 1 <= a.workers <= 4 or a.smoke_count < 1:
        raise ValueError("workers must be 1..4 and smoke-count positive")
    if a.action == "scan-one":
        config = read_json(out / "protocol.json")
        labels, _ = source_rows(reference)
        paths = {r["sha256"].lower(): r for r in read_csv(config["lineage"])}
        if (a.sha not in labels or a.apk is None or a.cache_output is None
                or a.apk.resolve() != Path(paths[a.sha]["absolute_path"]).resolve()
                or str(reference) != config["reference"]):
            raise ValueError("worker identity mismatch")
        destination = a.cache_output.resolve()
        if destination.name != a.sha + ".json.gz" or destination.parent not in {out / "cache", out / "smoke/cache"}:
            raise ValueError("worker output must be its own cache entry")
        scan_one(a.sha, a.apk, reference / f"features/frozen_views/{a.sha}.json.gz",
                 destination, config["contract_sha256"])
        return
    config, lineage = prepare(project, reference, a.lineage.resolve(), out, a.resume)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another controller owns this output; do not launch concurrently")
        try:
            verify_inputs(out)
            write_json(out / "integrity/before.json", {"assets": config["assets"]})
            status(out, "RUNNING", a.action)
            if a.action == "preflight":
                write_json(out / "integrity/before.json", {"assets": config["assets"]})
                status(out, "PREPARED", "preflight")
                print("STATIC_ARGUMENT_PREFLIGHT_PASS source=799 future=0", flush=True)
                return
            if a.action in {"smoke", "all"}:
                run_scans(project, reference, out, lineage, config, a.workers, a.smoke_count)
                if a.action == "smoke":
                    verify_inputs(out); status(out, "SMOKE_COMPLETED", "smoke"); return
            if a.action in {"extract", "all"}:
                run_scans(project, reference, out, lineage, config, a.workers)
            if a.action in {"extract", "feasibility", "probe", "all"}:
                gate = feasibility(reference, out, config)
                if a.action in {"extract", "feasibility"}:
                    verify_inputs(out); status(out, "EXTRACTION_COMPLETED", "feasibility", gate=gate["status"]); return
                if gate["status"] != "INPUT_READY_FOR_FIXED_PROBE":
                    finish(out, config, {"status": "INSUFFICIENT_STATIC_INPUT", "probe_executed": False, "feasibility": gate})
                    return
                from argument_probe import run_probe
                status(out, "RUNNING", "source_readout")
                result = run_probe(reference, out, config)
                finish(out, config, result)
        except Exception as exc:
            status(out, "FAILED", a.action, error_type=type(exc).__name__, message=str(exc))
            raise


if __name__ == "__main__":
    main()
