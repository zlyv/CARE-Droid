#!/usr/bin/env python3
"""Run safe year-partitioned AndroZoo downloads for AZ_2017_2022_Random15K.

This companion to az_download_random15k.py uses its verified SHA256 download
routine, but assigns exclusive locks and status journals to individual years.
It also takes a shared lock on the legacy global lock, so it cannot overlap
with the original, whole-dataset downloader. The --all mode runs up to four
download workers total, with separate OS processes per year.

No old APKs or manifests are modified or deleted. Requires the original
az_download_random15k.py in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait, as_completed
from datetime import datetime, timezone
from pathlib import Path

from az_download_random15k import (
    CSV_FIELDS,
    DEFAULT_KEYFILE,
    DEFAULT_ROOT,
    fetch_one,
    load_rows,
    sha256_file,
)

YEARS = tuple(range(2017, 2023))


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def fail_if_legacy_active(root: Path):
    """Acquire a shared legacy lock; old downloader uses an exclusive lock."""
    legacy = (root / "logs" / "downloader.lock").open("a+")
    try:
        fcntl.flock(legacy.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        legacy.close()
        raise SystemExit(
            "Old az_download_random15k.py is still running. Stop it cleanly "
            "and verify all its curl children have exited before starting this version."
        )
    return legacy


def run_year(args) -> int:
    root = args.root.expanduser().resolve()
    manifest = root / "manifests" / "selected_all.csv"
    rows = load_rows(manifest)
    if len(rows) != 15000:
        raise SystemExit(f"Frozen manifest must have 15000 rows; found {len(rows)}")
    chosen = [r for r in rows if int(r["year"]) == args.year]
    expected_count = 5000 if args.year == 2017 else 2000
    if len(chosen) != expected_count:
        raise SystemExit(f"Year {args.year}: expected {expected_count} rows, found {len(chosen)}")
    manifest_digest = sha256_file(manifest)
    approx_gib = sum(int(r.get("apk_size") or 0) for r in chosen) / (1024 ** 3)
    print(f"YEAR={args.year} TASKS={len(chosen)} WORKERS={args.workers} EST_GiB={approx_gib:.2f}", flush=True)
    print(f"FROZEN_MANIFEST_SHA256={manifest_digest}", flush=True)
    if args.dry_run:
        return 0

    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with fail_if_legacy_active(root):
        year_lock_path = logs / f"downloader_year{args.year}.lock"
        with year_lock_path.open("a+") as year_lock:
            try:
                fcntl.flock(year_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"Year {args.year} already has an active downloader")
            pinned = logs / "frozen_manifest.sha256"
            if not pinned.is_file() or pinned.read_text().strip() != manifest_digest:
                raise SystemExit("Frozen manifest pin missing or different; refusing to mix experiments")

            keyfile = args.key_file.expanduser()
            if not keyfile.is_file():
                raise SystemExit(f"AndroZoo key file missing: {keyfile}")
            key = keyfile.read_text(encoding="utf-8").strip()
            if not re.fullmatch(r"[a-fA-F0-9]{64}", key):
                raise SystemExit("AndroZoo key must be a 64-character hexadecimal string")

            journal = logs / f"download_status_year{args.year}.csv"
            new_journal = not journal.exists() or journal.stat().st_size == 0
            counts = Counter()
            failures_in_a_row = 0
            aborted = False
            attempts = 0
            pending = iter(chosen)
            started = time.monotonic()

            with journal.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new_journal:
                    writer.writeheader()
                    f.flush()
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    active = {}

                    def submit_next() -> bool:
                        nonlocal aborted
                        if shutil.disk_usage(root).free < args.min_free_gb * (1024 ** 3):
                            print(f"STOP: less than {args.min_free_gb:g} GiB free", flush=True)
                            aborted = True
                            return False
                        try:
                            row = next(pending)
                        except StopIteration:
                            return False
                        future = pool.submit(fetch_one, root, row, key, args.limit_rate_mbps)
                        active[future] = row
                        return True

                    for _ in range(args.workers):
                        if not submit_next():
                            break
                    while active:
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            row = active.pop(future)
                            try:
                                r, outcome, size, elapsed, detail = future.result()
                            except Exception as exc:
                                r, outcome, size, elapsed, detail = row, "FAIL", 0, 0, f"{type(exc).__name__}: {exc}"
                            attempts += 1
                            counts[outcome] += 1
                            failures_in_a_row = failures_in_a_row + 1 if outcome == "FAIL" else 0
                            writer.writerow({
                                "timestamp_utc": stamp(), "year": r["year"],
                                "label": r["label"], "sha256": r["sha256"],
                                "status": outcome, "bytes": size,
                                "seconds": f"{elapsed:.1f}", "detail": detail,
                            })
                            f.flush()
                            print(
                                f"[{attempts}/{len(chosen)}] {outcome} "
                                f"year={args.year} label={r['label']} "
                                f"sha={r['sha256'][:12]} bytes={size} "
                                f"sec={elapsed:.0f} {detail}",
                                flush=True,
                            )
                            if failures_in_a_row >= args.stop_after_consecutive_fails:
                                aborted = True
                        if not aborted:
                            for _ in range(args.workers - len(active)):
                                if not submit_next():
                                    break

            print(
                f"YEAR_SUMMARY={args.year} {dict(counts)} "
                f"ABORTED={aborted} SECONDS={time.monotonic()-started:.1f} "
                f"STATUS={journal}",
                flush=True,
            )
            return 2 if aborted or counts["FAIL"] else 0


def run_all(args) -> int:
    root = args.root.expanduser().resolve()
    if args.parallel_years * args.workers > 4:
        raise SystemExit("Shared-server limit: parallel_years * workers must be <= 4")
    # Do not launch the batch if the old downloader still owns the legacy lock.
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with fail_if_legacy_active(root):
        pass

    def run_child(year: int):
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--year", str(year), "--workers", str(args.workers),
            "--root", str(root), "--key-file", str(args.key_file),
            "--limit-rate-mbps", str(args.limit_rate_mbps),
            "--min-free-gb", str(args.min_free_gb),
            "--stop-after-consecutive-fails", str(args.stop_after_consecutive_fails),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        log_file = root / "logs" / f"year{year}_run.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n--- {stamp()} Starting year {year} ---\n")
            f.flush()
            completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
            f.write(f"--- {stamp()} Year {year} exit={completed.returncode} ---\n")
            f.flush()
        return (year, completed.returncode, log_file)

    print(
        f"STARTING YEARS=2017..2022 PARALLEL_YEARS={args.parallel_years} "
        f"WORKERS_PER_YEAR={args.workers} MAX_CONNECTIONS={args.parallel_years*args.workers}",
        flush=True,
    )
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel_years) as pool:
        futures = {pool.submit(run_child, y): y for y in YEARS}
        for fut in as_completed(futures):
            year, code, log_file = fut.result()
            print(f"YEAR_DONE={year} EXIT={code} LOG={log_file}", flush=True)
            if code:
                failed.append(year)
    print(f"FINISHED YEARS_FAILED={failed}", flush=True)
    return 2 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, choices=YEARS)
    group.add_argument("--all", action="store_true", help="Process six years concurrently")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--key-file", type=Path, default=DEFAULT_KEYFILE)
    p.add_argument("--parallel-years", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    p.add_argument("--limit-rate-mbps", type=int, default=2)
    p.add_argument("--stop-after-consecutive-fails", type=int, default=8)
    p.add_argument("--min-free-gb", type=float, default=150.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.min_free_gb < 0 or args.limit_rate_mbps < 1 or args.stop_after_consecutive_fails < 1:
        p.error("Invalid safety threshold")
    return run_all(args) if args.all else run_year(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted: verified APKs and .part files are preserved", file=sys.stderr)
        sys.exit(130)
