#!/usr/bin/env python3
"""Download the frozen AZ_2017_2022_Random15K manifest safely.

Features: independent data root, 1 or 2 workers, pilot across year/label,
verified SHA256, resumable .part downloads, a single-process filesystem lock,
recoverable status logs, and no API key in process arguments or log files.

This program neither edits the source manifest nor reads historical manifests.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path.home() / "datasets/android/AndroZoo/AZ_2017_2022_Random15K"
DEFAULT_KEYFILE = Path.home() / ".config/androzoo/api_key"
ENDPOINT = "https://androzoo.uni.lu/api/download"
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
CSV_FIELDS = ["timestamp_utc", "year", "label", "sha256", "status", "bytes", "seconds", "detail"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Manifest is empty")
    seen = set()
    for idx, row in enumerate(rows, start=2):
        sha = (row.get("sha256") or "").strip().lower()
        year = (row.get("year") or "").strip()
        label = (row.get("label") or "").strip()
        if not SHA_RE.fullmatch(sha) or year not in {str(y) for y in range(2017, 2023)} or label not in {"0", "1"}:
            raise ValueError(f"Invalid sha/year/label at CSV line {idx}")
        if sha in seen:
            raise ValueError(f"Duplicate SHA at CSV line {idx}: {sha}")
        seen.add(sha)
        row["sha256"] = sha
        row["year"] = year
        row["label"] = label
    return rows


def pilot_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0:
        return rows
    # Cover every year/label first instead of taking just the first 20 rows.
    buckets: dict[tuple[str, str], deque] = defaultdict(deque)
    for row in rows:
        buckets[(row["year"], row["label"])].append(row)
    keys = sorted(buckets)
    chosen = []
    while len(chosen) < count and keys:
        next_keys = []
        for k in keys:
            if not buckets[k]:
                continue
            chosen.append(buckets[k].popleft())
            if buckets[k]:
                next_keys.append(k)
            if len(chosen) >= count:
                break
        keys = next_keys
    return chosen


def disk_path(root: Path, row: dict[str, str]) -> Path:
    return root / "apks" / ("year" + row["year"]) / ("benign" if row["label"] == "0" else "malware") / (row["sha256"] + ".apk")


def verified(path: Path, expected_sha: str) -> bool:
    return path.is_file() and sha256_file(path) == expected_sha


def fetch_one(root: Path, row: dict[str, str], apikey: str, max_mbps: int):
    start = time.monotonic()
    final = disk_path(root, row)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")
    sha = row["sha256"]
    if verified(final, sha):
        return row, "SKIP_OK", final.stat().st_size, time.monotonic() - start, "already SHA256 verified"
    if final.exists():
        return row, "FAIL", final.stat().st_size, time.monotonic() - start, "Existing .apk SHA mismatch; not overwritten"

    # The API key is sent through curl's stdin config, never a command-line arg.
    # --location is deliberately omitted so redirects cannot forward the API key.
    curl_conf = 'data-urlencode = "apikey=' + apikey + '"\n' + 'data-urlencode = "sha256=' + sha + '"\n'
    command = [
        "curl", "-q", "-4", "--http1.1", "--noproxy", "*",
        "--fail", "--show-error", "--silent", "--get", "--config", "-",
        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "300",
        "--retry", "3", "--retry-delay", "10", "--retry-connrefused",
        "--limit-rate", f"{max_mbps}M",
        "--continue-at", "-", "--output", str(part),
        "--write-out", "HTTP=%{http_code}", ENDPOINT,
    ]
    try:
        completed = subprocess.run(command, input=curl_conf, text=True, capture_output=True, check=False)
        if completed.returncode == 33 and part.is_file():
            # Server ignored the Range request. Resume is impossible for this
            # partial file, so remove only this script's temporary download and
            # retry the same SHA once from byte zero.
            part.unlink()
            completed = subprocess.run(command, input=curl_conf, text=True, capture_output=True, check=False)
    except OSError as exc:
        return row, "FAIL", part.stat().st_size if part.exists() else 0, time.monotonic() - start, f"curl start error: {exc}"

    http_status = completed.stdout.strip().replace(apikey, "[REDACTED]")
    curl_error = (completed.stderr or "").strip().replace(apikey, "[REDACTED]")
    if completed.returncode != 0:
        reason = f"curl_exit={completed.returncode} {http_status} {curl_error[-260:]}"
        # Curl 33 indicates that byte-range resume is not supported.
        if completed.returncode == 33:
            reason += "; server may not support resuming this file"
        return row, "FAIL", part.stat().st_size if part.exists() else 0, time.monotonic() - start, reason
    if http_status != "HTTP=200" and http_status != "HTTP=206":
        return row, "FAIL", part.stat().st_size if part.exists() else 0, time.monotonic() - start, f"Unexpected response: {http_status}"
    if not part.is_file() or not verified(part, sha):
        size = part.stat().st_size if part.exists() else 0
        # A complete HTTP response with a bad SHA should not be resumed.
        if part.exists():
            part.rename(part.with_name(part.name + ".badsha." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        return row, "FAIL", size, time.monotonic() - start, f"SHA256 mismatch after {http_status}; part quarantined"
    os.replace(part, final)
    return row, "OK", final.stat().st_size, time.monotonic() - start, http_status + " SHA256 PASS"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--key-file", type=Path, default=DEFAULT_KEYFILE)
    ap.add_argument("--workers", type=int, choices=(1, 2), default=2)
    ap.add_argument("--pilot", type=int, default=0, help="Download N samples spread across year/label groups; 0=all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-rate-mbps", type=int, default=3, help="Max transfer rate per worker in MB/s")
    ap.add_argument("--stop-after-consecutive-fails", type=int, default=8)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    manifest = root / "manifests" / "selected_all.csv"
    rows = load_rows(manifest)
    manifest_hash = sha256_file(manifest)
    if len(rows) != 15000:
        raise SystemExit(f"Expected exactly 15000 rows in frozen manifest, got {len(rows)}")
    if args.pilot < 0 or args.limit_rate_mbps < 1 or args.stop_after_consecutive_fails < 1:
        raise SystemExit("Invalid numeric parameter")
    selected = pilot_rows(rows, args.pilot)
    counts = Counter((r["year"], r["label"]) for r in selected)
    total_estimated = sum(int(r.get("apk_size") or 0) for r in selected)
    print(f"MANIFEST={manifest}", flush=True)
    print(f"MANIFEST_SHA256={manifest_hash}", flush=True)
    print(f"TASKS={len(selected)} WORKERS={args.workers} ESTIMATED_APK_BYTES={total_estimated}", flush=True)
    print("GROUP_COUNTS=" + ", ".join(f"{y}/{l}:{n}" for (y, l), n in sorted(counts.items())), flush=True)
    if args.dry_run:
        return 0

    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Prevent multiple downloader instances from writing the same .part.
    lockfd = (logs / "downloader.lock").open("a+")
    try:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another downloader is active; no duplicate download has been started")

    pinned_manifest = logs / "frozen_manifest.sha256"
    if pinned_manifest.exists():
        if pinned_manifest.read_text().strip() != manifest_hash:
            raise SystemExit("Manifest changed since previous run; refusing to mix datasets")
    else:
        pinned_manifest.write_text(manifest_hash + "\n")

    if not args.key_file.is_file():
        raise SystemExit(f"API Key not found: {args.key_file}")
    apikey = args.key_file.read_text().strip()
    if not re.fullmatch(r"[a-fA-F0-9]{64}", apikey):
        raise SystemExit("API Key is not a 64-character hexadecimal string")

    status_path = logs / "download_status.csv"
    is_new = not status_path.exists() or status_path.stat().st_size == 0
    tally = Counter()
    attempts = 0
    failure_streak = 0
    aborted = False
    with status_path.open("a", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
            log.flush()
        pending = iter(selected)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            active = {}

            def submit_next():
                try:
                    row = next(pending)
                except StopIteration:
                    return False
                future = executor.submit(fetch_one, root, row, apikey, args.limit_rate_mbps)
                active[future] = row
                return True

            for _ in range(args.workers):
                submit_next()
            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    row = active.pop(future)
                    try:
                        r, outcome, size, elapsed, detail = future.result()
                    except Exception as exc:
                        r, outcome, size, elapsed, detail = row, "FAIL", 0, 0.0, f"Worker exception: {type(exc).__name__}: {exc}"
                    attempts += 1
                    tally[outcome] += 1
                    failure_streak = failure_streak + 1 if outcome == "FAIL" else 0
                    writer.writerow({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "year": r["year"], "label": r["label"], "sha256": r["sha256"],
                        "status": outcome, "bytes": size, "seconds": f"{elapsed:.1f}", "detail": detail,
                    })
                    log.flush()
                    print(f"[{attempts}/{len(selected)}] {outcome} year={r['year']} label={r['label']} sha={r['sha256'][:12]} bytes={size} sec={elapsed:.0f} {detail}", flush=True)
                    if failure_streak >= args.stop_after_consecutive_fails:
                        aborted = True
                if not aborted:
                    for _ in range(args.workers - len(active)):
                        if not submit_next():
                            break
    print(f"SUMMARY={dict(tally)} ABORTED={aborted} LOG={status_path}", flush=True)
    return 2 if aborted or tally["FAIL"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; verified files and .part files are kept for next run.", file=sys.stderr)
        sys.exit(130)
