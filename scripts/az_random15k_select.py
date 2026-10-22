"""Select reproducible balanced AndroZoo APK SHA256 lists for CARE-Droid.

Purpose: sample from a pre-existing AndroZoo manifest without excluding any
previous experimental SHA256. Applies per-year/class quotas, quarterly spread,
global SHA256 and package-name uniqueness, and generates random reserves.
It does NOT download APKs or establish APK/DEX near-duplicate independence.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

QUOTAS = {2017: 2500, 2018: 1000, 2019: 1000, 2020: 1000, 2021: 1000, 2022: 1000}
SHA_RE = re.compile(r"^[a-f0-9]{64}$")


def month_value(v):
    s = str(v or "").strip()
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 12:
            return n
        if len(s) == 6 and s[:4].isdigit() and s[4:].isdigit():
            n = int(s[-2:])
            return n if 1 <= n <= 12 else None
    m = re.match(r"^\d{4}[-/](\d{1,2})(?:[-/]|$)", s)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 12 else None
    return None


def label_value(v):
    s = str(v or "").strip().lower()
    return {"0": "benign", "1": "malware", "benign": "benign", "malware": "malware"}.get(s)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, cols):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--reserve-ratio", type=float, default=0.20)
    args = ap.parse_args()

    manifest = args.manifest.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not manifest.is_file():
        ap.error(f"Missing source manifest: {manifest}")
    if (out / "selected_all.csv").exists():
        ap.error(f"Output already exists: {out}/selected_all.csv; choose a new --out to preserve audit")
    if not 0 <= args.reserve_ratio <= 1:
        ap.error("--reserve-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    groups = defaultdict(list)
    input_count = Counter()
    pool_sha = set()
    all_columns = None

    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        all_columns = list(reader.fieldnames or [])
        required = {"year", "label", "sha256", "pkg_name"}
        if not required.issubset(all_columns):
            ap.error(f"Manifest missing fields: {sorted(required - set(all_columns))}")
        for row in reader:
            try:
                year = int(str(row["year"]).strip())
            except (ValueError, TypeError):
                continue
            label = label_value(row["label"])
            if year not in QUOTAS or label is None:
                continue
            sha = str(row["sha256"]).strip().lower()
            if not SHA_RE.fullmatch(sha):
                input_count["invalid_sha"] += 1
                continue
            if sha in pool_sha:
                input_count["duplicate_sha"] += 1
                continue
            pool_sha.add(sha)
            row["sha256"] = sha
            row["year"] = str(year)
            row["label_name"] = label
            row["pkg_name"] = str(row.get("pkg_name") or "").strip()
            mon = month_value(row.get("month"))
            quarter = (mon - 1) // 3 + 1 if mon else 0
            row["_quarter"] = quarter
            groups[year, label, quarter].append(row)
            input_count[f"{year}_{label}"] += 1
            if quarter == 0:
                input_count["missing_quarter"] += 1

    for g in groups.values():
        rng.shuffle(g)

    # Nonempty package names may occur only once across all selected years.
    used_sha = set()
    used_pkg = set()
    picked = []
    summaries = []
    fallback = Counter()

    def take(row, year, label, target, dest, kind):
        sha = row["sha256"]
        pkg = row["pkg_name"].lower()
        if sha in used_sha or (pkg and pkg in used_pkg):
            return False
        used_sha.add(sha)
        if pkg:
            used_pkg.add(pkg)
        r = dict(row)
        r["sampling_quarter"] = str(row["_quarter"])
        r["sampling_seed"] = str(args.seed)
        r["sampling_role"] = kind
        dest.append(r)
        return True

    for year, half in QUOTAS.items():
        for label in ("benign", "malware"):
            selected = []
            # Quarter-balanced targets, when the pool allows it.
            for quarter in (1, 2, 3, 4):
                target_q = half // 4 + int(quarter <= half % 4)
                current_q = 0
                for row in groups[year, label, quarter]:
                    if current_q >= target_q:
                        break
                    if take(row, year, label, half, selected, "selected"):
                        current_q += 1

            if len(selected) < half:
                extras = [r for q in range(5) for r in groups[year, label, q]]
                rng.shuffle(extras)
                for row in extras:
                    if len(selected) >= half:
                        break
                    if take(row, year, label, half, selected, "selected"):
                        fallback[year, label] += 1

            if len(selected) != half:
                raise SystemExit(
                    f"INSUFFICIENT {year} {label}: selected {len(selected)} < {half}; "
                    "no CSV written. Cannot satisfy package uniqueness."
                )
            picked.extend(selected)
            summaries.append({
                "year": year, "label": label, "pool": input_count[f"{year}_{label}"],
                "selected": len(selected), "quarter_counts": dict(sorted(
                    Counter(r["sampling_quarter"] for r in selected).items()
                )), "fallback_count": fallback[year, label]
            })

    if len(picked) != 15000:
        raise SystemExit(f"Expected 15000, got {len(picked)}")

    reserves = []
    for year, half in QUOTAS.items():
        for label in ("benign", "malware"):
            need = round(half * args.reserve_ratio)
            pool = [r for q in range(5) for r in groups[year, label, q]]
            rng.shuffle(pool)
            reserved = []
            for row in pool:
                if len(reserved) >= need:
                    break
                take(row, year, label, need, reserved, "reserve")
            reserves.extend(reserved)
            if len(reserved) < need:
                print(f"WARNING: reserve shortage {year} {label}: {len(reserved)}/{need}")

    # Only now write artifacts, after all main quotas were met.
    out.mkdir(parents=True, exist_ok=True)
    cols = list(all_columns)
    for extra in ("sampling_quarter", "sampling_seed", "sampling_role"):
        if extra not in cols:
            cols.append(extra)

    write_csv(out / "selected_all.csv", picked, cols)
    for year in QUOTAS:
        part = [r for r in picked if int(r["year"]) == year]
        write_csv(out / f"selected_{year}.csv", part, cols)
        (out / f"selected_{year}.sha256.txt").write_text(
            "".join(r["sha256"] + "\n" for r in part), encoding="utf-8"
        )
    write_csv(out / "reserve_candidates.csv", reserves, cols)
    (out / "selected_all.sha256.txt").write_text(
        "".join(r["sha256"] + "\n" for r in picked), encoding="utf-8"
    )
    audit = {
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha_file(manifest),
        "seed": args.seed,
        "historical_sha_exclusion": False,
        "cross_year_package_unique": True,
        "dex_near_duplicates_checked": False,
        "quarter_target": "roughly equal quarters; shortage filled randomly from same year/label",
        "total_selected": len(picked), "total_reserve": len(reserves),
        "missing_quarter_in_pool": input_count["missing_quarter"],
        "invalid_sha_in_pool": input_count["invalid_sha"],
        "duplicate_sha_in_pool": input_count["duplicate_sha"],
        "summary": summaries,
    }
    (out / "sampling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"SELECTED={len(picked)} RESERVES={len(reserves)} OUT={out}")
    print("YEAR  LABEL     POOL  SELECTED  Q1/Q2/Q3/Q4/UNK  BACKFILL")
    for s in summaries:
        q = s["quarter_counts"]
        qs = "/".join(str(q.get(str(i), 0)) for i in [1, 2, 3, 4, 0])
        print(f'{s["year"]}  {s["label"]:<8} {s["pool"]:>5}  {s["selected"]:>5}     {qs:<20} {s["fallback_count"]:>3}')
    print("Only manifest random sampling and package/sha uniqueness were checked.")
    print("No APK downloaded; prior samples were NOT excluded.")


if __name__ == "__main__":
    main()
