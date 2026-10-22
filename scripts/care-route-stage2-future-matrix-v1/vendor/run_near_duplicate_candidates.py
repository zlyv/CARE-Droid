#!/usr/bin/env python3
"""Build compact, conservative near-duplicate APK candidate clusters.

This pass is deliberately a candidate/triage pass.  It extracts a bounded set
of normalized DEX string tokens, computes deterministic MinHash signatures,
uses LSH to avoid an O(n^2) all-pairs scan, and combines only high-confidence
similarity edges with the existing exact-Dex/identity blocking assignments.

It never writes call graphs or raw DEX data.  The output is not a claim that
the selected threshold is scientifically calibrated; the audit keeps that
item as a blocking reason until a sample of candidate edges is reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


HEX_RE = re.compile(r"0x[0-9a-f]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\b")


def read_csv(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_uleb128(data: bytes, offset: int):
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 35:
            break
    raise ValueError("invalid ULEB128")


def dex_strings(data: bytes, max_items: int):
    """Read the DEX string pool without requiring Androguard.

    The string_ids_size/off fields are part of the fixed DEX header.  A
    fallback printable-string extractor is used for malformed/odex-like DEX
    inputs so one unusual sample does not abort the inventory.
    """
    if len(data) >= 0x40 and data[:4] == b"dex\n":
        string_count = struct.unpack_from("<I", data, 0x38)[0]
        string_off = struct.unpack_from("<I", data, 0x3C)[0]
        if string_count <= 2_000_000 and string_off + string_count * 4 <= len(data):
            if string_count <= max_items:
                indexes = range(string_count)
            else:
                # Deterministic sampling across the complete string pool. It
                # avoids walking hundreds of thousands of mostly irrelevant
                # strings while retaining coverage of the whole DEX.
                indexes = sorted({
                    (index * string_count) // max_items
                    for index in range(max_items)
                })
            for index in indexes:
                item_off = struct.unpack_from("<I", data, string_off + index * 4)[0]
                if item_off >= len(data):
                    continue
                try:
                    _, cursor = read_uleb128(data, item_off)
                    end = data.find(b"\0", cursor)
                    if end < 0:
                        continue
                    raw = data[cursor:end]
                    yield raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            return

    for match in re.finditer(rb"[ -~]{3,}", data):
        yield match.group(0).decode("ascii", errors="ignore")


def normalize_text(value: str):
    value = value.strip().lower()
    if len(value) < 3 or len(value) > 240:
        return ""
    value = HEX_RE.sub("<hex>", value)
    value = NUMBER_RE.sub("<num>", value)
    return value


def token_hash(value: str):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def extract_token_ids(row, max_strings_per_dex: int):
    """Extract one hashed token per normalized DEX string."""
    token_ids = set()
    dex_count = 0
    try:
        with zipfile.ZipFile(row["absolute_path"], "r") as archive:
            for name in archive.namelist():
                lower_name = name.lower()
                if not (lower_name.startswith("classes") and lower_name.endswith(".dex")):
                    continue
                dex_count += 1
                for text in dex_strings(archive.read(name), max_strings_per_dex):
                    normalized = normalize_text(text)
                    if normalized:
                        token_ids.add(token_hash(normalized))
        if dex_count == 0:
            return set(), "NO_DEX", "", ""
        if not token_ids:
            return set(), "NO_TOKENS", "", ""
        return token_ids, "OK", "", ""
    except Exception as exc:
        return set(), "ERROR", type(exc).__name__, str(exc)[:500]


def signature_for_apk(
    row,
    max_tokens: int,
    max_strings_per_dex: int,
    signature_size: int,
    allowed_token_ids=None,
):
    result = {
        "sha256": row["sha256"],
        "absolute_path": row["absolute_path"],
        "package_name": row.get("package_name", ""),
        "certificate_sha256": row.get("certificate_sha256", ""),
        "version_code": row.get("version_code", ""),
        "year": row.get("year", ""),
        "label": row.get("label", row.get("label_name", "")),
        "raw_token_count": 0,
        "token_count": 0,
        "minhash": "",
        "feature_status": "ERROR",
        "error_type": "",
        "error_message": "",
    }
    token_ids, status, error_type, error_message = extract_token_ids(row, max_strings_per_dex)
    result["raw_token_count"] = len(token_ids)
    result["error_type"] = error_type
    result["error_message"] = error_message
    if status != "OK":
        result["feature_status"] = status
        return result
    try:
        if allowed_token_ids is not None:
            token_ids.intersection_update(allowed_token_ids)
        if len(token_ids) > max_tokens:
            token_ids = set(sorted(token_ids)[:max_tokens])
        result["token_count"] = len(token_ids)
        if not token_ids:
            result["feature_status"] = "NO_TOKENS"
            return result
        # Bottom-k MinHash: one deterministic token hash, retaining only the
        # smallest k values. This avoids the previous 64x inner hash loop.
        bottom_k = sorted(token_ids)[:signature_size]
        result["minhash"] = ",".join(f"{value:016x}" for value in bottom_k)
        result["feature_status"] = "OK"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    return result


def parse_signature(value: str):
    return tuple(int(part, 16) for part in value.split(",") if part)


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def estimated_jaccard(left, right, left_count, right_count):
    if not left or not right:
        return 0.0
    overlap = len(set(left).intersection(right))
    signature_size = max(len(left), len(right))
    if left_count <= signature_size and right_count <= signature_size:
        union_count = left_count + right_count - overlap
        return overlap / union_count if union_count else 0.0
    if left_count > signature_size and right_count > signature_size:
        return overlap / signature_size
    # One set is smaller than k and the other is larger than k.  This
    # denominator is intentionally conservative for thresholding.
    return overlap / max(left_count, right_count)


def main():
    parser = argparse.ArgumentParser(description="Build conservative near-duplicate candidate clusters")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--fingerprints", required=True, type=Path)
    parser.add_argument("--blocking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--max-strings-per-dex", type=int, default=12000)
    parser.add_argument("--signature-size", type=int, default=128)
    parser.add_argument("--max-df-ratio", type=float, default=0.05)
    parser.add_argument("--same-package-threshold", type=float, default=0.80)
    parser.add_argument("--cross-package-threshold", type=float, default=0.92)
    parser.add_argument("--max-bucket-size", type=int, default=200)
    args = parser.parse_args()

    lineage_rows = read_csv(args.lineage)
    fingerprint_rows = {row["sha256"]: row for row in read_csv(args.fingerprints)}
    blocking_rows = {row["sha256"]: row for row in read_csv(args.blocking)}
    rows = []
    for row in lineage_rows:
        fingerprint = fingerprint_rows.get(row["sha256"], {})
        merged = dict(row)
        merged.update({key: value for key, value in fingerprint.items() if value})
        if row["sha256"] in blocking_rows:
            merged["base_block_id"] = blocking_rows[row["sha256"]].get("block_id", "")
        rows.append(merged)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_every = max(1, args.progress_every)
    print(f"pass 1/2: collecting document frequencies from {len(rows)} APKs", flush=True)
    document_frequency = Counter()
    first_pass_status = Counter()
    for index, row in enumerate(rows, start=1):
        token_ids, status, _, _ = extract_token_ids(
            row, max(100, args.max_strings_per_dex)
        )
        first_pass_status[status] += 1
        if status == "OK":
            document_frequency.update(token_ids)
        if index % progress_every == 0 or index == len(rows):
            print(f"pass 1/2 processed {index}/{len(rows)}", flush=True)

    max_df_count = max(2, int(len(rows) * max(0.001, args.max_df_ratio)))
    allowed_token_ids = {
        token_id for token_id, count in document_frequency.items()
        if count <= max_df_count
    }
    print(
        f"pass 1/2 complete: unique_tokens={len(document_frequency)} "
        f"allowed_tokens={len(allowed_token_ids)} max_df_count={max_df_count}",
        flush=True,
    )

    print("pass 2/2: building IDF-filtered signatures", flush=True)
    signatures = []
    for index, row in enumerate(rows, start=1):
        signatures.append(signature_for_apk(
            row,
            max(100, args.max_tokens),
            max(100, args.max_strings_per_dex),
            max(16, args.signature_size),
            allowed_token_ids,
        ))
        if index % progress_every == 0 or index == len(rows):
            print(f"pass 2/2 processed {index}/{len(rows)}", flush=True)

    signature_by_sha = {row["sha256"]: row for row in signatures}
    valid = [row for row in signatures if row["feature_status"] == "OK"]
    parsed = {row["sha256"]: parse_signature(row["minhash"]) for row in valid}

    # Eight bands of sixteen bottom-k MinHash values. Oversized buckets are ignored,
    # because common SDK/library tokens are not useful duplicate evidence.
    buckets = defaultdict(list)
    band_count = 8
    band_width = max(1, args.signature_size // band_count)
    for row in valid:
        signature = parsed[row["sha256"]]
        for band in range(band_count):
            start = band * band_width
            buckets[(band, signature[start:start + band_width])].append(row["sha256"])

    candidate_pairs = set()
    skipped_buckets = 0
    for members in buckets.values():
        if len(members) > args.max_bucket_size:
            skipped_buckets += 1
            continue
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                candidate_pairs.add(tuple(sorted((left, right))))

    accepted = []
    for left_sha, right_sha in sorted(candidate_pairs):
        left = signature_by_sha[left_sha]
        right = signature_by_sha[right_sha]
        similarity = estimated_jaccard(
            parsed[left_sha],
            parsed[right_sha],
            int(left.get("token_count", 0) or 0),
            int(right.get("token_count", 0) or 0),
        )
        same_package = bool(left.get("package_name")) and left.get("package_name") == right.get("package_name")
        threshold = args.same_package_threshold if same_package else args.cross_package_threshold
        if similarity < threshold:
            continue
        accepted.append({
            "left_sha256": left_sha,
            "right_sha256": right_sha,
            "estimated_jaccard": f"{similarity:.4f}",
            "threshold": f"{threshold:.4f}",
            "same_package": str(same_package).lower(),
            "left_package_name": left.get("package_name", ""),
            "right_package_name": right.get("package_name", ""),
            "left_block_id": left.get("base_block_id", ""),
            "right_block_id": right.get("base_block_id", ""),
        })

    union_find = UnionFind(row["sha256"] for row in signatures)
    base_members = defaultdict(list)
    for row in blocking_rows.values():
        if row.get("block_id"):
            base_members[row["block_id"]].append(row["sha256"])
    for members in base_members.values():
        for member in members[1:]:
            union_find.union(members[0], member)
    for edge in accepted:
        union_find.union(edge["left_sha256"], edge["right_sha256"])

    components = defaultdict(list)
    for row in signatures:
        components[union_find.find(row["sha256"])].append(row["sha256"])
    component_ids = {
        root: f"NDBLK{index:06d}"
        for index, root in enumerate(sorted(components), start=1)
    }
    accepted_by_sha = defaultdict(list)
    for edge in accepted:
        accepted_by_sha[edge["left_sha256"]].append(edge)
        accepted_by_sha[edge["right_sha256"]].append(edge)

    assignment_rows = []
    for root, members in sorted(components.items()):
        fuzzy = any(member in accepted_by_sha for member in members)
        for sha in members:
            source = signature_by_sha[sha]
            assignment_rows.append({
                "sha256": sha,
                "absolute_path": source["absolute_path"],
                "base_block_id": source.get("base_block_id", ""),
                "near_duplicate_block_id": component_ids[root],
                "near_duplicate_block_size": len(members),
                "near_duplicate_reason": "MINHASH_HIGH_CONFIDENCE|BASE_BLOCK" if fuzzy else "BASE_BLOCK",
            })

    write_csv(
        output / "near_duplicate_signatures.csv",
        signatures,
        [
            "sha256", "absolute_path", "package_name", "certificate_sha256",
            "version_code", "year", "label", "raw_token_count", "token_count", "minhash",
            "feature_status", "error_type", "error_message",
        ],
    )
    write_csv(
        output / "near_duplicate_edges.csv",
        accepted,
        [
            "left_sha256", "right_sha256", "estimated_jaccard", "threshold",
            "same_package", "left_package_name", "right_package_name",
            "left_block_id", "right_block_id",
        ],
    )
    write_csv(
        output / "near_duplicate_assignments.csv",
        assignment_rows,
        [
            "sha256", "absolute_path", "base_block_id", "near_duplicate_block_id",
            "near_duplicate_block_size", "near_duplicate_reason",
        ],
    )

    sizes = [len(members) for members in components.values()]
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_NEAR_DUPLICATE_CANDIDATE_CLUSTERING",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "fingerprints_csv": str(args.fingerprints.expanduser().resolve()),
        "blocking_csv": str(args.blocking.expanduser().resolve()),
        "apk_count": len(signatures),
        "first_pass_feature_status_counts": dict(first_pass_status),
        "feature_status_counts": dict(Counter(row["feature_status"] for row in signatures)),
        "valid_signature_count": len(valid),
        "candidate_pair_count": len(candidate_pairs),
        "accepted_edge_count": len(accepted),
        "skipped_oversized_lsh_bucket_count": skipped_buckets,
        "near_duplicate_component_count": len(components),
        "near_duplicate_component_size_max": max(sizes) if sizes else 0,
        "near_duplicate_component_size_mean": sum(sizes) / len(sizes) if sizes else 0.0,
        "parameters": {
            "max_tokens": args.max_tokens,
            "max_strings_per_dex": args.max_strings_per_dex,
            "max_df_ratio": args.max_df_ratio,
            "max_df_count": max_df_count,
            "unique_token_count": len(document_frequency),
            "allowed_token_count": len(allowed_token_ids),
            "minhash_size": args.signature_size,
            "lsh_bands": 8,
            "lsh_rows_per_band": 8,
            "same_package_threshold": args.same_package_threshold,
            "cross_package_threshold": args.cross_package_threshold,
            "max_bucket_size": args.max_bucket_size,
        },
        "outputs": {
            "signatures": str(output / "near_duplicate_signatures.csv"),
            "edges": str(output / "near_duplicate_edges.csv"),
            "assignments": str(output / "near_duplicate_assignments.csv"),
        },
        "blocking_reasons": [
            "near_duplicate_threshold_requires_manual_validation",
            "first_seen_utc_not_joined",
            "static_parse_and_api_graph_coverage_not_available",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "DEX string-pool tokens are used as a compact similarity signal",
            "MinHash/LSH candidate generation avoids an all-pairs scan",
            "raw DEX bytes and call graphs are not persisted",
            "near-duplicate assignments are conservative leakage-blocking constraints, not semantic family labels",
        ],
    }
    (output / "near_duplicate_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
