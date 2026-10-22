#!/usr/bin/env python3
"""Bind human approval to the exact passing lineage and Ego gate files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def combined_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("approve", "check"))
    parser.add_argument("--root", type=Path, default=Path(
        "artifacts/care-source-recovery-stage2only-v1/preflight"))
    args = parser.parse_args()
    gates = [
        args.root / "source_lineage_integrity_gate.json",
        args.root / "ego_smoke_gate.json",
    ]
    for path in gates:
        if not path.is_file():
            raise SystemExit(f"missing preflight gate: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("passed"):
            raise SystemExit(f"preflight gate did not pass: {path}")
        for artifact, expected_hash in value.get("artifact_hashes", {}).items():
            artifact_path = Path(artifact)
            if not artifact_path.is_file():
                raise SystemExit(f"gate-bound artifact is missing: {artifact_path}")
            if file_digest(artifact_path) != expected_hash:
                raise SystemExit(f"gate-bound artifact changed after audit: {artifact_path}")
    expected = combined_digest(gates)
    approval_path = args.root / "MANUAL_APPROVAL.json"
    if args.mode == "approve":
        approval_path.write_text(json.dumps({
            "approved": True,
            "combined_gate_sha256": expected,
            "scope": "permit full 799 source extraction/training only after both preflight reports were inspected",
        }, indent=2) + "\n", encoding="utf-8")
        print("Approved exact combined preflight gates:", expected)
        return
    if not approval_path.is_file():
        raise SystemExit("manual preflight approval is absent")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not approval.get("approved") or approval.get("combined_gate_sha256") != expected:
        raise SystemExit("manual approval does not match the current preflight gates")
    print("Manual approval matches current combined preflight gates:", expected)


if __name__ == "__main__":
    main()
