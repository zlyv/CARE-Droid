#!/usr/bin/env python3
"""Bind manual approval to the exact passing Host/Ego smoke report."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("approve", "check"))
    parser.add_argument("--smoke", type=Path, default=Path(
        "artifacts/care-source-recovery-stage2only-v1/smoke"))
    args = parser.parse_args()
    gate_path = args.smoke / "smoke_gate.json"
    approval_path = args.smoke / "MANUAL_APPROVAL.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise SystemExit("cannot approve a failed smoke gate")
    expected = digest(gate_path)
    if args.mode == "approve":
        approval_path.write_text(json.dumps({
            "approved": True, "smoke_gate_sha256": expected,
            "scope": "approve full 799 extraction after human inspection only",
        }, indent=2) + "\n", encoding="utf-8")
        print("Approved exact smoke gate:", expected)
        return
    if not approval_path.is_file():
        raise SystemExit(1)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not approval.get("approved") or approval.get("smoke_gate_sha256") != expected:
        raise SystemExit(1)
    print("Manual smoke approval matches current gate:", expected)


if __name__ == "__main__":
    main()
