from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tasa_droid.config import config_summary, load_project_config
from tasa_droid.p0 import run_p0_scan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tasa-droid",
        description="TASA-Droid temporal Android malware detection pipeline",
    )
    command = parser.add_subparsers(dest="command", required=True)
    p0 = command.add_parser("p0", help="P0 data inventory and audit")
    p0_command = p0.add_subparsers(dest="p0_command", required=True)

    validate = p0_command.add_parser("validate-config", help="validate and resolve data paths")
    validate.add_argument("--config", required=True, type=Path)

    scan = p0_command.add_parser("scan", help="scan APK inventory without modifying APKs")
    scan.add_argument("--config", required=True, type=Path)
    scan.add_argument("--output", required=True, type=Path)
    scan.add_argument("--hash-mode", choices=("sha256", "none"), default="sha256")
    scan.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_project_config(args.config)
        if args.command == "p0" and args.p0_command == "validate-config":
            print(json.dumps(config_summary(config), ensure_ascii=False, indent=2))
            return 0
        if args.command == "p0" and args.p0_command == "scan":
            audit = run_p0_scan(
                config=config,
                output_dir=args.output,
                hash_mode=args.hash_mode,
                workers=args.workers,
            )
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    parser.error("unsupported command")
    return 2

