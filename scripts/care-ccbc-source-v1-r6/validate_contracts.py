#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from common import OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    paths = sorted((OUTPUT / "cache/contracts").glob("*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ok = [row for row in rows if row.get("status") == "OK"]
    if len(rows) < args.expected or len(ok) < args.expected:
        errors = [row for row in rows if row.get("status") != "OK"][:5]
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: contract audit rows={len(rows)} ok={len(ok)} expected={args.expected} errors={errors}")
    if any(row.get("raw_component_name_emitted") for row in ok):
        raise RuntimeError("BLOCKED_BY_CONTRACT: raw component name entered extracted feature artifact")
    if any(row.get("call_graph_constructed") for row in ok):
        raise RuntimeError("BLOCKED_BY_CONTRACT: contract extractor constructed a new call graph")
    print(f"CCBC_CONTRACT_{args.expected}_APK_AUDIT_PASS rows={len(rows)} ok={len(ok)}")


if __name__ == "__main__":
    main()
