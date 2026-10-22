#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
import traceback
from pathlib import Path

from common import write_gzip_json
from fullgraph import build_full_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        graph = build_full_graph(Path(args.apk))
        graph["sha256"] = args.sha256.casefold()
        write_gzip_json(Path(args.output), graph)
        status = {
            "sha256": args.sha256.casefold(),
            "status": "OK",
            "graph_path": str(Path(args.output).resolve()),
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            **graph["stats"],
        }
    except Exception as exc:
        status = {
            "sha256": args.sha256.casefold(),
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc(limit=20),
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    temporary = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    print("FULLGRAPH_BUILD_ONE " + json.dumps({key: status.get(key) for key in ("sha256", "status", "method_count", "edge_count", "wall_seconds", "error_type")}, sort_keys=True), flush=True)
    if status["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
