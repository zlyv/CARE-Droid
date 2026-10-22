#!/usr/bin/env python3
from __future__ import annotations

from common import discover_assets, load_apk_paths
from select_pairs import select


def main() -> None:
    assets = discover_assets(); pairs = select()
    wanted = {row[key] for row in pairs for key in ("error_sha256", "control_sha256")}
    apk_map, audit = load_apk_paths(wanted)
    if len(apk_map) != len(wanted):
        missing = sorted(wanted - set(apk_map))
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: selected APK paths {len(apk_map)}/{len(wanted)} missing={missing[:10]}")
    import androguard  # noqa: F401
    print("HIGH_CONFIDENCE_OBSERVABILITY_PREFLIGHT_PASS")
    print({"pairs": len(pairs), "unique_apks": len(wanted), "resolved_apks": len(apk_map),
           "assets": {key: str(value) for key, value in assets.items()}, "path_audit": audit})


if __name__ == "__main__": main()
