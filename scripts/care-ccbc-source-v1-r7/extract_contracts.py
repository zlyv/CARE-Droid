#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from common import OUTPUT, dump_json, read_json, write_jsonl_gz
from contracts import lifecycle_names, parse_manifest_xml


def silence_androguard() -> None:
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass


def load_apk(path: Path):
    silence_androguard()
    from androguard.core.apk import APK
    return APK(str(path))


def dex_methods(apk) -> dict[str, list[tuple[str, str]]]:
    """Return descriptor -> [(method_name, descriptor)]. No Analysis object."""
    try:
        from androguard.core.dex import DEX
    except ImportError:
        from androguard.core.bytecodes.dvm import DalvikVMFormat as DEX
    result: dict[str, list[tuple[str, str]]] = {}
    for raw in apk.get_all_dex():
        vm = DEX(raw)
        for method in vm.get_methods():
            cls = str(method.get_class_name())
            name = str(method.get_name())
            descriptor = str(method.get_descriptor())
            result.setdefault(cls, []).append((name, descriptor))
    return result


def canonical_method(cls: str, name: str, descriptor: str) -> str:
    return f"{cls}->{name}{descriptor}"


def method_aliases(canonical: str) -> list[str]:
    # Endpoint cache IDs are uint64(big-endian SHA256(canonical_signature)).
    # canonical_signature removes whitespace and case-folds the full method reference.
    normalized = re.sub(r"\s+", "", canonical).casefold()
    values = set()
    for value in (canonical, normalized):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        values.update((digest, digest[:16], str(int(digest[:16], 16))))
    return sorted(values)


def extract_one(sha256: str, path: Path) -> dict:
    started = time.perf_counter()
    apk = load_apk(path)
    root = apk.get_android_manifest_xml()
    parsed = parse_manifest_xml(root, apk.get_package())
    methods = dex_methods(apk)
    components = []
    total_entry_methods = 0
    for component in parsed["components"]:
        cls = component.pop("class_descriptor")
        names = lifecycle_names(component["type"])
        entries = []
        for name, descriptor in methods.get(cls, []):
            if name in names:
                canonical = canonical_method(cls, name, descriptor)
                entries.append({"callback": name, "aliases": method_aliases(canonical)})
        total_entry_methods += len(entries)
        components.append({
            "component_id": len(components),
            "type": component["type"],
            "manifest_kind": component["manifest_kind"],
            "tokens": component["tokens"],
            "weak_tokens": component["weak_tokens"],
            "exported": component["exported"],
            "enabled": component["enabled"],
            "entry_methods": entries,
            "mapping_class_hash": hashlib.sha256(cls.encode("utf-8")).hexdigest(),
        })
    elapsed = time.perf_counter() - started
    return {
        "sha256": sha256, "status": "OK", "components": components,
        "weak_permissions": parsed["weak_permissions"],
        "uses_features": parsed["uses_features"],
        "component_count": len(components),
        "entry_method_count": total_entry_methods,
        "extraction_seconds": elapsed,
        "analysis_object_constructed": False,
        "call_graph_constructed": False,
        "raw_component_name_emitted": False,
        "future_rows_loaded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    paths = read_json(OUTPUT / "apk_paths.json")
    items = sorted(paths.items())
    if args.limit:
        items = items[:args.limit]
    shard_dir = OUTPUT / "cache/contracts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, (sid, raw_path) in enumerate(items, 1):
        target = shard_dir / f"{sid}.json"
        if target.is_file() and target.stat().st_size:
            try:
                old = json.loads(target.read_text(encoding="utf-8"))
                if old.get("status") == "OK":
                    continue
            except Exception:
                pass
        try:
            row = extract_one(sid, Path(raw_path))
        except Exception as exc:
            failed += 1
            row = {"sha256": sid, "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}
        target.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        if index <= 5 or index % 25 == 0 or index == len(items):
            print(f"CCBC_CONTRACT_EXTRACTION {index}/{len(items)} failed={failed}", flush=True)
    rows = [json.loads((shard_dir / f"{sid}.json").read_text(encoding="utf-8")) for sid, _ in items]
    write_jsonl_gz(OUTPUT / "cache/contracts_2017.jsonl.gz", rows)
    ok = [row for row in rows if row.get("status") == "OK"]
    summary = {
        "rows": len(rows), "ok": len(ok), "failed": len(rows) - len(ok),
        "component_total": sum(int(row.get("component_count", 0)) for row in ok),
        "entry_method_total": sum(int(row.get("entry_method_count", 0)) for row in ok),
        "mean_seconds": sum(float(row.get("extraction_seconds", 0)) for row in ok) / max(len(ok), 1),
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "contract_extraction_summary.json", summary)
    if len(ok) != len(rows):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: contract extraction failures {len(rows)-len(ok)}")
    print("CCBC_CONTRACT_EXTRACTION_DONE")


if __name__ == "__main__":
    main()

