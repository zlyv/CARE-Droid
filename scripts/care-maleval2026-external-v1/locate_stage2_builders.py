#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from common import OUTPUT, PROJECT, SOURCE_RESOLVED, dump_json, file_record, read_json, sha256_file


def files_named(name):
    return sorted(path for path in PROJECT.glob("scripts/**/" + name) if path.is_file())


def locate_by_hash(wanted, patterns):
    for pattern in patterns:
        for path in sorted(PROJECT.glob(pattern)):
            if path.is_file() and sha256_file(path) == wanted:
                return path
    return None


def main():
    resolved = read_json(SOURCE_RESOLVED); mapping = Path(resolved["mapping"])
    schema = read_json(PROJECT / "artifacts/p1-features/feature_schema.json")
    priority = locate_by_hash(schema["priority_sha256"], ("**/*.yaml", "**/*.yml"))
    compact = files_named("run_compact_feature_extraction_ag.py")
    endpoint = files_named("run_endpoint_centered_tensor_cache_ag.py")
    edge = []
    for path in sorted(PROJECT.glob("**/*.yaml")) + sorted(PROJECT.glob("**/*.yml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("edge_rules"), list): edge.append(path)
        except Exception:
            continue
    if not mapping.is_file() or priority is None or not compact or not endpoint or not edge:
        raise RuntimeError("BLOCKED_BY_CONTRACT: cannot resolve frozen stage-II builders/configuration")
    edge.sort(key=lambda path: (0 if "edge" in path.name.casefold() else 1,
                                0 if "semantic" in str(path).casefold() else 1,
                                len(str(path)), str(path)))
    selection = {"mapping": str(mapping.resolve()), "priority": str(priority.resolve()),
                 "edge_semantics": str(edge[0].resolve()), "compact_builder": str(compact[0].resolve()),
                 "endpoint_builder": str(endpoint[0].resolve())}
    dump_json(OUTPUT / "stage2_builders.json", selection)
    manifest_path = OUTPUT / "baseline_asset_manifest_before.json"
    manifest = read_json(manifest_path)
    known = {row["absolute_path"] for row in manifest}
    for purpose, value in selection.items():
        path = Path(value).resolve()
        if str(path) not in known:
            manifest.append(file_record(path, "stage2_" + purpose))
            known.add(str(path))
    dump_json(manifest_path, manifest)
    print("MALEVAL_STAGE2_BUILDERS_PASS", selection, flush=True)


if __name__ == "__main__":
    main()

