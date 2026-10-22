#!/usr/bin/env python3
"""Resumable label-free P31 extraction for the CICMalDroid2020 cohort."""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import yaml

from common import OUTPUT, SOURCE_RESOLVED, norm, read_csv

SHARDS = OUTPUT / "cache/p31_shards"
TARGET_RE = re.compile(r"(L[^;]+;)->([^\s(]+)\(")


def suppress():
    os.environ["LOGURU_LEVEL"] = "CRITICAL"; logging.disable(logging.CRITICAL)
    try:
        from loguru import logger
        logger.remove(); logger.disable("androguard")
    except ImportError:
        pass


def canon(value):
    return str(value or "").strip().casefold().replace(" ", "")


def safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def dex_class():
    try:
        from androguard.core.dex import DEX
        return DEX
    except ImportError:
        from androguard.core.bytecodes.dvm import DalvikVMFormat
        return DalvikVMFormat


def apk_class():
    try:
        from androguard.core.apk import APK
        return APK
    except ImportError:
        from androguard.core.bytecodes.apk import APK
        return APK


def has_code(method):
    return callable(getattr(method, "get_code", None)) and safe(method.get_code) is not None


def unwrap(candidate, vm):
    if has_code(candidate): return candidate
    getter = getattr(candidate, "get_method", None); decoded = safe(getter) if callable(getter) else None
    if decoded is not None and has_code(decoded): return decoded
    cls = safe(getattr(candidate, "get_class_name", lambda: ""), "")
    name = safe(getattr(candidate, "get_name", lambda: ""), "")
    desc = safe(getattr(candidate, "get_descriptor", lambda: ""), "")
    resolver = getattr(vm, "get_method_descriptor", None)
    decoded = safe(lambda: resolver(cls, name, desc)) if callable(resolver) else None
    if isinstance(decoded, (list, tuple)): decoded = decoded[0] if decoded else None
    if decoded is not None and has_code(decoded): return decoded
    index_getter = getattr(candidate, "get_method_idx", None); by_index = getattr(vm, "get_method_by_idx", None)
    decoded = safe(lambda: by_index(index_getter())) if callable(index_getter) and callable(by_index) else None
    return decoded if decoded is not None and has_code(decoded) else None


def instructions(method):
    getter = getattr(method, "get_instructions_idx", None)
    iterator = getter() if callable(getter) else ((0, x) for x in getattr(method, "get_instructions", lambda: ())())
    for _, ins in iterator:
        yield str(safe(getattr(ins, "get_name", lambda: ""), "") or ""), str(safe(getattr(ins, "get_output", lambda: ""), "") or "")


def match(rule, cls, name):
    if rule.get("rule_kind", "api") != "api": return False
    cls, name = canon(cls), canon(name)
    prefix = [canon(x) for x in rule.get("class_prefixes", [])]
    contains = [canon(x) for x in rule.get("class_contains", [])]
    methods = [canon(x) for x in rule.get("method_names", [])]
    method_prefix = [canon(x) for x in rule.get("method_prefixes", [])]
    return bool(prefix or contains) and bool(methods or method_prefix) and \
        (any(cls.startswith(x) for x in prefix) or any(x in cls for x in contains)) and \
        (name in methods or any(name.startswith(x) for x in method_prefix))


def extract(row, rules, role_index):
    APK, DEX = apk_class(), dex_class(); path = Path(row["absolute_path"]); started = time.perf_counter()
    role_methods = np.zeros(31, np.float64); sensitive_methods = mapped = class_n = method_n = 0
    apk = APK(str(path))
    for blob in apk.get_all_dex():
        vm = DEX(blob)
        try: class_n += len(list(vm.get_classes()))
        except Exception: pass
        methods = list(vm.get_methods()); method_n += len(methods); candidates = []
        try:
            for dex_cls in vm.get_classes(): candidates.extend(list(dex_cls.get_methods()))
        except Exception: candidates = []
        if not candidates: candidates = methods
        for candidate in candidates:
            method = unwrap(candidate, vm)
            if method is None: continue
            found = set()
            for opname, output in instructions(method):
                if not opname.casefold().startswith("invoke-"): continue
                hits = TARGET_RE.findall(output)
                if not hits: continue
                cls, name = hits[-1]
                roles = {role_index[rule["role"]] for rule in rules if match(rule, cls, name)}
                if roles: mapped += 1; found.update(roles)
            if found: sensitive_methods += 1; role_methods[list(found)] += 1
        vm = None; gc.collect()
    if sensitive_methods: role_methods /= sensitive_methods
    audit = np.asarray([path.stat().st_size, class_n, method_n, sensitive_methods, mapped,
                        time.perf_counter() - started], float)
    return role_methods.astype(np.float32), audit


def atomic(path, **arrays):
    temporary = path.with_suffix(".tmp.npz"); np.savez_compressed(temporary, **arrays); os.replace(temporary, path)


def main():
    suppress(); rows = read_csv(OUTPUT / "dataset_manifest_label_free.csv")
    if not rows: raise RuntimeError("BLOCKED_BY_CONTRACT: empty manifest")
    resolved = json.loads(SOURCE_RESOLVED.read_text(encoding="utf-8"))
    mapping_path = Path(resolved["mapping"]); mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    roles = list(mapping["roles"]); role_index = {role: i for i, role in enumerate(roles)}
    rules = [rule for rule in mapping["rules"] if rule.get("rule_kind", "api") == "api"]
    if len(roles) != 31: raise RuntimeError("BLOCKED_BY_CONTRACT: Role31 mapping changed")
    SHARDS.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (SHARDS / (norm(row["sha256"]) + ".npz")).is_file()]
    print(f"CIC_P31 total={len(rows)} completed={len(rows)-len(pending)} pending={len(pending)}", flush=True)
    failures = []
    for index, row in enumerate(pending, 1):
        try:
            p31, audit = extract(row, rules, role_index)
            atomic(SHARDS / (norm(row["sha256"]) + ".npz"), p31=p31, audit=audit)
        except Exception as exc:
            failures.append({"sha256": row["sha256"], "error_type": type(exc).__name__, "error": str(exc)})
        if index % 10 == 0 or index == len(pending):
            print(f"CIC_P31 {index}/{len(pending)} failed={len(failures)}", flush=True)
    if failures:
        (OUTPUT / "p31_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: P31 failures={len(failures)}")
    ids, p31, audit = [], [], []
    for row in rows:
        sid = norm(row["sha256"])
        with np.load(SHARDS / (sid + ".npz"), allow_pickle=False) as archive:
            p31.append(archive["p31"]); audit.append(archive["audit"])
        ids.append(sid)
    np.savez_compressed(OUTPUT / "cicmaldroid2020_p31.npz", sha256=np.asarray(ids),
        role_prevalence=np.asarray(p31, np.float32), audit_values=np.asarray(audit, float),
        audit_names=np.asarray(["apk_bytes", "class_count", "method_ref_count", "sensitive_method_count",
                                "mapped_invoke_count", "total_seconds"]), role_names=np.asarray(roles))
    print(f"CIC_P31_EXTRACTION_DONE rows={len(rows)} labels_loaded=false", flush=True)


if __name__ == "__main__":
    main()

