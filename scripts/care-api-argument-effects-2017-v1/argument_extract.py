"""Read-only DEX extraction. Only application data is parsed; never executed."""
from __future__ import annotations
from collections import Counter, defaultdict
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from argument_core import analyze_method, aggregate_events, legacy_id, exact_reference, RUNTIME_REFS
from argument_io import read_json, write_json, sha256, status, signature

TARGET_IDS = {legacy_id(x) for x in RUNTIME_REFS} | {legacy_id("Ljava/lang/ProcessBuilder;->start()Ljava/lang/Process;")}
METHOD_REF = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+")


def decode_method(method):
    """Normalize Androguard operands, retaining case-sensitive signatures."""
    code = method.get_code()
    if code is None:
        return [], [], None
    objects = list(method.get_instructions_idx())
    by_offset = {int(off): obj for off, obj in objects}
    rows = []
    for offset, obj in objects:
        name = str(obj.get_name()).lower()
        row = {"offset": int(offset), "name": name, "registers": []}
        row["payload"] = "payload" in name
        if row["payload"]:
            rows.append(row); continue
        try:
            operands = obj.get_operands()
            row["registers"] = [int(v[1]) for v in operands if int(v[0]) == 0]
            if name.endswith("/range") and hasattr(obj, "AA") and hasattr(obj, "CCCC"):
                row["registers"] = list(range(int(obj.CCCC), int(obj.CCCC) + int(obj.AA)))
            literals = [int(v[1]) for v in operands if int(v[0]) == 1]
            if literals:
                row["literal"] = literals[0]
            if name.startswith("const-string"):
                strings = [str(v[2]) for v in operands if int(v[0]) == 257 and len(v) > 2]
                if not strings:
                    raise ValueError("missing decoded string operand")
                row["string"] = strings[0]
            output = str(obj.get_output())
            if name.startswith("invoke-"):
                refs = METHOD_REF.findall(re.sub(r"\s+", "", output))
                row["reference"] = refs[-1] if refs else ""
                if not row["reference"]:
                    raise ValueError("missing method reference")
            if name in {"new-instance", "new-array"} or name.startswith("filled-new-array"):
                types = re.findall(r"\[*L[^;\s]+;", output)
                row["type"] = types[-1] if types else ""
            if name.startswith(("if-", "goto")):
                row["target"] = int(offset) + 2 * int(obj.get_ref_off())
            if name in {"packed-switch", "sparse-switch"}:
                payload = by_offset.get(int(offset) + 2 * int(obj.get_ref_off()))
                if payload is None or str(payload.get_name()).lower() != name + "-payload":
                    raise ValueError("missing/mismatched switch payload")
                row["switch_targets"] = [int(offset) + 2 * int(v) for v in payload.get_targets()]
        except Exception as exc:
            row["decode_error"] = type(exc).__name__
        rows.append(row)
    # Use the already audited exception-table adapter without generic common imports.
    count = int(code.get_tries_size())
    if not count:
        return rows, [], None
    try:
        handlers = code.get_handlers()
        base = int(handlers.get_off())
        by = {int(h.get_off()) - base: h for h in handlers.get_list()}
        regions = []
        for entry in code.get_tries():
            handler = by[int(entry.get_handler_off())]
            targets = [2 * int(p.get_addr()) for p in handler.get_handlers()]
            if int(handler.get_size()) <= 0:
                targets.append(2 * int(handler.get_catch_all_addr()))
            regions.append({"start_offset_bytes": 2 * int(entry.get_start_addr()),
                            "end_offset_bytes": 2 * (int(entry.get_start_addr()) + int(entry.get_insn_count())),
                            "handlers": [{"target_offset_bytes": x} for x in targets]})
        if len(regions) != count:
            raise ValueError("try-table count")
        return rows, regions, None
    except Exception as exc:
        return rows, [], type(exc).__name__


def selected_hosts(view):
    return {hid: set(view["pairs_by_host"][hid]) & TARGET_IDS
            for hid in view["host_ids"] if set(view["pairs_by_host"][hid]) & TARGET_IDS}


def scan_one(sid, apk_path, view_path, output_path, contract):
    if sha256(apk_path) != sid:
        raise ValueError("APK content SHA mismatch: " + sid)
    view = read_json(view_path)
    if view["sha256"] != sid or int(view.get("year", -1)) != 2017:
        raise ValueError("frozen view SHA mismatch")
    selected = selected_hosts(view)
    methods, hosts = defaultdict(list), []
    started = time.monotonic()
    if selected:
        from loguru import logger
        logger.remove()
        from androguard.core.apk import APK
        from androguard.core.dex import DEX
        apk = APK(str(apk_path))
        for dex_name in sorted(apk.get_dex_names()):
            dex = DEX(apk.get_file(dex_name))
            for cls in dex.get_classes():
                for method in cls.get_methods():
                    reference = exact_reference(str(cls.get_name()) + "->" + str(method.get_name()) + str(method.get_descriptor()))
                    hid = legacy_id(reference)
                    if hid in selected:
                        methods[hid].append((method, dex_name, reference))
        for hid, targets in selected.items():
            candidates = methods[hid]
            row = {"host_id": hid, "expected_targets": len(targets), "events": []}
            if len(candidates) != 1:
                row["status"] = "AMBIGUOUS_HOST_ID" if candidates else "HOST_NOT_FOUND"
            else:
                method, dex_name, reference = candidates[0]
                rows, regions, error = decode_method(method)
                analysis = analyze_method(rows, targets, regions, error)
                row.update(analysis)
                row["dex_name"] = dex_name
                row["method_signature_sha256"] = __import__("hashlib").sha256(reference.encode()).hexdigest()
                observed = {event["endpoint_id"] for event in row["events"]}
                row["unrecovered_targets"] = len(targets - observed)
            hosts.append(row)
    if sha256(apk_path) != sid:
        raise ValueError("APK changed while being parsed: " + sid)
    events = [e for host in hosts for e in host["events"]]
    record = {"schema": __import__("argument_core").SCHEMA, "sha256": sid, "year": 2017,
              "contract_sha256": contract, "view_sha256": sha256(view_path),
              "apk_sha256_before_after": sid, "status": "SCANNED" if selected else "NO_TARGETS",
              "hosts": hosts, "summary": aggregate_events(events), "seconds": time.monotonic() - started}
    record["payload_sha256"] = signature(record)
    write_json(output_path, record)
    return record


def validate_cache(path, sid, view_path, contract):
    obj = read_json(path)
    payload = {k: v for k, v in obj.items() if k != "payload_sha256"}
    if (obj.get("sha256") != sid or obj.get("year") != 2017 or obj.get("contract_sha256") != contract
            or obj.get("view_sha256") != sha256(view_path) or obj.get("apk_sha256_before_after") != sid
            or obj.get("payload_sha256") != signature(payload)):
        raise ValueError("stale or foreign extraction cache: " + str(path))
    return obj


def run_scans(project, reference, out, lineage, config, workers=2, smoke=0):
    """One child per APK bounds parser memory and enforces a real timeout."""
    ids = sorted(lineage)
    if smoke:
        candidates = []
        for sid in ids:
            view = read_json(reference / f"features/frozen_views/{sid}.json.gz")
            if selected_hosts(view):
                candidates.append((len(view["host_ids"]), sid))
        ids = [sid for _, sid in sorted(candidates)[:smoke]]
        if not ids:
            write_json(out / "smoke/no_targets.json", {"status": "NO_COMMAND_HOSTS"})
            return
    folder = out / ("smoke/cache" if smoke else "cache")
    folder.mkdir(parents=True, exist_ok=True)
    def one(sid):
        view = reference / f"features/frozen_views/{sid}.json.gz"
        output = folder / (sid + ".json.gz")
        if output.exists():
            return validate_cache(output, sid, view, config["contract_sha256"])
        if not selected_hosts(read_json(view)):
            return scan_one(sid, lineage[sid]["absolute_path"], view, output, config["contract_sha256"])
        cmd = [sys.executable, "-B", str(Path(__file__).with_name("run_experiment.py")),
               "scan-one", "--project", str(project), "--reference", str(reference), "--output", str(out),
               "--sha", sid, "--apk", lineage[sid]["absolute_path"], "--cache-output", str(output)]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=900, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("static parser timeout for " + sid) from exc
        if result.returncode:
            # Parser stderr can contain app literals; retain the failure class only.
            raise RuntimeError("static parser child failed for " + sid + "; exit=" + str(result.returncode))
        return validate_cache(output, sid, view, config["contract_sha256"])
    failures, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, sid): sid for sid in ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append({"sha256": sid, "error": str(exc)})
            done += 1
            if done % 10 == 0 or done == len(ids):
                print("STATIC_ARGUMENT_PROGRESS", done, "/", len(ids), "failed", len(failures), flush=True)
                status(out, "RUNNING", "smoke" if smoke else "extraction", completed=done, total=len(ids), failed=len(failures))
    if failures:
        write_json(out / ("smoke/failures.json" if smoke else "reports/extraction_failures.json"), failures, replace=True)
        raise RuntimeError("APK parser failures; no samples may be silently excluded")
    if smoke:
        records = [read_json(folder / (sid + ".json.gz")) for sid in ids]
        write_json(out / "smoke/result.json", {
            "status": "REAL_APK_PARSE_COMPLETED", "apks": len(ids),
            "events": sum(r["summary"]["events"] for r in records),
            "resolved": sum(r["summary"]["resolved"] for r in records),
        }, replace=True)
