#!/usr/bin/env python3
"""Conservative intraprocedural S1/S2/S3 extraction for frozen CARE Host methods.

The input is decoded Dalvik instructions for one method and the exact frozen sensitive
invoke targets for that Host. The output contains categorized constants, return-use
states, and basic-block-local register def-use proofs. It does not infer CFG reachability
as data flow and reports uncertainty rather than following heap, alias, field, native,
reflection, exception, or cross-method propagation.
"""
from __future__ import annotations

import re
from collections import defaultdict

from common import canonical, stable_id

CONTROL = ("if-", "goto", "packed-switch", "sparse-switch", "throw")
MOVES = ("move", "move-object", "move-wide", "move/from16", "move-object/from16", "move-wide/from16",
         "move/16", "move-object/16", "move-wide/16")


def _return_type(reference):
    value = canonical(reference)
    return value.rsplit(")", 1)[-1] if ")" in value else ""


def categorize_string(value):
    """Return only a finite category; never return the original literal."""
    text = str(value or "").strip()
    low = text.casefold()
    if low.startswith("content://"):
        return "CONTENT_URI"
    if re.match(r"^(https?|wss?|ftp)://", low) or re.match(r"^[a-z0-9.-]+\.[a-z]{2,}([/:]|$)", low):
        return "NETWORK_ENDPOINT"
    if low.startswith(("android.intent.action.", "android.settings.", "android.provider.telephony.")):
        return "INTENT_ACTION"
    if re.match(r"^l[^;]+;$", low) or ("/" in low and low.endswith(("activity", "service", "receiver"))):
        return "COMPONENT_NAME"
    if any(x in low for x in ("aes", "rsa", "sha-", "sha1", "sha256", "md5", "hmac", "desede", "pkcs")):
        return "CRYPTO_ALGORITHM"
    if low.startswith(("/data/data/", "/data/user/", "/data/user_de/")) or "getfilesdir" in low:
        return "APP_PRIVATE_PATH"
    if low.startswith(("/sdcard/", "/storage/emulated/", "/mnt/sdcard/")):
        return "SHARED_STORAGE_PATH"
    if any(x in low for x in ("/dcim/", "/pictures/", "/movies/", "/music/", "/download/")):
        return "MEDIA_PATH"
    if text:
        return "STATIC_STRING"
    return "CONSTANT_UNKNOWN_CATEGORY"


def _constant(row):
    name = row.get("name", "")
    if name.startswith("const-string"):
        return ("CONST", categorize_string(row.get("string", "")))
    if name.startswith("const") and "class" not in name and "method" not in name:
        if "literal" not in row:
            return ("CONST", "CONSTANT_UNKNOWN_CATEGORY")
        value = int(row["literal"])
        return ("CONST", "BOOLEAN_CONSTANT" if value in (0, 1) else "INTEGER_CONSTANT")
    return None


def analyze_method(instructions, frozen_pairs, tries_size=0, maximum_instructions=4096):
    """Extract S1/S2/S3 with conservative straight/basic-block-local register tracking."""
    if len(instructions) > maximum_instructions:
        return {"status": "RESOURCE_LIMIT", "instruction_count": len(instructions), "events": [], "dependencies": [],
                "unknown_reasons": ["METHOD_INSTRUCTION_LIMIT"]}
    state = {}
    events = []
    by_event = {}
    dependencies = []
    unknown = set()
    pending = None
    other_pending = False
    branch_tainted = False

    def mark_use(token, use):
        event_id = token[1]
        if event_id in by_event:
            by_event[event_id]["return_uses"].add(use)

    for index, row in enumerate(instructions):
        name = str(row.get("name", "")).casefold()
        regs = list(map(int, row.get("registers", []) or []))
        if row.get("decode_error"):
            unknown.add("INSTRUCTION_DECODE_ERROR")
        if pending is not None and not name.startswith("move-result"):
            by_event[pending]["return_uses"].add("RETURN_IGNORED")
            pending = None
        if other_pending and not name.startswith("move-result"):
            other_pending = False
        if name.startswith("if-"):
            for reg in regs:
                token = state.get(reg)
                if token and token[0] == "RESULT":
                    mark_use(token, "RESULT_USED_IN_CONDITION")
        if name.startswith(CONTROL) or name in ("monitor-enter", "monitor-exit"):
            for token in list(state.values()):
                if token and token[0] == "RESULT":
                    mark_use(token, "UNKNOWN")
            state.clear()
            branch_tainted = True
        const = _constant(row)
        if const and regs:
            state[regs[0]] = const
            continue
        if name in MOVES and len(regs) >= 2:
            value = state.get(regs[1])
            state.pop(regs[0], None)
            if value:
                if value[0] == "RESULT":
                    value = ("RESULT", value[1], int(value[2]) + 1)
                state[regs[0]] = value
            continue
        if name.startswith("move-result"):
            if pending is not None and regs:
                state[regs[0]] = ("RESULT", pending, 0)
                by_event[pending]["return_uses"].add("RESULT_STORED")
                pending = None
            elif other_pending:
                if regs:
                    # Any move-result writes its destination. A result from an unrelated
                    # call must therefore kill an older sensitive-result token in it.
                    state.pop(regs[0], None)
                other_pending = False
            else:
                unknown.add("ORPHAN_MOVE_RESULT")
            continue
        if name.startswith("invoke-"):
            ref = canonical(row.get("reference", ""))
            target = str(stable_id(ref)) if ref else ""
            result_tokens = [state.get(reg) for reg in regs if state.get(reg, (None,))[0] == "RESULT"]
            for token in result_tokens:
                mark_use(token, "RESULT_PASSED_TO_CALL")
            if target in frozen_pairs:
                event_id = f"e{len(events)}"
                categories = sorted({state[reg][1] for reg in regs if state.get(reg, (None,))[0] == "CONST"})
                if not categories:
                    categories = ["UNKNOWN"]
                event = {"event_id": event_id, "offset": int(row.get("offset", -1)), "endpoint_node_id": target,
                         "role_ids": sorted(map(int, frozen_pairs[target])), "s1_categories": categories,
                         "return_uses": set(), "branch_tainted_before_event": branch_tainted}
                events.append(event)
                by_event[event_id] = event
                for token in result_tokens:
                    src = by_event.get(token[1])
                    if src:
                        for sr in src["role_ids"]:
                            for tr in event["role_ids"]:
                                dependencies.append({"source_role": sr, "target_role": tr,
                                  "dependency_type": "DIRECT_REGISTER_FLOW" if int(token[2]) == 0 else "TRACKED_LOCAL_COPY_FLOW",
                                  "confidence_status": "EXACT_BASIC_BLOCK_REGISTER_DEF_USE", "support_count": 1})
                if _return_type(ref) != "v":
                    pending = event_id
                other_pending = False
            else:
                other_pending = True
            for reg in regs:
                # An invoke may alter object state; do not propagate object aliases through it.
                if state.get(reg, (None,))[0] == "CONST":
                    continue
            continue
        if name.startswith("return"):
            for reg in regs:
                token = state.get(reg)
                if token and token[0] == "RESULT":
                    mark_use(token, "RESULT_RETURNED")
        if name.startswith(("iput", "sput", "aput")):
            for reg in regs[:1]:
                token = state.get(reg)
                if token and token[0] == "RESULT":
                    mark_use(token, "RESULT_STORED")
        # Conservative kill for opcodes that define their first register.
        if regs and (name.startswith(("new-", "iget", "sget", "aget", "array-length", "instance-of", "cmp", "neg-", "not-"))
                     or "-to-" in name or name.startswith(("add-", "sub-", "mul-", "div-", "rem-", "and-", "or-", "xor-", "shl-", "shr-", "ushr-"))):
            state.pop(regs[0], None)
    if pending is not None:
        by_event[pending]["return_uses"].add("RETURN_IGNORED")
    if tries_size:
        unknown.add("EXCEPTIONAL_FLOW_NOT_PROPAGATED")
    priority = ["RESULT_PASSED_TO_CALL", "RESULT_RETURNED", "RESULT_USED_IN_CONDITION", "RESULT_STORED", "RETURN_IGNORED", "UNKNOWN"]
    for event in events:
        uses = set(event.pop("return_uses"))
        if not uses:
            uses.add("RETURN_IGNORED" if _return_type(next((r.get("reference", "") for r in instructions if int(r.get("offset", -2)) == event["offset"]), "")) == "v" else "UNKNOWN")
        event["s2_uses"] = sorted(uses)
        event["s2_primary"] = next((value for value in priority if value in uses), "UNKNOWN")
    merged = defaultdict(int)
    for row in dependencies:
        key = (row["source_role"], row["target_role"], row["dependency_type"], row["confidence_status"])
        merged[key] += 1
    dependencies = [{"source_role": k[0], "target_role": k[1], "dependency_type": k[2],
                     "confidence_status": k[3], "support_count": value} for k, value in sorted(merged.items())]
    status = "PARTIAL" if unknown else "SUCCESS"
    return {"status": status, "instruction_count": len(instructions), "events": events,
            "dependencies": dependencies, "unknown_reasons": sorted(unknown)}

