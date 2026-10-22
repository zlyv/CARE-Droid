#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any

from common import canonical

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+", re.I)
MOVE = re.compile(r"^move(?:-wide|-object)?(?:/from16|/16)?$")
MOVE_RESULT = re.compile(r"^move-result(?:-wide|-object)?$")
RETURN = re.compile(r"^return(?:-void|-wide|-object)?$")
EXACT_INVOKES = {"invoke-static", "invoke-static/range", "invoke-direct", "invoke-direct/range", "invoke-super", "invoke-super/range"}
DYNAMIC_INVOKES = {"invoke-virtual", "invoke-virtual/range", "invoke-interface", "invoke-interface/range"}
FORBIDDEN_METHOD_NAMES = {"<init>", "<clinit>"}


def parse_method_reference(output: str) -> str | None:
    matches = METHOD_REFERENCE.findall(canonical(output))
    return canonical(matches[-1]) if matches else None


def split_reference(reference: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(L[^;]+;)->([^\s(]+)(\([^)]*\).+)", canonical(reference), re.I)
    return match.groups() if match else None


def return_type(descriptor: str) -> str:
    value = canonical(descriptor)
    return value.split(")", 1)[1] if ")" in value else ""


def register_numbers(output: str) -> list[int]:
    prefix = str(output).split("L", 1)[0]
    numbers = [int(value) for value in re.findall(r"\bv(\d+)\b", prefix, re.I)]
    if ".." in prefix and len(numbers) == 2 and numbers[1] >= numbers[0]:
        return list(range(numbers[0], numbers[1] + 1))
    return numbers


def api_rule_matches(rule: dict[str, Any], class_name: str, method_name: str) -> bool:
    if rule.get("rule_kind", "api") != "api":
        return False
    class_prefixes = [canonical(x) for x in rule.get("class_prefixes", [])]
    class_contains = [canonical(x) for x in rule.get("class_contains", [])]
    method_names = [canonical(x) for x in rule.get("method_names", [])]
    method_prefixes = [canonical(x) for x in rule.get("method_prefixes", [])]
    class_ok = any(class_name.startswith(x) for x in class_prefixes) or any(x in class_name for x in class_contains)
    method_ok = method_name in method_names or any(method_name.startswith(x) for x in method_prefixes)
    return bool(class_ok and method_ok and (class_prefixes or class_contains))


def match_role(reference: str, mapping: dict[str, Any]) -> dict[str, Any] | None:
    parts = split_reference(reference)
    if not parts:
        return None
    class_name, method_name, _ = parts
    for rule in mapping.get("rules", []):
        if api_rule_matches(rule, class_name, method_name):
            return {
                "rule_id": str(rule.get("rule_id", "")),
                "role": str(rule.get("role", "other")),
                "confidence": float(rule.get("confidence", 0.0)),
            }
    return None


def analyze_forwarder(method: dict[str, Any]) -> dict[str, Any]:
    name = canonical(method.get("name"))
    access = canonical(method.get("access"))
    records = list(method.get("instructions") or [])
    rejection = []
    if name in FORBIDDEN_METHOD_NAMES:
        rejection.append("constructor_or_class_initializer")
    if any(flag in access for flag in ("native", "abstract", "synchronized")):
        rejection.append("unsafe_access_flag")
    if not records:
        rejection.append("no_code")
    names = [canonical(row.get("name")) for row in records]
    allowed = lambda op: op == "nop" or MOVE.fullmatch(op) or MOVE_RESULT.fullmatch(op) or RETURN.fullmatch(op) or op.startswith("invoke-")
    disallowed = sorted({op for op in names if not allowed(op)})
    if disallowed:
        rejection.append("disallowed_opcode:" + ",".join(disallowed[:8]))
    invokes = [i for i, op in enumerate(names) if op.startswith("invoke-")]
    returns = [i for i, op in enumerate(names) if RETURN.fullmatch(op)]
    if len(invokes) != 1:
        rejection.append("invoke_count_not_one")
    if len(returns) != 1:
        rejection.append("return_count_not_one")
    if rejection:
        return {"candidate": False, "reasons": rejection}
    invoke_index, return_index = invokes[0], returns[0]
    if invoke_index >= return_index:
        return {"candidate": False, "reasons": ["invoke_not_before_return"]}
    invoke_name = names[invoke_index]
    if invoke_name not in EXACT_INVOKES | DYNAMIC_INVOKES:
        return {"candidate": False, "reasons": ["unsupported_invoke_opcode"]}
    target = parse_method_reference(str(records[invoke_index].get("output", "")))
    if not target:
        return {"candidate": False, "reasons": ["target_reference_unparsed"]}
    target_parts = split_reference(target)
    if not target_parts or target_parts[1] in FORBIDDEN_METHOD_NAMES:
        return {"candidate": False, "reasons": ["target_constructor_or_unparsed"]}

    registers_size = int(method.get("registers_size", 0))
    ins_size = int(method.get("ins_size", 0))
    if ins_size <= 0 or registers_size < ins_size:
        return {"candidate": False, "reasons": ["invalid_parameter_registers"]}
    params = list(range(registers_size - ins_size, registers_size))
    origin = {register: register for register in params}
    for index in range(invoke_index):
        op = names[index]
        if op == "nop":
            continue
        regs = register_numbers(str(records[index].get("output", "")))
        if not MOVE.fullmatch(op) or len(regs) < 2 or regs[1] not in origin:
            return {"candidate": False, "reasons": ["preinvoke_dataflow_not_pure"]}
        origin[regs[0]] = origin[regs[1]]
        if "wide" in op:
            if regs[1] + 1 not in origin:
                return {"candidate": False, "reasons": ["wide_source_not_parameter"]}
            origin[regs[0] + 1] = origin[regs[1] + 1]
    invoke_registers = register_numbers(str(records[invoke_index].get("output", "")))
    invoke_origins = [origin.get(register) for register in invoke_registers]
    if any(value is None for value in invoke_origins) or invoke_origins != params:
        return {"candidate": False, "reasons": ["arguments_not_exact_passthrough"]}

    tail = [(names[i], register_numbers(str(records[i].get("output", "")))) for i in range(invoke_index + 1, len(records)) if names[i] != "nop"]
    wrapper_return = return_type(str(method.get("descriptor", "")))
    target_return = return_type(target_parts[2])
    if wrapper_return != target_return:
        return {"candidate": False, "reasons": ["return_type_changed"]}
    if wrapper_return == "v":
        if len(tail) != 1 or tail[0][0] != "return-void":
            return {"candidate": False, "reasons": ["void_return_not_direct"]}
    else:
        if len(tail) != 2 or not MOVE_RESULT.fullmatch(tail[0][0]) or not RETURN.fullmatch(tail[1][0]):
            return {"candidate": False, "reasons": ["result_not_directly_returned"]}
        if len(tail[0][1]) != 1 or len(tail[1][1]) != 1 or tail[0][1][0] != tail[1][1][0]:
            return {"candidate": False, "reasons": ["returned_register_changed"]}
    tier = "VERIFIED_STRICT" if invoke_name in EXACT_INVOKES else "DECLARED_TARGET_ONLY"
    return {
        "candidate": True,
        "tier": tier,
        "target_reference": target,
        "invoke_opcode": invoke_name,
        "parameter_word_count": len(params),
        "argument_register_count": len(invoke_registers),
        "return_type": wrapper_return,
        "proof_flags": {
            "single_invoke": True,
            "straight_line": True,
            "no_extra_opcode": True,
            "exact_argument_forwarding": True,
            "direct_return": True,
            "exact_dispatch": tier == "VERIFIED_STRICT",
        },
        "reasons": [] if tier == "VERIFIED_STRICT" else ["dynamic_dispatch_not_strict"],
    }
