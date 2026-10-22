"""Conservative result-centred dependency slicing over Dalvik instructions.

This module deliberately separates dependency evidence from call proximity.
Only register def-use, method return/argument substitution, and field
write/read propagation can connect sensitive roles into one behavior unit.
Call reachability, same-method membership, and same-class membership never do.

The implementation is a bounded, path-insensitive feasibility analyser.  It
does not claim FlowDroid/Soot-level alias precision.  Its outputs therefore
carry explicit evidence and limitation flags suitable for a dry-run audit.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable


REGISTER_RE = re.compile(r"\b[vp](\d+)\b", re.IGNORECASE)
RANGE_RE = re.compile(
    r"\b([vp])(\d+)\s*\.\.\s*([vp])(\d+)\b", re.IGNORECASE
)
METHOD_REF_RE = re.compile(r"(L[^\s,]+;->[^\s,(]+\([^)]*\)\S*)")
FIELD_REF_RE = re.compile(r"(L[^\s,]+;->[^\s,]+\s+\S+)")

TokenState = dict[str, frozenset[str]]


@dataclass(frozen=True)
class InstructionRecord:
    offset: int
    name: str
    output: str


@dataclass(frozen=True)
class MethodIR:
    signature: str
    parameter_registers: tuple[str, ...]
    instructions: tuple[InstructionRecord, ...]


@dataclass(frozen=True)
class BehaviorUnitTemplate:
    sink_id: str
    sink_role: str
    upstream_tokens: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass
class MethodSummary:
    return_state: TokenState
    field_writes: dict[str, TokenState]
    units: tuple[BehaviorUnitTemplate, ...]


@dataclass(frozen=True)
class SlicePolicy:
    max_iterations: int = 8
    max_units_per_method: int = 256
    max_tokens_per_value: int = 64


SINK_METHODS = {
    "network_transmission_or_upload": {
        "write", "send", "sendto", "execute", "enqueue",
    },
    "sms_send_receive": {
        "sendtextmessage", "sendmultiparttextmessage", "senddatamessage",
    },
    "command_execution": {"exec", "start", "startpipeline"},
    "dynamic_code_loading": {"loadclass", "defineclass"},
}


def canonical_reference(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def stable_id(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def registers(output: str) -> list[str]:
    """Return Dalvik registers, expanding invoke-range notation."""

    text = str(output or "")
    expanded: list[str] = []
    consumed: list[tuple[int, int]] = []
    for match in RANGE_RE.finditer(text):
        left_prefix, left, right_prefix, right = match.groups()
        if left_prefix.casefold() != right_prefix.casefold():
            continue
        start, end = int(left), int(right)
        if end >= start and end - start <= 255:
            expanded.extend(
                f"{left_prefix.casefold()}{index}" for index in range(start, end + 1)
            )
            consumed.append(match.span())
    for match in REGISTER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        expanded.append(match.group(0).casefold())
    return expanded


def method_reference(output: str) -> str:
    match = METHOD_REF_RE.search(str(output or ""))
    return canonical_reference(match.group(1)) if match else ""


def field_reference(output: str) -> str:
    match = FIELD_REF_RE.search(str(output or ""))
    return canonical_reference(match.group(1)) if match else ""


def method_name_from_reference(reference: str) -> str:
    value = canonical_reference(reference)
    if "->" not in value:
        return ""
    return value.split("->", 1)[1].split("(", 1)[0]


def copy_state(state: TokenState | None) -> TokenState:
    return {token: frozenset(evidence) for token, evidence in (state or {}).items()}


def merge_states(
    states: Iterable[TokenState | None], max_tokens: int = 64
) -> TokenState:
    merged: dict[str, set[str]] = defaultdict(set)
    for state in states:
        for token, evidence in (state or {}).items():
            merged[token].update(evidence)
    return {
        token: frozenset(merged[token])
        for token in sorted(merged)[:max_tokens]
    }


def add_evidence(state: TokenState, evidence: str) -> TokenState:
    return {
        token: frozenset(set(values) | {evidence})
        for token, values in state.items()
    }


def role_tokens(state: TokenState) -> set[str]:
    return {token[5:] for token in state if token.startswith("role:")}


def _substitute(
    state: TokenState,
    arguments: list[TokenState],
    relation: str,
    max_tokens: int,
) -> TokenState:
    values: list[TokenState] = []
    for token, evidence in state.items():
        if token.startswith("param:"):
            try:
                index = int(token.split(":", 1)[1])
            except ValueError:
                continue
            if index < len(arguments):
                values.append(add_evidence(arguments[index], relation))
        else:
            values.append({token: frozenset(set(evidence) | {relation})})
    return merge_states(values, max_tokens)


def _unit_key(unit: BehaviorUnitTemplate) -> tuple:
    return (
        unit.sink_id, unit.sink_role, unit.upstream_tokens, unit.evidence,
    )


def _deduplicate_units(
    units: Iterable[BehaviorUnitTemplate], cap: int
) -> tuple[BehaviorUnitTemplate, ...]:
    unique = {_unit_key(unit): unit for unit in units}
    return tuple(unique[key] for key in sorted(unique)[:cap])


def is_sink(role: str, reference: str) -> bool:
    return method_name_from_reference(reference) in SINK_METHODS.get(role, set())


def _analyse_method(
    method: MethodIR,
    summaries: dict[str, MethodSummary],
    field_state: dict[str, TokenState],
    map_roles: Callable[[str], set[str]],
    policy: SlicePolicy,
) -> MethodSummary:
    register_state: dict[str, TokenState] = {
        register: {f"param:{index}": frozenset({"METHOD_PARAMETER"})}
        for index, register in enumerate(method.parameter_registers)
    }
    pending_result: TokenState = {}
    return_values: list[TokenState] = []
    field_writes: dict[str, list[TokenState]] = defaultdict(list)
    units: list[BehaviorUnitTemplate] = []

    def state_for(register: str) -> TokenState:
        return register_state.get(register, {})

    for instruction in method.instructions:
        name = instruction.name.casefold()
        output = instruction.output
        regs = registers(output)

        if name.startswith("move-result"):
            if regs:
                register_state[regs[0]] = copy_state(pending_result)
            pending_result = {}
            continue

        if name.startswith("invoke-"):
            reference = method_reference(output)
            arguments = [state_for(register) for register in regs]
            incoming = merge_states(arguments, policy.max_tokens_per_value)
            mapped_roles = sorted(map_roles(reference)) if reference else []

            for sink_role in mapped_roles:
                if not is_sink(sink_role, reference):
                    continue
                evidence = set()
                for values in incoming.values():
                    evidence.update(values)
                if incoming:
                    evidence.add("ARGUMENT_DEF_USE")
                units.append(BehaviorUnitTemplate(
                    sink_id=stable_id(method.signature, instruction.offset, reference),
                    sink_role=sink_role,
                    upstream_tokens=tuple(sorted(incoming)),
                    evidence=tuple(sorted(evidence)),
                ))

            callee = summaries.get(reference)
            if callee is not None:
                pending_result = _substitute(
                    callee.return_state, arguments,
                    "INTERPROCEDURAL_RETURN", policy.max_tokens_per_value,
                )
                for nested in callee.units:
                    nested_state = {
                        token: frozenset(nested.evidence)
                        for token in nested.upstream_tokens
                    }
                    substituted = _substitute(
                        nested_state, arguments,
                        "INTERPROCEDURAL_ARGUMENT", policy.max_tokens_per_value,
                    )
                    evidence = set(nested.evidence)
                    for values in substituted.values():
                        evidence.update(values)
                    units.append(BehaviorUnitTemplate(
                        sink_id=nested.sink_id,
                        sink_role=nested.sink_role,
                        upstream_tokens=tuple(sorted(substituted)),
                        evidence=tuple(sorted(evidence)),
                    ))
            else:
                pending_result = copy_state(incoming)

            for role in mapped_roles:
                token_state = {
                    f"role:{role}": frozenset({"SENSITIVE_API_RESULT"})
                }
                pending_result = merge_states(
                    [pending_result, token_state], policy.max_tokens_per_value
                )
            continue

        pending_result = {}
        if name.startswith(("iput", "sput")):
            reference = field_reference(output)
            if reference and regs:
                field_writes[reference].append(
                    add_evidence(state_for(regs[0]), "FIELD_WRITE")
                )
            continue
        if name.startswith(("iget", "sget")):
            if regs:
                reference = field_reference(output)
                register_state[regs[0]] = add_evidence(
                    field_state.get(reference, {}), "FIELD_READ"
                )
            continue
        if name.startswith("return"):
            if regs:
                return_values.append(
                    add_evidence(state_for(regs[0]), "METHOD_RETURN")
                )
            continue
        if name.startswith("move"):
            if regs:
                register_state[regs[0]] = (
                    copy_state(state_for(regs[1])) if len(regs) > 1 else {}
                )
            continue
        if name.startswith("check-cast"):
            continue
        if name.startswith(("const", "new-instance", "new-array")):
            if regs:
                register_state[regs[0]] = {}
            continue
        if name.startswith("aget"):
            if regs:
                register_state[regs[0]] = merge_states(
                    [state_for(register) for register in regs[1:]],
                    policy.max_tokens_per_value,
                )
            continue
        if name.startswith("aput"):
            if len(regs) > 1:
                register_state[regs[1]] = merge_states(
                    [state_for(regs[0]), state_for(regs[1])],
                    policy.max_tokens_per_value,
                )
            continue
        if regs and (
            name.startswith((
                "add-", "sub-", "mul-", "div-", "rem-", "and-", "or-",
                "xor-", "shl-", "shr-", "ushr-", "neg-", "not-", "cmp",
                "array-length", "instance-of",
            ))
            or "/2addr" in name or "/lit" in name
        ):
            register_state[regs[0]] = merge_states(
                [state_for(register) for register in regs[1:]],
                policy.max_tokens_per_value,
            )

    writes = {
        field: merge_states(values, policy.max_tokens_per_value)
        for field, values in field_writes.items()
    }
    return MethodSummary(
        return_state=merge_states(return_values, policy.max_tokens_per_value),
        field_writes=writes,
        units=_deduplicate_units(units, policy.max_units_per_method),
    )


def _summary_fingerprint(
    summaries: dict[str, MethodSummary], field_state: dict[str, TokenState]
) -> tuple:
    method_rows = []
    for signature, summary in sorted(summaries.items()):
        method_rows.append((
            signature,
            tuple(sorted((token, tuple(sorted(ev))) for token, ev in summary.return_state.items())),
            tuple(_unit_key(unit) for unit in summary.units),
        ))
    field_rows = tuple(
        (field, tuple(sorted((token, tuple(sorted(ev))) for token, ev in state.items())))
        for field, state in sorted(field_state.items())
    )
    return tuple(method_rows), field_rows


def slice_program(
    methods: Iterable[MethodIR],
    map_roles: Callable[[str], set[str]],
    policy: SlicePolicy | None = None,
) -> dict:
    """Build result-centred, overlapping behavior units for one program."""

    policy = policy or SlicePolicy()
    method_map = {canonical_reference(method.signature): method for method in methods}
    summaries = {
        signature: MethodSummary({}, {}, tuple()) for signature in method_map
    }
    field_state: dict[str, TokenState] = {}
    converged = False
    iterations = 0
    previous = None
    for iterations in range(1, policy.max_iterations + 1):
        next_summaries = {
            signature: _analyse_method(
                method, summaries, field_state, map_roles, policy
            )
            for signature, method in method_map.items()
        }
        field_values: dict[str, list[TokenState]] = defaultdict(list)
        for summary in next_summaries.values():
            for field, state in summary.field_writes.items():
                # Cross-method field evidence is accepted only for resolved
                # semantic roles; unresolved caller parameters are not guessed.
                resolved = {
                    token: evidence for token, evidence in state.items()
                    if token.startswith("role:")
                }
                if resolved:
                    field_values[field].append(resolved)
        next_fields = {
            field: merge_states(values, policy.max_tokens_per_value)
            for field, values in field_values.items()
        }
        fingerprint = _summary_fingerprint(next_summaries, next_fields)
        summaries, field_state = next_summaries, next_fields
        if fingerprint == previous:
            converged = True
            break
        previous = fingerprint

    all_units = _deduplicate_units(
        (unit for summary in summaries.values() for unit in summary.units),
        max(1, len(method_map) * policy.max_units_per_method),
    )
    grouped: dict[str, list[BehaviorUnitTemplate]] = defaultdict(list)
    for unit in all_units:
        grouped[unit.sink_id].append(unit)

    retained_units = []
    for sink_id, candidates in sorted(grouped.items()):
        resolved = [
            unit for unit in candidates
            if role_tokens({token: frozenset() for token in unit.upstream_tokens})
        ]
        selected = resolved or candidates[:1]
        for unit in selected:
            upstream_roles = sorted(role_tokens({
                token: frozenset() for token in unit.upstream_tokens
            }))
            roles = sorted(set(upstream_roles) | {unit.sink_role})
            evidence = sorted(set(unit.evidence))
            retained_units.append({
                "sink_id": sink_id,
                "sink_role": unit.sink_role,
                "roles": roles,
                "order": len(roles),
                "dependency_status": (
                    "RESOLVED_DEPENDENCY" if upstream_roles
                    else "UNRESOLVED_SINGLETON"
                ),
                "evidence": evidence,
            })

    return {
        "units": retained_units,
        "method_count": len(method_map),
        "field_with_resolved_role_count": len(field_state),
        "fixed_point_iterations": iterations,
        "fixed_point_converged": converged,
        "analysis_characteristics": {
            "path_sensitive": False,
            "control_dependence_supported": False,
            "points_to_alias_supported": False,
            "intent_bundle_supported": False,
            "local_register_def_use_supported": True,
            "interprocedural_argument_return_supported": True,
            "resolved_field_write_read_supported": True,
            "call_proximity_used_as_dependency": False,
            "same_method_used_as_dependency": False,
            "same_class_used_as_dependency": False,
        },
    }
