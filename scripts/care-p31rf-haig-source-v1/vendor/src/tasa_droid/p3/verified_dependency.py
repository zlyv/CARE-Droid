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

    # Index in the Dalvik invoke register list.
    # For invoke-virtual/interface, register 0 is normally the receiver.
    sink_argument_index: int = -1

    # direct_payload:
    #     The selected argument directly carries the value of interest.
    #
    # object_carrier:
    #     The selected argument is a request/packet-like object whose
    #     internal payload is not yet reconstructed.
    #
    # unresolved:
    #     The current analysis cannot identify an explicit payload argument.
    sink_argument_kind: str = "unknown"

    # Method name only, e.g. connect/execute/write/loadclass.
    # Full API signatures are still not persisted.
    sink_method_name: str = ""

    # Coarse library family only; full class/API signature is not persisted.
    sink_api_family: str = "other"

    # Explicit reason when this sink cannot currently be verified.
    resolution_hint: str = ""


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
        # Keep this list aligned with the curated semantic mapping's
        # network-result methods. The semantic mapper decides whether the
        # receiver class is a network API; this set only decides whether the
        # mapped call is a result-centred sink.
        "connect", "execute", "enqueue", "newcall",
        "getoutputstream", "send", "sendto", "write",
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


def origin_tokens(state: TokenState) -> set[str]:
    return {
        token for token in state
        if token.startswith("origin:")
    }


def unknown_tokens(state: TokenState) -> set[str]:
    return {
        token for token in state
        if token.startswith("unknown:")
    }


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
        unit.sink_id,
        unit.sink_role,
        unit.sink_argument_index,
        unit.sink_argument_kind,
        unit.sink_method_name,
        unit.sink_api_family,
        unit.upstream_tokens,
        unit.evidence,
        unit.resolution_hint,
    )


def _deduplicate_units(
    units: Iterable[BehaviorUnitTemplate], cap: int
) -> tuple[BehaviorUnitTemplate, ...]:
    unique = {_unit_key(unit): unit for unit in units}
    return tuple(unique[key] for key in sorted(unique)[:cap])



def api_family_from_reference(reference: str) -> str:
    value = canonical_reference(reference)

    families = (
        ("ljava/net/", "java_net"),
        ("ljavax/net/", "javax_net"),
        ("lokhttp3/", "okhttp3"),
        ("lretrofit2/", "retrofit2"),
        ("lorg/apache/http/", "apache_http"),
    )

    for prefix, family in families:
        if value.startswith(prefix):
            return family

    return "other"


def is_sink(role: str, reference: str) -> bool:
    return method_name_from_reference(reference) in SINK_METHODS.get(role, set())



# ---------------------------------------------------------------------------
# Result-centred sink argument policy.
#
# IMPORTANT:
# We intentionally do NOT merge every invoke argument into one sink state.
# Only arguments with an explicit policy are treated as direct payloads.
#
# This prevents receiver/destination/callback registers from creating
# artificial source -> sink dependencies.
# ---------------------------------------------------------------------------

SINK_ARGUMENT_POLICY = {
    "network_transmission_or_upload": {
        # OutputStream/Writer-style writes.
        "write": (
            (0, "direct_payload"),
        ),

        # These objects may contain a payload, but P70 does not yet perform
        # object-field/points-to reconstruction for request/packet objects.
        "newcall": (
            (0, "object_carrier"),
        ),
        "send": (
            (0, "object_carrier"),
        ),
        "sendto": (
            (0, "object_carrier"),
        ),
    },

    "sms_send_receive": {
        # SmsManager.sendTextMessage(
        #   destinationAddress,
        #   scAddress,
        #   text,
        #   sentIntent,
        #   deliveryIntent
        # )
        "sendtextmessage": (
            (2, "direct_payload"),
        ),

        # SmsManager.sendMultipartTextMessage(..., parts, ...)
        "sendmultiparttextmessage": (
            (2, "direct_payload"),
        ),

        # SmsManager.sendDataMessage(
        #   destinationAddress,
        #   scAddress,
        #   destinationPort,
        #   data,
        #   ...
        # )
        "senddatamessage": (
            (3, "direct_payload"),
        ),
    },

    "command_execution": {
        # Runtime.exec(command...)
        "exec": (
            (0, "direct_payload"),
        ),
    },

    "dynamic_code_loading": {
        # ClassLoader.loadClass(name)
        "loadclass": (
            (0, "direct_payload"),
        ),

        # Typical defineClass(name, bytes, offset, length).
        # byte[] is the behaviorally important input.
        "defineclass": (
            (1, "direct_payload"),
        ),
    },
}


NON_PAYLOAD_ENDPOINTS = {
    "network_transmission_or_upload": {
        "connect": (
            "effect_only",
            "TRANSPORT_CONNECT_EFFECT",
        ),
        "getoutputstream": (
            "sink_handle",
            "OUTPUT_STREAM_ACQUISITION",
        ),
    },
}


NO_EXPLICIT_PAYLOAD_SINKS = {
    "network_transmission_or_upload": {
        "connect",
        "execute",
        "enqueue",
        "getoutputstream",
    },
    "command_execution": {
        "start",
        "startpipeline",
    },
}



TYPE_RE = re.compile(
    r"\[*("
    r"L[^;]+;"
    r"|[ZBCSIJFD]"
    r")"
)


def method_parameter_types(reference: str) -> list[str]:
    """Parse Dalvik parameter descriptors conservatively.

    canonical_reference() lower-cases method references, therefore this
    parser accepts both upper- and lower-case descriptor letters.

    Examples:
        Ljava/lang/String;
        [B
        [[I
        Lorg/apache/http/HttpRequest;
    """

    value = canonical_reference(reference)

    if "(" not in value or ")" not in value:
        return []

    descriptor = value.split("(", 1)[1].split(")", 1)[0]

    result: list[str] = []
    index = 0

    primitive_types = {
        "z", "b", "c", "s",
        "i", "j", "f", "d",
    }

    while index < len(descriptor):
        start_index = index

        # Array dimensions.
        while (
            index < len(descriptor)
            and descriptor[index] == "["
        ):
            index += 1

        if index >= len(descriptor):
            break

        kind = descriptor[index].casefold()

        # Object descriptor.
        if kind == "l":
            end_index = descriptor.find(
                ";",
                index,
            )

            if end_index < 0:
                # Malformed descriptor: stop rather than guessing.
                break

            result.append(
                descriptor[
                    start_index:end_index + 1
                ]
            )

            index = end_index + 1
            continue

        # Primitive descriptor.
        if kind in primitive_types:
            result.append(
                descriptor[
                    start_index:index + 1
                ]
            )

            index += 1
            continue

        # Unknown descriptor symbol.
        # Advance one position to avoid an infinite loop.
        index += 1

    return result

def carrier_selection(
    instruction_name: str,
    sink_role: str,
    reference: str,
    register_count: int,
) -> tuple[list[tuple[int, str]], str] | None:
    """Return carrier register positions for known API families.

    This stage only identifies the carrier object and exposes its existing
    provenance. It does not claim object-field reconstruction.
    """

    if sink_role != "network_transmission_or_upload":
        return None

    family = api_family_from_reference(reference)
    method = method_name_from_reference(reference)
    params = method_parameter_types(reference)

    is_static = instruction_name.startswith("invoke-static")
    receiver_offset = 0 if is_static else 1

    # --------------------------------------------------------
    # Apache HttpClient / HttpRequestExecutor:
    #
    # execute(HttpUriRequest ...)
    # execute(HttpHost, HttpRequest ...)
    #
    # Find the first request-like Java parameter instead of
    # blindly assuming logical arg0.
    # --------------------------------------------------------
    if family == "apache_http" and method == "execute":
        for logical_index, param in enumerate(params):
            lower = param.casefold()

            if "request" not in lower:
                continue

            actual_index = logical_index + receiver_offset

            if actual_index >= register_count:
                return [], "CARRIER_ARGUMENT_INDEX_OUT_OF_RANGE"

            return [
                (actual_index, "argument_carrier")
            ], ""

        return [], "APACHE_EXECUTE_REQUEST_ARGUMENT_NOT_FOUND"

    # --------------------------------------------------------
    # OkHttp:
    #
    # Call.enqueue(Callback)
    #
    # Callback is not payload. Request/body is already bound
    # to the Call object, so the carrier is the receiver.
    # --------------------------------------------------------
    if family == "okhttp3" and method == "enqueue":
        if is_static or register_count < 1:
            return [], "OKHTTP_RECEIVER_NOT_AVAILABLE"

        return [
            (0, "receiver_carrier")
        ], ""

    # Optional future-compatible handling for Call.execute()
    # when it has no Java parameters.
    if (
        family == "okhttp3"
        and method == "execute"
        and not params
    ):
        if is_static or register_count < 1:
            return [], "OKHTTP_RECEIVER_NOT_AVAILABLE"

        return [
            (0, "receiver_carrier")
        ], ""

    # --------------------------------------------------------
    # java.net DatagramSocket.send(DatagramPacket)
    # --------------------------------------------------------
    if family == "java_net" and method == "send":
        for logical_index, param in enumerate(params):
            if "datagrampacket" not in param.casefold():
                continue

            actual_index = logical_index + receiver_offset

            if actual_index >= register_count:
                return [], "CARRIER_ARGUMENT_INDEX_OUT_OF_RANGE"

            return [
                (actual_index, "argument_carrier")
            ], ""

    return None


def sink_argument_selection(
    instruction_name: str,
    sink_role: str,
    reference: str,
    register_count: int,
) -> tuple[list[tuple[int, str]], str]:
    """Return Dalvik invoke-register positions relevant to a sink.

    The configured indexes are logical Java parameter indexes excluding
    the receiver. For non-static invokes, +1 converts them to the Dalvik
    invoke register list because regs[0] is normally the receiver.

    A sink without an explicit payload argument is retained as unresolved
    rather than guessed.
    """

    method_name = method_name_from_reference(reference)

    endpoint_spec = NON_PAYLOAD_ENDPOINTS.get(
        sink_role, {}
    ).get(method_name)

    if endpoint_spec is not None:
        endpoint_kind, endpoint_reason = endpoint_spec

        return [], (
            "NON_PAYLOAD_ENDPOINT:"
            f"{endpoint_kind}:"
            f"{endpoint_reason}"
        )

    carrier = carrier_selection(
        instruction_name=instruction_name,
        sink_role=sink_role,
        reference=reference,
        register_count=register_count,
    )

    if carrier is not None:
        return carrier

    if method_name in NO_EXPLICIT_PAYLOAD_SINKS.get(sink_role, set()):
        return [], "NO_EXPLICIT_PAYLOAD_ARGUMENT"

    policy = SINK_ARGUMENT_POLICY.get(sink_role, {}).get(method_name)

    if policy is None:
        return [], "UNMODELED_SINK_ARGUMENT_POLICY"

    is_static = instruction_name.startswith("invoke-static")
    receiver_offset = 0 if is_static else 1

    selected: list[tuple[int, str]] = []

    for logical_index, kind in policy:
        actual_index = logical_index + receiver_offset

        if actual_index < 0 or actual_index >= register_count:
            return [], (
                "SINK_ARGUMENT_INDEX_OUT_OF_RANGE:"
                f"{method_name}:logical={logical_index}:"
                f"registers={register_count}"
            )

        selected.append((actual_index, kind))

    return selected, ""


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

                selections, unresolved_reason = sink_argument_selection(
                    instruction_name=name,
                    sink_role=sink_role,
                    reference=reference,
                    register_count=len(arguments),
                )

                sink_id = stable_id(
                    method.signature,
                    instruction.offset,
                    reference,
                )

                sink_method_name = method_name_from_reference(
                    reference
                )

                sink_api_family = api_family_from_reference(
                    reference
                )

                # Keep unsupported/no-explicit-payload sinks as unresolved
                # evidence instead of guessing from all invoke arguments.
                if not selections:
                    endpoint_kind = "unresolved"

                    if unresolved_reason.startswith(
                        "NON_PAYLOAD_ENDPOINT:"
                    ):
                        parts = unresolved_reason.split(":", 2)

                        if len(parts) >= 2:
                            endpoint_kind = parts[1]

                    units.append(BehaviorUnitTemplate(
                        sink_id=sink_id,
                        sink_role=sink_role,
                        upstream_tokens=tuple(),
                        evidence=(unresolved_reason,),
                        sink_argument_index=-1,
                        sink_argument_kind=endpoint_kind,
                        sink_method_name=sink_method_name,
                        sink_api_family=sink_api_family,
                        resolution_hint=unresolved_reason,
                    ))
                    continue

                for argument_index, argument_kind in selections:
                    argument_state = copy_state(arguments[argument_index])

                    evidence = set()
                    for values in argument_state.values():
                        evidence.update(values)

                    if argument_state:
                        evidence.add("SINK_ARGUMENT_DEF_USE")
                    else:
                        evidence.add("EMPTY_SINK_ARGUMENT_STATE")

                    # Carrier objects are intentionally NOT treated as
                    # verified payload dependencies until object-sensitive
                    # reconstruction is implemented.
                    if argument_kind == "object_carrier":
                        evidence.add("OBJECT_CARRIER_NOT_EXPANDED")
                        upstream_tokens = tuple()
                        resolution_hint = "OBJECT_CARRIER_NOT_EXPANDED"

                    elif argument_kind in {
                        "argument_carrier",
                        "receiver_carrier",
                    }:
                        upstream_tokens = tuple(
                            sorted(argument_state)
                        )
                        evidence.add("CARRIER_PROVENANCE")
                        resolution_hint = ""

                    else:
                        upstream_tokens = tuple(
                            sorted(argument_state)
                        )
                        resolution_hint = ""

                    units.append(BehaviorUnitTemplate(
                        sink_id=sink_id,
                        sink_role=sink_role,
                        upstream_tokens=upstream_tokens,
                        evidence=tuple(sorted(evidence)),
                        sink_argument_index=argument_index,
                        sink_argument_kind=argument_kind,
                        sink_method_name=sink_method_name,
                        sink_api_family=sink_api_family,
                        resolution_hint=resolution_hint,
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
                        sink_argument_index=nested.sink_argument_index,
                        sink_argument_kind=nested.sink_argument_kind,
                        sink_method_name=nested.sink_method_name,
                        sink_api_family=nested.sink_api_family,
                        resolution_hint=nested.resolution_hint,
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
        if name.startswith("const"):
            if regs:
                register_state[regs[0]] = {
                    "origin:constant": frozenset({
                        "CONSTANT_OR_LITERAL"
                    })
                }
            continue

        if name.startswith(("new-instance", "new-array")):
            if regs:
                # Object/array contents are not reconstructed by P70.
                # Keep an explicit unknown token so that lack of a mapped
                # sensitive role cannot be mistaken for negative evidence.
                register_state[regs[0]] = {
                    "unknown:local_allocation": frozenset({
                        "LOCAL_OBJECT_CONTENT_UNMODELED"
                    })
                }
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
            if any(
                role != unit.sink_role
                for role in role_tokens({
                    token: frozenset() for token in unit.upstream_tokens
                })
            )
        ]
        selected = resolved or candidates[:1]
        for unit in selected:
            upstream_roles = sorted(role_tokens({
                token: frozenset() for token in unit.upstream_tokens
            }))
            roles = sorted(set(upstream_roles) | {unit.sink_role})
            evidence = sorted(set(unit.evidence))
            cross_role = [
                role for role in upstream_roles if role != unit.sink_role
            ]
            upstream_state = {
                token: frozenset()
                for token in unit.upstream_tokens
            }

            origins = sorted(origin_tokens(upstream_state))
            unknowns = sorted(unknown_tokens(upstream_state))

            parameter_tokens = sorted(
                token for token in unit.upstream_tokens
                if token.startswith("param:")
            )

            unresolved_reason = ""

            if unit.sink_argument_kind in {
                "effect_only",
                "sink_handle",
            }:
                dependency_status = "NON_PAYLOAD_ENDPOINT"
                unresolved_reason = ""

            elif unit.sink_argument_kind in {
                "argument_carrier",
                "receiver_carrier",
            }:
                if cross_role:
                    dependency_status = (
                        "CARRIER_CROSS_ROLE_CANDIDATE"
                    )
                    unresolved_reason = ""

                else:
                    dependency_status = "UNRESOLVED"

                    if unit.resolution_hint:
                        unresolved_reason = unit.resolution_hint
                    elif unknowns:
                        unresolved_reason = unknowns[0]
                    elif parameter_tokens:
                        unresolved_reason = (
                            "UNRESOLVED_CARRIER_PARAMETER"
                        )
                    elif upstream_roles:
                        unresolved_reason = (
                            "CARRIER_PROVENANCE_WITHOUT_CROSS_ROLE"
                        )
                    elif not unit.upstream_tokens:
                        unresolved_reason = (
                            "EMPTY_CARRIER_PROVENANCE"
                        )
                    else:
                        unresolved_reason = (
                            "UNRESOLVED_CARRIER_PROVENANCE"
                        )

            elif cross_role:
                dependency_status = "VERIFIED_CROSS_ROLE_DEPENDENCY"

            elif upstream_roles:
                dependency_status = "VERIFIED_SAME_ROLE_DEPENDENCY"

            elif origins and not unknowns and not parameter_tokens:
                # "No sensitive source" means no source represented by the
                # configured semantic mapping. It does NOT mean benign.
                dependency_status = "RESOLVED_NO_SENSITIVE_SOURCE"

            else:
                dependency_status = "UNRESOLVED"

                if unit.resolution_hint:
                    unresolved_reason = unit.resolution_hint
                elif unknowns:
                    unresolved_reason = unknowns[0]
                elif parameter_tokens:
                    unresolved_reason = "UNRESOLVED_PARAMETER_OR_ENTRY"
                elif not unit.upstream_tokens:
                    unresolved_reason = "EMPTY_OR_UNMODELED_ARGUMENT_STATE"
                else:
                    unresolved_reason = "UNRESOLVED_OTHER"
            retained_units.append({
                "sink_id": sink_id,
                "sink_role": unit.sink_role,
                "source_roles": cross_role,
                "upstream_roles": upstream_roles,
                "sink_argument_index": unit.sink_argument_index,
                "sink_argument_kind": unit.sink_argument_kind,
                "sink_method_name": unit.sink_method_name,
                "sink_api_family": unit.sink_api_family,
                "roles": roles,
                "order": len(roles),
                "dependency_status": dependency_status,
                "unresolved_reason": unresolved_reason,
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
            "sink_argument_aware": True,
            "all_sink_arguments_merged": False,
            "object_carrier_payload_reconstruction": False,
            "explicit_unresolved_sink_policy": True,
        },
    }


# ---------------------------------------------------------------------------
# P70 verified-dependency development branch.
#
# This file was initially cloned from:
#   src/tasa_droid/p2/result_centered_slicing.py
#
# P2 is intentionally frozen for baseline reproducibility.
# New result-centered source -> sink verification logic must be implemented
# in this P3 copy only.
# ---------------------------------------------------------------------------

P3_ANALYSIS_VERSION = "p70_verified_dependency_v1"
