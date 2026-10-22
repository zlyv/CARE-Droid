from __future__ import annotations

import hashlib
from collections import defaultdict, deque


def _stable_id(text: str) -> str:
    return hashlib.sha256(
        str(text).encode("utf-8", errors="ignore")
    ).hexdigest()


def _plumbing_opcode(name: str) -> bool:
    """
    Very conservative first-generation definition.

    Allowed:
      nop
      move*
      move-result*
      check-cast
      exactly one invoke-*
      return*

    Deliberately NOT allowed:
      const*
      iget/sget/aget
      iput/sput/aput
      arithmetic
      if/switch/goto
      monitor
      throw
      new-instance
      array construction
      etc.

    Therefore false-positive contraction should be low.
    """
    n = str(name).strip().casefold()

    if not n:
        return False

    # Exception handling has real control semantics.
    if n == "move-exception":
        return False

    if n == "nop":
        return True

    if n == "check-cast":
        return True

    if n.startswith("move"):
        return True

    if n.startswith("invoke-"):
        return True

    if n.startswith("return"):
        return True

    return False


def build_neutral_transit_profiles(
    method_irs,
    node_records,
    edge_records,
    canonicalize,
):
    """
    Identify only high-confidence semantic-neutral forwarding methods.

    IMPORTANT:
    Method/class names are NEVER used as evidence of obfuscation.
    a(), b(), foo(), onSomething() are treated identically.

    Neutral requires:
      - internal method;
      - decoded bytecode available;
      - exactly one invoke;
      - exactly one return;
      - all opcodes are conservative plumbing opcodes;
      - exactly one unique graph predecessor;
      - exactly one unique graph successor;
      - no ICC/reflection incident edge;
      - not constructor/class-initializer.

    Anything uncertain => meaningful (cost 1).
    """

    ir_by_signature = {}

    for ir in method_irs:
        try:
            signature = canonicalize(ir.signature)
        except Exception:
            signature = ""

        if signature:
            ir_by_signature[signature] = ir

    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    special_edge_nodes = set()

    for edge in edge_records:
        source = edge.get("source")
        target = edge.get("target")

        if (
            source not in node_records
            or target not in node_records
        ):
            continue

        outgoing[source].add(target)
        incoming[target].add(source)

        if (
            bool(edge.get("icc_any"))
            or bool(edge.get("reflection_any"))
        ):
            special_edge_nodes.add(source)
            special_edge_nodes.add(target)

    profiles = {}
    reason_counts = defaultdict(int)

    internal_count = 0
    decoded_count = 0
    neutral_count = 0

    for signature, record in node_records.items():

        if record.get("external"):
            continue

        internal_count += 1

        ir = ir_by_signature.get(signature)

        reason = None
        instruction_count = 0
        invoke_count = 0
        return_count = 0
        forbidden_count = 0

        if ir is None:
            reason = "NO_BYTECODE_IR"

        else:
            decoded_count += 1

            instructions = tuple(ir.instructions)
            instruction_count = len(instructions)

            names = [
                str(inst.name).strip().casefold()
                for inst in instructions
            ]

            invoke_count = sum(
                name.startswith("invoke-")
                for name in names
            )

            return_count = sum(
                name.startswith("return")
                for name in names
            )

            forbidden_count = sum(
                not _plumbing_opcode(name)
                for name in names
            )

            method_name = str(
                record.get("method_name", "")
            )

            if method_name in {"<init>", "<clinit>"}:
                reason = "CONSTRUCTOR"

            elif not instructions:
                reason = "EMPTY_CODE"

            elif invoke_count != 1:
                reason = "INVOKE_COUNT"

            elif return_count != 1:
                reason = "RETURN_COUNT"

            elif forbidden_count:
                reason = "MEANINGFUL_OPCODE"

            elif len(incoming[signature]) != 1:
                # Protect roots and join points.
                reason = "IN_DEGREE"

            elif len(outgoing[signature]) != 1:
                # Protect split/organizer points.
                reason = "OUT_DEGREE"

            elif signature in special_edge_nodes:
                reason = "SPECIAL_EDGE"

            elif record.get("roles"):
                reason = "SEMANTIC_ROLE"

            else:
                reason = "NEUTRAL_FORWARDER"

        neutral = reason == "NEUTRAL_FORWARDER"

        if neutral:
            neutral_count += 1

        reason_counts[reason] += 1

        profiles[signature] = {
            "neutral": neutral,
            "reason": reason,
            "instruction_count": instruction_count,
            "invoke_count": invoke_count,
            "return_count": return_count,
            "forbidden_count": forbidden_count,
            "in_degree": len(incoming[signature]),
            "out_degree": len(outgoing[signature]),
            "special_edge": (
                signature in special_edge_nodes
            ),
        }

    audit = {
        "internal_node_count": internal_count,
        "decoded_internal_count": decoded_count,
        "neutral_transit_count": neutral_count,
        "decoded_coverage": (
            decoded_count / internal_count
            if internal_count
            else 0.0
        ),
        "neutral_rate_all_internal": (
            neutral_count / internal_count
            if internal_count
            else 0.0
        ),
        "neutral_rate_decoded": (
            neutral_count / decoded_count
            if decoded_count
            else 0.0
        ),
        "reason_counts": dict(reason_counts),
    }

    neutral_flags = {
        signature: bool(profile["neutral"])
        for signature, profile in profiles.items()
    }

    return neutral_flags, profiles, audit


def build_semantic_upstream_view(
    node_records,
    edge_records,
    role_tier,
    endpoint_roles,
    semantic_hops=4,
    raw_hop_cap=16,
    k_in=8,
    endpoint_cap=128,
    node_cap=2048,
    edge_cap=4096,
    neutral_flags=None,
):
    """
    Wrapper-transparent reverse traversal.

    semantic distance:
        neutral transit caller -> +0
        meaningful caller      -> +1

    raw distance always increments by one and is used only as a
    safety cap against pathological wrapper chains.
    """

    # Reuse the existing deterministic endpoint-selection policy.
    from run_sensitive_upstream_tensor_cache_ag import (
        select_endpoints,
        edge_key,
    )

    neutral_flags = neutral_flags or {}
    allowed_roles = set(endpoint_roles)

    sensitive_all = {
        signature
        for signature, record in node_records.items()
        if record.get("roles")
        and any(
            role in allowed_roles
            for role in record.get("roles", [])
        )
    }

    endpoint_limit = min(
        max(1, endpoint_cap),
        max(1, node_cap),
    )

    selected_endpoints = select_endpoints(
        sensitive_all,
        node_records,
        role_tier,
        endpoint_limit,
    )

    endpoint_set = set(selected_endpoints)

    canonical_edges = {}
    incoming = defaultdict(list)

    for raw_edge in edge_records:
        key = edge_key(raw_edge)
        source, target = key[0], key[1]

        if (
            source not in node_records
            or target not in node_records
        ):
            continue

        edge = {
            "source": source,
            "target": target,
            "edge_type": key[2],
            "icc_any": key[3],
            "reflection_any": key[4],
        }

        canonical_edges[key] = edge
        incoming[target].append((source, key))

    def tier(signature):
        return min(
            (
                int(role_tier.get(role, 4))
                for role in
                node_records[signature].get(
                    "roles", []
                )
            ),
            default=4,
        )

    def caller_sort_key(signature):
        record = node_records[signature]

        return (
            0 if not record.get("external") else 1,
            0 if record.get("roles") else 1,
            tier(signature),
            _stable_id(signature),
        )

    for target in incoming:
        incoming[target].sort(
            key=lambda item:
            caller_sort_key(item[0])
        )

    # best[node] = (semantic_distance, raw_distance)
    best = {
        endpoint: (0, 0)
        for endpoint in selected_endpoints
    }

    queue = deque(selected_endpoints)
    traversed_edge_keys = set()

    while queue:
        current = queue.popleft()

        current_sem, current_raw = best[current]

        if current_raw >= raw_hop_cap:
            continue

        eligible = []

        for caller, key in incoming.get(current, []):
            caller_record = node_records[caller]

            # Same rule as the existing no-Entry extractor:
            # ordinary external/framework APIs are not caller context.
            if (
                caller_record.get("external")
                and caller not in endpoint_set
            ):
                continue

            eligible.append((caller, key))

        for caller, key in eligible[:k_in]:

            next_raw = current_raw + 1

            if next_raw > raw_hop_cap:
                continue

            is_neutral = (
                not node_records[caller].get("external")
                and bool(
                    neutral_flags.get(caller, False)
                )
            )

            step_cost = 0 if is_neutral else 1

            next_sem = current_sem + step_cost

            if next_sem > semantic_hops:
                continue

            traversed_edge_keys.add(key)

            candidate = (
                next_sem,
                next_raw,
            )

            previous = best.get(caller)

            if (
                previous is None
                or candidate < previous
            ):
                best[caller] = candidate

                # External APIs remain terminal anchors.
                if not node_records[caller].get(
                    "external"
                ):
                    if step_cost == 0:
                        queue.appendleft(caller)
                    else:
                        queue.append(caller)

    candidate_nodes = set(best)

    def node_sort_key(signature):
        record = node_records[signature]

        sem, raw = best.get(
            signature,
            (10**9, 10**9),
        )

        return (
            0 if signature in endpoint_set else 1,
            sem,
            raw,
            0 if not record.get("external") else 1,
            tier(signature),
            _stable_id(signature),
        )

    active = sorted(
        candidate_nodes,
        key=node_sort_key,
    )[:node_cap]

    active_set = set(active)

    node_truncated = (
        len(candidate_nodes) > len(active_set)
    )

    path_edges = [
        canonical_edges[key]
        for key in traversed_edge_keys
        if key in canonical_edges
        and canonical_edges[key]["source"]
            in active_set
        and canonical_edges[key]["target"]
            in active_set
    ]

    path_edges.sort(
        key=lambda edge: (
            best.get(
                edge["target"],
                (10**9, 10**9),
            ),
            best.get(
                edge["source"],
                (10**9, 10**9),
            ),
            _stable_id(
                "|".join(
                    map(str, edge_key(edge))
                )
            ),
        )
    )

    edge_truncated = len(path_edges) > edge_cap
    retained_edges = path_edges[:edge_cap]

    # Remove disconnected nodes after edge truncation,
    # but always preserve semantic endpoints.
    final_nodes = (
        set(selected_endpoints) & active_set
    )

    for edge in retained_edges:
        final_nodes.add(edge["source"])
        final_nodes.add(edge["target"])

    active = sorted(
        final_nodes,
        key=node_sort_key,
    )

    internal = {
        signature
        for signature in active
        if signature not in endpoint_set
        and not node_records[signature].get(
            "external"
        )
    }

    neutral_internal = {
        signature
        for signature in internal
        if neutral_flags.get(signature, False)
    }

    meaningful_internal = (
        internal - neutral_internal
    )

    active_sem = [
        best[s][0]
        for s in active
        if s in best
    ]

    active_raw = [
        best[s][1]
        for s in active
        if s in best
    ]

    summary = {
        "endpoint_count": len(
            set(selected_endpoints) & set(active)
        ),
        "retained_node_count": len(active),
        "retained_edge_count": len(
            retained_edges
        ),
        "internal_count": len(internal),
        "neutral_internal_count": len(
            neutral_internal
        ),
        "meaningful_internal_count": len(
            meaningful_internal
        ),
        "max_semantic_distance": (
            max(active_sem)
            if active_sem
            else 0
        ),
        "max_raw_distance": (
            max(active_raw)
            if active_raw
            else 0
        ),
        "node_truncated": bool(
            node_truncated
        ),
        "edge_truncated": bool(
            edge_truncated
        ),
    }

    semantic_distance = {
        s: best[s][0]
        for s in active
        if s in best
    }

    raw_distance = {
        s: best[s][1]
        for s in active
        if s in best
    }

    return (
        active,
        retained_edges,
        summary,
        semantic_distance,
        raw_distance,
    )
