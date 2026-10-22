#!/usr/bin/env python3
"""Pure helpers for lightweight within-method semantic extraction."""
from __future__ import annotations

import re
from itertools import combinations
import numpy as np

ROLE_N = 31
COOC_N = ROLE_N * (ROLE_N - 1) // 2
TRANSITION_N = ROLE_N * ROLE_N
TARGET_RE = re.compile(r"(L[^;]+;)->([^\s(]+)\(")


def canonical(value):
    return str(value or "").strip().casefold().replace(" ", "")


def api_rule_matches(rule, class_name, method_name):
    if rule.get("rule_kind", "api") != "api":
        return False
    cls, name = canonical(class_name), canonical(method_name)
    prefixes = [canonical(x) for x in rule.get("class_prefixes", [])]
    contains = [canonical(x) for x in rule.get("class_contains", [])]
    methods = [canonical(x) for x in rule.get("method_names", [])]
    method_prefixes = [canonical(x) for x in rule.get("method_prefixes", [])]
    return bool(prefixes or contains) and bool(methods or method_prefixes) and (
        any(cls.startswith(x) for x in prefixes) or any(x in cls for x in contains)
    ) and (name in methods or any(name.startswith(x) for x in method_prefixes))


def target_from_instruction(output):
    matches = TARGET_RE.findall(str(output or ""))
    return matches[-1] if matches else ("", "")


def roles_for_target(class_name, method_name, rules, role_index):
    return tuple(sorted({role_index[r["role"]] for r in rules
                         if api_rule_matches(r, class_name, method_name)}))


def collapse_adjacent(values):
    out = []
    for value in values:
        if not out or out[-1] != value:
            out.append(int(value))
    return out


def summarize_methods(method_role_sequences, role_n=ROLE_N):
    """Return scale-controlled prevalence, co-occurrence and transitions."""
    prevalence = np.zeros(role_n, np.float32)
    cooc = np.zeros((role_n, role_n), np.float32)
    transitions = np.zeros((role_n, role_n), np.float32)
    sensitive_methods = 0
    transition_total = 0
    mapped_invokes = 0
    for raw_events in method_role_sequences:
        # Each event is one invoke instruction and may conservatively map to
        # more than one role.  Never invent an ordering between roles assigned
        # to the same invoke.
        events = []
        raw_mapped_event_count = 0
        for event in raw_events:
            if isinstance(event, (int, np.integer)):
                event = (int(event),)
            event = tuple(sorted(set(int(x) for x in event)))
            if event:
                raw_mapped_event_count += 1
            if event and (not events or events[-1] != event):
                events.append(event)
        if not events:
            continue
        sensitive_methods += 1
        mapped_invokes += raw_mapped_event_count
        unique = sorted({role for event in events for role in event})
        prevalence[unique] += 1
        for a, b in combinations(unique, 2):
            cooc[a, b] += 1
        for left, right in zip(events[:-1], events[1:]):
            weight = 1.0 / (len(left) * len(right))
            for a in left:
                for b in right:
                    transitions[a, b] += weight
            transition_total += 1
    if sensitive_methods:
        prevalence /= sensitive_methods
        cooc /= sensitive_methods
    if transition_total:
        transitions /= transition_total
    upper = cooc[np.triu_indices(role_n, 1)]
    return prevalence, upper, transitions.reshape(-1), {
        "sensitive_method_count": sensitive_methods,
        "mapped_invoke_count": mapped_invokes,
        "transition_count": transition_total,
    }

