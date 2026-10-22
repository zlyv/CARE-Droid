#!/usr/bin/env python3
"""Host-local CFG witness primitives; reads/writes no files."""
from __future__ import annotations

from collections import defaultdict

EXIT = -1


def reachable(cfg, source, target):
    if cfg.get('status') != 'PARSED':
        return None
    if source not in cfg.get('reachable', set()) or target not in cfg.get('reachable', set()):
        return None
    return bool(cfg['desc'][source] & cfg['bits'][target])


def relation(cfg, a, b):
    """Classify instruction-position reachability; same-block order is inherent."""
    ab, ba = reachable(cfg, a, b), reachable(cfg, b, a)
    if ab is None or ba is None:
        return 'UNKNOWN', ab, ba
    if ab and ba:
        return 'BIDIRECTIONAL', True, True
    if ab:
        return 'A_TO_B', True, False
    if ba:
        return 'B_TO_A', False, True
    return 'NEITHER', False, False


def strongly_connected_components(cfg):
    """Deterministic iterative Kosaraju SCCs over reachable CFG nodes.

    An iterative implementation is required because real DEX methods can have
    more than Python's default recursion limit of reachable instructions.
    """
    if cfg.get('status') != 'PARSED':
        return [], set()
    nodes = sorted(n for n in cfg['reachable'] if n != EXIT)
    succ = {n: sorted(v for v in cfg['succ'][n] if v != EXIT and v in cfg['reachable']) for n in nodes}
    pred = {n: [] for n in nodes}
    for source in nodes:
        for target in succ[source]:
            pred[target].append(source)
    for node in nodes:
        pred[node].sort()

    visited = set(); order = []
    for root in nodes:
        if root in visited:
            continue
        visited.add(root); stack = [(root, 0)]
        while stack:
            node, offset = stack[-1]
            if offset < len(succ[node]):
                nxt = succ[node][offset]
                stack[-1] = (node, offset + 1)
                if nxt not in visited:
                    visited.add(nxt); stack.append((nxt, 0))
            else:
                stack.pop(); order.append(node)

    assigned = set(); components = []
    for root in reversed(order):
        if root in assigned:
            continue
        assigned.add(root); stack = [root]; component = []
        while stack:
            node = stack.pop(); component.append(node)
            for nxt in reversed(pred[node]):
                if nxt not in assigned:
                    assigned.add(nxt); stack.append(nxt)
        components.append(sorted(component))
    loop_nodes = set()
    for comp in components:
        if len(comp) > 1 or (len(comp) == 1 and comp[0] in succ[comp[0]]):
            loop_nodes.update(comp)
    return components, loop_nodes


def basic_blocks(cfg):
    """Map instruction offsets to conservative normal-flow basic-block ids."""
    if cfg.get('status') != 'PARSED':
        return {}
    nodes = sorted(n for n in cfg['reachable'] if n != EXIT)
    normal = {n: sorted(v for v in cfg['normal'].get(n, set()) if v != EXIT and v in cfg['reachable']) for n in nodes}
    pred = defaultdict(set)
    for a in nodes:
        for b in normal[a]: pred[b].add(a)
    leaders = {cfg['entry']}
    for n in nodes:
        if len(pred[n]) != 1: leaders.add(n)
        if len(normal[n]) != 1:
            leaders.update(normal[n])
        for nxt in normal[n]:
            if len(pred[nxt]) != 1: leaders.add(nxt)
    result = {}; block = 0
    for leader in sorted(leaders):
        if leader in result: continue
        current = leader
        while current not in result:
            result[current] = block
            nxt = normal.get(current, [])
            if len(nxt) != 1 or nxt[0] in leaders or len(pred[nxt[0]]) != 1: break
            current = nxt[0]
        block += 1
    for n in nodes:
        if n not in result: result[n] = block; block += 1
    return result


def witness_pair(cfg, a, b, branch_sets=None):
    rel, ab, ba = relation(cfg, a, b)
    blocks = basic_blocks(cfg)
    _, loops = strongly_connected_components(cfg)
    branch_sets = branch_sets or {}
    ca, cb = set(branch_sets.get(a, ())), set(branch_sets.get(b, ()))
    return {
        'relation': rel, 'a_reaches_b': ab, 'b_reaches_a': ba,
        'same_basic_block': blocks.get(a) == blocks.get(b) if a in blocks and b in blocks else None,
        'a_in_loop': a in loops if rel != 'UNKNOWN' else None,
        'b_in_loop': b in loops if rel != 'UNKNOWN' else None,
        'shared_verified_control_branch': bool(ca & cb) if rel != 'UNKNOWN' else None,
        'different_verified_control_sets': bool(ca or cb) and ca != cb if rel != 'UNKNOWN' else None,
    }

