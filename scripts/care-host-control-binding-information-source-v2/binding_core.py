#!/usr/bin/env python3
"""Pure label-free functions. Proof direction is not class direction."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np

VERIFIED = {'VERIFIED_CONTROL', 'VERIFIED_PERMISSION_GUARD'}
TIERS = VERIFIED | {'PARTIAL', 'UNKNOWN'}

def node_id(v):
    if isinstance(v, bool):
        raise RuntimeError('boolean node ID')
    n = int(str(v))
    if n < 0:
        raise RuntimeError('negative node ID')
    return str(n)

def proof_tables(record, frozen):
    by = defaultdict(list)
    seen = set()
    pairs = frozen['pairs']
    for e in record['events']:
        if e['event_id'] in seen:
            raise RuntimeError('duplicate sensitive event ID')
        seen.add(e['event_id'])
        if not e['frozen_aligned']:
            continue
        pair = node_id(e['host_node_id']) + ':' + node_id(e['endpoint_node_id'])
        if pair not in pairs or set(map(int, e['role_ids'])) != set(map(int, pairs[pair])):
            raise RuntimeError('frozen event/pair Role mismatch')
        if e['evidence_tier'] not in TIERS:
            raise RuntimeError('unrecognized evidence tier')
        by[pair].append(e)
    keys, hosts, roles, flags, observed, count, unknown, verified_count = [], [], [], [], [], [], [], []
    for pair in sorted(pairs):
        h, target = pair.split(':')
        if node_id(h) != h or node_id(target) != target:
            raise RuntimeError('noncanonical frozen pair ID')
        rr = tuple(sorted(set(map(int, pairs[pair]))))
        if not rr or any(r < 0 or r >= 31 for r in rr):
            raise RuntimeError('invalid frozen Role set')
        es = by.get(pair, [])
        vv = [e for e in es if e['evidence_tier'] in VERIFIED]
        for e in vv:
            if not e.get('controls'):
                raise RuntimeError('verified fact has no proof')
            for c in e['controls']:
                if not c.get('event_postdominates_allowing_successor') or not c.get('blocked_successors_cannot_reach_event'):
                    raise RuntimeError('invalid strict control proof')
        dominant = any(c.get('branch_dominates_event') and not c.get('bypass_path_possible') for e in vv for c in e['controls'])
        bypass = any(c.get('bypass_path_possible') for e in vv for c in e['controls'])
        keys.append(pair); hosts.append(h); roles.append(rr)
        flags.append((bool(vv), bool(es) and len(vv) == len(es), dominant, bypass))
        observed.append(bool(es)); count.append(len(es))
        unknown.append(sum(e['evidence_tier'] == 'UNKNOWN' for e in es))
        verified_count.append(len(vv))
    return dict(keys=keys, hosts=hosts, roles=roles, flags=np.asarray(flags, float).reshape(-1, 4),
                observed=np.asarray(observed, bool), count=np.asarray(count, int),
                unknown=np.asarray(unknown, int), verified_count=np.asarray(verified_count, int))

def aggregates(table, flags=None):
    flags = table['flags'] if flags is None else np.asarray(flags, float)
    evidence = np.zeros((4, 31)); den = np.zeros(31); total = np.zeros(31); unk = np.zeros(31)
    for i, rr in enumerate(table['roles']):
        if table['observed'][i]:
            for r in rr:
                den[r] += 1; evidence[:, r] += flags[i]
                total[r] += table['count'][i]; unk[r] += table['unknown'][i]
    evidence /= np.maximum(den, 1)[None, :]
    coverage = np.concatenate([(den > 0).astype(float), unk / np.maximum(total, 1),
                               [table['observed'].sum() / max(len(table['keys']), 1),
                                table['unknown'].sum() / max(table['count'].sum(), 1)]])
    return evidence.reshape(-1), coverage

def host_context(values, records, model_roles):
    x = np.asarray(values, float)
    mapping = np.asarray(model_roles, int)
    if x.ndim != 2 or x.shape[1] not in (105, 169) or mapping.shape != (31,) or sorted(mapping.tolist()) != list(range(31)):
        raise RuntimeError('frozen C43 context schema mismatch')
    if not np.isfinite(x).all():
        raise RuntimeError('nonfinite frozen C43 values')
    if not records:
        if x.shape != (1, x.shape[1]) or np.any(x[:, :31]) or np.any(x[:, 43:105]):
            raise RuntimeError('invalid empty Host placeholder')
        return {}
    if len(records) != len(x):
        raise RuntimeError('frozen Host count mismatch')
    if x.shape[1] == 169 and np.any(x[:, 105:]):
        raise RuntimeError('A0 compatibility API/Host-call slots must be zero')
    result = {}
    for i, r in enumerate(records):
        if not np.allclose(x[i, :31], r['presence'], atol=1e-6, rtol=0):
            raise RuntimeError('C43 SelfRole/Host order mismatch')
        if not np.allclose(x[i, 43:74], r['endpoint_counts'], atol=1e-6, rtol=0):
            raise RuntimeError('C43 endpoint Role sidecar mismatch')
        if not np.allclose(x[i, 74:105], r['directed_neighbor_role'], atol=1e-6, rtol=0):
            raise RuntimeError('C43 Relation31 mismatch')
        key = node_id(r['host_id'])
        if key in result:
            raise RuntimeError('duplicate frozen Host ID')
        tail = x[i, 31:43] / (1 + np.abs(x[i, 31:43]))
        result[key] = np.concatenate([x[i, :31], tail, x[i, 74:105]])
    return result

def exact_strata(table):
    groups = defaultdict(list)
    for i, rr in enumerate(table['roles']):
        if table['observed'][i]:
            groups[(rr, int(table['count'][i]), int(table['unknown'][i]))].append(i)
    return [groups[k] for k in sorted(groups)]

def seed_for(sid, repeat, seed):
    return int.from_bytes(hashlib.sha256(f'{sid}|{repeat}|{seed}'.encode()).digest()[:8], 'big')

def permutation(table, seed):
    rng = np.random.default_rng(seed)
    donor = np.arange(len(table['keys']))
    eligible = np.zeros(len(donor), bool)
    for group in exact_strata(table):
        by_host = defaultdict(list)
        for i in group:
            by_host[table['hosts'][i]].append(i)
        if len(by_host) < 2:
            continue
        for i in group:
            eligible[i] = True
        pools = {h: list(rng.permutation(by_host[h])) for h in sorted(by_host)}
        while True:
            remaining = [h for h in sorted(pools) if pools[h]]
            if len(remaining) < 2:
                break
            remaining = list(rng.permutation(remaining))
            block = np.asarray([pools[h].pop() for h in remaining], int)
            donor[block] = np.roll(block, int(rng.integers(1, len(block))))
    validate_permutation(table, donor)
    return donor, eligible

def validate_permutation(table, donor):
    donor = np.asarray(donor, int)
    n = len(table['keys'])
    if donor.shape != (n,) or not np.array_equal(np.sort(donor), np.arange(n)):
        raise RuntimeError('not a complete donor permutation')
    for i, j in enumerate(donor):
        if i == j:
            continue
        if table['hosts'][i] == table['hosts'][j] or table['roles'][i] != table['roles'][j]:
            raise RuntimeError('donor crosses Role set or remains in same Host')
        if not table['observed'][i] or table['count'][i] != table['count'][j] or table['unknown'][i] != table['unknown'][j]:
            raise RuntimeError('donor changes receiver observation profile')
    before = aggregates(table)
    after = aggregates(table, table['flags'][donor])
    if not all(np.array_equal(a, b) for a, b in zip(before, after)):
        raise RuntimeError('previous Role aggregates or coverage changed')

def joint_descriptor(table, context, flags=None):
    flags = table['flags'] if flags is None else np.asarray(flags, float)
    out = np.zeros((31, 4, 74))
    indices = defaultdict(list)
    for i, rr in enumerate(table['roles']):
        if table['hosts'][i] not in context:
            raise RuntimeError('pair has no exactly aligned Host context')
        if table['observed'][i]:
            for r in rr:
                indices[r].append(i)
    for r, ii in indices.items():
        a = flags[ii]
        x = np.asarray([context[table['hosts'][i]] for i in ii])
        out[r] = (a - a.mean(axis=0)).T @ (x - x.mean(axis=0)) / len(ii)
    if not np.isfinite(out).all():
        raise RuntimeError('nonfinite joint descriptor')
    return out

def summarize_repeat(table, context, sid, repeat, seed, tolerance):
    donor, eligible = permutation(table, seed_for(sid, repeat, seed))
    shuffled = table['flags'][donor]
    moved = donor != np.arange(len(donor))
    changed = np.any(shuffled != table['flags'], axis=1)
    diff = joint_descriptor(table, context) - joint_descriptor(table, context, shuffled)
    observed = int(table['observed'].sum())
    verified = table['flags'][:, 0] > 0
    return dict(repeat=repeat, frozen_pairs=len(donor), observed_pairs=observed,
                verified_pairs=int(verified.sum()), eligible_pairs=int(eligible.sum()), moved_pairs=int(moved.sum()),
                changed_pairs=int(changed.sum()), changed_verified_pairs=int((changed & verified).sum()),
                moved_fraction=float(moved.sum() / max(observed, 1)),
                joint_rms_change=float(np.sqrt(np.mean(diff * diff))),
                joint_max_abs_change=float(np.max(np.abs(diff))),
                binding_ready=bool(changed.any() and np.max(np.abs(diff)) > tolerance),
                invariants_pass=True), donor
