#!/usr/bin/env python3
"""Scan one frozen routed APK for Host-local CFG witnesses.

Reads one original APK and frozen Host/Role mapping. It never executes or
rewrites the APK. It writes one gzip JSON cache only inside the new artifact.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, sys, time, traceback
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'care-local-control-constraint-feasibility-v2'
sys.path.insert(0, str(PARENT)); sys.path.insert(0, str(HERE))
from common import canonical, stable_id
from control_core import build_cfg, controlling_branches
from dex_adapter import decode_method, matching_roles
from witness_core import strongly_connected_components, witness_pair

SCHEMA = 'care-cfg-rolegraph-witness-v1.0'
PAIR_CAP = 200000


def dump_gz(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'))
    temp.replace(path)


def scan(apk_path, sid, mapping, frozen):
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    try:
        from loguru import logger
        logger.remove()
    except ImportError: pass
    started = time.monotonic(); apk = APK(str(apk_path)); methods = []; counts = Counter(); dex_count = 0
    for dex_name in sorted(apk.get_dex_names()):
        raw = apk.get_file(dex_name); dex = DEX(raw); dsha = hashlib.sha256(raw).hexdigest(); dex_count += 1
        for cls in dex.get_classes():
            exact_cls = str(cls.get_name())
            for method in cls.get_methods():
                name = str(method.get_name()); desc = ''.join(str(method.get_descriptor()).split())
                signature = canonical(exact_cls + '->' + name + desc); hid = str(stable_id(signature))
                counts[hid] += 1; methods.append((method, dex_name, dsha, signature, hid))
    host_set = set(map(str, frozen['host_ids'])); host_rows = []; event_rows = []; pair_rows = []; failures = []
    recovered = set(); pair_cap_methods = 0
    for method, dex_name, dsha, signature, hid in methods:
        if hid not in host_set: continue
        if counts[hid] != 1:
            failures.append({'host_node_id': hid, 'method_signature': signature, 'reason': 'AMBIGUOUS_CANONICAL_HOST_ID'})
            continue
        recovered.add(hid)
        try: instructions, meta = decode_method(method)
        except Exception as exc:
            failures.append({'host_node_id': hid, 'method_signature': signature, 'reason': 'DEX_INSTRUCTION_DECODE_FAILED', 'detail': repr(exc)})
            continue
        cfg = build_cfg(instructions, meta['tries_size'], exception_regions=meta.get('exception_regions'), exception_error=meta.get('exception_error'))
        aligned = []
        for ins in instructions:
            if not ins['name'].startswith('invoke-') or not ins.get('reference'): continue
            ref = ins['reference']; target = str(stable_id(ref)); key = hid + ':' + target
            if key not in frozen['pairs']: continue
            role_names = matching_roles(ref, mapping)
            # Frozen role IDs are authoritative; matching must be nonempty and endpoint exact.
            if not role_names: continue
            event_id = hashlib.sha256(json.dumps([sid, dex_name, dsha, signature, ins['offset']], separators=(',', ':')).encode()).hexdigest()
            aligned.append({'event_id': event_id, 'offset': int(ins['offset']), 'opcode': ins['name'],
                            'reference': ref, 'role_ids': list(map(int, frozen['pairs'][key])),
                            'endpoint_node_id': target})
        components, loop_nodes = strongly_connected_components(cfg)
        blocks = len(set())
        branch_sets = {}
        for event in aligned:
            if cfg.get('status') == 'PARSED' and event['offset'] in cfg.get('reachable', set()):
                branch_sets[event['offset']] = [p['branch_offset'] for p in controlling_branches(cfg, event['offset'])]
            else: branch_sets[event['offset']] = []
            event_rows.append({'sha256': sid, 'dex_name': dex_name, 'dex_sha256': dsha,
                'host_node_id': hid, 'method_signature': signature, **event,
                'cfg_status': cfg.get('status'), 'cfg_reason': cfg.get('reason', ''),
                'cfg_sha256': cfg.get('cfg_sha256', ''),
                'entry_reachable': event['offset'] in cfg.get('reachable', set()),
                'in_loop': event['offset'] in loop_nodes if cfg.get('status') == 'PARSED' else None,
                'verified_control_branch_count': len(branch_sets[event['offset']])})
        potential = len(aligned) * (len(aligned) - 1) // 2
        pair_status = 'COMPLETE'
        if potential > PAIR_CAP:
            pair_status = 'PAIR_ENUMERATION_CAP'; pair_cap_methods += 1
        else:
            for a, b in combinations(aligned, 2):
                proof = witness_pair(cfg, a['offset'], b['offset'], branch_sets)
                for ra, rb in product(a['role_ids'], b['role_ids']):
                    pair_rows.append({'sha256': sid, 'dex_name': dex_name, 'dex_sha256': dsha,
                        'host_node_id': hid, 'method_signature': signature,
                        'event_a': a['event_id'], 'event_b': b['event_id'],
                        'offset_a': a['offset'], 'offset_b': b['offset'],
                        'role_a': int(ra), 'role_b': int(rb), **proof})
        role_set = sorted({r for e in aligned for r in e['role_ids']})
        relation_counts = Counter(r['relation'] for r in pair_rows if r['sha256'] == sid and r['host_node_id'] == hid)
        host_rows.append({'sha256': sid, 'dex_name': dex_name, 'dex_sha256': dsha,
            'host_node_id': hid, 'method_signature': signature,
            'instruction_count': len(instructions), 'cfg_status': cfg.get('status'),
            'cfg_reason': cfg.get('reason', ''), 'cfg_sha256': cfg.get('cfg_sha256', ''),
            'has_reachable_cycle': bool(loop_nodes), 'scc_count': len(components),
            'aligned_sensitive_calls': len(aligned), 'distinct_roles': len(role_set),
            'role_ids': role_set, 'sensitive_calls_in_loop': sum(e['offset'] in loop_nodes for e in aligned),
            'potential_event_pairs': potential, 'pair_status': pair_status,
            'a_to_b_pairs': relation_counts['A_TO_B'] + relation_counts['B_TO_A'],
            'bidirectional_pairs': relation_counts['BIDIRECTIONAL'],
            'neither_pairs': relation_counts['NEITHER'], 'unknown_pairs': relation_counts['UNKNOWN']})
    return {'schema': SCHEMA, 'status': 'OK', 'sha256': sid, 'labels_used': False,
        'future_rows_loaded': 0, 'dex_count': dex_count, 'frozen_hosts': len(host_set),
        'recovered_hosts': len(recovered), 'hosts': host_rows, 'events': event_rows,
        'pairs': pair_rows, 'failures': failures, 'pair_cap_methods': pair_cap_methods,
        'wall_seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser()
    for name in ('apk','sha','mapping','frozen','output','status'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); started = time.monotonic()
    try:
        import yaml
        result = scan(Path(args.apk), args.sha, yaml.safe_load(Path(args.mapping).read_text(encoding='utf-8')),
                      json.loads(Path(args.frozen).read_text(encoding='utf-8')))
        dump_gz(args.output, result)
        status = {'sha256': args.sha, 'status': 'OK', 'wall_seconds': time.monotonic()-started}
    except Exception as exc:
        status = {'sha256': args.sha, 'status': 'ERROR', 'error': repr(exc),
                  'traceback': traceback.format_exc(), 'wall_seconds': time.monotonic()-started}
    Path(args.status).parent.mkdir(parents=True, exist_ok=True)
    Path(args.status).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    if status['status'] != 'OK': raise SystemExit(1)


if __name__ == '__main__': main()

