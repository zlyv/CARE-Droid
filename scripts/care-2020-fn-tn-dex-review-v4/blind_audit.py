#!/usr/bin/env python3
"""Run only inside a sandbox with /review mounted and no repository access."""
import csv
import hashlib
import json
import pathlib
import re

BASE = pathlib.Path('/review')
HEADERS = ['case_id', 'reliably_interpretable_Dex_difference',
           'fact_and_method_location', 'possible_class_direction_before_unsealing',
           'is_fact_in_frozen_C43', 'uncertainty_or_counterexample', 'review_notes']
REF_HEADERS = ['case_id', 'side', 'dex', 'method', 'offset', 'reference', 'claim_id', 'certainty']

def exclusive_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')

def load_packets():
    paths = sorted((BASE / 'blind_packets').glob('*.json'))
    if [p.stem for p in paths] != [f'M{i:02d}' for i in range(1, 9)]:
        raise RuntimeError('FAIL_CLOSED: expected exactly eight packets M01..M08')
    packets = {}
    for p in paths:
        raw = p.read_text(encoding='utf-8')
        if re.search(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', raw, re.I):
            raise RuntimeError(f'FAIL_CLOSED: possible SHA leak in {p.name}')
        d = json.loads(raw)
        if d.get('case_id') != p.stem or set(d.get('sides', {})) != {'A', 'B'}:
            raise RuntimeError(f'FAIL_CLOSED: malformed packet {p.name}')
        packets[p.stem] = d
    return packets

def reference_index(packets):
    out = set()
    for case, d in packets.items():
        for side, s in d['sides'].items():
            for m in s.get('example_sensitive_methods', []):
                for c in m.get('sensitive_calls', []):
                    out.add((case, side, str(m['dex']), m['method'],
                             str(c['offset']), c['reference']))
    return out

def main():
    packets = load_packets()
    with (BASE / 'blind_review_working.csv').open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != HEADERS:
            raise RuntimeError('FAIL_CLOSED: existing review fields changed')
        rows = list(rd)
    if [r['case_id'] for r in rows] != list(packets):
        raise RuntimeError('FAIL_CLOSED: review rows differ from packets')
    if any(any(v.strip() for k, v in r.items() if k != 'case_id') for r in rows):
        raise RuntimeError('FAIL_CLOSED: working form unexpectedly populated')
    index = reference_index(packets)
    if not index:
        raise RuntimeError('FAIL_CLOSED: packets contain no verifiable citation entries')
    with (BASE / 'blind_evidence_references_template.csv').open('x', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(REF_HEADERS)
    audit = {
        'status': 'PASS_INTERNAL_PACKET_REFERENCES_ONLY',
        'packet_count': len(packets), 'case_ids': list(packets),
        'sides_per_case': 2, 'indexed_sensitive_invoke_references': len(index),
        'raw_apk_byte_verification': 'UNKNOWN: no label-free alias-to-APK manifest in blind workspace',
        'packet_completeness': 'PARTIAL: example_sensitive_methods may be truncated',
        'semantic_interpretation': 'NOT_PERFORMED',
        'blindness': 'No labels, predictions, sealed key, organizer records, or original APK paths mounted.'}
    exclusive_json(BASE / 'blind_reference_audit.json', audit)
    (BASE / 'REVIEW_INSTRUCTIONS.md').write_text(
        '# Independent blinded review\n\n'
        'Read each A and B independently before writing a pair conclusion. Fill the exact seven columns in '
        '`blind_review_working.csv`, then save a separate `blind_review_submission.csv`. '
        'Use UNKNOWN when the packets do not prove a condition, data flow, or class direction. '
        'For each decisive observed fact, add one row to `blind_evidence_references.csv` using the '
        'companion template. The citation must exactly match packet side, DEX index, method, invoke offset, '
        'and referenced API. Review of method names, permissions, or attention alone is not proof of intent.\n\n'
        'These packets are truncated examples. A missing call is not evidence of absence. This workspace '
        'cannot certify raw APK bytes because no anonymous alias-to-APK path map was provided. '
        'Do not access the project or key. Do not infer which side is FN/TN.\n', encoding='utf-8')
    print('BLIND_AUDIT_PASS cases=8 internal_refs=' + str(len(index)))

if __name__ == '__main__':
    main()
