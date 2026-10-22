#!/usr/bin/env python3
"""Manual post-review freezing; no key access or automatic unsealing."""
import csv
import hashlib
import json
import pathlib
from blind_audit import BASE, HEADERS, REF_HEADERS, load_packets, reference_index

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    submitted = BASE / 'blind_review_submission.csv'
    citations = BASE / 'blind_evidence_references.csv'
    frozen = BASE / 'blind_review_frozen.csv'
    manifest = BASE / 'blind_review_freeze_manifest.json'
    if frozen.exists() or manifest.exists():
        raise RuntimeError('FAIL_CLOSED: a freeze already exists; never overwrite')
    if not submitted.exists() or not citations.exists():
        print('AWAITING_BLIND_SEMANTIC_REVIEW: submission or citations file absent')
        return
    packets = load_packets()
    with submitted.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != HEADERS:
            raise RuntimeError('FAIL_CLOSED: form schema changed')
        rows = list(rd)
    if [r['case_id'] for r in rows] != list(packets):
        raise RuntimeError('FAIL_CLOSED: all eight cases must appear in fixed order')
    allowed = {'VERIFIABLE_BEHAVIOR_DIFFERENCE', 'STRUCTURAL_DIFFERENCE_ONLY',
               'NO_VERIFIABLE_DISTINCTION', 'INSUFFICIENT_EVIDENCE', 'IDENTITY_AMBIGUITY'}
    for row in rows:
        if any(not str(v).strip() for v in row.values()):
            raise RuntimeError('FAIL_CLOSED: every review field must be filled; use UNKNOWN explicitly')
        if row['reliably_interpretable_Dex_difference'] not in allowed:
            raise RuntimeError('FAIL_CLOSED: invalid case conclusion ' + row['case_id'])
        joined = ' '.join(row.values())
        import re
        if re.search(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', joined, re.I):
            raise RuntimeError('FAIL_CLOSED: SHA-like identifier in blinded form')
        if re.search(r'\b(FN|TN|TP|FP|malware|benign|malicious)\b', joined, re.I):
            raise RuntimeError('FAIL_CLOSED: label/prediction term in blinded form')
    with citations.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != REF_HEADERS:
            raise RuntimeError('FAIL_CLOSED: citation schema changed')
        ref_rows = list(rd)
    idx = reference_index(packets)
    for r in ref_rows:
        key = tuple(r[k] for k in REF_HEADERS[:6])
        if key not in idx:
            raise RuntimeError('FAIL_CLOSED: citation not present in packet: ' + repr(key))
        if not r['claim_id'].strip() or not r['certainty'].strip():
            raise RuntimeError('FAIL_CLOSED: citation lacks claim ID/certainty')
    cited = {r['case_id'] for r in ref_rows}
    for r in rows:
        if r['reliably_interpretable_Dex_difference'] in {
                'VERIFIABLE_BEHAVIOR_DIFFERENCE', 'STRUCTURAL_DIFFERENCE_ONLY'} and r['case_id'] not in cited:
            raise RuntimeError('FAIL_CLOSED: decisive difference lacks packet citation')
    # No automatic assertion that a packet citation proves behavior or class direction.
    frozen.write_bytes(submitted.read_bytes())
    with manifest.open('x', encoding='utf-8') as f:
        json.dump({'status': 'FROZEN_BLINDED_HUMAN_SUBMISSION',
                   'sha256': sha(frozen), 'citation_sha256': sha(citations),
                   'reviewed_cases': 8, 'validated_citations': len(ref_rows),
                   'raw_dex_verification': 'UNKNOWN',
                   'unsealed': False}, f, indent=2, sort_keys=True)
        f.write('\n')
    print('BLIND_REVIEW_FROZEN sha256=' + sha(frozen))

if __name__ == '__main__':
    main()
