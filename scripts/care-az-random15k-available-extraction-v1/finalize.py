#!/usr/bin/env python3
"""Verify reused assets and summarize extraction coverage without fitting models."""
from __future__ import annotations

import json

from common import OUT, digest, write_json


def main():
    before = json.loads((OUT / 'asset_manifest_before.json').read_text(encoding='utf-8'))
    changed = []
    for old in before:
        try: now = digest(old['path'])
        except OSError as exc:
            changed.append({'path': old['path'], 'error': str(exc)}); continue
        if now != old: changed.append({'before': old, 'after': now})
    integrity = {'status': 'PASS' if not changed else 'FAIL', 'checked': len(before),
        'changed': changed, 'scope': 'named prior caches, manifests, builders and protocol files'}
    write_json(OUT / 'baseline_integrity_check.json', integrity)
    if changed: raise RuntimeError('reused asset changed')
    pre = json.loads((OUT / 'coverage_before.json').read_text())
    new = json.loads((OUT / 'new_stage2_asset_audit.json').read_text())
    reused = pre['stage2_status'].get('REUSE_COMPLETE_2017', 0) + pre['stage2_status'].get('REUSE_COMPLETE_FUTURE_ROUTED', 0)
    total = reused + new['complete_intersection']
    decision = {'status': 'AVAILABLE_SNAPSHOT_EXTRACTION_COMPLETE',
        'p31_complete': pre['p31_complete'], 'stage2_reused_complete': reused,
        'stage2_new_complete': new['complete_intersection'], 'stage2_total_complete': total,
        'stage2_total_rate_of_p31': total/max(pre['p31_complete'], 1),
        'labels_used': False, 'model_training_performed': False,
        'dataset_limit': 'incomplete and extremely class-imbalanced download snapshot; extraction only',
        'next_action': 'use these assets only after defining a leakage-safe, class-supported experiment'}
    write_json(OUT / 'final_decision.json', decision)
    print('AZ_AVAILABLE_SNAPSHOT_PIPELINE_DONE', json.dumps(decision), flush=True)


if __name__ == '__main__': main()

