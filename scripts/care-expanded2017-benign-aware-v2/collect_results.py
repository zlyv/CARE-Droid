#!/usr/bin/env python3
"""Freeze this experiment's descriptive results and recheck named reused assets."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from audit_new_apks import OUT, file_record, read_csv, write_csv, write_json
from formal_integrity import verify as verify_formal


def verified_json(name):
    value = json.loads((OUT / name).read_text())
    if value.get('status') != 'PASS':
        raise RuntimeError('required phase did not pass: ' + name)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--collect', action='store_true', required=True)
    p.parse_args()
    source = verified_json('stage1_training_audit.json')
    router = verified_json('router_training_audit.json')
    stage2_data = verified_json('stage2_data_audit.json')
    stage2 = verified_json('stage2_training_audit.json')
    evaluation = verified_json('evaluation_audit.json')
    baseline = json.loads((OUT / 'baseline_integrity_check.json').read_text())
    changed = []
    for record in baseline['before']:
        current = file_record(record['path'])
        if any(current[k] != record[k] for k in ('sha256', 'size', 'mtime_ns')):
            changed.append({'path': record['path'], 'before': record, 'after': current})
    integrity = {'status': 'PASS' if not changed else 'FAIL',
                 'changed': changed, 'asset_count': len(baseline['before']),
                 'scope': 'six named reused audit/manifest files; prior audit content-verified 3989 APKs'}
    write_json(OUT / 'baseline_integrity_final.json', integrity)
    if changed:
        raise RuntimeError('reused asset integrity changed')
    formal_integrity = verify_formal()
    write_json(OUT / 'formal_asset_integrity_check.json', formal_integrity)
    if formal_integrity['status'] != 'PASS':
        raise RuntimeError('named formal asset integrity changed')
    all_rows = read_csv(OUT / 'cascade_results.csv')
    stage1_rows = read_csv(OUT / 'stage1_results.csv')
    routes = read_csv(OUT / 'router_results.csv')
    predictions = read_csv(OUT / 'per_apk_predictions.csv')
    missing = Counter(r['arm'] for r in predictions if r['fallback_reason'])
    if any(int(r['routed']) and not int(r['expert_available']) for r in predictions):
        result_status = 'COMPLETE_WITH_STAGE2_ASSET_FALLBACK'
    else:
        result_status = 'COMPLETE'
    if any(int(r['malware_n']) < 10 for r in all_rows if r['scope'] in ('2018','2019','2020','2021','2022')):
        quality = 'VERY_LOW_POSITIVE_COUNT_IN_NEW_EXTERNAL_YEARS'
    else:
        quality = 'ADEQUATE'
    final = {
        'status': result_status, 'formal_integrity': formal_integrity['status'],
        'formal_integrity_asset_count': formal_integrity['asset_count'],
        'train_2017_n': source['union_train_n'],
        'stage2_train_new_complete': stage2_data['new_train_complete'],
        'stage2_routed_test_complete': stage2_data['routed_test_complete'],
        'future_labels_used_for_selection': False,
        'label_definition': 'AndroZoo VirusTotal detection proxy: 0 benign, >=4 malware; not behavior truth',
        'external_label_quality': quality,
        'routed_without_expert_by_arm': dict(missing),
        'interpretation': 'Descriptive external evaluation. No future-year result selects an arm, threshold or route.'
    }
    write_json(OUT / 'final_decision.json', final)
    lines = ['# CARE-Droid expanded 2017 benign-aware external evaluation V2', '',
        'Status: **' + result_status + '**.', '',
        f"2017 union Stage-I training: {source['union_train_n']} APKs "
        f"({source['union_benign']} benign, {source['union_malware']} malware).",
        f"Stage-II complete new 2017 training APKs: {stage2_data['new_train_complete']}/"
        f"{stage2_data['new_train_total']}.",
        f"Complete routed new-future Stage-II APKs: {stage2_data['routed_test_complete']}/"
        f"{stage2_data['routed_test_total']}.", '',
        '## Interpretation limits', '',
        '- New AndroZoo labels are VirusTotal-detection proxies, not verified behavior truth.',
        '- 2018–2020 have only two malware APKs per year; 2021–2022 have one each. '
        'Year-specific malware recall and BA are extremely unstable; 2021–2022 are diagnostic only.',
        '- 2018–2021 have historically informed this research program; this run does not use '
        'future labels to choose a model, threshold, route, or representation.',
        '- No original APK, formal script, checkpoint, or cache was rewritten.', '',
        '## Annual cascade results', '',
        '| Arm | Cohort | Year | N | BA | AUC | FP | FN | Fixed | Broken | Net |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in all_rows:
        if r['scope'] not in ('2018','2019','2020','2021','2022','POOLED_2018_2020'): continue
        lines.append('| ' + ' | '.join(str(r.get(k,'')) for k in
                     ('arm','cohort','scope','n','BA','AUC','FP','FN','fixed','broken','net')) + ' |')
    lines += ['', '## Routing', '',
              '| Arm | Year | Routed | Stage-I errors | Captured errors | Error recall |',
              '|---|---:|---:|---:|---:|---:|']
    for r in routes:
        lines.append('| ' + ' | '.join(str(r.get(k,'')) for k in
                     ('arm','year','routed','stage1_errors','routed_errors','error_recall'))+' |')
    lines += ['', '## Safety', '',
              f"Reused audit/manifest integrity: **{integrity['status']}** ({integrity['asset_count']} files).",
              f"Named formal asset integrity: **{formal_integrity['status']}** "
              f"({formal_integrity['asset_count']} code, feature, data and checkpoint files).",
              'The earlier dataset audit content-verified the 3,989 APKs. This run reads them '
              'without installation, execution, rewriting or copying.', '',
              'See the CSV and JSON outputs in this directory for per-APK and module-level results.', '']
    with (OUT / 'final_report.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    with (OUT / 'reproduce_commands.md').open('x', encoding='utf-8') as f:
        f.write('# Reproduction\n\n'
                'Run only in a fresh independent output version; this version does not overwrite existing files.\n\n'
                '`bash scripts/care-expanded2017-benign-aware-v2/launch_batch_tmux.sh`\n\n'
                '`bash scripts/care-expanded2017-benign-aware-v2/launch_followon_tmux.sh`\n')
    print('EXPANDED_RESULTS_FROZEN', result_status, flush=True)


if __name__ == '__main__': main()
