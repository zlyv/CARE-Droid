#!/usr/bin/env python3
"""Integrity and honest stop status after sandboxed, label-free packet check."""
import hashlib
import json
import pathlib

ROOT = pathlib.Path('/home/boyuanjie/projects/tasa-droid')
OUT = ROOT / 'outputs/care-2020-fn-tn-dex-review-v4'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

def main():
    prior = json.loads((OUT / 'organizer_only/input_hashes_before.json').read_text())
    changes = [p for p, old in prior.items() if not pathlib.Path(p).is_file() or digest(pathlib.Path(p)) != old]
    audit = json.loads((OUT / 'blind_workspace/blind_reference_audit.json').read_text())
    status = 'AWAITING_BLIND_SEMANTIC_REVIEW' if not changes and audit['status'].startswith('PASS') else 'PRECHECK_FAILED'
    save('baseline_integrity_check.json', {'status': 'PASS' if not changes else 'FAIL',
         'asset_count': len(prior), 'changed': changes,
         'scope': 'source cohort CSV, original blank form, and eight original anonymous packets only; no APK opened'})
    save('blind_review_validation.json', {'status': 'AWAITING_HUMAN_REVIEW',
         'packet_reference_audit': audit['status'], 'raw_dex_reference_validation': 'UNKNOWN',
         'review_complete': False, 'freeze_authorized': False})
    save('blind_review_freeze_manifest.json', {'status': 'NOT_FROZEN',
         'reason': 'No independent blinded human semantic review submitted', 'sha256': None})
    save('final_decision.json', {'status': status, 'unsealed': False, 'model_training_performed': False,
         'apk_execution_performed': False, 'future_rows_loaded_for_model_selection': 0,
         'root_cause': 'NO_VERIFIED_SEMANTIC_ROOT_CAUSE',
         'reason': 'Anonymous packets and citations prechecked; genuine independent semantic review is still required',
         'next_action': 'Independent reviewer fills blind_workspace/blind_review_submission.csv and blind_evidence_references.csv; then manually invoke freeze_review.py inside sandbox. Do not unseal automatically.'})
    (OUT / 'final_report.md').write_text(
        '# CARE-Droid 2020 FN/TN DEX blind review v4 — pre-review state\n\n'
        f'Final current status: **{status}**. The 77-case organizer preflight matched the requested '
        '24 unresolved FN, 2 fixed FN, 48 stable TN, and 3 broken TN; all 77 SHA values are distinct. '
        'Eight anonymous A/B packets and the existing seven-column blank form were copied into a '
        'bubblewrap-isolated workspace. Internal packet invoke citations were indexed and checked without '
        'mounting labels, predictions, the sealed key, or original APK paths.\n\n'
        'Packet examples are truncated; they cannot prove the absence of a behavior. No anonymous '
        'alias-to-APK map was provided, so raw APK byte-level citation validation and DEX-to-C43 tracing '
        'remain UNKNOWN. The assistant previously encountered the mapping in an earlier task, so it cannot '
        'serve as an independent blind semantic reviewer here. No form was filled or frozen; no key was '
        'opened in this version. Post-unseal tables are intentionally not created.\n\n'
        'No APK was opened, modified, installed, or run; no model was trained or tuned. '
        f'Original input integrity: {"PASS" if not changes else "FAIL"}. '
        'Root cause: **NO_VERIFIED_SEMANTIC_ROOT_CAUSE**.\n', encoding='utf-8')
    print('CARE_2020_FN_TN_DEX_REVIEW_STATUS=' + status)

if __name__ == '__main__':
    main()
