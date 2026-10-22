#!/usr/bin/env python3
from __future__ import annotations

from common import (FUTURE_LABELS, OUT, SEALED_FEATURE_CANDIDATES, file_sha256,
                    first_existing, infer_year, load_future_cohort, norm, read_csv,
                    save_json, write_csv)


def canonical_label(value):
    v = str(value or '').strip().casefold()
    if v in ('1', 'malware', 'malicious'):
        return 'malicious'
    if v in ('0', 'benign'):
        return 'benign'
    return ''


def main():
    ref = OUT / 'preflight/p31_reference/future_predictions_diagnostic.csv'
    if not ref.is_file():
        raise RuntimeError('BLOCKED_BY_CONTRACT: build P31 future reference before unsealing evaluation labels')
    frozen_hash = file_sha256(ref)
    sealed = first_existing(SEALED_FEATURE_CANDIDATES, 'sealed 2017+future compact feature table')
    if not FUTURE_LABELS.is_file():
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: future diagnostic labels not found: {FUTURE_LABELS}')
    labels = read_csv(FUTURE_LABELS)
    label_map = {}
    for r in labels:
        sid = norm(r.get('sha256'))
        if not sid:
            continue
        yv = r.get('y', r.get('label', ''))
        if str(yv).strip() == '':
            continue
        label_map[sid] = int(float(yv))
    _, cohort = load_future_cohort()
    future_ids = {sid for sid, _ in cohort}
    if set(label_map).intersection(future_ids) != future_ids:
        missing = list(future_ids - set(label_map))
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: evaluation labels miss {len(missing)} future SHA; first={missing[:3]}')

    rows = read_csv(sealed)
    output = []
    seen_future = set()
    for row in rows:
        sid = norm(row.get('sha256'))
        year = infer_year(row)
        if year == 2017:
            label = canonical_label(row.get('metadata_label'))
            if not label:
                # Some source tables may use a generic label column.
                label = canonical_label(row.get('label'))
            if not label:
                raise RuntimeError(f'BLOCKED_BY_CONTRACT: source label missing for {sid}')
            row['metadata_label'] = label
            output.append(row)
        elif year in (2018, 2019, 2020, 2021):
            if sid not in future_ids:
                continue
            row['metadata_label'] = 'malicious' if label_map[sid] == 1 else 'benign'
            output.append(row); seen_future.add(sid)
    if seen_future != future_ids:
        missing = list(future_ids - seen_future)
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: sealed feature table misses {len(missing)} future SHA; first={missing[:3]}')
    source_n = sum(infer_year(r) == 2017 for r in output)
    future_n = len(output) - source_n
    if source_n != 799 or future_n != 1989:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: evaluation feature rows expected 799+1989, got {source_n}+{future_n}')
    out = OUT / 'preflight/compact_features_2017_2021_eval_only.csv'
    write_csv(out, output)
    after_hash = file_sha256(ref)
    if after_hash != frozen_hash:
        raise RuntimeError('BLOCKED_BY_CONTRACT: P31 routing reference changed after label unsealing')
    save_json(OUT / 'preflight/eval_label_unseal_audit.json', {
        'sealed_input': str(sealed),
        'evaluation_output': str(out),
        'source_rows': source_n,
        'future_rows': future_n,
        'future_labels_used_only_after_p31_predictions_and_routes_frozen': True,
        'p31_future_reference_sha256_before': frozen_hash,
        'p31_future_reference_sha256_after': after_hash,
    })
    print('P31_FUTURE_EVAL_FEATURES_PASS')
    print(f'eval_features={out}')
    print('future_labels_used_only_after_p31_predictions_and_routes_frozen=true')


if __name__ == '__main__':
    main()
