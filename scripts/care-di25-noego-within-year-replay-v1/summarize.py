#!/usr/bin/env python3
"""Apply frozen DI25 to same-year OOF expert predictions and summarize end-to-end CARE-Droid."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

PROJECT = Path('/home/boyuanjie/projects/tasa-droid')
ROOT = PROJECT / 'artifacts/care-di25-noego-within-year-replay-v1'
ROUTER = PROJECT / 'artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv'
EXPERT = ROOT / 'expert_anomaly_bounded/within_year_expert_predictions.csv'
EXPECTED_ROUTE_COUNTS = {2018:198, 2019:100, 2020:99, 2021:100}


def norm(x) -> str:
    return str(x or '').strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError('no rows to write')
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def metrics(y, p, base_pred=None):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out = {
        'n': int(len(y)),
        'ACC': float(accuracy_score(y, pred)),
        'BA': float(balanced_accuracy_score(y, pred)),
        'AUC': float(roc_auc_score(y, p)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }
    if base_pred is not None:
        base_pred = np.asarray(base_pred, dtype=np.int64)
        fixed = int(np.sum((base_pred != y) & (pred == y)))
        broken = int(np.sum((base_pred == y) & (pred != y)))
        out.update({
            'fixed': fixed,
            'broken': broken,
            'net': fixed - broken,
            'correction_precision': float(fixed / (fixed + broken)) if fixed + broken else 0.0,
        })
    else:
        out.update({'fixed':0,'broken':0,'net':0,'correction_precision':0.0})
    return out


def main() -> None:
    if not ROUTER.is_file() or not EXPERT.is_file():
        raise RuntimeError('BLOCKED_BY_CONTRACT: router or expert predictions missing')
    router = read_csv(ROUTER)
    expert = read_csv(EXPERT)
    if len(router) != 1989 or len(expert) != 1989:
        raise RuntimeError('BLOCKED_BY_CONTRACT: expected 1989 router/expert rows')
    rb = {norm(r['sha256']): r for r in router}
    eb = {norm(r['sha256']): r for r in expert}
    if len(rb) != 1989 or len(eb) != 1989 or set(rb) != set(eb):
        raise RuntimeError('BLOCKED_BY_CONTRACT: router/expert SHA cohort mismatch')

    rows = []
    for r in router:
        sha = norm(r['sha256'])
        e = eb[sha]
        if int(r['year']) != int(e['year']) or int(r['fold']) != int(e['fold']) or int(r['label']) != int(e['label']):
            raise RuntimeError('BLOCKED_BY_CONTRACT: label/year/fold mismatch for ' + sha)
        p_stage1 = float(r['p_stage1'])
        p_expert = float(e['p_expert'])
        route = int(float(r['route_DI']))
        p_final = p_expert if route == 1 else p_stage1
        label = int(r['label'])
        stage1_pred = int(p_stage1 >= 0.5)
        final_pred = int(p_final >= 0.5)
        rows.append({
            'sha256': sha,
            'year': int(r['year']),
            'fold': int(r['fold']),
            'label': label,
            'p_stage1': p_stage1,
            'route_DI': route,
            'p_expert': p_expert,
            'p_final': p_final,
            'stage1_pred': stage1_pred,
            'final_pred': final_pred,
            'fixed': int(stage1_pred != label and final_pred == label),
            'broken': int(stage1_pred == label and final_pred != label),
        })

    route_counts = {y: sum(r['route_DI'] for r in rows if r['year'] == y) for y in EXPECTED_ROUTE_COUNTS}
    if route_counts != EXPECTED_ROUTE_COUNTS:
        raise RuntimeError('BLOCKED_BY_CONTRACT: route counts changed: ' + repr(route_counts))

    metric_rows = []
    for scope in [2018, 2019, 2020, 2021, 'pooled']:
        part = rows if scope == 'pooled' else [r for r in rows if r['year'] == scope]
        y = np.asarray([r['label'] for r in part], dtype=np.int64)
        p1 = np.asarray([r['p_stage1'] for r in part], dtype=np.float64)
        pf = np.asarray([r['p_final'] for r in part], dtype=np.float64)
        base_pred = (p1 >= 0.5).astype(np.int64)
        base = metrics(y, p1)
        final = metrics(y, pf, base_pred=base_pred)
        routed_n = int(sum(r['route_DI'] for r in part))
        metric_rows.append({'scope': scope, 'method': 'Global24', 'routed_n': 0, 'routed_ratio': 0.0, **base})
        metric_rows.append({'scope': scope, 'method': 'DI25__Host__Anomaly__RGCondAttn__Bounded',
                            'routed_n': routed_n, 'routed_ratio': routed_n / len(part), **final})

    write_csv(ROOT / 'oof_predictions.csv', rows)
    write_csv(ROOT / 'metrics.csv', metric_rows)

    protocol = {
        'experiment': 'CARE-Droid no-Ego within-year replay v1',
        'status': 'development/diagnostic same-year OOF; not temporal generalization',
        'years': [2018,2019,2020,2021],
        'split': 'independent per-year fixed 5-fold OOF',
        'router': 'frozen within-year DI25 from existing within_year_router_inputs.csv',
        'route_counts': route_counts,
        'stage2': 'Host-only Anomaly + RoleGraph RGCondAttn + FixKeep bounded correction',
        'ego_consumed': False,
        'host_keys': ['host_values','host_offsets','real_host_count'],
        'seeds': [900000,900001,900002],
        'max_delta': 2.25,
        'learning_rate': 8e-4,
        'weight_decay': 5e-4,
        'max_epochs': 30,
        'keep_weight': 2.0,
        'fix_weight': 1.0,
        'keep_consistency': 0.5,
        'future_inter': 'rebuilt only for this replay into replay-local cache using recovered historical loader; lost canonical future_interz is not replaced',
        'apk_reextraction': False,
        'need_gate_used': False,
    }
    (ROOT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# CARE-Droid no-Ego within-year replay v1', '',
        'Independent same-year 5-fold OOF on 2018-2021. This is diagnostic/development evidence, not temporal generalization.', '',
        '| Scope | Global24 BA | CARE BA | Delta BA (pp) | Routed | Fixed | Broken | Net |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    by = {(str(r['scope']), r['method']): r for r in metric_rows}
    for scope in ['2018','2019','2020','2021','pooled']:
        b = by[(scope,'Global24')]
        c = by[(scope,'DI25__Host__Anomaly__RGCondAttn__Bounded')]
        lines.append(f"| {scope} | {b['BA']:.6f} | {c['BA']:.6f} | {(c['BA']-b['BA'])*100:+.2f} | {c['routed_n']} | {c['fixed']} | {c['broken']} | {c['net']} |")
    lines += ['', 'No Ego tensors are consumed. Need-Gate is not used in this replay.']
    (ROOT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('NOEGO_WITHIN_YEAR_SUMMARY_DONE')
    print((ROOT / 'summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
