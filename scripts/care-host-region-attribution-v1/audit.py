#!/usr/bin/env python3
"""Read-only frozen Direct-C43 replay and Host/graph intervention feasibility audit.

Role in this experiment: Phase A and G1/G2 gate. Writes only new audit output files;
never modifies APKs, frozen caches, checkpoints, or CARE-Droid source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/care_host_region_attribution_v1'
BASE = ROOT / 'outputs/care_multiyear_development_v1'
PRIOR = ROOT / 'outputs/care_crossyear_layerwise_diagnosis_v2'
SEEDS = (900000, 900001, 900002)
TAGS = ('T2_C1', 'T3_C1', 'T4_C1')


def write_json(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError(f'refusing to overwrite {path}')
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_text(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError(f'refusing to overwrite {path}')
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def asset(path):
    path = path.resolve(strict=True)
    st = path.stat()
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha256(path))


def protected_paths():
    names = [
        BASE / 'data.npz', BASE / 'feature_rows.pkl', BASE / 'dataset_audit.json',
        BASE / 'per_apk_predictions.csv', BASE / 'sample_manifest.csv',
        PRIOR / 'baseline_replay.json', PRIOR / 'per_apk_error_contract.csv',
        ROOT / 'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv',
        ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz',
        ROOT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
        ROOT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1/host_sidecar.py',
        ROOT / 'scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py',
        ROOT / 'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
        ROOT / 'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py',
        ROOT / 'scripts/care-multiyear-development-v1/audit_data.py',
        ROOT / 'scripts/care-multiyear-development-v1/myd_common.py',
        ROOT / 'scripts/care-multiyear-development-v1/protocol.py',
        ROOT / 'scripts/care-multiyear-development-v1/train_stage2.py',
        ROOT / 'scripts/care-crossyear-layerwise-diagnosis-v2/audit_replay.py',
        ROOT / 'scripts/care-multiyear-error-evidence-audit-v1/selection_intervention.py',
    ]
    names += [BASE / 'models/stage2' / tag / (str(seed) + '.pkl') for tag in TAGS for seed in SEEDS]
    return names


def audit_prior():
    return """# 既有 Host 干预重叠审计

- `selection_intervention.py` 在同一冻结 C1 中替换旧 C0 的关键 Host 索引、注意力分布或二者；所有 Host 仍由 C1 编码，未移除 Host，也未重算三背景。其 10 个稳定退步 FN 在这些回退下均未恢复。
- Host Binding/Role 匹配置乱改变 Host-控制或 Role 对应关系，用于可观测性或置乱对照；不是删除 Host 后完整重算冻结网络。
- Local Control Constraint、Static Evidence Fidelity 与 2020 FN/TN DEX 盲审属于静态证据定位，不是本轮冻结模型输入干预。盲审结果不能升级为恶意行为真值。
- Cross-Year Layerwise Information Diagnosis 已完整重放九个 Direct-C43 checkpoint，并导出层张量；它没有执行 Host 删除。

拟议的新实验与上述工作在操作定义上不同：在**同一冻结专家**内删除 Host/区域，重算动态内部背景，并保证全局 RoleGraph 与 Host 输入仍对应同一假定程序。若最后一项无法证明，不能把仅删 Host 袋行的结果称为合法完整 Host 干预；不得运行正式归因 Pilot。
"""


def endpoint_host_audit(data, selected_ids):
    """Read only endpoint payloads for selected cases; test count/order provenance."""
    import gzip
    import importlib.util
    module_path = ROOT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1/host_sidecar.py'
    spec = importlib.util.spec_from_file_location('hra_host_sidecar', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    endpoint_path = ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'
    wanted = set(selected_ids)
    seen = {}
    with gzip.open(endpoint_path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            sid = str(row.get('sha256', '')).casefold()
            if sid in wanted and row.get('status') == 'OK' and isinstance(row.get('payload'), dict):
                hosts = module.endpoint_hosts(row['payload'])
                if sid in seen and seen[sid] != hosts:
                    raise RuntimeError('conflicting endpoint cache duplicate ' + sid)
                seen[sid] = hosts
    index = {str(s).casefold(): i for i, s in enumerate(data['ids'])}
    rows = []
    for sid in selected_ids:
        i = index[sid]
        n = int(data['real_host_count'][i])
        saved = int(data['host_offsets'][i+1] - data['host_offsets'][i])
        endpoint = seen.get(sid)
        rows.append(dict(sha256=sid, real_hosts=n, saved_rows=saved,
                         endpoint_hosts=None if endpoint is None else len(endpoint),
                         count_equal=endpoint is not None and len(endpoint) == n,
                         placeholder_valid=saved == max(n, 1),
                         node_ids_retained_in_data_npz=False,
                         per_host_to_s7_provenance_retained_in_data_npz=False))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true', required=True)
    cli = parser.parse_args()
    if OUT.exists() or HERE != ROOT / 'scripts/care-host-region-attribution-v1':
        raise RuntimeError('target exists or code path unexpected; refusing overwrite')
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / 'tmp').mkdir(mode=0o700)
    os.environ['TMPDIR'] = str(OUT / 'tmp')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    start = time.monotonic()
    manifest = [asset(p) for p in protected_paths()]
    write_json('asset_manifest_before.json', manifest)
    write_text('prior_intervention_overlap_audit.md', audit_prior())
    sys.path.insert(0, str(ROOT / 'scripts/care-crossyear-layerwise-diagnosis-v2'))
    import diagnostic_common as dc
    mc, v4, _, data = dc.original()
    from train_stage2 import raw_bags, transform_saved
    from protocol import mean_opposition, top_budget
    original = read_csv(BASE / 'per_apk_predictions.csv')
    lookup = {(r['window'] + '_' + r['arm'], r['sha256']): r for r in original}
    index = {str(s): i for i, s in enumerate(data['ids'])}
    replay = []
    checkpoints = []
    maxdiff = 0.0
    selected_2020 = []
    for tag in TAGS:
        objs = [dc.checkpoint(BASE / 'models/stage2' / tag / (str(seed) + '.pkl'), mc, data) for seed in SEEDS]
        meta = objs[0]['metadata']
        ids = list(meta['validation_sha'])
        ix = np.asarray([index[s] for s in ids], dtype=int)
        raw = raw_bags(data, ix)
        score = mean_opposition(data['historical_members'][ix], data['historical_base'][ix])
        route = top_budget(score, data['ids'][ix], .25)
        ensemble = []
        tag_max = 0.0
        for seed, obj in zip(SEEDS, objs):
            if obj['metadata']['validation_sha'] != ids or obj['metadata']['train_sha'] != meta['train_sha']:
                raise RuntimeError('checkpoint order mismatch')
            args = argparse.Namespace(**obj['metadata']['args'])
            bags = transform_saved(v4, raw, obj['background'], args)
            graph = v4.transform_role_graphs(data['s7'][ix], obj['graph_scaler'])
            model = v4.RoleGraphConditionedEvidence(24, 'RGCondAttn', args)
            model.load_state_dict(obj['state_dict'])
            model.eval()
            p, _ = v4.predict_model(model, np.asarray(obj['global_validation'], np.float32),
                                     np.full(len(ix), .5), graph, bags, v4.torch.device('cpu'))
            diff = float(np.max(np.abs(p - obj['probability'])))
            if diff > 1e-6:
                raise RuntimeError(f'checkpoint probability replay failed {tag}/{seed}: {diff}')
            maxdiff = max(maxdiff, diff)
            tag_max = max(tag_max, diff)
            ensemble.append(p)
            checkpoints.append(dict(window=tag, seed=seed,
                                    checkpoint=str(BASE / 'models/stage2' / tag / (str(seed) + '.pkl')),
                                    train_n=len(meta['train_sha']), evaluation_n=len(ix),
                                    saved_probability_max_abs_difference=diff,
                                    shared_centers_shape=list(np.shape(obj['background']['state']['shared_centers'])),
                                    benign_centers_shape=list(np.shape(obj['background']['state']['benign_centers'])),
                                    host_input_width=int(bags[0].shape[1]),
                                    graph_node_shape=list(graph['node'].shape[1:]),
                                    graph_edge_shape=list(graph['edge'].shape[1:])))
            print('HRA_REPLAY', tag, seed, diff, flush=True)
        ens = np.mean(ensemble, axis=0)
        cats = Counter()
        stage1diff = 0.0
        routemismatch = 0
        ensemble_diff = 0.0
        for j, sid in enumerate(ids):
            row = lookup.get((tag.replace('_C1', '') + '_U3_FULL_ROLLING_C1', sid))
            if row is None:
                raise RuntimeError('missing original result ' + sid)
            stage1diff = max(stage1diff, abs(float(row['stage1_probability']) - float(data['historical_base'][ix[j]])))
            routemismatch += int(int(row['routed']) != int(route[j]))
            ensemble_diff = max(ensemble_diff, abs(float(row['expert_probability']) - float(ens[j])))
            if int(route[j]):
                y = int(row['label'])
                s = int(row['stage1_prediction'])
                e = int(float(ens[j]) >= .5)
                category = ('UNFIXED_FN' if y else 'UNFIXED_FP') if s != y and e != y else \
                           ('FIXED_FN' if y else 'FIXED_FP') if s != y else \
                           ('BROKEN_TN' if not y else 'BROKEN_TP') if e != y else 'STABLE_CORRECT'
                cats[category] += 1
                if tag == 'T3_C1' and s == 0:
                    selected_2020.append(sid)
        if stage1diff > 1e-8 or routemismatch or ensemble_diff > 1e-6:
            raise RuntimeError(f'baseline Stage-I/route/ensemble mismatch {tag}: {stage1diff}, {routemismatch}, {ensemble_diff}')
        replay.append(dict(window=tag, year=int(data['years'][ix[0]]), evaluation_n=len(ix),
                           routed=int(route.sum()), stage1_max_abs_difference=stage1diff,
                           route_mismatch=routemismatch, ensemble_max_abs_difference=ensemble_diff,
                           seed_max_abs_difference=tag_max, categories=dict(cats)))
    write_json('checkpoint_manifest.json', checkpoints)
    write_json('baseline_replay.json', dict(status='PASS', tolerance=1e-6,
                                            max_seed_probability_difference=maxdiff, windows=replay,
                                            stage1_probability_source='frozen historical_base; no Stage-I retraining',
                                            baseline_route_source='frozen historical_members and original top_budget'))
    if len(selected_2020) != 77:
        raise RuntimeError(f'2020 Stage-I-benign routed count {len(selected_2020)} != 77')
    graph_rows = endpoint_host_audit(data, selected_2020)
    count_ok = sum(r['count_equal'] and r['placeholder_valid'] for r in graph_rows)
    raw_dim = int(data['host_values'].shape[1])
    s7_dim = int(data['s7'].shape[1])
    keys = sorted(data)
    safe = (count_ok == len(graph_rows))
    graph_audit = dict(status='BLOCKED_FOR_FULL_HOST_REMOVAL',
        primary_2020_count=len(graph_rows), endpoint_host_count_aligned=count_ok,
        raw_host_width=raw_dim, s7_width=s7_dim,
        stored_bundle_keys=keys,
        selected_host_count_quantiles=np.quantile([r['real_hosts'] for r in graph_rows], [0,.25,.5,.75,1]).tolist(),
        endpoint_count_and_placeholder_consistent=safe,
        host_row_to_endpoint_node_id_in_prepared_bundle=False,
        per_host_s7_contribution_in_prepared_bundle=False,
        rolegraph_built_from_all_endpoint_hosts=True,
        rolegraph_recomputed_after_host_deletion=False,
        internal_background_can_recompute_from_reduced_bag=True,
        reason='The prepared 169-column Host rows and 1546-column S7 RoleGraph are stored as independent APK-level arrays. The frozen forward uses S7 node/edge/global to form Role query and token, while Host rows form local evidence. A row deletion with unchanged S7 would preserve the removed Host in graph-derived statistics. A verified per-Host graph/S7 rebuild and exact replay contract is not established. Such inputs are artificial partial-view perturbations, not the requested consistent Host removal.',
        next_requirement='Develop and separately validate endpoint-node-ID to each frozen Host row plus complete S7/global feature recomputation (including multiscale null statistics), then rerun the baseline equivalence and intervention self-test before any matched pilot.',
        pilot_authorized=False)
    write_json('host_graph_alignment_audit.json', graph_audit)
    write_json('intervention_validity.json', dict(valid_model_interventions=0,
        hypothetical_host_bag_only='ARTIFICIAL_INPUT_PERTURBATION',
        attempted_forward_on_inconsistent_graph=False,
        full_graph_transformation='INVALID_GRAPH_TRANSFORMATION until verified',
        internal_background_recomputation_performed=False))
    after = [asset(Path(item['path'])) for item in manifest]
    changed = [dict(before=b, after=a) for b, a in zip(manifest, after) if b != a]
    write_json('asset_manifest_after.json', after)
    write_json('baseline_integrity_check.json', dict(status='PASS' if not changed else 'FAIL',
        asset_count=len(manifest), changed=changed,
        scope='only listed reused checkpoint, cache, result and code files; no original APK was opened'))
    if changed:
        raise RuntimeError('protected asset changed')
    write_text('final_report.md', f'''# CARE-Droid Host 区域反事实归因审计 v1

## 最终状态

**INTERVENTION_NOT_SEMANTICALLY_VALID**。停在 G1/Phase B 合法性门槛；不运行 8 对 Pilot、全部路由或跨年 Host 归因。

九个正式 checkpoint 在 1e-6 容差内完成回放，最大概率差 {maxdiff:.9g}。Stage-I 缓存概率、MeanOpposition 路由、三种子集成也与冻结逐 APK 表一致。2020 年路由且 Stage-I 预测良性的 APK 为 {len(selected_2020)} 个。端点 Host 计数对齐 {count_ok}/{len(graph_rows)}。

## 为什么停

正式模型同时读取每 APK 的 Host 袋和由全 APK 端点结构构造的 S7 RoleGraph。后者的 31 个 Role 节点、边与全局统计进入查询和分类。`data.npz` 保存 Host 行、偏移量与整个 S7，却没有逐 Host 行到 S7 各分量的贡献映射。虽然删除 Host 行后可以重算动态 Internal Background 和注意力，冻结 S7 会让已删除 Host 的结构证据仍在另一条输入路径中。当前没有完成逐 Host 到端点节点身份的核实、完整 S7 重建和重放等价性验证，因此不能将其标记为 `VALID_MODEL_INTERVENTION`。

历史的关键 Host/注意力回退是另一种实验；它也没有解决上述全图一致性。不能复用其效应冒充删除归因。端点计数对齐不等于身份、边或 S7 等价。

## 对十个研究问题的回答

本轮没有合法 Host 删除样本，因此 Q1–Q9 的数量、类别方向、Attention 相关、组合效应及跨年稳定性均**不可验证**；不能以 0 代替未知。Q10：现在不具备修改 Direct-C43 的因果或类别区分证据。最小下一步是独立完成逐 Host→端点节点→S7 的可复核映射及完整重建自测试，之后再决定是否恢复 Pilot。若无法重建，应停止本分支，不能用局部袋行置零/删除替代。

原始 APK 没有被读取、修改、安装或运行；正式资产清单完整性为 {'PASS' if not changed else 'FAIL'}，范围仅限列入哈希清单的 {len(manifest)} 个文件。耗时 {time.monotonic()-start:.1f} 秒。
''')
    write_json('final_decision.json', dict(status='INTERVENTION_NOT_SEMANTICALLY_VALID',
        stopped_at='G1_PHASE_B_GRAPH_CONSISTENCY_GATE', baseline_replay='PASS',
        formal_integrity='PASS', valid_model_interventions=0,
        pilot_run=False, full_2020_run=False, crossyear_interventions_run=False,
        next_action=graph_audit['next_requirement']))
    write_text('reproduce_commands.md', '''# 复现审计（新版本号才能重新执行）\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\nPYTHONDONTWRITEBYTECODE=1 /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v1/audit.py --audit-only\n```\n\n该命令不可在现有输出目录中重复执行；检查结果请读取 `outputs/care_host_region_attribution_v1/final_decision.json`。\n''')
    print('HRA_FINAL_STATUS=INTERVENTION_NOT_SEMANTICALLY_VALID', flush=True)
    print('HRA_STOPPED_AT=G1_PHASE_B_GRAPH_CONSISTENCY_GATE', flush=True)


if __name__ == '__main__':
    main()
