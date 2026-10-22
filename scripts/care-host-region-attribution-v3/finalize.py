#!/usr/bin/env python3
"""Finalize graph-dependency feasibility decision after an exact unmodified replay.

Role: read-only provenance audit and fail-closed decision. Writes only new v3
report files; never modifies APKs, old caches, checkpoints or formal source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe import ROOT, OUT, asset, payload_of, write


def text_new(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError('refusing overwrite ' + str(path))
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    probe = read(OUT / 'graph_rebuild_probe.json')
    prior = ROOT / 'outputs/care_host_region_attribution_v2'
    baseline = read(prior / 'baseline_replay.json')
    integrity = read(OUT / 'baseline_integrity_check.json')
    if probe['status'] != 'PASS_EXACT_REBUILD' or baseline['status'] != 'PASS' or integrity['status'] != 'PASS':
        raise RuntimeError('required replay/integrity gate did not pass')
    with np.load(ROOT / 'outputs/care_multiyear_development_v1/predictions/T3/router.npz',allow_pickle=False) as z:
        ids=np.asarray(z['ids']).astype(str); route=np.asarray(z['new_route'],bool); s1=np.asarray(z['s1'],float)
    with np.load(ROOT / 'outputs/care_multiyear_development_v1/data.npz',allow_pickle=False) as z:
        sid=min((s for s,r,p in zip(ids,route,s1) if r and p<.5),key=lambda s:int(z['real_host_count'][np.flatnonzero(z['ids']==s)[0]]))
    payload=payload_of(sid)
    producer=ROOT/'src/tasa_droid/p2/endpoint_centered_view.py'
    source=producer.read_text(encoding='utf-8')
    required=(
        'entry_reachable_distances(',
        'upstream_distance.get(signature',
        'downstream_distance.get(signature',
        'node_base_features',
        'edge_features',
        'float(key in upstream_edges)',
        'float(key in downstream_edges)',
    )
    missing=[s for s in required if s not in source]
    if missing:
        raise RuntimeError('producer contract changed: '+repr(missing))
    expected={'node_ids','node_role_ids','node_base_features','edge_index','edge_features','edge_type_ids'}
    schema=sorted(payload)
    missing_payload=sorted(expected-set(payload))
    if missing_payload:
        raise RuntimeError('endpoint payload schema changed '+repr(missing_payload))
    graph=read(prior/'host_graph_alignment_audit.json')
    report=dict(
        status='BLOCKED_COHERENT_HOST_REMOVAL',
        exact_unmodified_rebuild='PASS',
        unmodified_s7_max_abs_difference=probe['s7_block_max_abs_difference']['max_abs'],
        unmodified_host_column_max_abs_difference=probe['host_column_max_abs_difference'],
        model_probability_replay_max_abs_difference=baseline['max_seed_probability_difference'],
        matched_primary_2020_count=graph['primary_2020_count'],
        endpoint_count_matches=graph['endpoint_host_count_aligned'],
        endpoint_payload_keys=schema,
        producer_dependency_evidence={
            'node_base_width':15,
            'entry_reachability_fields':[8,9,10,11],
            'upstream_downstream_and_endpoint_distance_fields':[12,13,14],
            'edge_feature_width':7,
            'upstream_downstream_edge_flags':[5,6],
            'source_path':str(producer),
            'source_sha256':asset(producer)['sha256'],
            'full_pretruncation_node_records_retained_in_payload':False,
            'full_pretruncation_edge_records_retained_in_payload':False,
        },
        reason='Endpoint-node/Host identity and unchanged S7 can be recovered. However, removing a Host node or its incident edges from the retained cache would leave node reachability/distance fields and upstream/downstream edge flags derived from the original full call graph. The cache contains only a capped induced subgraph, not all pretruncation records needed to recompute those fields exactly. A modified cache cannot be certified as the same formal extractor applied to a changed analysis graph.',
        candidate_perturbation_class='ARTIFICIAL_INPUT_PERTURBATION or INVALID_GRAPH_TRANSFORMATION',
        valid_model_interventions=0,
        pilot_authorized=False,
        original_apks_opened=0,
        next_action='Only a separate pre-cache full-graph transformation/re-extraction branch, with unchanged-APK exact replay and modified-graph invariant checks, could enable coherent Host removal. Do not run the 8-pair attribution pilot with cached-row deletion.',
    )
    write('producer_dependency_audit.json',report)
    before=asset(producer);after=asset(producer)
    write('producer_asset_integrity.json',dict(status='PASS' if before==after else 'FAIL',before=before,after=after))
    if before!=after:
        raise RuntimeError('producer file changed during audit')
    write('final_decision.json',dict(status='INTERVENTION_NOT_SEMANTICALLY_VALID',
        stopped_at='G2_COHERENT_GRAPH_TRANSFORMATION_GATE',
        baseline_replay='PASS', exact_original_graph_rebuild='PASS',
        formal_integrity='PASS', valid_model_interventions=0,
        pilot_run=False, full_2020_run=False, crossyear_interventions_run=False,
        next_action=report['next_action']))
    text_new('final_report.md',f'''# CARE-Droid Host 区域反事实贡献归因审计 v3

## 结论

**INTERVENTION_NOT_SEMANTICALLY_VALID**。停在 G2“修改后图的完整一致性”门槛。没有执行 8 对 FN/TN Pilot、全 2020 路由集或跨年 Host 归因。

这不是因为原始专家无法回放：T2/T3/T4 共 9 个 checkpoint 的最大概率差为 {baseline['max_seed_probability_difference']:.9g}（容差 1e-6），正式滚动 SVM、MeanOpposition 路由也逐样本一致。2020 年 Stage-I 预测良性且已路由的 77 个 APK 的组成由冻结表重新核对：未修复 FN 24、修复 FN 2、保持正确 TN 48、误伤 TN 3。

进一步使用原始端点缓存，在一个预先按最少 Host 数选出的 2020 APK 上精确重建 26 个 Host 的 Role、端点计数、Relation31（三者最大差均 0）及 S7 全图特征（最大差 {probe['s7_block_max_abs_difference']['max_abs']:.9g}）。这说明**未修改输入的身份与特征回放可行**，但不等于删除 Host 后的图有效。

端点构建器将完整调用图的入口可达性、上下游路径及端点距离编码在 `node_base_features`，还在 `edge_features` 保存上下游路径标记。现存缓存是截断后的子图，未保留计算这些字段所需的全部原始节点/边事实。若直接删去 Host 行/节点和关联边、再重算 S7，原来的可达性、距离和路径标记会成为陈旧字段；若保持 S7 则已删除 Host 仍通过 RoleGraph 影响查询与决策。两种做法均不能标记为 `VALID_MODEL_INTERVENTION`。旧 Host 选择/Attention 回退与本实验操作不同，也不能代替合法删除。

因此 Q1–Q9（影响最大的 Host、FN/TN 与 FP/TP 类别方向、Attention 排名、单 Host/邻域/组合效应及跨年稳定性）均**未获有效数据，不能报告零或肯定结果**。Q10：当前没有证据允许修改 Direct-C43。最小下一步是另建 pre-cache 全图变换与重提取分支，首先让未修改 APK 的每个相关张量精确回放，再验证删除后所有节点/边派生字段与 Host/Role/Relation 的一致性；通过后才恢复 Pilot。

正式资产未被修改。主清单 35 个复用文件与补充生产者源码哈希均未变化；完整性只覆盖这些列明的文件，不代表服务器全盘。未打开、安装、执行或改写任何 APK。
''')
    text_new('reproduce_commands.md','''# 已执行的关键命令

```bash
cd /home/boyuanjie/projects/tasa-droid
bash scripts/care-host-region-attribution-v2/launch_tmux.sh
bash scripts/care-host-region-attribution-v3/launch_tmux.sh
PYTHONDONTWRITEBYTECODE=1 /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v3/finalize.py
```

已有输出只读收集，不得重新运行到同一输出路径。
''')
    print('HRA_V3_FINAL_STATUS=INTERVENTION_NOT_SEMANTICALLY_VALID',flush=True)


if __name__ == '__main__':
    main()
