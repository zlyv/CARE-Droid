#!/usr/bin/env python3
"""BFA-V1 staged audit/retraining driver; reads frozen 2017–2021 caches and writes only the new BFA output, never opens APKs."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from bfa_core import (ARMS, COLS, OUT, ROOT, SCBE, SEEDS, SPLITS, audit_bags,
                      formal_split, immutable_csv, immutable_json, immutable_text,
                      load_data_contract, preflight, transform_arm, verify_assets)


def audit_only(resume):
    status = preflight(resume)
    if status['status'] != 'PASS':
        raise RuntimeError('frozen asset integrity failed')
    old = ROOT / 'outputs'
    experiments = [
        ('Prototype class/overlap', old / 'care_background_prototype_audit_v1/final_decision.json',
         '支持率、几何/覆盖重叠、增量probe、匹配FN/TN；没有当前网络三背景逐支中性化重新训练。'),
        ('TB-FRS', old / 'care-tb-frs-source-v1/final_decision.json',
         '软分配和有符号残差source信息测试；不是当前背景入口消融。'),
        ('Three-background SupCon', old / 'care_existing_three_background_contrastive_v2/final_decision.json',
         '变更对比损失与背景条件；不是保留BCE的背景删除。'),
        ('Layerwise diagnosis', old / 'care_crossyear_layerwise_diagnosis_v2/final_decision.json',
         '背景可读性与分层诊断；不是等容量重新训练消融。'),
        ('SCBE', SCBE / 'final_decision.json',
         '修改External标量及派生Residual，M3无稳定增量；没有Shared/Internal中性化。'),
    ]
    prior = ['# 既有背景实验边界', '', '本轮先核对可复用结果；只运行当前正式输入接口尚未回答的背景功能消融。', '']
    for name, path, boundary in experiments:
        if not path.is_file():
            raise RuntimeError('required prior experiment missing ' + str(path))
        decision = json.loads(path.read_text(encoding='utf-8'))
        prior += [f'- {name}: `{path}`；状态 `{decision.get("status", "UNKNOWN")}`。{boundary}']
    prior += ['', '另有早期 `run_care_joint_dualprototype_ablation.py` 原始列关闭实验，'
              '但其网络/Role约束、冻结路由、时间窗口和Direct-C43协议不同，不能替代本次等协议消融。',
              'Host注意力/选择回退也改变局部选择机制，不等价于独立取消三背景信息。', '']
    if not (OUT / 'prior_background_experiment_audit.md').exists():
        immutable_text(OUT / 'prior_background_experiment_audit.md', '\n'.join(prior))
    graph = {'inputs': ['Compact107', 'Role31', 'Relation31'],
             'banks': {'Shared32': 'fit_prototype_state on balanced training Host sample',
                       'External24': 'fit_prototype_state on benign training Host sample',
                       'Internal': 'internal_dominant_distance of the current APK Host set'},
             'nodes': {'External': {'parents': ['Compact107', 'Role31', 'External24'], 'host_index': 107},
                       'Internal': {'parents': ['Compact107', 'Shared32', 'APK Host set'], 'host_index': 108},
                       'Commonness': {'parents': ['Compact107', 'Role31', 'Shared32', 'training class-cluster prevalence'], 'host_index': 109},
                       'Residual': {'parents': ['External', 'Internal', 'Commonness'], 'host_index': 110},
                       'HostEncoder': {'parents': ['Compact107', 'External', 'Internal', 'Commonness', 'Residual']},
                       'RelationKey': {'parents': ['Relation31']},
                       'RoleGraphQuery': {'parents': ['RoleGraph']},
                       'Logit': {'parents': ['HostEncoder', 'RelationKey', 'RoleGraphQuery', 'Global24']}},
             'code': ['scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py:cols',
                      'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py:make_class',
                      'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py:RoleGraphConditionedEvidence']}
    if not (OUT / 'background_dependency_graph.json').exists():
        immutable_json(OUT / 'background_dependency_graph.json', graph)
    formula = '''# 正式三背景依赖与本轮干预边界

由 `haig_background.py:cols` 核对：Shared按Role兼容硬分配后，`Commonness = shared_commonness[assigned] * exp(-shared_RMS/shared_scale)`。External按Role兼容硬分配，`External = log1p(external_RMS / external_scale)`。Internal由当前APK的Host及主导Shared簇计算。`Residual = (1-clip(Commonness,0,1))*sqrt(max(External,0)*max(Internal,0))`。

```text
Shared32 ──> Commonness ───────┐
      └──> Internal ───────────┼──> Residual ─┐
External24 ─> External ────────┘              ├──> HostEncoder / Logit
Compact107 / Relation31 / RoleGraph ──────────┘
```

实际142维Host输入为`[Compact107, External, Internal, Commonness, Residual, Relation31]`。前111维进入`haig_models.py:Model.select_local_evidence`的Host Encoder；Relation31进入relation_key；Residual还参与注意力风险分数及四维背景统计摘要。因此背景在多个下游路径重复使用，不能将Commonness/Residual误认为独立第四/第五背景。

本轮B1、B2、B3是**训练期类盲等APK均值的原始标量信息中性化**，并严格重算Residual，输入宽度、网络参数与B0相同。B2同时中性化Shared派生的Commonness与Internal，属于**Shared来源信息组**消融，不能分离Shared→Internal与Shared→Commonness的单独效应，也不是重新拟合一个假想的Shared Bank。B4使用同Role训练Host的External响应置乱，重算Residual，作为同容量、保留边际尺度的负对照。所有模型重新训练；测试年绝不参与中性参照拟合。B0直接复用SCBE M0的同协议重新训练模型与checkpoint，不额外制造一个不同训练路径的“新基线”。

冻结模型的梯度及背景输入干预只代表数值敏感性；不代表真实APK级反事实或独立因果效应。没有运行时执行证据，也不把低异常解释为良性。
'''
    if not (OUT / 'formal_background_equations.md').exists():
        immutable_text(OUT / 'formal_background_equations.md', formula)
    if not (OUT / 'background_dependency_graph.md').exists():
        immutable_text(OUT / 'background_dependency_graph.md', formula)
    tensors = [{'name': 'Compact', 'indices': '0:107', 'width': 107, 'source': 'Host static compact', 'downstream': 'region_encoder'},
               {'name': 'External', 'indices': '107', 'width': 1, 'source': 'External24 benign bank', 'downstream': 'region_encoder, Residual, summary'},
               {'name': 'Internal', 'indices': '108', 'width': 1, 'source': 'APK-local dominant Shared cluster', 'downstream': 'region_encoder, Residual, summary'},
               {'name': 'Commonness', 'indices': '109', 'width': 1, 'source': 'Shared32 and training class prevalence', 'downstream': 'region_encoder, Residual, summary'},
               {'name': 'Residual', 'indices': '110', 'width': 1, 'source': 'deterministic External/Internal/Commonness', 'downstream': 'region_encoder, local score, summary'},
               {'name': 'Relation', 'indices': '111:142', 'width': 31, 'source': 'Host relation', 'downstream': 'relation_key'}]
    if not (OUT / 'background_tensor_manifest.csv').exists():
        immutable_csv(OUT / 'background_tensor_manifest.csv', tensors)
    data, _ = load_data_contract()
    if not (OUT / 'dataset_contract.json').exists():
        immutable_json(OUT / 'dataset_contract.json', {'status': 'PASS', 'n': len(data['ids']),
                       'year_counts': {str(y): int(np.sum(data['years'] == y)) for y in (2017, 2018, 2019, 2020, 2021)},
                       'new_downloaded_2017_2022_used': False, 'APK_parsing': False})
    print('BFA_AUDIT_PASS', status['checked'], flush=True)


def smoke(device):
    data, _ = load_data_contract()
    pack = formal_split(data, 'S1', SEEDS[0], device)
    report = {'status': 'PASS', 'split': 'S1', 'seed': SEEDS[0],
              'm0_max_probability_difference': pack['replay_diff'], 'arms': []}
    for arm in ARMS[1:]:
        a, b, how = transform_arm(pack['bags_train'], pack['bags_val'],
                                  pack['raw_train'], pack['raw_val'], arm, SEEDS[0])
        check = audit_bags(pack['bags_val'], b, arm)
        if check['changed_bags'] == 0:
            raise RuntimeError('smoke intervention changes no validation bags ' + arm)
        report['arms'].append({**how, **check})
    if not (OUT / 'intervention_smoke.json').exists():
        immutable_json(OUT / 'intervention_smoke.json', report)
    print('BFA_SMOKE_PASS', flush=True)


def train_phase(which, device, resume):
    from train import train_one
    if verify_assets()['status'] != 'PASS':
        raise RuntimeError('historical asset modified before BFA training')
    data, _ = load_data_contract()
    splits = SPLITS[:5] if which == 'source' else SPLITS[5:]
    for split in splits:
        for seed in SEEDS:
            pack = formal_split(data, split, seed, device)
            reference = pack['old']
            y = np.asarray(data['y'][pack['train']], int)
            ntrain = len(pack['train'])
            train_dir = OUT / 'models' / split / str(seed)
            train_dir.mkdir(parents=True, exist_ok=True)
            for arm in ARMS[1:]:
                path = train_dir / (arm + '.pkl')
                if path.exists():
                    if not resume:
                        raise RuntimeError('checkpoint exists without --resume: ' + str(path))
                    with path.open('rb') as f:
                        old = pickle.load(f)
                    if old['arm'] != arm or old['split'] != split or old['seed'] != seed or old['ids'] != reference['ids']:
                        raise RuntimeError('resume checkpoint identity mismatch')
                    print('BFA_RESUME', split, seed, arm, flush=True)
                    continue
                tr, va, how = transform_arm(pack['bags_train'], pack['bags_val'],
                                            pack['raw_train'], pack['raw_val'], arm, seed)
                validity = audit_bags(pack['bags_val'], va, arm)
                if validity['changed_bags'] == 0:
                    raise RuntimeError('arm changes no validation bags')
                if pack['dev'].type == 'cuda':
                    pack['v4'].torch.cuda.reset_peak_memory_stats(pack['dev'])
                start = time.perf_counter()
                pred, state, cost = train_one(pack['v4'], pack['args'], pack['dev'], seed,
                    int(reference['selected_epochs']), pack['global_train'], pack['global_val'],
                    pack['graph_train'], pack['graph_val'], tr, va, y)
                obj = {'arm': arm, 'split': split, 'seed': seed,
                       'ids': list(map(str, data['ids'][pack['val']])),
                       'train_n': ntrain, 'test_n': len(pack['val']),
                       'probability': pred, 'state_dict': state,
                       'selected_epochs': int(reference['selected_epochs']),
                       'parameter_n': int(cost['parameters']), 'cost': cost,
                       'feature_protocol': how, 'intervention_validity': validity,
                       'same_checkpoint_training_protocol_as_SCBE_M0': True,
                       'new_APKs_used': False, 'Stage1_and_Router_changed': False,
                       'elapsed_seconds': time.perf_counter() - start}
                with path.open('xb') as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('BFA_ARM_DONE', split, seed, arm,
                      'epochs=' + str(reference['selected_epochs']), flush=True)
            print('BFA_SPLIT_SEED_DONE', split, seed, flush=True)
    print('BFA_PHASE_DONE', which, flush=True)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    for flag in ('audit-only', 'smoke-test', 'attribution', 'full-attribution',
                 'train-source', 'train-rolling', 'ablation', 'evaluate', 'collect'):
        g.add_argument('--' + flag, action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if a.workers != 1:
        raise RuntimeError('single-worker protocol; do not compete with other tasks')
    if a.device == 'cuda':
        # This host's CUDA driver intermittently fails lazy initialization
        # after the historical feature bundle is loaded. Establish it first.
        import torch
        probe = torch.zeros(1, device='cuda')
        del probe
    if a.audit_only:
        audit_only(a.resume)
        return
    if not (OUT / 'asset_manifest_before.json').is_file() or verify_assets()['status'] != 'PASS':
        raise RuntimeError('run --audit-only first; historical assets must be unchanged')
    if a.smoke_test:
        smoke(a.device)
    elif a.train_source or a.ablation:
        train_phase('source', a.device, a.resume)
    elif a.train_rolling:
        train_phase('rolling', a.device, a.resume)
    elif a.attribution or a.full_attribution:
        from bfa_attribution import run
        run(a.device, a.resume, full=a.full_attribution)
    elif a.evaluate or a.collect:
        from bfa_evaluate import run
        run(a.resume)


if __name__ == '__main__':
    main()
