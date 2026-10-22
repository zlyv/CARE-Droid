#!/usr/bin/env python3
"""Freeze evidence-bounded Chinese report and decisions; write only new audit outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict

import numpy as np

from audit_assets import OUT,ROOT,TAGS,SEEDS,read_csv,write_csv,write_json,verify_manifest


def n(row,key):
    v=row.get(key)
    return float(v) if v not in (None,'','None','nan') else np.nan


def main():
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true',required=True);p.parse_args()
    for name in ('preflight.json','analysis_phase.json','error_phase.json'):
        if json.loads((OUT/name).read_text())['status']!='PASS':raise RuntimeError('phase not PASS: '+name)
    integrity=verify_manifest()
    write_json(OUT/'baseline_integrity_check.json',integrity)
    if integrity['status']!='PASS':raise RuntimeError('protected historical asset changed')
    support=read_csv(OUT/'prototype_class_support.csv')
    overlap=read_csv(OUT/'shared_external_overlap.csv')
    control=read_csv(OUT/'role_matched_overlap_controls.csv')
    probe=read_csv(OUT/'background_incremental_information.csv')
    probe_null=read_csv(OUT/'background_probe_controls.csv')
    intervals=read_csv(OUT/'paired_intervals.csv')
    association=read_csv(OUT/'background_error_association.csv')
    unique=read_csv(OUT/'prototype_unique_coverage.csv')
    dependency=read_csv(OUT/'background_dependency_metrics.csv')
    manifests=json.loads((OUT/'prototype_bank_manifest.json').read_text())['banks']
    types=Counter((r['bank'],r['support_type']) for r in support)
    if len(support)!=12*56 or len(manifests)!=12:
        raise RuntimeError('12-bank/56-Prototype count changed')
    bykey=defaultdict(list)
    for r in control:
        if r['sampled_shared_id']==r['true_nearest_shared_id']:continue
        if np.isfinite(n(r,'jaccard')):
            bykey[(r['tag'],r['seed'],r['external_id'])].append(n(r,'jaccard'))
    corrected=[]
    for r in overlap:
        null=bykey[(r['tag'],r['seed'],r['external_id'])]
        jj=n(r,'jaccard_train_APK_support')
        percentile=float(np.quantile(null,.95)) if null else None
        corrected.append({'tag':r['tag'],'seed':r['seed'],'external_id':r['external_id'],
            'nearest_shared_id':r['shared_id'],'real_jaccard':jj if np.isfinite(jj) else None,
            'role_matched_non_nearest_controls':len(null),'null95_jaccard':percentile,
            'real_exceeds_null95':bool(jj>percentile) if percentile is not None and np.isfinite(jj) else None,
            'control_boundary':'nearest itself excluded; absent if no alternative role-compatible Shared center'})
    write_csv(OUT/'role_matched_overlap_controls_corrected.csv',corrected)
    evaluable=[r for r in corrected if r['real_exceeds_null95'] is not None]
    above=sum(r['real_exceeds_null95'] for r in evaluable)
    close=sum(r['distance_close_unlabeled']=='True' for r in overlap)
    jhigh=sum(n(r,'jaccard_train_APK_support')>=.7 for r in overlap)
    unique_num=sum(int(r['unique_vs_any_role_compatible_shared_APKs']) for r in unique)
    unique_den=sum(int(r['external_support_APKs']) for r in unique)
    p2_p1=[];positive_by_tag=Counter();null_by_key=defaultdict(list)
    for r in probe_null:
        if r['control']=='TRAIN_EXTERNAL_ROLE_SIZE_SHUFFLE':
            null_by_key[(r['tag'],r['seed'])].append(n(r,'AUC'))
    by_probe=defaultdict(dict)
    for r in probe:by_probe[(r['tag'],r['seed'])][r['arm']]=n(r,'AUC')
    for key,value in by_probe.items():
        delta=value['P2_PLUS_EXTERNAL']-value['P1_PLUS_SHARED_COMMONNESS']
        good=delta>=.01 and value['P2_PLUS_EXTERNAL']>np.quantile(null_by_key[key],.95)
        positive_by_tag[key[0]]+=int(good)
        p2_p1.append({'tag':key[0],'seed':key[1],'P2_minus_P1_AUC':delta,
            'P2_exceeds_condition_shuffle95':bool(value['P2_PLUS_EXTERNAL']>np.quantile(null_by_key[key],.95)),
            'fixed_increment_rule_pass':bool(good)})
    write_csv(OUT/'background_increment_vs_controls.csv',p2_p1)
    ext_sig=[r for r in intervals if r['metric']=='mean_external'][0]
    shared_sig=[r for r in intervals if r['metric'] in ('high_support_shared_count','external_shared_overlap_fraction')]
    broad_overlap=bool(jhigh>=.5*len(overlap) and above>=.5*len(evaluable) and len(evaluable)>=.5*len(overlap))
    cross_year_info=bool(sum(positive_by_tag[t]>=2 for t in TAGS[1:])>=2)
    shared_fn_link=bool(any(n(r,'BH_q_across_predeclared_metrics')<.05 for r in shared_sig))
    stable_error_link=False
    for metric in ('mean_external','external_shared_overlap_fraction'):
        sub=[r for r in association if r['metric']==metric]
        # Validation years are previously observed development sets, so this is descriptive.
        if all(any(r['tag']==tag and n(r,'BH_q_within_tag_seed')<.05 for r in sub)
               for tag in ('T2_C1','T4_C1')):
            stable_error_link=True
    statuses=[]
    if broad_overlap:statuses.append('EXTERNAL_SHARED_REDUNDANCY_CANDIDATE')
    if cross_year_info:statuses.append('EXTERNAL_BENIGN_INFORMATION_COMPLEMENTARY')
    if shared_fn_link and stable_error_link:statuses.append('SHARED_PROTOTYPE_FN_ASSOCIATION_CANDIDATE')
    if jhigh>0 and not stable_error_link:statuses.append('OVERLAP_WITHOUT_ERROR_LINK')
    if not broad_overlap and not cross_year_info and not (shared_fn_link and stable_error_link):
        statuses.append('NO_VERIFIED_BACKGROUND_OVERLAP_BOTTLENECK')
    decision={'status':statuses[-1], 'diagnostic_labels':statuses,
        'formal_integrity':integrity['status'],'protected_asset_count':integrity['asset_count'],
        'banks':12,'shared_prototype_instances':384,'external_prototype_instances':288,
        'role_compatible_external_instances':len(overlap),
        'distance_close_instances':close,'APK_jaccard_at_least_0_7_instances':jhigh,
        'role_null_evaluable_instances':len(evaluable),'real_overlap_above_role_null95':above,
        'external_unique_support_occurrences':unique_num,
        'external_support_occurrences':unique_den,
        'external_unique_occurrence_fraction':unique_num/unique_den,
        '2020_matched_pairs':22,'2020_external_mean_FN_minus_TN':n(ext_sig,'mean_FN_minus_TN'),
        '2020_external_mean_BH_q':n(ext_sig,'BH_q_across_predeclared_metrics'),
        '2020_shared_overlap_metrics_BH_lt_0_05':shared_fn_link,
        'stable_2019_2021_error_link':stable_error_link,
        'probe_fixed_increment_seeds_by_tag':dict(positive_by_tag),
        'next_model_change_authorized':False,
        'recommendation':'保留原三背景；不据本次开发集审计新增去冗余或类别校准模块。',
        'future_labels_used_for_bank_or_thresholds':False,
        'independent_external_test':False}
    write_json(OUT/'final_decision.json',decision)
    runtime=[{'tag':m['tag'],'seed':m['seed'],'train_APKs':m['train_n'],
              'validation_APKs':m['validation_n'],'support_seconds':m['elapsed_seconds'],
              'new_DEX_scans':0,'GPU_used':False} for m in manifests]
    write_csv(OUT/'runtime_cost.csv',runtime)
    imp='# 三背景实际实现核查\n\n'
    imp+='- `scripts/run_care_joint_dualprototype_ablation.py::fit_prototype_state`：训练期每APK最多采样16个Host，按类别平衡形成Shared训练Host池，以MiniBatchKMeans得到32个107维中心；良性训练Host池形成24个107维External中心。中心不是单个真实Host。\n'
    imp+='- `scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py::split,profiles,hard,cols`：169维原始Host拆为107维Compact、31维Role权重和31维Relation。107维只在训练窗口内按Host训练集拟合均值/标准差；两类Bank均在这个107维空间。Role profile由训练Host对最近中心的分配累积，Host硬分配优先同主Role，否则回退最近中心。\n'
    imp+='- 同一源码中的`cols`输出4列External、Internal、Commonness、Residual，再与107维Compact、31维Relation拼成142维网络Host输入。空Host沿正式占位规则，零Role会回退最近中心。\n'
    imp+='- 四窗口、三种子均为独立Bank。中心跨窗口/种子没有共享坐标，禁止跨checkpoint直接算距离。来源Host不可精确追踪：`prototype_provenance.csv`只记录最近观测训练Host，绝非KMeans中心的唯一构成来源；排除自来源支持率不可实现，支持率为训练期描述。\n'
    imp+='- `outputs/care_existing_three_background_contrastive_v2/existing_three_background_audit.md`及本轮源码核查显示先前已审计三背景、背景Probe和负实验；本轮新增的是逐中心训练APK支持与跨Bank支持集对应，不重复把已有Prototype当新模块。\n'
    (OUT/'background_implementation_audit.md').write_text(imp,encoding='utf-8')
    formula='# 背景计算公式审计\n\n'
    formula+='''正式距离为107维均方差根 `RMS(x,c)=sqrt(mean((x-c)^2))`。Shared硬分配最小化RMS/该中心shared_scale（优先同Role），External硬分配最小化RMS（优先同Role）。`commonness_j=clip(2*min(prevalence_B_j,prevalence_M_j),0,1)`，Host commonness为选中Shared中心的commonness乘以`exp(-RMS/shared_scale)`；External为`log1p(RMS/external_scale)`；Internal由当前APK中主导Shared簇的袋内中心距离构成；Residual=`(1-clip(Commonness,0,1))*sqrt(max(External,0)*max(Internal,0))`。因此Residual在数学上由前三列确定，不是第四个独立事实。\n\n'''
    formula+='''本轮支持阈值不是正式分类阈值：仅用每个Bank训练APK的“匹配Host中位距离”的无标签q25/q50/q75固定，APK有至少一个被正式硬分配的Host位于阈值内则该原型获一次支持。训练KMeans中心的自来源无法排除。\n'''
    (OUT/'background_formula_audit.md').write_text(formula,encoding='utf-8')
    (OUT/'commonness_residual_audit.md').write_text(formula+'\n验证期APK均值的相关性见`background_dependency_metrics.csv`；相关、数学依赖、类别冗余三者不能混同。\n',encoding='utf-8')
    (OUT/'background_information_interpretation.md').write_text('''# 条件信息解读\n\n`background_incremental_information.csv`中的P0/P1/P2/P3均是训练窗口标签拟合的固定C=1线性诊断Probe，不是正式Direct-C43的因果消融。P2相对P1与Role/规模分层训练置乱及同维噪声对照见`background_increment_vs_controls.csv`。这只能描述冻结响应的可读增量；这些背景列本已进入Direct-C43。2018—2021为既有开发年份，不是独立盲测。\n''',encoding='utf-8')
    lines=['# CARE-Droid三背景Prototype审计 V1','',
        '**最终判断：'+decision['status']+'。不修改正式External Benign Background。**','',
        '本轮是冻结资产只读审计，无新模型训练、无APK重提取或动态执行。样本统计以APK为单位；三种子不是独立APK。', '',
        '## 对12个研究问题的逐项回答','',
        f"1. Shared中心并不普遍是两类高支持：{types[('Shared','HIGH_SUPPORT_SHARED')]}/384个Bank实例满足预设高支持共享定义；另有{types[('Shared','BENIGN_ASSOCIATED')]}/384个偏良性、{types[('Shared','INSUFFICIENT_SUPPORT')]}/384个支持不足。平衡采样来源不等于类别共享。",
        f"2. External中{types[('External','HIGH_SUPPORT_SHARED')]}/288个为高支持共享，不能把全部良性来源中心视为良性特异。",
        f"3. External中{types[('External','BENIGN_ASSOCIATED')]}/288个达到训练期良性关联标准，但仅是统计关联，不是Host良性语义真值。",
        f"4. 288个External Bank实例中{len(overlap)}个存在同主Role的Shared候选；其中{close}个最近跨Bank距离不超过无标签同Bank最近中心距离中位数。其余20个不可强行按Role匹配。",
        f"5. 最近同Role对中{jhigh}/{len(overlap)}个训练APK支持Jaccard≥0.70。向量接近和支持集相同不是同一结论。",
        f"6. 排除真实最近中心后的Role匹配对照仅{len(evaluable)}对有可评价的非最近合法候选；其中{above}对真实Jaccard高于对照95分位。原始随机对照含最近中心，不用于这一判断。不能推广到全部288个External实例。",
        f"7. External在可比较对中的独有支持为{unique_num}/{unique_den}次原型×APK支持发生（{unique_num/unique_den:.1%}）；这是发生次数而非去重APK总量。",
        '8. Commonness直接来自Shared分配及两类训练支持；Residual严格由External、Internal和Commonness决定。训练期类别支持并未作为单独输入列，已有响应相关性见背景依赖表；数学依赖不等于类别增量已被充分利用。',
        f"9. 2020固定22对FN/TN中，FN的External均值平均低{abs(n(ext_sig,'mean_FN_minus_TN')):.3f}（BH q={n(ext_sig,'BH_q_across_predeclared_metrics'):.4f}），但共享原型计数/External—Shared重叠比例均未过预设多重比较门槛；不能把‘更接近良性中心’等同于‘重叠导致FN’。",
        '10. 2019路由FP/TP与2021路由FN/TN均未获得跨三种子且校正后稳定的相同重叠—错误关系；详情见`background_error_association.csv`。',
        '11. 未获得支持修改External构建方式的因果或跨年稳健证据。局部重叠存在，误判根因未验证。',
        '12. 下一轮建议保留原三背景；若研究此假设，应先在未参与开发的新独立样本中冻结支持规则验证，不实施新的Prototype约束。','',
        '## 条件信息与先前负实验','',
        '固定线性Probe中External增量在C0的3/3种子通过预设0.01 AUC且超过置乱95分位，T3（验证2020）0/3通过；跨窗口不稳定。该Probe不是新CARE-Droid模型，也不能证明删除External无害。历史TB-FRS及三背景SupCon/Layerwise实验研究的是表示/训练机制，本轮研究的是逐中心训练APK覆盖与误判关联；统计对象不同，结果不矛盾。','',
        '## 解释与安全边界','',
        'KMeans中心来源APK/Host不可唯一恢复；最近训练Host仅是可复核的几何近邻，不能据此排除自来源。类别支持为训练内描述，不是独立泛化。2019—2021为已反复使用的开发年份，配对显著性只说明关联。没有任何单项API、Role或Prototype距离证明恶意意图。',
        f"已复核{integrity['asset_count']}个命名历史文件，包括12个checkpoint和3个抽样原始APK：**{integrity['status']}**。完整性结论不涵盖仓库全部文件或所有原始APK。",'']
    (OUT/'final_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'reproduce_commands.md').write_text('''# 复现命令\n\n只在全新版本目录运行；已有V1结果不得覆盖。\n\n`PYTHONDONTWRITEBYTECODE=1 /home/boyuanjie/envs/tif/bin/python scripts/care-background-prototype-audit-v1/audit_assets.py --audit-only`\n\n`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/boyuanjie/envs/tif/bin/python scripts/care-background-prototype-audit-v1/compute_prototype_support.py --prototype-support --device cpu --workers 1`\n\n`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/boyuanjie/envs/tif/bin/python scripts/care-background-prototype-audit-v1/analyze.py --collect --device cpu --workers 1`\n\n`PYTHONDONTWRITEBYTECODE=1 /home/boyuanjie/envs/tif/bin/python scripts/care-background-prototype-audit-v1/analyze_errors.py --error-association`\n\n`PYTHONDONTWRITEBYTECODE=1 /home/boyuanjie/envs/tif/bin/python scripts/care-background-prototype-audit-v1/collect_results.py --collect`\n''',encoding='utf-8')
    print('PROTOTYPE_AUDIT_FINAL',decision['status'],flush=True)


if __name__=='__main__':main()
