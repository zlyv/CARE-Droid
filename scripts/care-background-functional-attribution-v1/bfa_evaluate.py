#!/usr/bin/env python3
"""BFA-V1 paired APK-level evaluation; reads only frozen caches/new BFA checkpoints and writes only the new BFA output, never opens APKs."""
from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict

import numpy as np

from bfa_core import (ARMS, OUT, SCBE, SEEDS, SPLITS, immutable_csv, immutable_json,
                      immutable_text, load_data_contract, load_m0, verify_assets)
from metrics import cascade, paired_intervals
from train import split_contract


def model(split, seed, arm):
    if arm == ARMS[0]:
        return load_m0(split, seed), SCBE / 'models' / split / str(seed) / 'M0_REPRODUCED.pkl'
    path = OUT / 'models' / split / str(seed) / (arm + '.pkl')
    if not path.is_file():
        raise RuntimeError('BFA checkpoint missing: ' + str(path))
    with path.open('rb') as f:
        obj = pickle.load(f)
    if obj['arm'] != arm or obj['split'] != split or obj['seed'] != seed:
        raise RuntimeError('BFA checkpoint metadata mismatch')
    return obj, path


def get_group_metrics(rows, split, arm):
    return next(r for r in rows if r['split'] == split and r['arm'] == arm and r['seed'] == 'ENSEMBLE')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def run(resume):
    if verify_assets()['status'] != 'PASS':
        raise RuntimeError('historical asset changed before evaluation')
    if (OUT / 'final_decision.json').exists():
        if resume:
            print('BFA_EVALUATION_RESUME_COMPLETE', flush=True)
            return
        raise RuntimeError('final decision already exists')
    if not (OUT / 'background_logit_effects.csv').is_file():
        raise RuntimeError('full frozen attribution not finished')
    data, source = load_data_contract()
    rows, per, runtime, validity, conditional, fixed, paired = [], [], [], [], [], [], []
    distributions = {}
    source_order = np.flatnonzero(data['years'] == 2017)
    source_pos = {int(i): j for j, i in enumerate(source_order)}
    source_base = np.empty(len(source_order)); source_route = np.empty(len(source_order), bool)
    source_preds = {arm: np.empty(len(source_order)) for arm in ARMS}
    source_seen = {arm: np.zeros(len(source_order), bool) for arm in ARMS}
    for split in SPLITS:
        tr, va, base, route, score = split_contract(data, split, source)
        ids = list(map(str, data['ids'][va])); y = np.asarray(data['y'][va], int)
        for arm in ARMS:
            probabilities = []
            for seed in SEEDS:
                obj, path = model(split, seed, arm)
                if obj['ids'] != ids:
                    raise RuntimeError('checkpoint SHA order differs: ' + str(path))
                p = np.asarray(obj['probability'], float)
                if p.shape != (len(va),) or not np.isfinite(p).all():
                    raise RuntimeError('checkpoint invalid probability')
                probabilities.append(p)
                m = cascade(y, base, p, route)
                rows.append({'split': split, 'year': int(data['years'][va[0]]), 'arm': arm,
                             'seed': seed, 'train_n': len(tr), 'test_n': len(va),
                             'checkpoint': str(path), **m})
                if arm == ARMS[0]:
                    cost = obj['cost']; protocol = {'type': 'FROZEN_RETRAINED_M0'}
                else:
                    cost = obj['cost']; protocol = obj['feature_protocol']
                    validity.append({'split': split, 'seed': seed, 'arm': arm,
                                     **obj['intervention_validity'], **protocol})
                runtime.append({'split': split, 'seed': seed, 'arm': arm,
                                'selected_epochs': obj['selected_epochs'],
                                'training_seconds': cost['training_seconds'],
                                'inference_seconds': cost['inference_seconds'],
                                'parameters': cost['parameters'],
                                'gpu_peak_mib': cost['gpu_peak_mib'],
                                'checkpoint': str(path)})
            p = np.mean(probabilities, axis=0)
            distributions[(split, arm)] = (va, y, base, route, p)
            m = cascade(y, base, p, route)
            rows.append({'split': split, 'year': int(data['years'][va[0]]), 'arm': arm,
                         'seed': 'ENSEMBLE', 'train_n': len(tr), 'test_n': len(va),
                         'checkpoint': 'mean of three seed probabilities', **m})
            conditional.append({'split': split, 'arm': arm, 'FN_vs_TN_AUC': m['FN_vs_TN_AUC'],
                                'TP_vs_FP_AUC': m['TP_vs_FP_AUC'], 'Routed_AUC': m['Routed_AUC'],
                                'FN_n': m['FN_n'], 'TN_n': m['TN_n'],
                                'FP_n': m['FP_n'], 'TP_n': m['TP_n']})
            fixed.append({'split': split, 'arm': arm, 'Fixed': m['Fixed'],
                          'Broken': m['Broken'], 'Net': m['Net'],
                          'Fixed_FN': m['Fixed_FN'], 'Fixed_FP': m['Fixed_FP'],
                          'Broken_TN': m['Broken_TN'], 'Broken_TP': m['Broken_TP'],
                          'routed_errors': m['routed_errors'],
                          'unrouted_errors': m['unrouted_errors']})
            final = np.where(route, p, base)
            for i, sid in enumerate(ids):
                per.append({'split': split, 'year': int(data['years'][va[i]]),
                            'arm': arm, 'sha256': sid, 'label': int(y[i]),
                            'stage1_probability': float(base[i]),
                            'router_score': float(score[i]) if np.isfinite(score[i]) else None,
                            'routed': int(route[i]), 'expert_probability': float(p[i]),
                            'final_probability': float(final[i]),
                            'final_correct': int(int(final[i] >= .5) == int(y[i]))})
            if split.startswith('S'):
                positions = np.asarray([source_pos[int(i)] for i in va])
                source_preds[arm][positions] = p
                source_seen[arm][positions] = True
                if arm == ARMS[0]:
                    source_base[positions] = base
                    source_route[positions] = route
    for arm in ARMS:
        if not source_seen[arm].all():
            raise RuntimeError('source OOF incomplete')
        m = cascade(np.asarray(data['y'][source_order], int), source_base,
                    source_preds[arm], source_route)
        rows.append({'split': 'SOURCE_2017_OOF', 'year': 2017, 'arm': arm,
                     'seed': 'ENSEMBLE', 'train_n': 'five outer folds',
                     'test_n': 799, 'checkpoint': 'five folds × three seeds', **m})
        distributions[('SOURCE_2017_OOF', arm)] = (source_order,
            np.asarray(data['y'][source_order], int), source_base, source_route, source_preds[arm])
        conditional.append({'split': 'SOURCE_2017_OOF', 'arm': arm,
                            'FN_vs_TN_AUC': m['FN_vs_TN_AUC'],
                            'TP_vs_FP_AUC': m['TP_vs_FP_AUC'], 'Routed_AUC': m['Routed_AUC'],
                            'FN_n': m['FN_n'], 'TN_n': m['TN_n'],
                            'FP_n': m['FP_n'], 'TP_n': m['TP_n']})
        fixed.append({'split': 'SOURCE_2017_OOF', 'arm': arm,
                      'Fixed': m['Fixed'], 'Broken': m['Broken'], 'Net': m['Net'],
                      'routed_errors': m['routed_errors'], 'unrouted_errors': m['unrouted_errors']})
    for split in ('SOURCE_2017_OOF', 'T1', 'T2', 'T3', 'T4'):
        _, y, base, route, reference = distributions[(split, ARMS[0])]
        for arm in ARMS[1:]:
            _, yy, bb, rr, p = distributions[(split, arm)]
            if not (np.array_equal(y, yy) and np.array_equal(base, bb) and np.array_equal(route, rr)):
                raise RuntimeError('Stage-I or Router changed between arms')
            for result in paired_intervals(y, base, route, p, reference, repeats=500):
                paired.append({'split': split, 'candidate': arm,
                               'reference': ARMS[0], **result})
    # M0 must equal already frozen SCBE M0 metrics exactly; the same checkpoint is reused.
    with (SCBE / 'source2017_fivefold_results.csv').open(encoding='utf-8', newline='') as f:
        old_source = list(csv.DictReader(f))
    existing = next(r for r in old_source if r['split'] == 'SOURCE_2017_OOF'
                    and r['arm'] == 'M0_REPRODUCED' and r['seed'] == 'ENSEMBLE')
    current = get_group_metrics(rows, 'SOURCE_2017_OOF', ARMS[0])
    if abs(float(existing['BA']) - current['BA']) > 1e-12 or int(existing['FN']) != current['FN']:
        raise RuntimeError('M0 source metrics not reproduced')
    prior_replay = json.loads((SCBE / 'baseline_replay.json').read_text(encoding='utf-8'))
    immutable_json(OUT / 'baseline_replay_training.json', {
        'status': 'PASS', 'source_M0_metrics_exact': True,
        'source_M0_BA': current['BA'], 'source_M0_FN': current['FN'],
        'source_formal_comparison_is_descriptive': True,
        'SCBE_original_historical_replay': prior_replay,
        'training_phase_replayed_M0_validation_before_each_arm': True})
    # Report all years and capacity/seed controls; never treat seeds as independent APKs.
    get = lambda split, arm: get_group_metrics(rows, split, arm)
    scores = {}
    for arm in ARMS[1:]:
        ba = float(np.mean([get(t, arm)['BA'] - get(t, ARMS[0])['BA'] for t in ('T2', 'T3', 'T4')]))
        net = int(sum(get(t, arm)['Net'] - get(t, ARMS[0])['Net'] for t in ('T2', 'T3', 'T4')))
        years = int(sum(get(t, arm)['BA'] > get(t, ARMS[0])['BA'] for t in ('T2', 'T3', 'T4')))
        fn2020 = int(get('T3', arm)['FN'] - get('T3', ARMS[0])['FN'])
        source_gap = float(get('SOURCE_2017_OOF', arm)['BA'] - get('SOURCE_2017_OOF', ARMS[0])['BA'])
        seed_gains = []
        for seed in SEEDS:
            gain = np.mean([next(r['BA'] for r in rows if r['split'] == t and r['arm'] == arm and r['seed'] == seed) -
                            next(r['BA'] for r in rows if r['split'] == t and r['arm'] == ARMS[0] and r['seed'] == seed)
                            for t in ('T2', 'T3', 'T4')])
            seed_gains.append(float(gain))
        scores[arm] = {'T2_T4_mean_BA_gain': ba, 'T2_T4_Net_gain': net,
                       'positive_BA_years': years, '2020_FN_change': fn2020,
                       'source_BA_gain': source_gap, 'seed_mean_BA_gains': seed_gains,
                       'screen_pass': ba >= .005 and net >= 6 and years >= 2
                         and fn2020 <= 0 and source_gap >= -.01 and min(seed_gains) >= -.005}
    main_arms = ARMS[1:4]
    supported = [arm for arm in main_arms if scores[arm]['screen_pass']]
    # Capacity-matched shuffle must not explain away an apparent External benefit.
    if ARMS[1] in supported and scores[ARMS[1]]['T2_T4_mean_BA_gain'] <= scores[ARMS[4]]['T2_T4_mean_BA_gain']:
        supported.remove(ARMS[1])
    status = 'BACKGROUND_ABLATION_IMPROVES_CLASSIFICATION' if supported else 'BACKGROUND_ABLATION_NOT_RELIABLE'
    integrity = verify_assets()
    if integrity['status'] != 'PASS':
        raise RuntimeError('historical assets changed during BFA')
    immutable_csv(OUT / 'background_ablation_source2017.csv', [r for r in rows if r['split'].startswith('S') or r['split'] == 'SOURCE_2017_OOF'])
    immutable_csv(OUT / 'background_ablation_rolling.csv', [r for r in rows if r['split'].startswith('T')])
    immutable_csv(OUT / 'per_apk_predictions.csv', per)
    immutable_csv(OUT / 'fixed_broken_analysis.csv', fixed)
    immutable_csv(OUT / 'conditional_auc_analysis.csv', conditional)
    immutable_csv(OUT / 'paired_intervals.csv', paired)
    immutable_csv(OUT / 'runtime_cost.csv', runtime)
    immutable_csv(OUT / 'background_capacity_controls.csv', [r for r in rows if r['arm'] in (ARMS[0], ARMS[1], ARMS[4])])
    immutable_csv(OUT / 'background_interaction_results.csv', [
        {'status': 'NOT_IDENTIFIABLE_WITH_THIS_FACTORIAL',
         'reason': 'B2 jointly neutralizes Commonness and Internal; no independent, legal full-bank Shapley intervention'}])
    # Alias the pre-registered information-neutralization comparison for easy inspection.
    immutable_csv(OUT / 'background_error_association.csv', [
        {'arm': arm, **score} for arm, score in scores.items()])
    immutable_json(OUT / 'asset_integrity_final.json', integrity)
    decision = {'status': status, 'candidate_arms': supported,
                'scores': scores, 'formal_integrity': integrity['status'],
                'new_APKs_used': False, 'future_years_are_development': True,
                'Stage1_or_Router_changed': False,
                'B2_scope': 'Shared-derived Commonness+Internal joint information group, not pure Shared Bank deletion',
                'B5': 'not run: single-background changes do not establish an independently identifiable legal interaction',
                'formal_mainline_changed': False,
                'next_action': '保留正式三背景；任何候选都需独立新数据验证。' if not supported else
                               '仅开发集消融候选，先检查对照和独立测试，不替换正式模型。'}
    immutable_json(OUT / 'final_decision.json', decision)
    table = ['| 窗口 | 方案 | BA | AUC | FN | FP | Fixed | Broken | Net | FN-vs-TN AUC | TP-vs-FP AUC |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('SOURCE_2017_OOF', 'T1', 'T2', 'T3', 'T4'):
        for arm in ARMS:
            r = get(split, arm)
            table.append(f"| {split} | {arm} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {r['FN_vs_TN_AUC']:.4f} | {r['TP_vs_FP_AUC']:.4f} |")
    report = ['# CARE-Droid 三背景功能归因与受控消融 BFA-V1', '', f'最终状态：**{status}**。',
              '', '只用已有2017—2021开发缓存；未读取新APK、未改第一层/路由/正式checkpoint。B0复用SCBE M0同协议重新训练模型。',
              '', '## 固定路由下的端到端结果', '', *table, '', '## 预注册筛选与解释边界', '']
    for arm, score in scores.items():
        report.append(f"- {arm}: T2—T4平均BA差{score['T2_T4_mean_BA_gain']:+.5f}；Net差{score['T2_T4_Net_gain']:+d}；2020 FN差{score['2020_FN_change']:+d}；源域BA差{score['source_BA_gain']:+.5f}；正向年份{score['positive_BA_years']}/3；筛选{'通过' if score['screen_pass'] else '未通过'}。")
    report += ['', '所有变体保持142维输入、同一参数量、三种子、相同Epoch/BCE/阈值与冻结上游。B1/B3是单入口信息中性化；B2是Shared派生的Commonness+Internal联合信息中性化，不能单独归因于Shared Bank。B4是同Role训练Host External响应置乱。中性化后的某些Host联合背景响应未必对应真实可实现APK，故仅为模型输入实验，非程序因果反事实。',
               '冻结梯度和干预的逐APK结果、2020固定22对、2019 FP/TP、2021 FN/TN见对应CSV。Commonness与Residual不是独立背景；前者来自Shared，后者由三列决定。',
               'B5交互组未运行：在Shared→Internal/Commonness的依赖下，现有变体不能独立拆出纯背景主效应，不能硬做Shapley。',
               '2018—2021已多次用于方法开发；配对bootstrap只是探索性不确定性，不构成独立最终验证。没有任何结果自动授权修改正式三背景。',
               f'命名冻结资产SHA256核验：{integrity["status"]}，共{integrity["checked"]}项；原始APK未打开，不声称全服务器完整性。', '']
    sign_rows = csv_rows(OUT / 'seed_direction_consistency.csv')
    effect_rows = csv_rows(OUT / 'background_effect_group_summary.csv')
    def count_sign(split, branch, group):
        selected = [r for r in sign_rows if r['split'] == split and r['branch'] == branch and r['group'] == group]
        return sum(r['all_negative'] == 'True' for r in selected), len(selected)
    def effect(split, branch, group):
        selected = [r for r in effect_rows if r['split'] == split and r['branch'] == branch and r['group'] == group]
        if not selected:
            return '该组无样本'
        r = selected[0]
        return f"均值{float(r['mean_effect']):+.4f} logit，APK={r['APKs']}"
    def change(split, arm, metric):
        return get(split, arm)[metric] - get(split, ARMS[0])[metric]
    report += ['## Q1–Q15逐项研究判断', '',
        'Q1：Shared32通过Commonness及Internal的共享分配路径影响Residual，再进入Host Encoder、注意力风险分数和摘要；External24产生External标量，Internal来自APK本身的Host集合。确切计算见依赖图和公式文件。',
        'Q2：是。Residual是External、Internal、Commonness的确定性函数；Shared还参与Internal的簇分配，因此无法把三个路径解释为彼此独立的因果贡献。',
        f"Q3：2020未修复FN中External原始值相对类盲训练锚点在三种子均压低恶意Logit的APK为{count_sign('T3','External','UNFIXED_FN')[0]}/{count_sign('T3','External','UNFIXED_FN')[1]}；这是模型输入敏感性，不是恶意语义证明。",
        f"Q4：2020持续正确TN中同方向为{count_sign('T3','External','KEPT_TN')[0]}/{count_sign('T3','External','KEPT_TN')[1]}；必须比较固定22对表，而不能只看FN单侧。",
        f"Q5：2020未修复FN的Shared来源组同方向为{count_sign('T3','Shared_group','UNFIXED_FN')[0]}/{count_sign('T3','Shared_group','UNFIXED_FN')[1]}，Internal为{count_sign('T3','Internal','UNFIXED_FN')[0]}/{count_sign('T3','Internal','UNFIXED_FN')[1]}；两者路径重叠，不相加。",
        f"Q6：2019未修复FP External作用：{effect('T2','External','UNFIXED_FP')}；正确TP：{effect('T2','External','KEPT_TP')}。其他背景见逐APK表；这里是开发年份的描述性比较。",
        f"Q7：2021未修复FN External三种子均为负向者{count_sign('T4','External','UNFIXED_FN')[0]}/{count_sign('T4','External','UNFIXED_FN')[1]}；和2020同口径比较，不能宣称独立验证。",
        'Q8：冻结干预与重新训练消融回答不同问题；只有数值方向一致且公平消融通过筛选，才能提出背景功能候选，不能把两者硬解释成同一因果效应。',
        f"Q9：B1取消External有效变化并重算Residual后，T3的FN差{change('T3',ARMS[1],'FN'):+d}、FP差{change('T3',ARMS[1],'FP'):+d}、BA差{change('T3',ARMS[1],'BA'):+.5f}；T2–T4合计Net差{scores[ARMS[1]]['T2_T4_Net_gain']:+d}。",
        f"Q10：B2共同取消Shared派生的Commonness+Internal后，T3的FN差{change('T3',ARMS[2],'FN'):+d}、FP差{change('T3',ARMS[2],'FP'):+d}、BA差{change('T3',ARMS[2],'BA'):+.5f}；这不是纯Shared Bank效应。",
        f"Q11：B3单独中性化Internal并重算Residual后，T3的FN差{change('T3',ARMS[3],'FN'):+d}、FP差{change('T3',ARMS[3],'FP'):+d}、BA差{change('T3',ARMS[3],'BA'):+.5f}。",
        'Q12：本轮只能识别按指定训练期锚点中性化后是否损失预测信息；不能证明其他背景不可替代。各年BA、条件AUC和置信区间见表格。',
        'Q13：未获得可独立识别的背景交互效应。Shared→Internal/Commonness依赖使简易独立Shapley分解无效，B5未强行执行。',
        'Q14：模型宽度、参数量、训练种子、Epoch、BCE和路由固定，且设置同Role置乱对照；但中性化可能改变联合输入分布，无法完全排除分布外解释。',
        f"Q15：预注册候选为{supported if supported else '无'}；最终状态{status}。这批年份属于开发数据，任何候选均不自动改变正式三背景。", '']
    immutable_text(OUT / 'final_report.md', '\n'.join(report))
    immutable_text(OUT / 'reproduce_commands.md', '# BFA-V1复现命令\n\n本V1结果不可覆盖；请在新版本目录复跑。\n\n`bash scripts/care-background-functional-attribution-v1/run.sh --audit-only`\n`bash scripts/care-background-functional-attribution-v1/run.sh --smoke-test --device cuda`\n`bash scripts/care-background-functional-attribution-v1/run.sh --attribution --device cuda`\n`bash scripts/care-background-functional-attribution-v1/run.sh --full-attribution --device cuda`\n`bash scripts/care-background-functional-attribution-v1/run.sh --train-source --resume --device cuda`\n`bash scripts/care-background-functional-attribution-v1/run.sh --train-rolling --resume --device cuda`\n`bash scripts/care-background-functional-attribution-v1/run.sh --evaluate --resume`\n')
    print('BFA_FINAL', status, flush=True)

