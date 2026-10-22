#!/usr/bin/env python3
"""LFC-V1 read-only provenance preflight. Reads LiteStage and frozen C43 assets; writes audit/hash manifests only in LFC_OUT. No APK execution, extraction or training."""
from __future__ import annotations
import json
from core import *

ASSETS=[LITE/'final_decision.json',LITE/'stage1_oof_predictions.csv',LITE/'future_frozen_predictions.csv',LITE/'feature_extraction_cost.csv',LITE/'source_selection.json',HDBC/'source2017_oof_predictions.csv',HDBC/'future2018_2021_predictions.csv',HDBC/'final2017_frozen.json',lc.SOURCE_PRED,lc.P31,lc.MAPPING]
def main():
    ids,y,fold,pred,expert=source_index();s=shard('source');f=shard('future')
    if set(ids)!=set(s):raise RuntimeError('source shard alignment')
    if len(f)!=1989 or len(set(f))!=1989:raise RuntimeError('future shard coverage')
    frozen=json.loads((HDBC/'final2017_frozen.json').read_text())
    if frozen['train_years']!=[2017]:raise RuntimeError('future C43 not source-only')
    old=json.loads((LITE/'final_decision.json').read_text())
    if old['asset_integrity']!='PASS' or not old['future_complete']:raise RuntimeError('LiteStage incomplete')
    if old['selected_global_arm']!='L4_LOG':raise RuntimeError('frozen LiteStage changed')
    historical=legacy_source()
    for arm in ('L1_MANIFEST','L5_P31_RF','LITE_NESTED_SELECTED'):
        if sum((arm,sid) in historical for sid in ids)!=799:raise RuntimeError('OOF missing '+arm)
    before=[sha(p) for p in ASSETS];put_json('asset_manifest_before.json',before)
    # E0 correction: the LiteStage L4 extraction persisted invocation count in dex_meta[4].
    # Its L2/L3 cached vectors therefore cannot be accepted as pure pre-invoke features.
    report=f'''# LFC-V1 现有LiteStage资产审计\n\n2017=799，固定五折；未来2018–2021=1989。两组LiteStage JSON shard均完整，不重提取APK。Direct-C43使用HDBC H0的同折2017 OOF概率；2018–2021来自2017-only T1冻结模型，覆盖所有APK。新的路由在测试前仅使用第一层概率。\n\n上一轮93.49%是**逐外层折内选择的混合配置**，不是单一L4_LOG模型：各折选择记录在`outputs/care_lite_stage_v1/source_selection.json`。全2017训练后冻结的未来配置为L4_LOG。它依赖Manifest、DEX解析、方法引用解析、方法体invoke扫描；不构建调用图或Host。2017实测共享扫描平均总耗时约6.78秒/APK，OS页缓存未受控。Manifest-only约0.184秒，引用路径的阶段计时约0.184+0.762+1.277秒，但阶段之和不是独立重跑的冷启动成本。P31历史计时约5.53秒，环境不匹配。\n\n**重要修正：**LiteStage shard的`dex_meta[4]`是全方法体`invoke_n`，上轮L2_DEXMETA及L3各臂训练向量包含该字段，因此不能把其预测作为纯元信息／纯引用低成本候选。LFC新Ref候选只使用Manifest、Role引用和不含invoke的DEX MethodId数量；重新拟合2017固定五折RF，不用旧L2/L3 OOF冒充公平对照。上轮L1_MANIFEST、正式P31-RF及嵌套BestLiteStage的预测可以合法复用。\n\nC43第二层全样本冻结预测可按新路由选择，不需要重新训练或错误折填补。新系统仍是共享冻结专家诊断，不声称已完成按新Router配套训练。Host/调用图/RoleGraph增量提取成本缺少同APK计时，端到端成本保持UNVERIFIED。\n'''
    put_text('existing_litestage_asset_audit.md',report)
    put_json('preflight.json',dict(status='PASS',source=799,fold_sizes=[int((fold==i).sum()) for i in range(1,6)],benign=int((y==0).sum()),malware=int((y==1).sum()),future=1989,C43_all_source_predictions=True,future_C43_train_years=[2017],new_APK_extraction=False,old_L2_L3_cost_leakage='dex_meta[4]=invoke_n',legacy_lite_BA=old['source_lite_BA']))
    print('LFC_PREFLIGHT_PASS source=799 future=1989 frozen_C43=true',flush=True)
if __name__=='__main__':main()
