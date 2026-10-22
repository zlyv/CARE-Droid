#!/usr/bin/env python3
"""LiteStage-V1 provenance/preflight. Reads frozen code, mapping, lineages and prediction indices; writes new audit only; no APK parsing or training."""
from __future__ import annotations
import json
from common import *

def main():
    lineage,source=source_contract();expert=frozen_expert_source()
    future=readcsv(FUTURE_LINEAGE)
    if len(future)!=1989 or len({r['sha256'] for r in future})!=1989:raise RuntimeError('future lineage')
    from yaml import safe_load
    mapping=safe_load(MAPPING.read_text(encoding='utf-8'))
    if len(mapping['roles'])!=31:raise RuntimeError('mapping')
    paths=[SOURCE_LINEAGE,FUTURE_LINEAGE,P31,FUTURE_P31,SOURCE_PRED,HDBC/'source2017_oof_predictions.csv',HDBC/'future2018_2021_predictions.csv',MAPPING,
      ROOT/'scripts/care-p31-classifier-family-v1/stage1_core.py',ROOT/'scripts/care-litelocal-p31-diagnostics-v1/extract_future_p31.py',ROOT/'scripts/care-scbe-v1/train.py',ROOT/'scripts/care-multiyear-development-v1/train_stage2.py']
    before=OUT/'asset_manifest_before.json'
    if before.exists():
        old=json.loads(before.read_text(encoding='utf-8'))
        if old!=[digest(p) for p in paths]:raise RuntimeError('historical asset SHA changed')
    else:writejson('asset_manifest_before.json',[digest(p) for p in paths])
    if set(lineage)!=set(expert):raise RuntimeError('C43 matched source SHA')
    entries=[('L1_MANIFEST','MANIFEST_ONLY','APK constructor parses AXML; permission/components/sdk','NO'),('L2_DEXMETA','DEX_METADATA','L1 + DEX parser sections, class/method IDs','NO'),('L3_ROLE_REF','DEX_REFERENCE_SCAN','L2 + MethodId table; referenced != invoked','NO'),('L4_ROLE_INVOKE','DEX_INVOKE_SCAN','L2 + method body invoke scan, Role31 mapping','NO'),('L5_P31_RF','DEX_INVOKE_SCAN','historic method-level Role31 prevalence from DEX invoke scan','NO'),('DIRECT_C43','HOST_GRAPH_REQUIRED','frozen Host/RoleGraph and three backgrounds','YES')]
    writecsv('feature_dependency_matrix.csv',[dict(feature=a,cost_class=b,actual_operation=c,call_graph_required=d) for a,b,c,d in entries])
    writetext('feature_provenance_audit.md',f'# LiteStage特征与正式协议\n\n2017=799（400良性、399恶意），固定五折160/160/160/160/159；C43 H0与Source SHA完全一致。使用正式Role31 YAML `{MAPPING}`，未重新选Role。P31旧抽取器 `scripts/care-litelocal-p31-diagnostics-v1/extract_future_p31.py` 在DEX方法体遍历invoke，计算敏感方法的Role prevalence；显式不创建Analysis/调用图。新L3仅扫描MethodId引用，不能等同调用发生。新L4重用同一Role规则和invoke语义。L1-L4程序不调用任何图构建器。\n\nP31-RF沿用历史内层校准的2017 OOF，58错；P31-SVM为97错。Direct-C43共享专家为SCBE M0等BCE在全体外层训练APK训练、Stage-I logit置零的直接判断；它的OOF Fold与新第一层匹配，且训练不依赖当前Router，因此可在同Fold共享专家。路由更换仍需明确标注为新系统，不可用SVM Fixed/Broken直接平移。\n')
    writetext('cost_baseline_report.md','# 成本基线\n\nP31已经只扫描DEX invoke、不构建完整调用图；因此LiteRoleInvoke31不能以“省去调用图”作为新增成本优势。L1只解析Manifest，L2解析DEX元信息，L3迭代MethodId表，L4遍历invoke。真实逐APK计时将在本轮提取后统计；旧P31的`audit_values/total_seconds`是历史抽取计时，环境/缓存状态未控制，不能直接作因果成本差。第二层调用图/Host构建成本若无同APK可复核计时，端到端节省标UNVERIFIED。\n')
    print('LITESTAGE_PREFLIGHT_PASS source=799 future=1989 p31_callgraph=false',flush=True)
if __name__=='__main__':main()
