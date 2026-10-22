#!/usr/bin/env python3
"""Correct the Phase-0 gate and document class-readiness without touching v1 evidence.

This CARE-Droid organizer reads the immutable v1 audit outputs only. It does not
extract APK features, download samples, train, or use future labels for selection.
"""
import collections
import csv
import hashlib
import json
import pathlib

ROOT = pathlib.Path('/home/boyuanjie/projects/tasa-droid')
PREVIOUS = ROOT / 'outputs/care_expanded2017_crossyear_v1'
OUTPUT = ROOT / 'outputs/care_expanded2017_crossyear_v2'
SOURCES = [PREVIOUS / n for n in ('dataset_audit.json', 'dataset_manifest.csv',
    'deduplication_audit.csv', 'dataset_overlap_matrix.csv', 'label_provenance.json',
    'baseline_integrity_check.json')]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, data):
    with (OUTPUT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')

def main():
    if OUTPUT.exists():
        raise RuntimeError('FAIL_CLOSED: v2 output already exists')
    if any(not p.is_file() for p in SOURCES):
        raise RuntimeError('FAIL_CLOSED: prior audit file missing')
    before = {str(p): sha(p) for p in SOURCES}
    audit = json.loads(SOURCES[0].read_text())
    if audit['downloaded_apk_count'] != 3989 or audit['old_2017_count'] != 799:
        raise RuntimeError('FAIL_CLOSED: audit cohort changed')
    with SOURCES[1].open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    counts = collections.Counter((int(r['year']), r['label']) for r in rows)
    overlaps = [r for r in rows if 'OLD_NEW_2017_SHA_OVERLAP' in r['exclusion_reasons']]
    if len(overlaps) != 6 or any(r['year'] != '2017' or r['label'] != '0' for r in overlaps):
        raise RuntimeError('FAIL_CLOSED: overlap correction assumptions did not reproduce')
    if sum(counts[y, '1'] for y in (2018, 2019, 2020)) != 6:
        raise RuntimeError('FAIL_CLOSED: test class composition changed')
    OUTPUT.mkdir(mode=0o700)
    with (OUTPUT / 'readiness_by_year.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['year', 'downloaded_benign', 'downloaded_malware', 'new_data_training_or_BA_readiness'])
        for y in range(2017, 2023):
            w.writerow([y, counts[y, '0'], counts[y, '1'], 'NOT_READY_INCOMPLETE_MALWARE_COHORT'])
    after = {str(p): sha(p) for p in SOURCES}
    changed = [p for p in before if before[p] != after[p]]
    old_integrity = json.loads(SOURCES[-1].read_text())
    dump('baseline_integrity_check.json', {
        'status': 'PASS' if not changed and old_integrity['status'] == 'PASS' else 'FAIL',
        'v1_audit_files_hashed': len(SOURCES), 'changed': changed,
        'v1_formal_input_integrity': old_integrity,
        'boundary': 'v2 only reinterprets completed audit; no APK or formal model writes'})
    dump('final_decision.json', {
        'status': 'BLOCKED_INCOMPLETE_MALWARE_DOWNLOAD', 'stopped_at': 'PHASE_0_DATA_AUDIT',
        'v1_gate_correction': 'Six old/new 2017 benign SHA overlaps are valid deduplication, not a hard failure.',
        'actual_blocker': 'Only two newly downloaded malware APKs in each 2017-2020 year; S2 5-fold calibration and robust annual BA comparison are not supportable.',
        'new_2017_by_label': {'benign': counts[2017, '0'], 'malware': counts[2017, '1']},
        'union_2017_total_sha': audit['union_2017_sha_count'],
        'novel_test_counts': audit['novel_test_by_year'],
        'model_training_started': False, 'full_feature_extraction_started': False,
        'next_action': 'Complete balanced selected-malware downloads under the already frozen manifest, then run a new independent audit version. Do not tune on future labels.'})
    (OUTPUT / 'final_report.md').write_text(
        '# CARE-Droid 新旧 2017 联合训练：Phase 0 决策更正\n\n'
        '**最终状态：BLOCKED_INCOMPLETE_MALWARE_DOWNLOAD。停在数据审计；未做特征全量提取、训练或测试。**\n\n'
        '已对 3989 个实际下载 APK 核验内容 SHA256，检查清单年份、标签、APK 大小，并计算 '
        'classes*.dex 精确指纹。原 2017 为 799 个；新 2017 为 1910 个，其中良性 1908、恶意 2。'
        '新旧 2017 有 6 个良性 SHA 重叠，正常去重后 UNION_2017 为 2703 个。'
        'v1 把该可预期重叠作为硬失败是不正确的，本版仅修正决策解释，不改写 v1 文件。\n\n'
        '新 2018 为 810/2（良/恶），2019 为 629/2，2020 为 630/2；'
        '2021、2022 各 1/1。新测试与明确列出的历史开发样本 SHA 重叠 4 个，'
        '故 NOVEL_TEST 数量分别为 812、629、630。40 个以上精确 DEX 重复组只是'
        '描述性检查，不能宣称已排除近重复代码。\n\n'
        '上游清单以 dex_date 指定年份，以 VirusTotal 检测数 0/≥4 指定良/恶，'
        '1–3 排除；这是 AV 共识代理标签，不是人工行为真值。当前下载明显偏向良性。'
        '只有 2 个新恶意训练样本无法执行要求的 S2 严格多折校准；各测试年只有 2 个恶意'
        '样本，BA 与恶意召回率极不稳定。继续 U0–U3 会把下载进度误当作泛化性能，故按合同停止。\n\n'
        '安全性：v2 仅读取 v1 审计文件；v1 对 7 个复用清单的前后哈希检查 PASS；'
        '正式模型、APK、缓存及历史结果未修改。需要先完成冻结清单中各年的恶意 APK 下载，'
        '然后用新版本重新审计类别、重复和标签冲突，审计通过后再做 smoke test。\n',
        encoding='utf-8')
    print('EXPANDED2017_FINAL_STATUS=BLOCKED_INCOMPLETE_MALWARE_DOWNLOAD')

if __name__ == '__main__':
    main()
