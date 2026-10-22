#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
from common import *


def main():
    before=json.loads((OUTPUT/"baseline_asset_manifest_before.json").read_text(encoding="utf-8"));after=[];changed=[]
    for old in before:
        new=file_record(Path(old["absolute_path"]),old["purpose"]);after.append(new)
        if old["sha256"]!=new["sha256"] or old["size_bytes"]!=new["size_bytes"]:changed.append({"path":old["absolute_path"]})
    dump(OUTPUT/"baseline_asset_manifest_after.json",after);dump(OUTPUT/"baseline_integrity_check.json",{"status":"PASS" if not changed else "FAIL","changed":changed})
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen input changed")
    rows=read_csv(OUTPUT/"diagnostic_results.csv");comparison=read_csv(OUTPUT/"diagnostic_comparison.csv")
    lines=["# SVM consensus selective-correction future diagnostics","",
        "The 2-of-3 and 3-of-3 rules were frozen before reading future results. The 2017 source decision remains NOT SUPPORTED.",
        "Frozen 2017-to-future and independent same-year OOF are reported separately.",""]
    for protocol,title in (("frozen_2017_to_future","Frozen 2017 → future"),("independent_same_year_fixed_5fold","Independent same-year OOF")):
        lines += [f"## {title}","","| Year | Arm | BA | fixed | broken | net |","|---|---|---:|---:|---:|---:|"]
        for scope in [str(y) for y in YEARS]+["pooled"]:
            for arm in ARMS:
                row=next(r for r in rows if r["protocol"]==protocol and r["scope"]==scope and r["arm"]==arm)
                lines.append(f"| {scope} | {arm} | {float(row['BA']):.6f} | {row['fixed']} | {row['broken']} | {row['net']} |")
        lines.append("")
    lines += ["## Decision","","**DIAGNOSTIC_COMPLETE_SOURCE_DECISION_UNCHANGED**","",
        "Future diagnostics do not authorize changing the source-selected rule. Formal inputs remained unchanged."]
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    dump(OUTPUT/"final_summary.json",{"decision":json.loads((OUTPUT/"diagnostic_decision.json").read_text()),
        "comparison":comparison,"formal_integrity":"PASS"})
    print("CARE_SVM_CONSENSUS_DIAGNOSTICS_PIPELINE_DONE")
    print("report="+str(OUTPUT/"report.md"))


if __name__=="__main__":main()
