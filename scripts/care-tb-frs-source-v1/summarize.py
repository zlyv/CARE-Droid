#!/usr/bin/env python3
from common import *
def main():
    d=read_json(OUTPUT/"final_decision.json"); integ=read_json(OUTPUT/"baseline_integrity_check.json"); r=d["results"]
    lines=["# CARE-Droid TB-FRS source-only information probe","",f"Final status: **{d['status']}**.","",
      "This experiment preserves the formal 32 shared and 24 benign prototypes, Role compatibility, SVM MeanOpposition Top25 route and Direct-C43 reference. It tests representation compression only; no future or external data are loaded.","",
      "## Directional information","","| Arm | FN-vs-TN AUC | TP-vs-FP AUC | Host-count Spearman |","|---|---:|---:|---:|"]
    for arm in ("B0_DIRECT_C43_REFERENCE",)+ARMS:
        x=r[arm]; lines.append(f"| {arm} | {x['FN_vs_TN_AUC']:.6f} | {x['TP_vs_FP_AUC']:.6f} | {x['host_count_spearman']} |")
    lines += ["","## Frozen decision", "",f"- Integration candidate: `{d['candidate_for_separate_integration']}`.",f"- End-to-end run in this package: **{d['end_to_end_run']}**.","- A future run is forbidden until a separately specified source integration passes.",f"- Formal asset integrity: **{integ['status']}**.",""]
    ensure(OUTPUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("TBFRS_REPORT_DONE status="+d["status"],flush=True)
if __name__=="__main__":main()
