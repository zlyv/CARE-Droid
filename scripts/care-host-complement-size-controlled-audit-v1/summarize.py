#!/usr/bin/env python3
from audit_common import *

CORE=("A","B","C","ABC")

def main():
    metrics_rows=read_csv(ROOT/"reports/metrics.csv");m={r["variant"]:r for r in metrics_rows if r["year"]=="pooled"}
    boot={(r["baseline"],r["candidate"]):r for r in read_csv(ROOT/"reports/paired_bootstrap.csv")}
    cond={r["variant"]:r for r in read_csv(ROOT/"reports/conditional_separation.csv")}
    corr=read_csv(ROOT/"reports/feature_size_correlations.csv")
    decisions={};lines=["CARE-Droid Host Complement Size-Controlled Audit v1","","Strict same-year five-fold OOF diagnostic; no CARE-Droid training/replay and no APK reanalysis.",""]
    for name in CORE:
        inc=boot[("SizeOnly",f"{name}_PlusSize")];res=boot[("Random",f"{name}_Residualized")];c=cond[f"{name}_Residualized"]
        incremental=float(inc["delta_auc_ci_low"])>0 and float(inc["delta_pr_auc_ci_low"])>0
        residual_general=float(res["delta_auc_ci_low"])>0 and float(res["delta_pr_auc_ci_low"])>0
        residual_conditional=float(c["ci_low"])>0
        count=sum((incremental,residual_general,residual_conditional))
        status="SIZE_INDEPENDENT_SUPPORTED" if count==3 else ("PARTIAL_SIZE_INDEPENDENT_SIGNAL" if count>=2 else "SIZE_CONFOUNDED_OR_UNRESOLVED")
        decisions[name]={"status":status,"incremental_beyond_size":incremental,"residual_general_signal":residual_general,"residual_conditional_signal":residual_conditional}
        lines.append(f"{name}: Original AUC={float(m[name+'_Original']['ROC_AUC']):.6f}; Size+Candidate AUC={float(m[name+'_PlusSize']['ROC_AUC']):.6f}; Residualized AUC={float(m[name+'_Residualized']['ROC_AUC']):.6f}")
        lines.append(f"  incremental ΔAUC={float(inc['delta_auc']):.6f} CI=[{float(inc['delta_auc_ci_low']):.6f},{float(inc['delta_auc_ci_high']):.6f}]; ΔPR={float(inc['delta_pr_auc']):.6f} CI=[{float(inc['delta_pr_auc_ci_low']):.6f},{float(inc['delta_pr_auc_ci_high']):.6f}]")
        lines.append(f"  residual CS={float(c['conditional_separation']):.6f} CI=[{float(c['ci_low']):.6f},{float(c['ci_high']):.6f}]; decision={status}")
    top={}
    for name in ("A","B","C"):
        rows=[r for r in corr if r["candidate"]==name];top[name]=max(rows,key=lambda r:float(r["abs_rho"]))
        lines.append(f"{name} strongest size association: {top[name]['feature']} vs {top[name]['size_variable']}, rho={float(top[name]['rho']):.6f}")
    lines += ["","CORRECTED PRIOR DECISION:","PRIMARY_HOST_COMPLEMENT_CANDIDATE=NO_UNIQUE_WINNER",
              "Reason: A did not outperform B/ABC on general OOF metrics, and the prior automatic winner rule did not require CI-supported superiority over all alternatives.",
              "This audit diagnoses association after size control; it does not authorize a neural module or CARE-Droid prediction change."]
    report="\n".join(lines)+"\n";write_text(ROOT/"reports/SIZE_CONTROLLED_AUDIT_REPORT.txt",report)
    write_json(ROOT/"reports/decision.json",{"candidate_decisions":decisions,"corrected_primary_candidate":"NO_UNIQUE_WINNER","method_change_authorized":False})
    manifest=load_json(ROOT/"preflight/source_manifest.json");after={p:sha256(Path(p)) for p in manifest["input_sha256"]};changed=[p for p,h in manifest["input_sha256"].items() if after[p]!=h]
    write_text(ROOT/"audit/original_sha256_after.txt","".join(f"{h}  {p}\n" for p,h in after.items()));write_json(ROOT/"audit/no_original_modification.json",{"passed":not changed,"changed":changed})
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: original artifacts changed")
    print("HOST_COMPLEMENT_SIZE_CONTROLLED_AUDIT_DONE");print(report,end="");print("All original dependency SHA256 values unchanged.")
if __name__=="__main__":main()

