"""Descriptive frozen-expert and background-response analysis; no fitting or routing decisions."""
from __future__ import annotations
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from common import *

def main():
    rows=read_csv(OUTPUT/'expert_capability.csv');b=read_csv(OUTPUT/'background_geometry.csv')
    out=[];responses=[]
    for tag in ['T2_C1','T3_C1','T4_C1']:
        r=[x for x in rows if x['tag']==tag];routed=[x for x in r if int(x['routed'])];stage_errors=[x for x in r if not int(x['stage1_correct'])];correctable=[x for x in stage_errors if int(x['expert_correct'])]
        counts=Counter(x['category'] for x in routed)
        out.append(dict(tag=tag,validation_APKs=len(r),routed=len(routed),StageI_errors=len(stage_errors),routed_StageI_errors=sum(not int(x['stage1_correct']) for x in routed),routed_correctable_errors=counts['FIXED_FN']+counts['FIXED_FP'],routed_unfixed_errors=counts['UNFIXED_FN']+counts['UNFIXED_FP'],routed_broken=counts['BROKEN_TN']+counts['BROKEN_TP'],unrouted_correctable_errors=sum(int(x['router_missed_correctable']) for x in r),all_validation_C43_correctable=len(correctable),label_aware_oracle_fixed_at_same_budget=min(len(correctable),len(routed)),oracle_is_non_deployable=True))
        br=[x for x in b if x['tag']==tag and int(x['seed'])==900000];ids={x['sha256']:x for x in r};lab=np.asarray([int(x['label']) for x in br]);host=np.asarray([int(x['host_n']) for x in br]);rr=np.asarray([int(ids[x['sha256']]['routed']) for x in br])
        for col in ['external','internal','commonness','residual']:
            val=np.asarray([float(x[col]) for x in br]);rho=spearmanr(val,host).statistic
            for scope,mask in [('all',np.ones(len(br),bool)),('routed',rr==1)]:
                responses.append(dict(tag=tag,scope=scope,response=col,n=int(mask.sum()),benign_mean=float(val[mask&(lab==0)].mean()),malware_mean=float(val[mask&(lab==1)].mean()),descriptive_malware_AUC=float(roc_auc_score(lab[mask],val[mask])),size_spearman=float(rho) if np.isfinite(rho) else None,training_on_validation_labels=False))
    write_csv(OUTPUT/'expert_bottleneck_diagnosis.csv',out);write_csv(OUTPUT/'background_response_diagnosis.csv',responses)
    print('ETBC_FROZEN_DIAGNOSTICS_DONE',out,flush=True)

if __name__=='__main__':main()
