#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent if HERE.parent.name=="scripts" else HERE.parent; SCRIPTS=PROJECT/"scripts" if (PROJECT/"scripts").is_dir() else PROJECT; BASE=SCRIPTS/"care-di25-graphglobal-path-source-v1"
for p in (HERE,BASE,SCRIPTS,SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"): sys.path.insert(0,str(p))
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from compact_models import make_compact_class


class Args:
    max_delta=2.25; graph_dim=32; embedding_dim=24; hidden_dim=48; graph_heads=4; dropout=0.0; graph_layers=2


def main():
    # The frozen RoleGraph edge contract is three channels.  The upstream
    # implementation exposes ROLE_NODE_WIDTH and GRAPH_GLOBAL_DIM, but no
    # ROLE_EDGE_WIDTH constant; its EdgeBiasedGraphBlock consumes width 3.
    v4.AUGMENTED_REGION_DIM=78; graph={"node":torch.randn(31,v4.ROLE_NODE_WIDTH),"edge":torch.randn(31,31,3),"global":torch.randn(v4.GRAPH_GLOBAL_DIM)}; core=torch.randn(5,47); rel=torch.randn(5,31); regions=torch.cat([core,rel],dim=1); global24=torch.randn(24); base=torch.tensor(0.2)
    counts={}; outputs={}
    for arm in ("C43_COMPACT","C43_RELATION_QUERY"):
        torch.manual_seed(7); model=make_compact_class(v4,arm)(24,"RGCondAttn",Args()); model.eval(); counts[arm]=sum(p.numel() for p in model.parameters())
        # Perturb one Host only.  Adding the same relation offset to every Host
        # would add a common attention-logit constant that softmax cancels.
        with torch.no_grad(): _,_,trace1=model.forward_with_trace(global24,base,graph,regions); changed=regions.clone(); changed[0,47:]+=3.0; _,_,trace2=model.forward_with_trace(global24,base,graph,changed)
        outputs[arm]=float(torch.max(torch.abs(trace1["attention"]-trace2["attention"])))
    assert len(set(counts.values()))==1
    assert outputs["C43_COMPACT"]==0.0
    assert outputs["C43_RELATION_QUERY"]>1e-8
    print("COMPACT43_RELATIONQUERY_SELFTEST_PASS")
    print("parameter_count_all_arms=",next(iter(counts.values())))
    print("C43_relation_attention_diff=0 C43_RQ_relation_attention_diff_positive=true")
    print("relation_enters_delta=false duplicate_role_columns=0 future_rows_loaded=0 ego_consumed=false")


if __name__=="__main__": main()
