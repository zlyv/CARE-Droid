#!/usr/bin/env python3
import numpy as np
from feature_contract import GROUPS,STAGE1_VARIANTS,ROUTER_VARIANTS,indices


def main():
    names=[name for values in GROUPS.values() for name in values]
    assert len(names)==24 and len(set(names))==24
    expected={"FULL24":24,"DROP_MANIFEST":14,"DROP_CODE_CALL":21,"DROP_SENSITIVE":19,"DROP_GRAPH":18,
              "ONLY_MANIFEST":10,"ONLY_CODE_CALL":3,"ONLY_SENSITIVE":5,"ONLY_GRAPH":6}
    for variant,width in expected.items():assert len(indices(names,variant))==width
    base=np.asarray([.9,.1]);member=np.asarray([[.8,.7,.6,.9],[.2,.3,.4,.1]])
    decision=(base>=.5).astype(float);score=np.where(decision[:,None]==1,1-member,member).mean(1)
    assert np.allclose(score,[.25,.25])
    assert ROUTER_VARIANTS==STAGE1_VARIANTS[:5]
    print("GLOBAL24_UTILITY_COST_SELFTEST_PASS")
    print("stage1_variants="+",".join(STAGE1_VARIANTS))
    print("router_variants="+",".join(ROUTER_VARIANTS))
    print("source_only=true future_rows_loaded=0 stage2_training_invoked=false")


if __name__=="__main__":main()
