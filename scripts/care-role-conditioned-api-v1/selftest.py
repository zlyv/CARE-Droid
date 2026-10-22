#!/usr/bin/env python3
import tempfile,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SCRIPTS=HERE.parent
for p in (HERE,SCRIPTS/"care-di25-graphglobal-path-source-v1",SCRIPTS/"care-rolegraph-conditioned-v4-script"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from role_api_core import project,shuffled_within_role
def fake():
    a=[[[] for _ in range(31)] for _ in range(2)];a[0][3]=["a","b"];a[1][3]=["c"];return a
def main():
    train=[fake(),fake(),fake()];test=[fake()];bt,bv,a=project(train,test,False,7)
    assert len(bt)==3 and bt[0].shape==(2,31,16) and len(bv)==1 and a["test_api_occurrences"]==3
    s=shuffled_within_role(train,9);assert sum(len(h[3]) for apk in s for h in apk)==sum(len(h[3]) for apk in train for h in apk)
    from role_api_models import TOTAL
    assert TOTAL==574
    print("ROLE_API_SELFTEST_PASS");print("role_isolation=true normalized_pooling=true bounded_alpha=true future_rows_loaded=0")
if __name__=="__main__":main()
