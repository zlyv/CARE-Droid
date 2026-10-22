"""Mechanical protocol checks; no model training or APK access."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from diagnostic_common import *

def main():
    y=np.array([1,0,1,0]);p=np.array([.8,.2,.7,.1])
    assert roc_auc_score(y,p)==1.0
    assert OUTPUT.parent==(PROJECT/'outputs').resolve()
    assert len(set(TAGS))==3 and len(set(SEEDS))==3
    mc,v4,args,data=original()
    assert len(data['ids'])==2788 and len(set(data['ids'].tolist()))==2788
    assert np.all(np.diff(data['host_offsets'])>=1)
    for tag in TAGS:
        obj=checkpoint(BASE/'models/stage2'/tag/'900000.pkl',mc,data)
        ix={str(s):i for i,s in enumerate(data['ids'])}
        tr=np.array([ix[s] for s in obj['metadata']['train_sha']]);va=np.array([ix[s] for s in obj['metadata']['validation_sha']])
        assert max(data['years'][tr])<min(data['years'][va])
        assert not set(tr)&set(va)
    print('CLD_SELFTEST_PASS',flush=True)
if __name__=='__main__':main()
