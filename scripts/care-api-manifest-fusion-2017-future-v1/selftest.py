#!/usr/bin/env python3
"""Deterministic unit tests for CARE API/Manifest fusion contracts and model replay."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from common import API_DIM, build_vocab, metric, permute_rows, shuffle_binding, vectorize_api
from model import make_model


def main():
    import torch
    records={
      'a':{'hosts':[{'events':[{'endpoint':'e1','family':'f1','role_key':'1','count':2},{'endpoint':'e2','family':'f2','role_key':'1','count':1}]}]},
      'b':{'hosts':[{'events':[{'endpoint':'e1','family':'f1','role_key':'1','count':1},{'endpoint':'e3','family':'f3','role_key':'2','count':3}]}]},
      'c':{'hosts':[{'events':[{'endpoint':'e2','family':'f2','role_key':'1','count':1}]}]},
    }
    v=build_vocab(records,['a','b','c'],'endpoint',min_df=1);assert v['dimension']==API_DIM and v['index']['e1']>0
    m,movable=shuffle_binding(records,['a','b','c'],v,7);assert m['e1']!=m['e2'] and {'e1','e2'}.issubset(movable) and m['e3']=='e3'
    x,a=vectorize_api(records,['a','b','c'],v,'endpoint');xs,b=vectorize_api(records,['a','b','c'],v,'endpoint',m);assert x.shape==xs.shape==(3,API_DIM) and np.allclose(x.sum(1),xs.sum(1))
    ids=np.array(['a','b','c']);z=np.arange(405,dtype=np.float32).reshape(3,135);p=permute_rows(z,ids,3);assert p.shape==z.shape and sorted(map(tuple,p))==sorted(map(tuple,z)) and not np.array_equal(p,z)
    model=make_model(torch);model.eval();base=torch.tensor([-.2,.3]);code=torch.randn(2,113);api=torch.randn(2,512);man=torch.randn(2,135)
    with torch.no_grad():out=model(base,code,api,man)
    assert torch.equal(out,base)
    assert metric([0,1],[.1,.9])['BA']==1.
    print('UNIT_TEST_PASS')
if __name__=='__main__':main()

