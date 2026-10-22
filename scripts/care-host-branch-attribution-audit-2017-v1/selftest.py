#!/usr/bin/env python3
"""Unit tests for JSON safety, readout slices, contribution identities and matching helpers."""
from __future__ import annotations
import importlib.util,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit.py'));a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
assert a.C_SL==slice(57,81) and a.A_SL==slice(81,105) and a.SUMMARY_SL==slice(105,113)
assert a.jv(np.bool_(True)) is True and a.jv(np.int64(2))==2 and a.jv(np.float32(1.5))==1.5
l11,l10,l01,l00=4.,3.,2.,1.;pc=.5*((l10-l00)+(l11-l01));pa=.5*((l01-l00)+(l11-l10));inter=l11-l10-l01+l00
assert abs((l00+pc+pa)-l11)<1e-12 and inter==0
assert abs(a.cosine(np.ones(3),np.ones(3))-1)<1e-12
frame=pd.DataFrame([{'comparison':'x'}]);assert not frame.empty and frame.groupby('comparison').size().to_dict()=={'x':1}
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/'x.json';a.wjson(p,{'b':np.bool_(False),'i':np.int64(3),'x':np.array([1,2])});assert json.loads(p.read_text())=={'b':False,'i':3,'x':[1,2]}
print('UNIT_TEST_PASS',flush=True)
