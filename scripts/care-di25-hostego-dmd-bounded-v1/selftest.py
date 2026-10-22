#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import dmd_training as d

y=np.asarray([0,1,0,1]);p=np.asarray([.9,.1,.1,.9])
a,mask,m,wrong=d.targets(y,p,3,torch.device("cpu"))
assert a.tolist()==[0,2,1,1] and mask.tolist()==[True]*4 and wrong.tolist()==[True,True,False,False]
a2,mask2,m2,wrong2=d.targets(y,p,2,torch.device("cpu"))
assert a2.tolist()==[0,1,0,1] and mask2.tolist()==wrong2.tolist()
assert not bool(mask2[~wrong2].any()) and bool(torch.all((m>=0)&(m<=2.25)))
q=torch.softmax(torch.tensor([4.,1.,-2.]),0);direction=q[2]-q[0];magnitude=2.25*torch.sigmoid(torch.tensor(2.))
assert abs(float(direction*magnitude))<=2.25+1e-6 and abs(float(q.sum())-1)<1e-7
print("DMD_BOUNDED_SELFTEST_PASS")

