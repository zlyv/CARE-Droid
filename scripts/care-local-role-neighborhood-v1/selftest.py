#!/usr/bin/env python3
import numpy as np
from neighborhood import apk_scores,role_scores
y=np.array([0,0,1,1,0,1]);train=np.arange(4);test=np.arange(4,6)
apk=np.array([[1,0],[.9,.1],[0,1],[.1,.9],[.95,.05],[.05,.95]],float);apk=apk/np.linalg.norm(apk,axis=1,keepdims=True)
s,_,_=apk_scores(apk,y,train,test,k=2)
assert s[4]<0 and s[5]>0
s_excluded,_,_=apk_scores(apk,y,train,test,k=2,exclude_ge=.8)
assert np.isnan(s_excluded[4]) and np.isnan(s_excluded[5])
role=np.zeros((6,31,43));mass=np.zeros((6,31));role[:,0,:2]=apk;mass[:,0]=1
s,_,_,sup=role_scores(role,mass,y,train,test,k=2,min_support=2)
assert s[4]<0 and s[5]>0 and sup[0][3]
s,_,_,sup=role_scores(role,mass,y,train,test,k=2,min_support=3)
assert np.isnan(s[4]) and not sup[0][3]
print("LOCAL_ROLE_NEIGHBOR_SELFTEST_PASS")

