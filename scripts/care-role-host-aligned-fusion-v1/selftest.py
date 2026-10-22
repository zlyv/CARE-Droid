#!/usr/bin/env python3
"""Synthetic invariants for static Host-own-Role31 binding controls; writes nothing."""
import numpy as np
from rhaf_common import role_binding,augment_bags

raw=np.zeros((5,169),np.float32); raw[0,43]=2; raw[1,44]=1; raw[2,43:45]=1; raw[4,45]=3
bag=np.zeros((5,142),np.float32); ids=['a']
base=role_binding(raw)
assert base.shape==(5,31) and np.allclose(base.sum(1),[1,1,1,0,1])
for arm in ('F1_ALIGNED_BIPARTITE','F2_WITHIN_APK_BINDING_SHUFFLE','F3_APK_ROLE_MARGINAL_CONTROL'):
    out,audit=augment_bags([bag],[raw],ids,arm,900000)
    assert out[0].shape==(5,173) and audit['role_marginal_max_abs_error']<1e-6
print('ROLE_HOST_ALIGNED_FUSION_SELFTEST_PASS')
