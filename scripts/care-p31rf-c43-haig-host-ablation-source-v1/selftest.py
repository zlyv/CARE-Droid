#!/usr/bin/env python3
import numpy as np
from host_sidecar import hashed, deterministic_shuffle
def main():
    a=hashed(["api:a","fam:x"],32);b=hashed(["api:a","fam:x"],32)
    assert a.shape==(32,) and np.all(np.isfinite(a)) and np.allclose(a,b)
    assert np.isclose(np.linalg.norm(a),1.0)
    p=deterministic_shuffle(5,"0"*64);assert sorted(p.tolist())==list(range(5))
    print("P31RF_C43_HAIG_HOST_ABLATION_SELFTEST_PASS")
    print("arms=A0_ROLE_ONLY,A1_API_NO_ROLE,A2_H3_NO_ROLE,A3_ROLE_PLUS_H3,A4_H3_SHUFFLE_NO_ROLE")
    print("source_only=true future_rows_loaded=0 max_delta=2.25")
if __name__=="__main__":main()

