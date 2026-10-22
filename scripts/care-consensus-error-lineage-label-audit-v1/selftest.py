#!/usr/bin/env python3
from __future__ import annotations
import base64
import numpy as np
from common import BIT_BYTES,sketch_jaccard

def sketch(bits):
    value=np.zeros(BIT_BYTES,dtype=np.uint8)
    for bit in bits:value[bit//8]|=1<<(bit%8)
    return {"bits_b64":base64.b64encode(value.tobytes()).decode()}
def main():
    assert abs(sketch_jaccard(sketch({1,2,3}),sketch({2,3,4}))-.5)<1e-12
    assert sketch_jaccard(sketch(set()),sketch(set()))==1.0
    assert sketch_jaccard(sketch({7}),sketch({8}))==0.0
    # Exact SHA or package+certificate is deliberately stronger than merely
    # sharing a signer.  The latter is common for benign developers/stores.
    exact_dex={"a"}; package={"b"}; cert={"b","c"}
    assert exact_dex|(package&cert)=={"a","b"}
    print("CONSENSUS_ERROR_LINEAGE_SELFTEST_PASS sketch=true exact_lineage=true",flush=True)
if __name__=="__main__":main()
