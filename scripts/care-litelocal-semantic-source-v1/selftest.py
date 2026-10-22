#!/usr/bin/env python3
import numpy as np
from local_semantic_core import *

def main():
    cls, name = target_from_instruction("v0, v1, Ljava/lang/Runtime;->exec(Ljava/lang/String;)Ljava/lang/Process;")
    assert cls == "Ljava/lang/Runtime;" and name == "exec"
    p, c, t, audit = summarize_methods([[(0,),(0,),(1,),(1,),(2,)], [(1,),(2,)], []])
    assert p.shape == (31,) and c.shape == (465,) and t.shape == (961,)
    assert np.isclose(p[0], .5) and np.isclose(p[1], 1) and audit["sensitive_method_count"] == 2
    assert np.isclose(t.reshape(31,31)[0,1], 1/3) and np.isclose(t.sum(), 1)
    print("LITELOCAL_SEMANTIC_SOURCE_SELFTEST_PASS")
    print("raw_blocks=role_method_prevalence31,within_method_cooccurrence465,directed_adjacent_transition961")
    print("model_arms=P31,COOC16,TRANS16,FUSED63")
    print("future_rows_loaded=0 callgraph_constructed=false analysis_object_constructed=false")
    print("raw_size_features_in_model=false identity_features=false stage2_training_invoked=false")

if __name__ == "__main__": main()

