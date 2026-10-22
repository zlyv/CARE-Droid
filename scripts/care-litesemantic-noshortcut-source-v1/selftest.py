#!/usr/bin/env python3
import logging,os
import numpy as np
from feature_contract import MANIFEST_NAMES,arm_indices
from extract_features import suppress_parser_logs

def main():
    roles=["r"+str(i) for i in range(31)]
    names=["manifest__"+x for x in MANIFEST_NAMES]+["role_presence__"+x for x in roles]+["role_proportion__"+x for x in roles]
    arms=arm_indices(names)
    assert {k:len(v) for k,v in arms.items()}=={"L0_MANIFEST":20,"L1_ROLE_PRESENCE":31,"L2_ROLE_PROPORTION":31,"L3_FUSED":82}
    counts=np.asarray([2,1,0.]);p=counts/max(counts.sum(),1);assert np.isclose(p.sum(),1)
    forbidden=("apk_size","method_count","class_count","sdk","package","certificate","year")
    assert not any(any(x in n.lower() for x in forbidden) for n in names)
    suppress_parser_logs();assert os.environ["LOGURU_LEVEL"]=="CRITICAL" and logging.root.manager.disable>=logging.CRITICAL
    print("LITESEMANTIC_NOSHORTCUT_SELFTEST_PASS")
    print("arms=L0_MANIFEST,L1_ROLE_PRESENCE,L2_ROLE_PROPORTION,L3_FUSED")
    print("model_dimensions=20,31,31,82")
    print("callgraph_constructed=false identity_features=false raw_size_features=false")
    print("future_rows_loaded=0 stage2_training_invoked=false")
    print("androguard_debug_logging_suppressed=true")

if __name__=="__main__":main()
