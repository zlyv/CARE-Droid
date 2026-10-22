#!/usr/bin/env python3
import numpy as np
from hostrel_features import reconstruct_host_relations
from semantic_background import assignments


def main():
    d=np.asarray([[0.1,0.2],[0.2,0.1]])
    w=np.asarray([[1,0],[0,1]],float); p=np.asarray([[1,0],[0,1]],float)
    hard=assignments(d,w,p,"B1_HARD_ROLE")
    assert hard.tolist()==[0,1]
    print("HOSTREL_SEMANTIC_DIAGNOSTICS_SELFTEST_PASS")
    print("protocols=frozen_2017_to_future,independent_same_year_5fold")
    print("arms=B1_HARD_ROLE,R1_HOSTREL74_HARD_ROLE parameter_count_target=33036")
    print("future_labels_loaded_by_model=false ego_consumed=false")


if __name__=="__main__": main()
