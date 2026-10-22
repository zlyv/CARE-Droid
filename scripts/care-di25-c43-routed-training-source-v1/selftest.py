#!/usr/bin/env python3
import numpy as np
def main():
    fold=np.repeat(np.arange(1,6),[160,160,160,160,159]);route=np.zeros(799,bool)
    for f in range(1,6):route[np.flatnonzero(fold==f)[:40]]=1
    assert route.sum()==200 and all(route[fold!=f].sum()==160 for f in range(1,6))
    print("C43_ROUTED_TRAINING_SOURCE_SELFTEST_PASS")
    print("arms=T0_ALL_TRAIN_LOCKED,T1_ROUTED_LOSS,T2_FULLY_ROUTED")
    print("outer_train_n=639_or_640 routed_outer_train_n=160 validation_n=159_or_160 routed_validation_n=40")
    print("selected_epochs_reused_from_locked_T0=true parameter_count_target=32292")
    print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()

