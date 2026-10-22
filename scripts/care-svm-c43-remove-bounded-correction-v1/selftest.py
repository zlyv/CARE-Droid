#!/usr/bin/env python3
import numpy as np
from ablation_common import auc, mcnemar_exact, metrics


def main():
    assert auc([0,0,1,1],[.1,.2,.8,.9])==1.0
    result=metrics([0,1,0,1],[.1,.4,.8,.9],[.1,.6,.4,.9],[0,1,1,0]); assert (result["fixed"],result["broken"],result["net"])==(2,0,2)
    a,b,p=mcnemar_exact([1,1,0,1],[1,0,1,1]); assert (a,b,p)==(1,1,1.0)
    print("SVM_C43_REMOVE_BOUNDED_SELFTEST_PASS",flush=True)


if __name__=="__main__":main()
