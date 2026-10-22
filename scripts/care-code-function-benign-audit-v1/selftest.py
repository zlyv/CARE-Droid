#!/usr/bin/env python3
from common import CONTEXT_FAMILIES,DIRECT_SENSITIVE,FAMILIES
def main():
    assert set(CONTEXT_FAMILIES).isdisjoint(DIRECT_SENSITIVE)
    assert set(CONTEXT_FAMILIES)|set(DIRECT_SENSITIVE)==set(FAMILIES)
    assert "UI_INTERACTION" in CONTEXT_FAMILIES and "LOCATION_GEO" not in CONTEXT_FAMILIES
    print("CODE_FUNCTION_BENIGN_AUDIT_SELFTEST_PASS context_families="+str(len(CONTEXT_FAMILIES)))
if __name__=="__main__":main()
