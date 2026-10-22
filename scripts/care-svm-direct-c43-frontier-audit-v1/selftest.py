#!/usr/bin/env python3
from run_audit import contiguous_regions
def main():
    rows=[{"threshold":.1,"ok":1,"fixed":2,"broken":1,"net":1},{"threshold":.11,"ok":1,"fixed":3,"broken":1,"net":2},{"threshold":.12,"ok":1,"fixed":3,"broken":0,"net":3},{"threshold":.14,"ok":1,"fixed":4,"broken":2,"net":2}]
    r=contiguous_regions(rows,"ok");assert len(r)==2 and r[0]["grid_points"]==3 and abs(r[0]["width"]-.02)<1e-12
    print("DIRECT_C43_FRONTIER_SELFTEST_PASS unified_threshold=true contiguous_region=true",flush=True)
if __name__=="__main__":main()
