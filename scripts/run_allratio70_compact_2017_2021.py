#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_frozen_anchor_stable70_count10_150_2017_2021 as stable
import run_stable70_vs_allratio70_2017 as allratio


# ============================================================
# 唯一修改：
# 当前 Stable70 -> 全比例70维 Compact
# ============================================================

stable.base.compact_matrix_named = (
    allratio.compact_matrix_allratio70
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/allratio70-compact-2017-2021",
        ])

    print(
        "============================================================"
    )
    print(
        "All-ratio70 Compact: train 2017 -> test 2018-2021"
    )
    print(
        "Final Compact = 24 global + 46 semantic ratios"
    )
    print(
        "Reported Compact result must use base_probability only"
    )
    print(
        "============================================================"
    )

    stable.base.main()
