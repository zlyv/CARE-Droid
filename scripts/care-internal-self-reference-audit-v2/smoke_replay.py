#!/usr/bin/env python3
"""Read-only exact replay smoke test for the first frozen S1 sample.

This script opens no APK and writes no files. It verifies that the v2
Internal reconstruction matches the formal frozen bag before the full audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-internal-self-reference-audit-v2'
sys.path[:0] = [str(HERE), str(ROOT / 'scripts/care-background-functional-attribution-v1')]

from bfa_core import formal_split, load_data_contract  # noqa: E402
from run_audit import hard_assignment, self_exclusion  # noqa: E402


def main() -> None:
    data, _ = load_data_contract()
    pack = formal_split(data, 'S1', 900000, device='cpu', train_data=False)
    maximum = 0.0
    checked = min(5, len(pack['bags_val']))
    for bag, raw in zip(pack['bags_val'][:checked], pack['raw_val'][:checked]):
        x, assignment = hard_assignment(pack, bag, raw)
        effect = self_exclusion(
            x,
            assignment,
            pack['args'].internal_background_coverage,
            repeats=1,
            random_seed=0,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(effect['original_internal'] - bag[:, 108]))),
        )
    if maximum > 2e-5:
        raise RuntimeError(f'V2_REPLAY_SMOKE_FAIL max_abs={maximum}')
    print(f'V2_REPLAY_SMOKE_PASS samples={checked} max_abs={maximum:.12g}')


if __name__ == '__main__':
    main()
