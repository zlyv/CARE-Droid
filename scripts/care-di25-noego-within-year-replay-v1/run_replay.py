#!/usr/bin/env python3
"""Launch the recovered historical within-year Anomaly+Bounded arm in an isolated replay output."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path('/home/boyuanjie/projects/tasa-droid')
RUNNER = PROJECT / 'scripts/care-route-stage2-within-year-matrix-v1/run_within_year_arm.py'
ROUTER = PROJECT / 'artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv'
RESOLVED = PROJECT / 'artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json'
ROOT = PROJECT / 'artifacts/care-di25-noego-within-year-replay-v1'
EXPERT = ROOT / 'expert_anomaly_bounded'
INTER = ROOT / 'shared/future_interz_rebuilt_for_replay.npz'


def main() -> None:
    EXPERT.mkdir(parents=True, exist_ok=True)
    INTER.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUNNER),
        '--representation', 'anomaly',
        '--head', 'bounded',
        '--router-inputs', str(ROUTER),
        '--resolved', str(RESOLVED),
        '--future-inter-cache', str(INTER),
        '--rebuild-future-inter-cache',
        '--seeds', '900000,900001,900002',
        '--max-epochs', '30',
        '--learning-rate', '8e-4',
        '--weight-decay', '5e-4',
        '--max-delta', '2.25',
        '--keep-weight', '2.0',
        '--fix-weight', '1.0',
        '--keep-consistency', '0.5',
        '--device', 'auto',
        '--output', str(EXPERT),
    ]
    print('NOEGO_WITHIN_YEAR_REPLAY_START', flush=True)
    print('expert_output =', EXPERT, flush=True)
    print('replay_inter_cache =', INTER, flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)
    pred = EXPERT / 'within_year_expert_predictions.csv'
    if not pred.is_file():
        raise RuntimeError('BLOCKED_BY_CONTRACT: expected expert prediction file missing')
    print('NOEGO_WITHIN_YEAR_EXPERT_DONE', flush=True)


if __name__ == '__main__':
    main()
