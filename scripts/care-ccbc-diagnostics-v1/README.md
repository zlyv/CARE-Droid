# CARE-Droid CCBC frozen-future and same-year diagnostics v1

This package is a post-selection diagnostic for the frozen CCBC source protocol hash
`1940d7d426777fb8a8450c38dbc6cf2958067e8bb69d0b4b214c65986accf5be`.

- Frozen: train/freeze on 2017, diagnose 2018--2021.
- Same-year: independent fixed five-fold OOF inside each year; this is not temporal generalization.
- Future results cannot modify the source conclusion, features, thresholds, router budget, or model.
- Formal artifacts are hash-audited before/after and never intentionally written.
- CCBC source remains `PARTIALLY_SUPPORTED`; source A3 net=-5 versus formal A0 net=+4.

The run requires the already completed `outputs/care-ccbc-source-v3` and
`artifacts/care-droid-formal-diagnostics-v1` assets. It extracts future Manifest contracts,
recovers exact component-entry reachability using the existing endpoint graph, replays frozen
A0 attention without overwriting formal outputs, and evaluates the same REAL/null arms.

Run with `launch_tmux.sh`. The pipeline resumes per-APK contract shards and completed caches.
