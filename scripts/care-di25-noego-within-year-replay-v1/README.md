# CARE-Droid no-Ego within-year replay v1

Purpose: replay the current no-Ego CARE-Droid core under **independent per-year five-fold OOF** for 2018, 2019, 2020 and 2021.

Pipeline:

`Global24 -> within-year DI25 -> Host-only Anomaly + RoleGraph/RGCondAttn -> FixKeep bounded correction`

This is **same-year development/diagnostic evaluation**, not temporal generalization.

## Frozen inputs

- Router: `artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv`
- 1989 rows total: 2018=793, 2019=399, 2020=397, 2021=400
- DI25 counts: 2018=198, 2019=100, 2020=99, 2021=100
- Host archive: `artifacts/care-future-local-assets-v2/regions_2018_2021.npz`
- Consumed Host keys only: `host_values`, `host_offsets`, `real_host_count`
- Ego is not loaded.

The historical canonical `future_interz.npz` is not recreated at its old location. The recovered historical loader is instructed to rebuild a **replay-local** cache at:

`artifacts/care-di25-noego-within-year-replay-v1/shared/future_interz_rebuilt_for_replay.npz`

## Frozen model protocol

- representation: anomaly
- head: bounded
- seeds: 900000,900001,900002
- max_delta: 2.25
- max_epochs: 30
- lr: 8e-4
- weight_decay: 5e-4
- keep_weight: 2.0
- fix_weight: 1.0
- keep_consistency: 0.5

## Run

Upload/unzip under `scripts/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid

CUDA_VISIBLE_DEVICES=1 \
  bash scripts/care-di25-noego-within-year-replay-v1/launch_tmux.sh
```

Monitor:

```bash
tail -F artifacts/care-di25-noego-within-year-replay-v1.console.log
```

Outputs:

- `artifacts/care-di25-noego-within-year-replay-v1/oof_predictions.csv`
- `artifacts/care-di25-noego-within-year-replay-v1/metrics.csv`
- `artifacts/care-di25-noego-within-year-replay-v1/protocol.json`
- `artifacts/care-di25-noego-within-year-replay-v1/summary.md`

This replay does not use Need-Gate.
