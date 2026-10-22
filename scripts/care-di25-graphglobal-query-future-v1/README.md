# CARE-Droid frozen GraphGlobal query-only future diagnostic v1

This package freezes the human-approved `Q_QUERY_ONLY` source candidate and runs one
diagnostic 2017-to-2018--2021 evaluation. `N0_NO_GRAPHGLOBAL` is run concurrently as
a mechanism control. It is not a future-year model-selection candidate.

Both arms use frozen DI25, Host-only anomaly evidence, 113-D bounded correction,
`max_delta=2.25`, and seeds `900000,900001,900002`. Q permits GraphGlobal to affect
Host attention only; it has no direct GraphGlobal-to-delta path. No Ego tensor is read.

Future labels are absent from training, early stopping, routing, and prediction. The
summarizer opens labels only after both 1989-row label-free prediction files exist.
The future years have already been inspected and remain diagnostic-only.

`future_interz.npz` is not required in advance. If absent, the tmux job builds it once
from the existing label-sealed future compact rows and endpoint cache, validates its
1989-row/465-D/961-D contract, and only then launches the two GPU arms. Both arms reuse
that cache read-only, preventing a parallel-writer race.

Upload the ZIP to `/home/boyuanjie/projects/`, then run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-graphglobal-query-future-v1.zip -d scripts/
bash scripts/care-di25-graphglobal-query-future-v1/launch_tmux.sh
tail -F artifacts/care-di25-graphglobal-query-future-v1.console.log
```

The launcher uses GPU 0 for N0 and GPU 1 for Q. It is resumable by exact 1989-row
prediction files. The final report is:

`artifacts/care-di25-graphglobal-query-future-v1/results/REPORT.txt`
