# CARE-Droid Host-only correction-input ablation v1

This experiment answers whether each direct input block of the 113-D bounded
correction head adds information after removing Host/Ego fusion.

## Locked protocol

- strict 2017 five-fold OOF only;
- frozen DI25 route, 200/799 APKs;
- Host-only Anomaly representation;
- no Ego cache, projection, or gate in the model;
- fixed logit bound `2.25*tanh(raw_delta)`;
- seeds `0,1,2`;
- identical 113-wide correction head and identical parameter count in every arm;
- a removed block is multiplied by a fixed zero mask during both training and inference;
- no 2018--2021 row is loaded.

Before interpreting any ablation, the `FULL` arm must reproduce the locked M0
Host-only metrics and the actual 33,388-parameter count. Failure is reported as
`BLOCKED_BY_REPRODUCTION`; the ablations are not summarized.

The six primary leave-one-block-out arms remove Global24, Stage-I logit, role
graph token, critical Host, attended Host, or the eight scalar summaries. Two
secondary controls retain only Stage-I context or only structural evidence.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-host-input-replay-v1.zip -d scripts/
bash scripts/care-di25-host-input-replay-v1/launch_tmux.sh
tail -F artifacts/care-di25-host-input-replay-v1.console.log
```

The launcher is arm-resumable. Completed `source_predictions.csv` files are
not retrained. Final outputs are under
`artifacts/care-di25-host-input-replay-v1/results/`.

Do not use diagnostic/future years to select blocks. A bootstrap interval that
crosses zero is inconclusive and does not prove that a block is useless.
