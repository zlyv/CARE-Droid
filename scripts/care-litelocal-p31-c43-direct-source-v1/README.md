# P31-DI25 C43-RQ direct-classification source experiment

This strict-2017 experiment isolates the Stage-II decision mechanism. It keeps
the locked P31 Mean Committee, P31 MeanOpposition DI25, C43-RQ representation,
outer folds, three seeds, full outer-train training scope, backgrounds and
parameter count unchanged.

The control outputs:

`logit(P31) + 2.25*tanh(raw_head)`

The candidate outputs:

`raw_head`

and trains that head using ordinary binary cross entropy. P31 logit remains an
input feature but is not added to the final output. The direct expert replaces
P31 only for the 200 routed APKs.

Run:

```bash
unzip -o ../care-litelocal-p31-c43-direct-source-v1.zip -d scripts/
bash scripts/care-litelocal-p31-c43-direct-source-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31-c43-direct-source-v1.console.log
```

Result:

`artifacts/care-litelocal-p31-c43-direct-source-v1/results/REPORT.txt`
