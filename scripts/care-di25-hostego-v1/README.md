# CARE-Droid frozen DI25 Host/Ego source ablation v1

This package changes exactly one methodological variable: the local Host
representation. M0/M2 use the frozen Host-only representation; M1/M3 add the
directed one-hop neighbourhood in the already audited sensitive-endpoint Host
graph. Global24, DI25, the graph-guided local attention, Anomaly/MalBen
evidence, FixKeep training, three seeds, five source folds and the logit-space
bound `2.25*tanh(raw_delta)` remain frozen.

The Ego encoder first embeds every Host with the same LocalEncoder. Incoming
and outgoing neighbour Host embeddings are averaged separately, projected by
one small `Linear(48,24)+ReLU`, and fused by a scalar sigmoid gate. Hosts with
no directed neighbour take an explicit exact `h_local=h_host` branch. Ego
never enters the Host scaler or either prototype definition.

`Direct` is not retrained. It is the sigmoid of the same trained correction
head's raw output, applied only to the exact same DI25 rows. This isolates the
decision integration from representation/training.

Run in two deliberate stages:

```bash
bash scripts/care-di25-hostego-v1/launch_tmux.sh m01
tail -F artifacts/care-di25-hostego-v1.m01.console.log

# Only after inspecting M0/M1:
bash scripts/care-di25-hostego-v1/launch_tmux.sh m23
tail -F artifacts/care-di25-hostego-v1.m23.console.log
```

No future-year file is accepted by this source runner.

## Final method freeze and validation

The final internal method name is
`CARE-Droid-DI25-HostEgo-Anomaly-Bounded`.  The final-validation scripts do
not introduce another method variant.  They protect the locked source files
by SHA-256, reuse the canonical DI25 route, and write only derived audits or
new predefined-seed runs.

Run the short gates first:

```bash
bash scripts/care-di25-hostego-v1/run_final_validation_preflight.sh
```

After inspecting `FINAL_VALIDATION_PREFLIGHT_PASS`, start the source seed job:

```bash
bash scripts/care-di25-hostego-v1/launch_seed_robustness_tmux.sh
tail -F artifacts/care-di25-hostego-v1/seed_robustness.console.log
```

Only after all R0--R4 checksums and the seed summary pass, start the separate
diagnostic-only 2018--2021 summarization:

```bash
bash scripts/care-di25-hostego-v1/launch_future_diagnostic_tmux.sh
tail -F artifacts/care-di25-hostego-v1/diagnostic_future.console.log
```

The future workflow freezes label-free probabilities before opening the
post-hoc label file.  These years were previously inspected, so its report is
explicitly diagnostic-only.  The 2022--2024 runner is fail-closed and cannot
run without audited assets, explicit provenance, and the confirmation flag.
