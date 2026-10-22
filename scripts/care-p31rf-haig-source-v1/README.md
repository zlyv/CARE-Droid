# CARE-Droid HAIG v1

Host-centred Fine-grained API Interaction Graph experiment. It preserves the
locked P31-RF/DI25 contracts and does not alter P70/P71. The extractor rebuilds
DEX method IR because the frozen endpoint cache does not contain canonical API
signatures or bytecode order.

Edges are deliberately named interactions:

- `SAME_HOST_ORDER`: bytecode order within one Host, not data flow.
- `HOST_CALL`: direct resolved invocation between sensitive Hosts, not data flow.

The source probe compares H0 Role, H1 API identity, H2 same-Host interaction,
H3 direct Host-call interaction, topology-preserving shuffled controls, and a
count-only control. Source selection is written before any future labels are
opened. Frozen 2017-to-2018/2021 and independent same-year five-fold diagnostics
run afterward and cannot alter the selected source representation.

Run:

```bash
unzip -o ../care-p31rf-haig-source-v1.zip -d scripts/
bash scripts/care-p31rf-haig-source-v1/launch_tmux.sh
tail -F artifacts/care-p31rf-haig-source-v1.console.log
```

Re-running is resumable: every APK graph is an independent gzip JSON checkpoint.
