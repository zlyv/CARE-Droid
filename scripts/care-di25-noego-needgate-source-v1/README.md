# CARE-Droid DI25 No-Ego Need-Gate Source v1

This package replaces neither the failed future/same-year package nor any old artifact. It is a new strict-2017 source-only contract.

`run_preflight_only.sh` still runs **only** synthetic self-tests and the real
source preflight.  Formal training is a separate, explicit entry point and is
never started by unpacking the package or by the preflight runner.

## Closed read whitelist

The code accepts no path arguments. Its only possible artifact reads are the seven paths in `source_contract.py`:

1. canonical 2017 DI25 router;
2. compact features and schema;
3. source P72 OOF manifest;
4. dedicated Host-only NPZ;
5. generic source regions NPZ (allowed but deliberately not consumed by preflight/model input);
6. 2017 endpoint cache.

No future file, within-year router, Ego NPZ, future row or future label has a fallback path.

## Source loader provenance

- Cohort/labels/folds are keyed exclusively by `care-source-di25-reconstructed-v1/source_router_inputs.csv`.
- The Host loader follows the audited `care-source-local-assets-v2/audit_source_assets_v2.py` and historical `care-rolegraph-conditioned-v4-script::get_source_host_bags` schema, but reads the dedicated `host_regions_2017.npz` directly.
- Its selected NPZ keys are exactly `sha256`, `host_values`, `host_offsets`, and `real_host_count`.
- The leakage-safe inner-OOF contract mirrors `care-rolegraph-conditioned-v4-script::inner_oof_global24`: each outer-train cohort is split into four inner folds; preprocessing and Global24 fitting occur only on each inner-fit subset; each outer-train APK receives exactly one inner-validation probability. Frozen router `p_stage1` is reserved for outer validation.

## Current execution

Upload the zip into `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-noego-needgate-source-v1.zip -d scripts/
bash scripts/care-di25-noego-needgate-source-v1/run_preflight_only.sh \
  2>&1 | tee artifacts/care-di25-noego-needgate-source-v1.preflight.console.log
```

This intentionally runs no formal training and creates no tmux session.

## Explicit formal run (only after inspecting the preflight PASS)

Direct foreground execution:

```bash
cd /home/boyuanjie/projects/tasa-droid
bash scripts/care-di25-noego-needgate-source-v1/run_formal_on_server.sh \
  2>&1 | tee artifacts/care-di25-noego-needgate-source-v1.console.log
```

Physical GPU 1 in tmux (it is correctly exposed to the process as `cuda:0`):

```bash
cd /home/boyuanjie/projects/tasa-droid
CUDA_VISIBLE_DEVICES=1 \
  bash scripts/care-di25-noego-needgate-source-v1/launch_formal_tmux.sh
```

The formal runner repeats selftest and preflight before training.  It writes
only under `artifacts/care-di25-noego-needgate-source-v1/`; the console log is
the adjacent `artifacts/care-di25-noego-needgate-source-v1.console.log`.
