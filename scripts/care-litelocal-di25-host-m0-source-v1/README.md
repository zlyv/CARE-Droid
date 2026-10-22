# CARE-Droid LiteLocal Stage-I + DI25 + existing Host-M0 bounded correction

This is a **strict 2017 source-only integration experiment** for the modified first layer.

## What is actually tested

Pipeline:

```text
APK
  -> LiteLocal FUSED63 Stage-I
       P31 + COOC16 + TRANS16
  -> DI25 (fold-wise Top25% MeanOpposition)
  -> only routed samples use the established CARE Host-M0 expert
       existing Global24 Stage-II context
       RoleGraph
       critical/attended Host evidence
  -> existing FixKeep bounded logit correction
  -> final probability
```

Important: the old code columns `A25__score` and `A25__routed` are used only as compatibility slots required by the existing CARE reader. They contain the **new LiteLocal DI25** values. The experiment also writes explicit `DI25_LiteLocal__score` and `DI25_LiteLocal__routed` columns.

## Leakage contract

For outer fold `k`:

- outer validation Stage-I probability: train on the other four folds, identical to the LiteLocal source audit;
- Stage-II outer training probability: strict inner OOF inside those four folds; for each inner validation fold, only the other three folds train LiteLocal;
- Stage-II never receives a Stage-I training probability from a model that saw the held-out outer fold;
- no 2018--2021 row is loaded.

The existing Stage-II implementation is reused rather than rewritten:

```text
scripts/care-di25-host-input-replay-v1/run_arm.py --arm FULL
```

Thus the Host/RoleGraph/FixKeep/bounded-correction implementation stays identical to the established M0 experiment.

## Required existing project inputs

```text
artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz
artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv
scripts/care-litelocal-semantic-source-v1/run_source_oof.py
scripts/care-di25-host-input-replay-v1/run_arm.py
```

If available, the script reuses:

```text
artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv
```

for the already-audited outer OOF predictions. It still builds strict nested Stage-I training predictions required by Stage-II.

## Install

Upload this ZIP to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-di25-host-m0-source-v1.zip -d scripts/
```

## Preflight only

```bash
/home/boyuanjie/envs/tif/bin/python \
  scripts/care-litelocal-di25-host-m0-source-v1/selftest.py

/home/boyuanjie/envs/tif/bin/python \
  scripts/care-litelocal-di25-host-m0-source-v1/preflight.py
```

Do this before a long run.

## Run one Stage-I definition

Example on GPU 1:

```bash
bash scripts/care-litelocal-di25-host-m0-source-v1/launch_tmux.sh MEAN_COMMITTEE 1
```

Monitor:

```bash
tail -F artifacts/care-litelocal-di25-host-m0-source-v1/MEAN_COMMITTEE.console.log
```

For a faster one-seed diagnostic first:

```bash
SEEDS=0 bash scripts/care-litelocal-di25-host-m0-source-v1/launch_tmux.sh MEAN_COMMITTEE 1
```

After the one-seed run passes, remove its output directory before the formal 3-seed run or use a fresh output root.

Supported Stage-I definitions:

```text
LR_ONLY
MEAN_COMMITTEE
MEDIAN_COMMITTEE
```

Use the definition selected by your strict source Stage-I audit. Do **not** choose it from future-year results.

## Main outputs

For `MEAN_COMMITTEE`:

```text
artifacts/care-litelocal-di25-host-m0-source-v1/MEAN_COMMITTEE/
  integration_summary.csv
  integration_per_fold.csv
  integration_predictions.csv
  REPORT_LITELOCAL_INTEGRATION.txt
  integration_contract.json
  preflight/litelocal_di25_reference/source_predictions.csv
  arms/FULL/source_predictions.csv
```

The summary reports:

- LiteLocal Stage-I BA/AUC/F1;
- DI25 routed count, error recall and routing precision;
- end-to-end BA/AUC/F1;
- fixed / broken / net correction counts;
- per-fold results.

## What this experiment does NOT prove

It is source-only. It can decide whether the modified Stage-I works coherently with the existing Stage-II and bounded correction without leakage. It does **not** justify choosing a model from 2018--2021 diagnostics and it does not by itself prove a runtime-cost advantage. Runtime/memory/failure-rate profiling should be a separate experiment.
