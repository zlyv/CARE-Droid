# CARE-Droid LFC-V1

Only first-stage RF candidates and predeclared uncertainty routing change. Direct-C43 is never retrained. All old LiteStage shards, baseline predictions and C43 probability tables are read-only. The LFC output must be a unique `outputs/care_lfc_v1/run_*` child. No APK is opened by this package.

The previous LiteStage `dex_meta[4]` equals `invoke_n`. LFC reference-only arms explicitly exclude it and are refitted on the original fixed source folds. The existing ManifestOnly, P31-RF, and nested BestLiteStage 2017 OOF predictions are reused.

To start, create a unique empty run directory, set `LFC_OUT` to it and execute `bash scripts/care-lfc-v1/run.sh`. To resume an interrupted run: `LFC_OUT=/absolute/run/path bash scripts/care-lfc-v1/run.sh --resume`. Status: `bash scripts/care-lfc-v1/status.sh /absolute/run/path` or `tail -n 50 /absolute/run/path/run.log`.

2017 candidate and routing comparisons are descriptive across predeclared arms, not a post-hoc unbiased best-arm estimate. No future label is used to fit RF or choose roles. The Host/call-graph/RoleGraph incremental cost is unavailable, so no full-system cost saving is claimed.
