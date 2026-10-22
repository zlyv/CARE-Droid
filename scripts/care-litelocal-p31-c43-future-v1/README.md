# CARE-Droid P31 -> P31-DI25 -> C43-RQ future diagnostic v1

This package evaluates the **actual P31 Stage-I** from the uploaded strict-source experiment on 2018-2021, followed by the same P31 MeanOpposition DI25 routing and the same C43 relation-query bounded correction family.

It does **not** use FUSED63.

Pipeline:

```text
2017 P31 (799 labeled source APKs)
        |
        | train calibrated LR / RF / XGB / SVM on source only
        v
2018-2021 P31
        |
        v
P31 Mean Committee Stage-I
        |
        v
MeanOpposition DI25
Top25% independently within each year
2018:198, 2019:100, 2020:99, 2021:100
        |
        v
C43_RELATION_QUERY Stage-II
relation -> attention, not directly -> delta
        |
        v
bounded correction, max_delta=2.25
```

## Important protocol

- Stage-I training uses **2017 labels only**.
- Future P31 probabilities and DI25 routes are written **before future labels are unsealed**.
- Future labels are then joined only for diagnostic evaluation / the legacy future evaluator's reporting path.
- 2018-2021 results are **diagnostic only**. Do not use them to select a router, threshold, representation, C43 variant, or correction architecture.
- The script refuses to replace P31 with `role_node_*` counts. A real future `role_prevalence[:,31]` cache must exist.

## Required source experiment

Run the strict source package first so this exists:

```text
artifacts/care-litelocal-p31-c43-source-v1/preflight/p31_reference/source_predictions.csv
```

## Install

Upload the ZIP to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-p31-c43-future-v1.zip -d scripts/
```

## Preflight-first run

The package automatically searches for a future semantic NPZ that contains:

```text
sha256
role_prevalence  # N x 31
```

and covers all 1,989 future APKs.

If your correct cache has a non-standard filename, set it explicitly:

```bash
export FUTURE_P31_CACHE=/absolute/or/repo/path/to/the_future_semantic_cache.npz
```

C43 future region cache is also discovered automatically. If multiple candidates exist, set:

```bash
export C43_FUTURE_REGION_CACHE=/path/to/the_correct_compact43_relation_future_cache.npz
```

Then run on physical GPU 1:

```bash
bash scripts/care-litelocal-p31-c43-future-v1/launch_tmux.sh 1
```

Monitor:

```bash
tail -F artifacts/care-litelocal-p31-c43-future-v1.console.log
```

The script performs all gates before the expensive Stage-II run. If a required future P31 or C43 cache is missing/ambiguous, it exits before GPU training and prints the candidate paths instead of guessing.

## Main report

```text
artifacts/care-litelocal-p31-c43-future-v1/results/REPORT.txt
```

It reports for each year and pooled 2018-2021:

- P31 Stage-I BA/AUC/F1;
- Stage-I error count;
- P31-DI25 routed count;
- routing error recall and precision;
- final BA/AUC/F1 after C43-RQ bounded correction;
- fixed / broken / net;
- delta BA.
