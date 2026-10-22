# LiteLocal + source-only learned error-risk router + existing Host-M0

This experiment changes **only the routing rule** relative to the completed fixed-DI25 source integration.

## Pipeline

```text
LiteLocal FUSED63 Stage-I
        |
        | four member probabilities + committee descriptors
        v
Learned Stage-I error-risk router
        |
        | predicts P(Stage-I is wrong)
        | threshold learned on source outer-train only
        v
route / keep
        |
        v
existing Host-M0 FULL expert (unchanged)
        |
existing FixKeep bounded correction (unchanged)
```

The router is **not** constrained to 25%.  Each held-out outer fold receives the threshold learned only from the other four folds.

## Router target

For strict Stage-I inner-OOF predictions inside an outer training split:

```text
error_target = 1 if Stage-I predicted label != true source label else 0
```

The router sees only information available before Stage-II:

```text
p_stage1
p_LR, p_RF, p_XGB, p_SVM
abs_margin
binary_entropy
member_std
member_range
vote_disagreement
mean_opposition
max_member_deviation
```

Model: standardized class-balanced logistic regression.

## Learned threshold

Inside each outer-training split, the router itself is cross-fitted over the remaining source folds.  The threshold is selected from those router OOF risks by:

1. maximize balanced accuracy for detecting Stage-I errors;
2. if tied, choose the lower route rate;
3. if still tied, choose the higher threshold.

Then the router is refit on all outer-training rows and applied once to the untouched outer-validation fold.

Therefore the reported source routing result is strict outer OOF. No future row is loaded.

## Reuse the already-computed strict Stage-I cache

Default:

```text
artifacts/care-litelocal-di25-host-m0-source-v1/stage1_cache/strict_fused63_members.npz
```

So the 20 strict nested LiteLocal fits from the completed fixed-DI25 experiment do not need to be repeated.

## Install

Upload ZIP to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-learnedrouter-host-m0-source-v1.zip -d scripts/
```

## Preflight first

```bash
/home/boyuanjie/envs/tif/bin/python \
  scripts/care-litelocal-learnedrouter-host-m0-source-v1/selftest.py

/home/boyuanjie/envs/tif/bin/python \
  scripts/care-litelocal-learnedrouter-host-m0-source-v1/preflight.py
```

Expected:

```text
LITELOCAL_LEARNED_ROUTER_HOST_M0_SELFTEST_PASS
LITELOCAL_LEARNED_ROUTER_HOST_M0_PREFLIGHT_PASS
```

## Run

The completed Stage-I definition was `MEAN_COMMITTEE`, so on GPU 1:

```bash
bash \
scripts/care-litelocal-learnedrouter-host-m0-source-v1/launch_tmux.sh \
MEAN_COMMITTEE 1
```

Monitor:

```bash
tail -F \
artifacts/care-litelocal-learnedrouter-host-m0-source-v1/MEAN_COMMITTEE.console.log
```

A quick one-seed wiring run is possible with `SEEDS=0`, but the formal result should use the same 3 Stage-II seeds as the fixed-DI25 experiment.

## Main outputs

```text
artifacts/care-litelocal-learnedrouter-host-m0-source-v1/MEAN_COMMITTEE/
  router_thresholds_by_outer_fold.csv
  router_comparison.csv
  integration_summary.csv
  integration_per_fold.csv
  integration_predictions.csv
  REPORT_LEARNED_ROUTER_INTEGRATION.txt
  integration_contract.json
  preflight/learned_router_reference/source_predictions.csv
  arms/FULL/source_predictions.csv
```

`router_comparison.csv` directly compares the learned source-only router with fixed DI25 on the exact same LiteLocal Stage-I OOF predictions.

## What to inspect first

The first decision is **not** whether final BA improved. Inspect in this order:

1. learned route rate (is it materially below/above 25%?);
2. Stage-I error recall;
3. routing precision / error enrichment;
4. then Fixed / Broken / Net after the unchanged Host-M0 expert.

This isolates whether replacing the fixed 25% budget actually improves routing before redesigning the correction model.
