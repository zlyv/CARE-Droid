# CARE-Droid OBCBC Stage-B static feasibility package

This package implements only the label-free Stage-B audit authorized by the
frozen `NOVELTY_PLAUSIBLE` literature decision.

It reuses the existing 200 routed-APK full app-defined method graphs. It does
not execute APKs, rebuild the formal baseline, read labels, train a model, load
future rows, install software, or modify formal assets.

## Inputs

- formal P31-RF/DI25 predictions (SHA/fold/routed columns only);
- frozen endpoint Host cache;
- frozen C43 attention;
- full-graph TRCBE Host assignment;
- 200 existing complete graph cache files;
- 200 routed APKs for read-only Manifest/DEX/resource parsing.

## Outputs

The output directory is `outputs/care-obcbc-source-v1/`. The scientific gate
is in `feasibility_decision.json`; the overall handoff is in
`final_decision.json`.

Only `STATICALLY_FEASIBLE` authorizes a separate source-probe package.
