# CARE-Droid v4: Role-Graph Conditioned Dual-Stream

This bundle contains a source-only 2017 ablation for replacing flat
`S7 + CARE` concatenation with a structured global-to-local mechanism.

## Faithful S7 mapping

`S7 = Stats120 + PairRaw465 + InterZ961` is mapped as follows:

* `Stats120[:93]` -> 31 role nodes x 3 statistics;
* `Stats120[93:120]` -> 27 graph-level statistics;
* `PairRaw465` -> the 465 undirected off-diagonal role pairs;
* `InterZ961` -> the complete 31 x 31 directed relation channel.

No raw call graph is rebuilt and no future labels are used.

## Variants

* `RGConcat`: graph token and local evidence meet only in the final correction
  head. This is the structured-concatenation control.
* `RGCondAttn`: the graph token directly conditions local-region attention.
* `RGCondDual`: additionally chooses a critical region and builds a second
  context stream relative to it.

All candidates retain dual-normal prototypes, FixKeep weighting, a bounded
delta, the frozen role-free Global24 first layer, and frozen A25 routing. Unlike
CARE v3, the candidate does not use ordinary Joint-S7 as its internal base;
ordinary Joint-S7 and CARE v3 are references.

## Required existing scripts

The repository `scripts/` directory must already contain the CARE v3 bundle
and its dependencies, including:

* `run_care_bounded_fixkeep_ablation.py`
* `run_care_joint_dualprototype_ablation.py`
* `run_care_local_dualpool_mil.py`
* `run_care_local_region_asset_audit.py`
* the Global24/S7 dependency scripts used by those files.

The source-only runner can safely reuse the prefix of the established combined
2017--2021 local-region cache. It never overwrites that combined cache.

## Recommended first run

Run only `RGCondDual`, one seed, and a short epoch budget as a code smoke test.
Then run all three variants on strict 2017. Do not run future diagnostics until
the source-side comparison has been frozen.
