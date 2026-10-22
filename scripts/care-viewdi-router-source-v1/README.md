# CARE-Droid leave-one-view-out DI router: strict 2017 source audit

This lightweight experiment changes only the routing score. It does not parse
APKs, construct call graphs, train Stage II, read Ego tensors, or load future
rows. Four calibrated L2 logistic-regression members receive the same frozen
Global24 feature matrix, but each omits one semantic group: Manifest (10),
Code/Call (3), Sensitive Behavior (5), or Graph Composition (6).

The primary score is the mean committee probability mass opposing the frozen
Stage-I hard decision. An exact 25% is selected separately in each fixed 2017
outer fold and compared with canonical frozen `route_DI` and A25.

The primary question is whether the candidate captures more frozen Stage-I
errors at the same budget. This is only a route diagnostic. A positive result
does not prove that the captured errors are repairable by Stage II.

From the repository root (ZIPs are uploaded to `/home/boyuanjie/projects/`):

```bash
unzip -o ../care-viewdi-router-source-v1.zip -d scripts/
bash scripts/care-viewdi-router-source-v1/launch_tmux.sh
tail -F artifacts/care-viewdi-router-source-v1.console.log
```

Expected cost: CPU-only, 100 small LR fits over 799 rows and at most 21
features. No GPU is used. Typical runtime should be minutes rather than hours.

Main result:

```bash
cat artifacts/care-viewdi-router-source-v1/REPORT.txt
```
