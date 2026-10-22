# CARE-Droid Local Role Neighborhood Probe v1

This is the final source-only representation-geometry probe on the locked P31-RF -> RF-DI25 -> Role C43-RQ line. It does not re-extract APKs, retrain C43, change the router, load future years, or introduce a benefit gate.

The available fold-safe representation is the label-free 43-D Compact Host content grouped by SelfRole31. Learned internal C43 hidden vectors and per-Role neural attention were not saved, so the experiment explicitly does not claim to retrieve them. Every validation query searches only the other four folds.

The primary settings are fixed before evaluation: cosine similarity, `k=5`, per-class per-Role support >=5, 20 deterministic matched-null repetitions, and near-duplicate sensitivity excluding similarities >=0.99. Integration is conditional and cannot run unless the strict source probe gate passes.

```bash
bash scripts/care-local-role-neighborhood-v1/launch_tmux.sh
tail -F artifacts/care-local-role-neighborhood-v1.console.log
```

