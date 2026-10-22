# CARE RGCondAttn Direct-L2 source ablation v1

This is the strict 2017 source-side test of the independent second-layer
classifier.  Copy the directory under the repository `scripts/` directory.
The established CARE v3/v4 scripts and caches must already be present.

`run_on_server.sh` first checks that changing the Global24 vector and its
first-layer logit leaves the Direct-L2 output exactly unchanged.

The expert input is exactly `[g, h_critical, e, m]`.  It excludes Global24 and
the first-layer logit, predicts an absolute malware logit, uses ordinary BCE,
and has no bounded residual or FixKeep objective.  The frozen A25 route is
unchanged: non-routed rows keep Global24 and routed rows use Direct-L2.

Run:

```bash
bash scripts/care-rgcondattn-direct-l2-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-rgcondattn-direct-l2-source-v1.console.log
```

Do not inspect future labels before making the source-side decision.
