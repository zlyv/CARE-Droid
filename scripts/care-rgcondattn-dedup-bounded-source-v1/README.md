# CARE bounded RGCondAttn deduplication source ablation v1

This strict 2017 experiment changes only the RGCondAttn implementation:

- `context == attended` is no longer concatenated twice;
- RGCondDual-only `critical_query` and `context_key` modules are removed;
- Global24, A25, role graph, local evidence, FixKeep, and bounded logit
  correction remain unchanged.

The cleaned correction input is `[xG, z1, g, h_critical, e, m]` with width
113.  Under the established dimensions it has 33,388 trainable parameters,
down from 35,740.

Run:

```bash
bash scripts/care-rgcondattn-dedup-bounded-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-rgcondattn-dedup-bounded-source-v1.console.log
```

Do not inspect future labels until the source-side decision is frozen.
