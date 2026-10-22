# CARE-Droid Context-Sensitive Multi-Hop Behavioral Witness v1

This is an independent, resumable experiment package. It never overwrites the frozen CARE-Droid baseline.

## Scientific distinction

The formal C43 line uses direct Host relations and a bounded endpoint neighborhood. Earlier full-graph TRCBE used context-insensitive reachability and collapsed a Host reached from both user and autonomous roots into `BOTH`.

This package retains call-site tokens and bounded call strings:

- maximum root-to-Host depth: 8 app-method calls;
- context: root family plus the latest three call-site tokens;
- maximum 16 contexts per node and root category;
- downstream effect audit: three app-method calls;
- unchanged A0 C43 attention for Host localization;
- no package, class, method, library or exact API identity enters the direction model.

It compares REAL depth-8 witnesses with a ONE_HOP ablation, root-label shuffle, a degree-preserving callee reassignment across call sites, count-only and path-only controls at identical model capacity. The callee shuffle preserves edge count, caller out-degree, callee in-degree and call-site count while breaking the real multi-hop composition.

## Strict gates

1. Build 2017 features only.
2. Stop if label-free Host alignment/contextual-attention coverage fails.
3. Run strict 2017 fixed-fold side-specific probes.
4. Decide source support from both directional REAL-vs-null gates and A0 end-to-end net/broken.
5. Freeze `protocol.json`, its hash, and the source decision.
6. Because this run was explicitly authorized to include diagnostics, build 2018–2021 features after freezing and run frozen 2017→future plus independent same-year five-fold diagnostics even when source is negative. A negative source result remains negative and cannot be rescued by those diagnostics.

Future diagnostics never participate in method selection.

## Optional environment variables

- `CMW_SOURCE_APK_INDEX`, `CMW_SOURCE_APK_ROOTS`
- `CMW_FUTURE_APK_INDEX`, `CMW_FUTURE_APK_ROOTS`
- `CMW_GRAPH_WORKERS` (default 2)
- `CMW_GRAPH_TIMEOUT` (default 1800 seconds per APK)
- explicit source/future prediction, endpoint and attention paths documented in `common.py`

The package first reuses `outputs/care-ccbc-source-v3/apk_paths.json` and `outputs/care-ccbc-diagnostics-v1/apk_paths.json`, so explicit variables should normally be unnecessary on the established server.

## Run

```bash
bash scripts/care-contextual-multihop-witness-v1/launch_tmux.sh
tail -F outputs/care-contextual-multihop-witness-v1.console.log
```

The graph and per-APK feature caches are checkpointed. Restarting the same command resumes completed APKs.
