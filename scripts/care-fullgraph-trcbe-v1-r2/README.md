# CARE-Droid full-graph TRCBE v1-r2

This package answers one question before any label-based experiment: can the
existing server assets/toolchain provide a complete app-defined method graph
that restores callback-root-to-Host paths?

The pipeline is strictly gated:

1. read-only asset/toolchain audit;
2. deterministic label-free 15-APK pilot;
3. only after pilot PASS, build the formal routed 200 graph cache;
4. rerun label-free TRCBE coverage with the frozen taxonomy;
5. stop at `FEASIBLE_FOR_DIRECTION_PROBE` or a negative engineering decision.

It does not install or download anything, does not use Android SDK, does not
run APKs, does not read future years, and never writes an old cache. The graph
contains every app-defined DEX method and caller-to-callee app-internal invoke
edge; app-local class-hierarchy expansion handles virtual/interface calls. Its
declared limitations are reflection, native code, dynamically loaded code and
framework internals.

This materialization package intentionally does not fabricate a direction
probe. If and only if the frozen 20% coverage gates pass, it returns
`FEASIBLE_FOR_DIRECTION_PROBE` and preserves the 200 graphs for the next
strictly label-aware package.

Revision r2 fixes one audit-only denominator bug found by the first pilot:
APKs with zero reconstructed sensitive Hosts are now reported separately and
are not treated as 0%-aligned. Exact Host alignment is pooled over actual Host
instances, while a separate gate requires at least ten Host-evaluable pilot
APKs. No graph feature, root rule, or scientific threshold was relaxed.
