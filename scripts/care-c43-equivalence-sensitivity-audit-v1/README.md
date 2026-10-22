# CARE-Droid C43 equivalence-sensitivity proxy audit v1

Revision r4 fixes recursive S7-loader interception during transformed-view
reconstruction. It pins the untouched source loader explicitly and saves one
checkpoint per transform/repeat so completed S7 views can be resumed safely.

Revision r5 applies transforms to the graph inside the endpoint-cache record's
`payload` envelope. The earlier v1 output applied no structural transform and
is retained as an invalid/inconclusive implementation run. r5 writes to the new
`outputs/care-c43-equivalence-sensitivity-audit-v2` directory.

Revision r6 keeps the original view under the strict frozen-reference check,
but reconstructs transformed routed rows with the same frozen S7 definition,
null-model seed/budget, and clipping rule. This permits the intentionally
changed PairRaw/InterZ values instead of treating them as cache corruption.

Revision r7 restores the builder-only null-model parameters that the frozen
inference CLI does not expose. It verifies sample count, swap budget, and clip
range against the formal multiscale `audit.json`, freezes the original builder
seed, and records the resolved protocol before transformed reconstruction.

Revision r8 supports formal feature directories where `audit.json` was not
retained. It parses the actually imported frozen builder with Python AST,
requires all five defaults to match the registered protocol, and records the
builder path and SHA256. If metadata is present it is checked additionally.

This package runs a source-2017-only, cache-level proxy audit. It never modifies protected assets, never reads future cohorts, and does not train EC-C43.

Frozen system: calibrated P31-SVM, SVM MeanOpposition Top25%, Direct-C43 at threshold 0.5.

The model is trained once per outer fold on original data. The same weights evaluate the original validation view and proxy-transformed endpoint/RoleGraph views. Compact43 Host descriptors remain frozen. Therefore this package measures only RoleGraph/query and Host-order sensitivity; it does **not** claim real transformed-APK semantic equivalence or verified robustness.

Transforms:

- `T0_ORDER`: exact node/edge and Host-row permutation check.
- `T1_WRAPPER_SENSITIVE`: proxy subdivision of edges into sensitive endpoints.
- `T2_DEAD_SUBGRAPH`: disconnected neutral subgraph insertion.
- `T3_DEAD_EDGE_NEAR_HOST`: neutral edge/node insertion near sensitive Hosts.
- `N1_WRAPPER_FAR`: edit-count/location control away from sensitive endpoints.

Only `GO_REAL_APK_PILOT` authorizes a separate 20--30 APK bytecode transformation pilot. It does not authorize EC-C43 training.
