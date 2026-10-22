# CARE-Droid static Role–Host aligned fusion v1

This source and rolling-development experiment uses the already cached exact Host-own-Role31 binding. It performs no dynamic analysis, opens no APK, changes no Stage-I/router/background, and adds no contrastive loss.

Arms:

- `F0_FROZEN_DIRECT_C43`: frozen formal baseline.
- `F1_ALIGNED_BIPARTITE`: exact node-level bidirectional residual fusion.
- `F2_WITHIN_APK_BINDING_SHUFFLE`: equal-capacity binding null preserving the APK Host-role row multiset and Role marginals.
- `F3_APK_ROLE_MARGINAL_CONTROL`: equal-capacity control retaining APK Role marginals without Host identity binding.

This is the final distinct end-to-end binding test after the compact HBI readout was not supported. If F1 does not beat both controls, no fusion tuning is authorized.
