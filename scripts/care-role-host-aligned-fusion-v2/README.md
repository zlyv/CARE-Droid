# CARE-Droid static Role–Host aligned fusion v2

This source and rolling-development experiment uses the already cached exact Host-own-Role31 binding. It performs no dynamic analysis, opens no APK, changes no Stage-I/router/background, and adds no contrastive loss.

Arms:

- `F0_FROZEN_DIRECT_C43`: frozen formal baseline.
- `F1_ALIGNED_BIPARTITE`: exact node-level bidirectional residual fusion.
- `F2_WITHIN_APK_BINDING_SHUFFLE`: equal-capacity binding null preserving the APK Host-role row multiset and Role marginals.
- `F3_APK_ROLE_MARGINAL_CONTROL`: equal-capacity control retaining APK Role marginals without Host identity binding.

This is the final distinct end-to-end binding test after the compact HBI readout was not supported. If F1 does not beat both controls, no fusion tuning is authorized.

## v2 numerical correction

v1 stopped during T3/900000 F3 validation because float32 reduction order reported an absolute marginal error of 0.0018158. A read-only replay found the float64-audited error was 7.56e-06. v2 constructs the F3 APK mean from a float64 reference and audits marginals in float64 with a fixed 1e-5 tolerance; all model tensors remain float32. v1 outputs are retained unchanged.
