# CARE-Droid BFA-V1

This experiment uses only the frozen 2017–2021 development cache and SCBE M0
checkpoints. It opens no APK, does not alter Stage-I, router, formal code, or
existing checkpoint. All new files go below
`outputs/care_background_functional_attribution_v1/`.

Run `run.sh --audit-only`, then `run.sh --smoke-test --device cuda`, then
`launch_tmux.sh`. The detached pipeline performs the 8-pair attribution pilot,
full T2–T4 attribution, source and rolling B1–B4 retraining, and evaluation.

B0 is the already independently retrained SCBE M0 checkpoint, not a newly
invented baseline. All ablations retain the 142-wide input/model capacity and
use the same three seeds, fixed routes, epochs, optimizer, BCE and threshold.
External, Shared-derived group and Internal primitive channels are neutralized
with class-blind train-only equal-APK means; Residual is always recomputed.
B4 role-matched External shuffling controls for input size/marginal scale.

These are model-input information neutralizations, not genuine APK-level
counterfactuals or mathematically pure bank deletions. Shared has a dependent
path to Internal, so B2 neutralizes both Shared-derived signals as a group.
No prior result justifies treating background statistics as semantic benign
certificates. If checks fail, the pipeline stops and leaves old assets intact.
