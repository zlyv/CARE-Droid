# Host-region attribution feasibility audit v1

Read-only Phase A/G1 audit. It replays nine frozen Direct-C43 checkpoints, checks routed cohorts and Host/RoleGraph provenance, hashes actual reused assets before and after, then fails closed if coherent Host removal cannot be validated. It never trains or perturbs inconsistent graph inputs.

Run with `bash scripts/care-host-region-attribution-v1/launch_tmux.sh` from the project root. All resulting files are in `outputs/care_host_region_attribution_v1/`; no output is overwritten.
