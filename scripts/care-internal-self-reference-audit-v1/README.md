# CARE-Droid Internal self-reference audit v1

Read-only, source-and-rolling-development audit of whether the frozen Internal
background is attenuated because each Host participates in the APK-local
centroid against which that same Host is scored.

The package:

- reads only frozen 2017--2021 feature caches and M0 checkpoints;
- never opens, installs, executes, or rewrites APKs;
- trains no model and changes no formal threshold or route;
- keeps the original dominant-cluster set and original within-APK scale fixed;
- compares leave-self-out with matched removal of another Host from the same
  selected cluster;
- writes only `outputs/care_internal_self_reference_audit_v1/`.

Run with `launch_tmux.sh`; inspect with `collect_results.sh` after completion.

