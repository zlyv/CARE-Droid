# SVM Direct-C43 frozen representation collision audit v1

This package replays the existing source-only Direct-C43 protocol solely to export its 113-D fused representation and penultimate hidden activation. It then runs strict outer-fold nearest-neighbor, local-purity, simple frozen-probe, P31/count-null and matched-shuffle audits. It never loads future data and does not select a deployable model or threshold.

Run with `bash scripts/care-svm-c43-representation-collision-audit-v1/launch_tmux.sh`.
