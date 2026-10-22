# Manifest four-member disagreement routing v1

Independent, read-only-input development audit. LR/RF/XGB/SVM each serve as
Stage-I on the same frozen 135 Manifest features. Three routing rules use only
four calibrated probabilities: committee probability standard deviation,
opposition of the other three to the current Stage-I decision, and Stage-I
uncertainty control. Each rule has a source-crossfit 25%-reference threshold;
an exact same-count uncertainty control is also evaluated. The Direct-C43
expert is the previously frozen, SHA-aligned probability, never retrained.

2017 uses the fixed five outer folds. Every router reference threshold is
fitted from a three-fold crossfit inside the corresponding outer training set.
2018–2021 use only full-2017 fits/reference thresholds; their labels are read
only after all model and routing rules have been frozen. These years are
historical development diagnostics, not independent final validation.

No APK is opened, installed, executed, or rewritten. All writes are under a
new `outputs/care_manifest_four_member_disagreement_v1/run_*` directory.

Run in tmux with `MFD_OUT` set to a fresh output child and execute `run.sh`.
The script resumes completed seed/fold fits. Inspect `run.log`,
`yearly_metrics.csv`, `stage1_summary.csv`, `final_report.md`, and
`final_decision.json`. Do not select a route from future-year outcomes.
