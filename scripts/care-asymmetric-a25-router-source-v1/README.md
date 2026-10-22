# CARE-Droid class-asymmetric A25 source audit

This experiment changes only how the fixed 25% A25 budget is divided between
APKs predicted benign and APKs predicted malicious.

For every fixed 2017 outer fold, five-fold inner OOF Global24 probabilities are
generated from the outer training rows. Candidate class allocations keep the
total route budget unchanged. Selection is lexicographic and has no weighted
hyperparameter: maximize captured inner-OOF errors, minimize the worse
accepted-side risk, maximize the worse side-specific error recall, then prefer
the allocation closest to equal routing rates across predicted-label sides.

The learned predicted-benign budget fraction is transferred to the held-out
outer fold. Its labels are not used for selection. The base and fixed
deduplicated bounded RGCondAttn expert are reused unchanged. No 2018--2021 row
is loaded.

Prerequisites:

```text
scripts/care-multiview-conformal-router-source-v1/
artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv
```

Run from the repository root:

```bash
bash scripts/care-asymmetric-a25-router-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-asymmetric-a25-router-source-v1.console.log
```
