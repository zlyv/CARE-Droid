# CARE-Droid P31-C43 FP benign-explanation gap audit v1

Strict-2017 read-only post-hoc audit. It asks whether the 20 routed false
positives contain benign-direction evidence in the already frozen endpoint
cache that is absent from the current C43 representation.

The primary comparison is FP versus TP because both cohorts are routed and
predicted malware by P31. Routed TN is retained as a benign reference. For each
outer fold, class centroids are built only from correct routed APKs in the other
four folds. No neural model is trained or invoked.

Primary evidence groups are normalized provenance, entry/edge context, and
semantic organization. Raw node/edge/endpoint/Host counts are audit-only and
are evaluated separately as `SIZE_ONLY`.

Run from the project root:

```bash
bash scripts/care-litelocal-p31-c43-fp-benign-gap-audit-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31-c43-fp-benign-gap-audit-v1.console.log
```
