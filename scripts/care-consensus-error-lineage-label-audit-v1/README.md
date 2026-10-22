# CARE-Droid consensus-error software-lineage and label-consistency audit v1

This is a read-only 2017--2021 audit of the 93 APKs jointly misclassified by
LR, RF, XGB, SVM and Direct-C43.  It does not train a classifier, alter routing,
or relabel any sample.

The earlier `care-p31-lineage-host-audit-v1` studied source-2017 lineage inside
P31-equivalent groups.  This experiment is non-duplicative: it first indexes
all 2,788 APKs using package, signer and exact DEX hashes, then applies that
earlier audit's verified 262,144-bit DEX fingerprint implementation only to
consensus errors and deterministic matched neighbors.

Mixed labels in one lineage are reported as ambiguity evidence.  They are not
automatically called label noise because repackaging, rider injection and real
behavior changes are plausible alternatives.

## Runtime

CPU only.  Four metadata workers are used by default.  Set `CELA_WORKERS` to a
smaller value if memory is constrained.  Per-APK metadata and deep fingerprints
are checkpointed and safely reused after interruption.
