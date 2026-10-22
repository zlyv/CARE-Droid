# CARE-Droid benign Host-Role contrastive pilot v1

This experiment is distinct from the historical APK-level Role-weighted
SupCon. It adds an auxiliary training loss on projected **Host** embeddings
after the Direct-C43 region encoder.

- Positive pair: two high-confidence, same-dominant-Role Hosts from different
  training-period benign APKs.
- Negative pair: high-confidence Hosts with different dominant Roles, from
  different training-period benign APKs.
- Malware-bag Hosts are not assigned pseudo-positive or pseudo-negative Host
  labels. Ambiguous/multi-Role Hosts (maximum normalized Role weight < 0.8)
  are excluded from this auxiliary loss, but remain in the classifier BCE.
- At most four eligible Hosts per benign APK enter each batch's auxiliary loss.
- H1 uses actual training Role labels; H2 shuffles the same batch's Role labels
  while retaining counts, model capacity, optimizer and contrastive weight.
- The original three background values, RoleGraph, Direct-C43 model and
  equal-weight APK BCE are preserved. The 16-D projection head is discarded
  at inference.
- The projection input is a 24-D Host vector encoded from Compact107 plus
  four background responses; explicit Host Role31 weights are not directly
  input into that projection.
- Frozen 2017-only / rolling train windows predict 2018-2021. Those years are
  development data, not independent final validation. The same Manifest-RF
  Stage-I outputs and routes are applied to B0, H1 and H2.

This pilot tests whether Host Role geometry improves detection. It cannot
by itself establish "benign evidence": same-Role compactness may also remove
class-relevant within-Role differences. Compare classification, fixed/broken,
cross-year consistency and the shuffled Role control before considering any
model change.

No APK is opened or executed, no feature is re-extracted, and no original
checkpoint or code is modified. New files are restricted to
`outputs/care_benign_host_role_contrastive_v1/`.
