# Blinded Host-direction review protocol

This is an evidence audit, not model training. Do not open `host_review_key_sealed.csv` until every reviewer form is frozen and hashed.

Review `host_review_items_blinded.csv` grouped by `review_id`. Use only the displayed Critical/top-attention Host roles, exact framework/API invocations, and app caller/callee context.

Allowed `sample_direction` values:

- `MALWARE_SUPPORT`: the displayed behavior and local context independently support correction toward malware.
- `BENIGN_SUPPORT`: the displayed behavior and local context independently support correction toward benign.
- `AMBIGUOUS`: evidence supports both directions or intent cannot be determined.
- `INSUFFICIENT`: the static context is too incomplete to determine direction.

Rules:

1. Do not infer benignness from a permission, common API, high/low Host count, or component count alone.
2. Do not infer maliciousness from sensitivity or background execution alone.
3. Exact APIs are explanatory evidence only; do not apply a memorized malware/benign API list.
4. Record the decisive call relation or semantic reason in `primary_evidence`.
5. Use `confidence` in `{LOW, MEDIUM, HIGH}`.
6. Prefer two independent reviewers. Freeze both forms before opening the key.
7. Report coverage, abstention, agreement, FN-side accuracy and FP-side accuracy separately.

Predeclared continuation gate after unblinding:

- at least 60% of selected samples receive a non-abstaining direction from both reviewers;
- Cohen's kappa at least 0.40 on non-abstained overlap;
- malware-side and benign-side direction accuracy are each above 0.60;
- neither side is explained by one reviewer using API/Host count;
- results are not driven by a single fold.

Failure of this gate means that the existing static Host evidence is not sufficiently interpretable for stable bidirectional correction. It does not authorize a larger model.
