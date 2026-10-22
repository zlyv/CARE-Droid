# CARE-Droid PASEA audit v1

This package performs only:

1. historical equivalence audit;
2. verified TPL/signature-tool inventory;
3. optional imported Owner assignment and independent precision audit;
4. formal-asset integrity verification.

It does **not** read labels, train a model, run fixed/broken analysis, or load future rows.

The prior `care-obcbc-source-v1` result is treated as PARTIAL because its Owner labels are identity-free corpus-reuse/namespace heuristics. PASEA proceeds only when concrete TPL/version identity and independently validated high-confidence Owner assignments exist.

Optional explicit inputs:

```bash
export PASEA_TPL_TOOL=/absolute/path/to/verified/tool
export PASEA_TPL_PROFILE_DB=/absolute/path/to/profile/database
export PASEA_OWNER_ASSIGNMENTS=/absolute/path/to/owner_assignments.csv
export PASEA_OWNER_VALIDATION=/absolute/path/to/owner_validation.csv
```

`owner_assignments.csv` requires:

```text
sha256,host_id,owner,library_id,confidence,is_sensitive,is_critical,attention_weight
```

`owner_validation.csv` requires:

```text
truth_owner,pred_owner,high_confidence
```

Run:

```bash
bash scripts/care-pasea-audit-v1/launch_tmux.sh
```

