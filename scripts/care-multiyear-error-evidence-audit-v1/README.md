# Multi-Year Same-APK Error Evidence Audit

This branch investigates new 2019–2021 failures, not another 2017 modality
presence screen or generic representation collision probe. It fits no model,
changes no threshold/route/feature, and never installs, executes or rewrites APKs.

Frozen cohort: 35 persistent FN, 18 persistent FP, 3 persistent broken TP,
7 persistent broken TN and 10 FN that C0 repaired but C1 no longer repairs.
2018 is excluded because C0/C1/C2 are the same model there. All three experts
are compared on the same Stage-I probabilities and routed APKs.

The branch retrieves saved background states, scalers, NN weights and global
validation inputs. Numerical probability replay must pass before interpretation.
Every Host is aligned by frozen endpoint-node identity plus full Role/Relation
columns, not guessed by method order. Same-year correct anchors and opposite-label
controls are matched using a fixed raw descriptor. Matching relaxation and reused
controls are disclosed; they do not create independent statistical units.

All original models, caches, scripts, predictions and inspected APKs are protected
by SHA/size/mtime manifests. Only new output is written. No Git command is used.

```bash
cd /home/boyuanjie/projects/tasa-droid
CUDA_VISIBLE_DEVICES=0 bash scripts/care-multiyear-error-evidence-audit-v1/run.sh --device cuda --workers 2
```

Output default: `outputs/care_multiyear_error_evidence_audit_v1/`.
Existing output is never overwritten. After an engineering failure preserve it
and explicitly choose a fresh version, for example:

```bash
MEEA_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_multiyear_error_evidence_audit_v2 \
CUDA_VISIBLE_DEVICES=0 bash scripts/care-multiyear-error-evidence-audit-v1/run.sh --device cuda --workers 2
```

Static inspection reads only methods selected by per-seed critical Host and
ensemble top3 attention. Androguard DEX parsing is bounded to two workers,
180 seconds per APK, 128MiB per DEX and 512MiB total uncompressed DEX.
Unknown/missing/ambiguous mappings are retained, never guessed. Branch presence
is not control dependence; API occurrence is not executed behavior. Non-permission
string literals are redacted so dossiers do not expose embedded credentials.

`private/model_behavior.csv` contains score/critical-Host observations, not verified
causes. Raw training-neighbor composition is an observation sensitive to N, not
Host truth. Independent-model hidden coordinates are not compared, and old/new
preprocessing or heads are never swapped into incompatible models.

For semantic review open **only** `blind/review_form.csv`, `blind/cases/` and
`blind/code/`. Cite method, DEX identity and instruction offset for every fact.
Do not open `private/blind_review_key.csv` or model traces until review is frozen.
Application method names may naturally reveal function; blinding is only to
dataset labels, predictions and target identity. Permission checks and weak
anomaly are never benign evidence. No general benign/malware certificate is
inferred from these snippets. Unknown is a legitimate review outcome.

An audit with complete replay/code but no independently verified class-direction
fact stops at causes unresolved. It does not authorize feature/model training.

Actual completed CPU run is v2 (v1 preserves a PyTorch CUDA initialization failure).
Use `--device cpu` where CUDA initialization fails; do not relax replay tolerance.
Additional independent helpers: `selection_intervention.py` (CPU only) performs
within-C1 selection diagnostics without swapping model coordinates;
`inspect_identity.py` resolves ambiguous selected IDs but never repairs originals;
`review_tools.py --priority-packet` makes a label-free review packet;
`finalize_report.py` compiles the completed evidence and packages that packet.
Set the same explicit `MEEA_OUTPUT` for these helpers. Each refuses to overwrite its
completed outputs. Reproduction requires a fresh numbered output root.

Casefolded IDs are used here solely to reproduce the frozen identity contract.
They are not an assertion that case-distinct DEX definitions are equivalent.
