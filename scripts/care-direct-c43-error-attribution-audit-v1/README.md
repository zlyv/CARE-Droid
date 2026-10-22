# CARE-Droid Direct-C43 Source Error Attribution Audit v1

## Scope and evidence contract

This is a source-2017, strictly read-only, hypothesis-generation audit. It is not
an experiment to train a classifier or select a new model, route, threshold,
feature, hyperparameter, or category-specific decision rule.

The mandatory input is the existing frozen **2017 Direct-C43** prediction CSV,
not the previous temporal audit's 2017-2021 combined `sample_direction_audit.csv`.
The program never loads the 2018-2021 future predictions or future labels.

Known frozen source checks are mandatory (and cause a hard failure if different):

- 799 APKs, 400 benign / 399 malicious;
- P31-SVM source confusion: TN356 FP44 FN53 TP346;
- Top25 routed: 200 total; 84 Stage-I mistakes;
- Direct-C43 routed: fixed49 broken17 unfixed35 stable-correct99;
- End-to-end Direct-C43 confusion: TN378 FP22 FN43 TP356;
- Routed C43 FN-vs-TN AUC 0.7556910569105691;
- Routed C43 TP-vs-FP AUC 0.8450996677740864.

The old source predictions use `Global24__prob` and `A25__routed` column names.
Those names are treated as *unverified legacy aliases* unless a separately
supplied authoritative P31-SVM reference matches source values and route exactly.
Confusion-matrix equality alone does not prove model origin.

## Install on the existing Linux machine

```bash
cd /home/boyuanjie/projects/tasa-droid

test ! -e scripts/care-direct-c43-error-attribution-audit-v1 \
  && test ! -e outputs/care-direct-c43-error-attribution-audit-v1 \
  || { echo 'Existing script/output directory. Stop without overwriting.'; exit 1; }

unzip -q ../care-direct-c43-error-attribution-audit-v1-package-r1.zip -d scripts/
chmod +x scripts/care-direct-c43-error-attribution-audit-v1/*.sh

bash scripts/care-direct-c43-error-attribution-audit-v1/run_on_server.sh
bash scripts/care-direct-c43-error-attribution-audit-v1/collect_results.sh
```

Alternatively, start detached:

```bash
bash scripts/care-direct-c43-error-attribution-audit-v1/launch_tmux.sh
tail -F outputs/care-direct-c43-error-attribution-audit-v1.console.log
```

If using any optional inputs, set their variables **on the first run**. To run
a second variant later, use a new versioned `ECA_OUTPUT` directory (for example,
`.../outputs/care-direct-c43-error-attribution-audit-v1-with-embeddings`).
The completed base audit is deliberately never overwritten.

`ECA_PROJECT`, `ECA_PYTHON`, `ECA_SOURCE`, `ECA_ARCHITECTURE`, and `ECA_OUTPUT`
can override defaults. Paths and inputs are guarded; rerunning cannot overwrite
an existing completed `final_decision.json`.

## Mandatory analysis outputs

- `final_decision.json`: validated frozen source counts, conditions, and gates;
- `baseline_integrity_check.json`: protected input hashes before/after;
- `REPORT.md`: evidence boundary and methodological explanation;
- `source_2017_sample_phenotypes.csv`: all 799 source APKs with group, score,
  wrong confidence and conditional per-sample rank-inversion fraction;
- `source_2017_group_summary.csv`: 35 unfixed, 17 broken, 49 fixed, 99 stable;
- `source_2017_score_proximity_pairs.csv`: descriptive near-score examples,
  **not** representation distances or training pairs;
- `manual_review_blinded_template.csv`: 52 error cases, no label/error group;
- `review_case_key_SOURCE_ONLY.csv`: source-only ground truth/group lookup to
  be withheld until independent DEX reviewers have completed their inspection.

For each routed Stage-I prediction side, `pairwise_inversion` is the fraction of
opposite-label routed cases whose C43 score outranks the current case in the
wrong class direction (with half credit for ties). This is a *label-informed
retrospective diagnostic*, not a deployable individual error detector.

## Optional frozen C43 embedding review (same outer fold only)

Only use source-2017 **OOF test** hidden states from the formal frozen Direct-C43
replay. Different outer folds have different model parameters and therefore
their embedding coordinates are NOT interchangeable. The script never computes
nearest-neighbor distances across different outer folds.

For example, supply an exported `.npz` with:

- `sha256`: an `N`-entry Unicode array of source SHAs;
- `fold`: an `N`-entry fold-ID array from the same replay;
- `embedding`: a finite `N x D` numerical embedding array.

Or a CSV with `sha256,fold,z_0,z_1,...`. Explicitly set:

```bash
ECA_EMBEDDINGS=/absolute/path/to/source_2017_oof_c43_hidden.npz \
bash scripts/care-direct-c43-error-attribution-audit-v1/run_on_server.sh
```

Do not invent this path: export it from the original frozen replay if it does
not already exist. Input provenance must be established. If no valid embeddings
are available, the baseline audit remains useful, but it makes **no new
embedding-geometry claim**. Resulting `same_fold_geometry.csv` is descriptive
and is not directly comparable with earlier train-referenced kNN purity unless
the neighbor-pool construction is identical.

## Optional source-only static descriptors

Prepare a source-2017 CSV, e.g. `sha256,year,host_count,method_count,role_count`.
Only explicitly allowlisted numeric, label-independent features will be read:

```bash
ECA_STATIC_FEATURES=/absolute/path/to/2017_static_features.csv \
ECA_FEATURE_COLUMNS=host_count,method_count,role_count \
bash scripts/care-direct-c43-error-attribution-audit-v1/run_on_server.sh
```

All rows must belong to the source cohort. No model fitting, feature selection,
multiple-testing optimization, or inferential causal claim is made. Descriptive
group summaries are exported only.

## Optional source APK archive evidence

Create a CSV `sha256,year,apk_path` with the exact original source-2017 APK
locations. Partial coverage is permitted but the output is explicitly marked
partial. APK file contents are SHA256-verified before inspection:

```bash
ECA_APK_INDEX=/absolute/path/to/source_2017_apk_index.csv \
bash scripts/care-direct-c43-error-attribution-audit-v1/run_on_server.sh
```

The optional scan only reports ZIP facts such as `.so`, multiple `classes*.dex`,
embedded code archives, and raw DEX class-name string hints for loaders,
reflection and JNI library loading. These are **NOT proof that the code runs**,
that a sensitive flow is present, or that C43 omitted a causal feature. A
source-to-C43 extraction-trace check is necessary for any such claim.

## Optional source Stage-I reference identity check

The original temporal-audit script reads `Global24__prob` and `A25__routed`
for 2017. If there is a separately trusted source-2017 P31-SVM OOF export, make
a CSV with `sha256,year,stage1_prob,routed` and pass:

```bash
ECA_STAGE1_REFERENCE=/absolute/path/to/canonical_p31_svm_source_oof.csv \
bash scripts/care-direct-c43-error-attribution-audit-v1/run_on_server.sh
```

The audit hard-stops unless all 799 probabilities match within 1e-12 and the
binary route flags match exactly. Without this optional reference, the report
says `UNVERIFIED_LEGACY_COLUMN_ALIAS`, even when all frozen metrics match.

## Manual static-evidence confirmation workflow

1. Run the base audit and copy `manual_review_blinded_template.csv` to a
   separately named review CSV. Do not edit the generated original or the
   frozen prediction files.
2. First inspect source DEX, method instructions, sensitive API mapping, Host
   selection and RoleGraph/Compact field extraction without opening
   `review_case_key_SOURCE_ONLY.csv`.
3. For each observed candidate, fill in: reviewer; source artifact path and
   its SHA256; exact method/instruction evidence; observed static fact; exact
   C43 mapping or parser that was checked; and independent verification notes.
4. Use one of: `UNREVIEWED`, `NOT_DETERMINED`, `STATIC_OBSERVATION_GAP`,
   `EXTRACTION_INCOMPLETE`, `OBSERVED_IN_C43_BUT_AMBIGUOUS`, `LABEL_DISPUTE`,
   or `OTHER_DOCUMENTED`.
5. Run the review validator:

```bash
ECA_REVIEW_INPUT=/absolute/path/to/my_completed_review.csv \
bash scripts/care-direct-c43-error-attribution-audit-v1/collect_review_results.sh
```

The review validator verifies all 52 case identities and evidence file hashes.
If the evidence path is an `.apk`, it also requires the APK hash to match the
source case SHA. It does **not** certify reviewer reasoning or prove causality.

## Explicit boundaries

- Do not mix 2018-2021 future diagnostics with the 2017 source audit.
- Do not choose new model or route parameters using old future results.
- A score's wrong side of 0.5 is not itself a proof of a directionally missing
  static feature. Similarly, retrospective class-pair inversions, C43 hidden
  collision, or absence of a trace do not prove an unavoidable Bayes error.
- `STATIC_OBSERVATION_GAP` means a reviewer has documented an observation and
  missing C43 field/extraction mapping; even with hash checks, this remains a
  *candidate mechanism*, not a validated cause of the prediction.
- Do not call first-layer errors unreachable merely because the 25% route did
  not include them. The primary target is the source **routed** set.
- The 54 and 43 previous categories remain excluded until their original
  generating definitions/code are independently recovered.

## r2 self-test hotfix

The synthetic self-test now explicitly isolates `ECA_OUTPUT` and optional APK input variables from the live server environment. `run_audit.py` path guards are **unchanged**. If r1 stopped at self-test, replacing only `selftest.py` with r2 is sufficient; no formal source data or trained model needs changing.
