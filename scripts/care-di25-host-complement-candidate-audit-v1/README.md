# CARE-Droid Host Complement Candidate Audit v1

This package performs a fully isolated, read-only diagnostic audit of three
label-free Host complement feature families on the frozen 2018--2021
same-year OOF cohort:

- **A**: intra-Host semantic-role composition;
- **B**: Host structural position in the frozen bounded endpoint graph;
- **C**: short-range Host coordination at ordinary-node gaps 0, 1 and 2.

It does not train or replay CARE-Droid, Global24, the existing R4 probes, or
any neural network. It does not re-analyse APKs. The only fitted models are
fixed L2 logistic-regression diagnostic probes, trained independently within
each year under strict five-fold OOF isolation and only on the frozen DI25
routed cohort.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-host-complement-candidate-audit-v1.zip -d scripts/
bash scripts/care-di25-host-complement-candidate-audit-v1/launch_tmux.sh
tail -F artifacts/care-di25-host-complement-candidate-audit-v1.console.log
```

## Contract highlights

- old artifacts are hashed before and after the audit;
- future labels never enter feature extraction;
- the probe target is `NeedCorrection = (Stage-I prediction != label)`;
- no OOF Graph probability is used as a meta-classifier input;
- 10,000-iteration stratified paired bootstrap is used for incremental tests;
- a candidate is `SUPPORTED` only when both general OOF signal and
  Graph-negative conditional complementarity have confidence support;
- all conclusions remain post-hoc development/diagnostic evidence.

The final report is written to:

`artifacts/care-di25-host-complement-candidate-audit-v1/reports/HOST_COMPLEMENT_CANDIDATE_REPORT.txt`
