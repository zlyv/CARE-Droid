# CARE-Droid Host Complementarity Audit v1

Completely isolated, read-only post-hoc diagnostic over saved strict same-year
OOF need-correction probe probabilities. It asks whether Critical or Attended
Host/Ego probes uniquely flag Wrong-Direction cases missed by the Graph probe.

This package does not train/replay CARE-Droid or any probe, write a checkpoint,
search a threshold, modify DI25, or change a prediction. The fixed historical
probe threshold is exactly `0.5`.

The 2018–2021 same-year OOF cohorts are development/diagnostic data. They are
not described as unseen, future, zero-day, or an independent temporal test.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-hostego-complementarity-audit-v1.zip -d scripts/
bash scripts/care-di25-hostego-complementarity-audit-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostego-complementarity-audit-v1.console.log
```

After `HOST_COMPLEMENTARITY_AUDIT_DONE`, press `Ctrl+C` to leave `tail -F`.

```bash
cat artifacts/care-di25-hostego-complementarity-audit-v1/reports/COMPLEMENTARITY_REPORT.txt
cat artifacts/care-di25-hostego-complementarity-audit-v1/reports/COMPLEMENTARITY_DECISION.txt
```

The decision is a diagnostic classification only. Even a supported result does
not authorize or establish a deployable Host complementary-evidence module.
