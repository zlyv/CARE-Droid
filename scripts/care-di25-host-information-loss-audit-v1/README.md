# CARE-Droid Host information-loss audit v1

Label-free, read-only audit of the frozen 2017 Host representation. It checks
constant/duplicate dimensions and reconstructs endpoint multiplicity,
directional Host degrees, and cached ICC/reflection/non-direct/upstream/
downstream context without reanalyzing APKs.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-host-information-loss-audit-v1.zip -d scripts/
bash scripts/care-di25-host-information-loss-audit-v1/run_on_server.sh
cat artifacts/care-di25-host-information-loss-audit-v1/REPORT.txt
```

No label, model training/inference, GPU, Ego, or future row is used. The audit
does not automatically authorize the proposed 74-D Host-V2 representation.
