# CARE-Droid Q correction-input interaction audit v1

Read-only strict-2017 audit of the complete Host-held-present factorial:

- H: Host only
- HR: Host + direct RoleToken
- HC: Host + Global24/Stage-I logit
- HRC: Full Q

It computes aggregate and fold-level interaction contrasts, exact paired tests,
and per-APK transitions showing Host harms prevented or benefits lost by Full Q.
No model training/inference, GPU, Ego, route change, or future row is used.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-q-correction-interaction-audit-v1.zip -d scripts/
bash scripts/care-di25-q-correction-interaction-audit-v1/run_on_server.sh
cat artifacts/care-di25-q-correction-interaction-audit-v1/REPORT.txt
```
