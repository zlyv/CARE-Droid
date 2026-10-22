# P31-DI25 C43-RQ failure-mechanism audit

Read-only strict-2017 audit. It decomposes the 46 routed P31 errors into:

- fixed by bounded correction;
- theoretically unreachable under the 2.25 logit bound;
- wrong/zero correction direction;
- correct direction but insufficient magnitude.

It also cross-tabulates Bounded and Direct fixes, and audits the 154 routed
P31-correct APKs that each mechanism preserves or breaks.

No training, inference, route modification, threshold search, future data, or
Ego tensor is used.

Run:

```bash
unzip -o ../care-litelocal-p31-c43-failure-audit-v1.zip -d scripts/
bash scripts/care-litelocal-p31-c43-failure-audit-v1/run_on_server.sh
cat artifacts/care-litelocal-p31-c43-failure-audit-v1/REPORT.txt
```
