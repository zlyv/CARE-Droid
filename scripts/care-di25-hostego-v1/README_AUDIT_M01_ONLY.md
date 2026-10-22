# M0/M1 post-hoc audit-only patch

This patch reads the already completed M0/M1 artifacts. It does not call the
training runner, prediction inference, M2 or M3. The only model operation is
parameter-free model construction for `sum(p.numel() for p in
model.parameters())`.

Run:

```bash
bash scripts/care-di25-hostego-v1/launch_audit_m01_tmux.sh
tail -F artifacts/care-di25-hostego-v1.audit-m01.console.log
```

The hard result lock is evaluated before audit acceptance. The saved M0/M1
prediction files are SHA-256 hashed before and after the audit. Outputs are
written only below `artifacts/care-di25-hostego-v1/audits/`.
