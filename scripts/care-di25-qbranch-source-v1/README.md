# CARE-Droid Q-Query branch isolation source v1

This is a strict-2017, frozen-DI25, Host-only ablation. It answers whether the
RoleToken branch or the selected Host branch is sufficient for bounded
correction, and whether GraphGlobal-guided attention helps Host-only evidence.

Arms (all instantiate the same 33,036 parameters and 113-D correction head):

- `FULL_Q`: locked completed `Q_QUERY_ONLY` reference; never retrained here.
- `ROLE_ONLY`: only the 32-D RoleToken reaches the correction head.
- `HOST_ONLY_Q`: CriticalHost + AttendedHost + Summary, with GraphGlobal query.
- `HOST_ONLY_N0`: identical Host inputs, without GraphGlobal query.
- `STRUCTURE_ONLY_Q`: RoleToken + Host inputs, no Global24 or Stage-I logit.
- `STAGE1_CONTEXT_ONLY`: Global24 + Stage-I logit only.

The masks are fixed and parameter-free. No Ego tensor or future row is read.
The source loader, fold-safe inner-OOF Stage-I construction, training loop and
bounded correction are inherited from the already validated GraphGlobal-path
source package.

Server usage from `/home/boyuanjie/projects/tasa-droid`:

```bash
unzip -o ../care-di25-qbranch-source-v1.zip -d scripts/
bash scripts/care-di25-qbranch-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-qbranch-source-v1.console.log
```

Final report:

```bash
cat artifacts/care-di25-qbranch-source-v1/results/REPORT.txt
```
