# CARE-Droid Q-Query drop-direct-RoleToken source v1

One strict-2017 arm tests the missing architecture:

```text
RoleToken + GraphGlobal -> Host attention
Global24 + Stage-I logit + selected Host + summary -> bounded delta
RoleToken -X-> bounded delta
```

It keeps the same 33,036 parameters and 113-D interface, with the 32-D direct
RoleToken slot masked to zero (81 active dimensions). Frozen DI25 routes exactly
200/799 APKs. No Ego or future row is consumed. FULL_Q is reused, not retrained.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-q-drop-role-source-v1.zip -d scripts/
bash scripts/care-di25-q-drop-role-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-q-drop-role-source-v1.console.log
```
