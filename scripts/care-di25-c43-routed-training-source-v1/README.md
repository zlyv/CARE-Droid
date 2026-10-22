# CARE-Droid C43-RQ routed-training source ablation v1

Strict 2017-only comparison of Stage-II training scope. It reuses the locked C43-RQ source result as T0 and trains two capacity-matched candidates:

- `T1_ROUTED_LOSS`: neural loss uses the 160 routed outer-train APKs; fold-safe Host backgrounds and Stage-II graph scaler still use every outer-train APK.
- `T2_FULLY_ROUTED`: neural loss, Host backgrounds, and Stage-II graph scaler use only the 160 routed outer-train APKs.

Both candidates generate expert probabilities for every validation APK, but final correction remains restricted to the frozen 40-per-fold DI25 route. The exact selected epoch for every fold and seed is inherited from locked T0, eliminating routed-cohort early-stop retuning. T0 is not retrained. No future row or Ego tensor is consumed.

```bash
cd /home/boyuanjie/projects/tasa-droid
bash scripts/care-di25-c43-routed-training-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-c43-routed-training-source-v1.console.log
```
