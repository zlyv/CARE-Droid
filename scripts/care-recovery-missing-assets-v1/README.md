# CARE-Droid missing asset recovery v1

This source-only recovery restores two immutable bundled assets, performs the
one unavoidable append-resumable Compact76 extraction for the fixed 799 APKs,
refits the strict source OOF committee, and reconstructs DI25 by its frozen
formula. It does not read 2018--2021 data and does not retrain Stage II.

Run in tmux:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-recovery-missing-assets-v1.zip -d scripts/
bash scripts/care-recovery-missing-assets-v1/launch_tmux.sh
tail -F artifacts/care-recovery-missing-assets-v1.console.log
```

The final hard gate is:

```text
MISSING_ASSETS_RECOVERY_GATE_PASS
RECOVERY_ASSET_AUDIT missing=0
CARE_DROID_MISSING_ASSETS_RECOVERY_DONE
```

The future Compact76 file is intentionally never accepted as a source-year
replacement.
