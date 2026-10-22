# Reproduction

```bash
cd /home/boyuanjie/projects/tasa-droid

test ! -e scripts/care-consensus-error-lineage-label-audit-v1 || {
  echo "script directory exists; stop to avoid overwrite"
  exit 1
}

test ! -e outputs/care-consensus-error-lineage-label-audit-v1 || {
  echo "output directory exists; choose a new version before running"
  exit 1
}

unzip -q ../care-consensus-error-lineage-label-audit-v1-package-r2.zip -d scripts/
chmod +x scripts/care-consensus-error-lineage-label-audit-v1/*.sh

CELA_WORKERS=4 \
bash scripts/care-consensus-error-lineage-label-audit-v1/launch_tmux.sh

tail -F outputs/care-consensus-error-lineage-label-audit-v1.console.log
```

After completion:

```bash
bash scripts/care-consensus-error-lineage-label-audit-v1/collect_results.sh
```
