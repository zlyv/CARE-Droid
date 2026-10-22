# Reproduction

```bash
cd /home/boyuanjie/projects/tasa-droid

test ! -e scripts/care-static-blindspot-modality-audit-v1 || exit 1
test ! -e outputs/care-static-blindspot-modality-audit-v1 || exit 1

unzip -q ../care-static-blindspot-modality-audit-v1-package-r1.zip -d scripts/
chmod +x scripts/care-static-blindspot-modality-audit-v1/*.sh

SBMA_WORKERS=3 bash scripts/care-static-blindspot-modality-audit-v1/launch_tmux.sh
tail -F outputs/care-static-blindspot-modality-audit-v1.console.log
```

After completion:

```bash
bash scripts/care-static-blindspot-modality-audit-v1/collect_results.sh
```
