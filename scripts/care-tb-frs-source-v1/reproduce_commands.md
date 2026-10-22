# Server reproduction

```bash
cd /home/boyuanjie/projects/tasa-droid

test ! -e scripts/care-tb-frs-source-v1 || {
  echo "scripts/care-tb-frs-source-v1 already exists; stop to avoid overwrite"
  exit 1
}

test ! -e outputs/care-tb-frs-source-v1 || {
  echo "outputs/care-tb-frs-source-v1 already exists; stop to avoid overwrite"
  exit 1
}

unzip -q ../care-tb-frs-source-v1-package-r1.zip -d scripts/
chmod +x scripts/care-tb-frs-source-v1/*.sh
bash scripts/care-tb-frs-source-v1/launch_tmux.sh
tail -F outputs/care-tb-frs-source-v1.console.log
```

After completion:

```bash
bash scripts/care-tb-frs-source-v1/collect_results.sh
```
