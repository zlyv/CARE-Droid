# Reproduce

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -q ../care-svm-c43-representation-collision-audit-v1-package-r1.zip -d scripts/
chmod +x scripts/care-svm-c43-representation-collision-audit-v1/*.sh
bash scripts/care-svm-c43-representation-collision-audit-v1/launch_tmux.sh
tail -F outputs/care-svm-c43-representation-collision-audit-v1.console.log
```

After completion:

```bash
bash scripts/care-svm-c43-representation-collision-audit-v1/collect_results.sh
```
