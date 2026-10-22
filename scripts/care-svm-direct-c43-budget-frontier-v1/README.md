# SVM Direct-C43 Budget–Response Frontier v1

This is a source-2017 descriptive audit. It reuses the already frozen strict-OOF Direct-C43 expert probability for every one of the 799 APKs and changes only the admission count along the unchanged four-member, SVM-anchored MeanOpposition ordering.

It does not train a model, tune or select a deployment budget, create a benefit gate, load future data, or infer runtime from the route fraction. Top25% must exactly reproduce fixed=49, broken=17 and net=32 before any frontier row is accepted.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid

test ! -e scripts/care-svm-direct-c43-budget-frontier-v1 || {
  echo "脚本目录已存在，停止以避免覆盖"; exit 1;
}
test ! -e outputs/care-svm-direct-c43-budget-frontier-v1 || {
  echo "输出目录已存在，停止以避免覆盖"; exit 1;
}

unzip -q ../care-svm-direct-c43-budget-frontier-v1-package-r1.zip -d scripts/
chmod +x scripts/care-svm-direct-c43-budget-frontier-v1/*.sh
bash scripts/care-svm-direct-c43-budget-frontier-v1/launch_tmux.sh
tail -F outputs/care-svm-direct-c43-budget-frontier-v1.console.log
```

Collect:

```bash
bash scripts/care-svm-direct-c43-budget-frontier-v1/collect_results.sh
```

This should normally finish in under one minute and does not use a GPU.

