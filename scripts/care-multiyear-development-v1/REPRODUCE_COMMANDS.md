# Commands for the already-deployed server branch

```bash
cd /home/boyuanjie/projects/tasa-droid

# All 2788 cached samples: counts, SHA overlap, labels, Role order, shapes,
# finite values, original-file SHA/size/mtime verification.
bash scripts/care-multiyear-development-v1/run_experiments.sh --audit-only --resume

# Protocol tests and a real original-network smoke (no GPU contention).
PYTHONDONTWRITEBYTECODE=1 /home/boyuanjie/envs/tif/bin/python -B scripts/care-multiyear-development-v1/selftest.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash scripts/care-multiyear-development-v1/run_experiments.sh --smoke-test --device cpu --resume

# These CPU phases have already completed; --resume verifies and reuses models.
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage stage1 --resume
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage router --resume

# NOT launched during implementation: check resources and run actual deep training.
nvidia-smi
CUDA_VISIBLE_DEVICES=0 \
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage stage2 --resume

# Run only after every requested expert window completes.
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage cascade --resume
bash scripts/care-multiyear-development-v1/run_experiments.sh --collect-only --resume
```

GPU0 is only an example: check the live occupancy before use. Neither user's
GPU process is terminated by this package. Use `--windows T1` for a first single
window. A CPU alternative keeps the exact architecture but will be slower.
The deep training phase is explicitly pending; smoke is not a Q3 result.

Reports are under `/home/boyuanjie/projects/tasa-droid/outputs/care_multiyear_development_v1/`.
`stage1_results.csv` and primary rows of `router_results.csv` are actual completed
rolling development results. Empty `stage2_results.csv`, `cascade_results.csv`
and `ablation_results.csv` mark pending deep experiments, not zero improvement.
Synthetic evaluation-unit fixtures are isolated in `selftest-fixtures` and are
never collected into experimental metrics. Early engineering-build checkpoints
are retained in `models_build_*`; only current `models/` are resumable by this code.
