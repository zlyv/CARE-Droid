# RCDB-V1 autonomous run

Launch from the project root after the foreground smoke test:

```bash
RCDB_DEVICE=cuda CUDA_VISIBLE_DEVICES=1 bash scripts/care-rcdb-v1/launch.sh
```

The launcher prints the actual `SESSION` and `OUTPUT` paths. Check progress:

```bash
tmux ls
tmux attach -t ACTUAL_SESSION
tail -n 50 ACTUAL_OUTPUT/run.log
cat ACTUAL_OUTPUT/run_state.json
bash scripts/care-rcdb-v1/status.sh ACTUAL_OUTPUT
find ACTUAL_OUTPUT/models -type f -name '*.pkl' | wc -l
```

Detach safely from tmux with `Ctrl+B`, then `D`. A missing tmux session after
completion is normal; check `exit_code.txt` and `completed.flag`.

Resume a failed run, preserving validated checkpoints and the same output:

```bash
tmux new-session -d -s NEW_SESSION "RCDB_OUTPUT='ACTUAL_OUTPUT' RCDB_RESUME=1 RCDB_DEVICE=cuda CUDA_VISIBLE_DEVICES=1 bash /home/boyuanjie/projects/tasa-droid/scripts/care-rcdb-v1/run.sh"
```

Do not change the code or original assets between run and resume; the asset
manifest fails closed if they change. Final results appear at
`ACTUAL_OUTPUT/final_report.md` and `ACTUAL_OUTPUT/final_decision.json`.
