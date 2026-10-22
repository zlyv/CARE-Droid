#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

bash scripts/care-di-budget-bound-selection-v1/run_source_grid_on_server.sh

echo "Source-only selection finished. Inspect and freeze selected_source_config.json."
echo "Run run_selected_future_on_server.sh separately only after the source decision is frozen."
