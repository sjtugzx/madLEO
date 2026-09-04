#!/usr/bin/env bash
# MAD-LEO reference subset pipeline: four stages as described in Methods.
# Usage: bash scripts/data_pipeline/run_reference.sh   (dry-run, prints the plan)
#        DRY_RUN=0 bash scripts/data_pipeline/run_reference.sh   (execute)
set -euo pipefail
cd "$(dirname "$0")/../.."

set -a
[ -f .env ] && source .env
source scripts/data_pipeline/params/reference.env
set +a

echo "== MAD-LEO reference pipeline (SUBSET=$SUBSET, DRY_RUN=$DRY_RUN) =="
echo "write targets: TABLES_DIR=$TABLES_DIR (plus configured data/interim paths)"

python scripts/data_pipeline/pipeline/stage1_collection.py
python scripts/data_pipeline/pipeline/stage2_processing.py
python scripts/data_pipeline/pipeline/stage3_aggregation_alignment.py
python scripts/data_pipeline/pipeline/stage4_label_construction.py
