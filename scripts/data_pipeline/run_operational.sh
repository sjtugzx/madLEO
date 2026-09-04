#!/usr/bin/env bash
# MAD-LEO operational subset pipeline (Starlink). No label construction:
# the operational subset carries no mission-reported events.
set -euo pipefail
cd "$(dirname "$0")/../.."

set -a
[ -f .env ] && source .env
source scripts/data_pipeline/params/operational.env
set +a

echo "== MAD-LEO operational pipeline (SUBSET=$SUBSET, DRY_RUN=$DRY_RUN) =="

python scripts/data_pipeline/pipeline/stage1_collection.py
python scripts/data_pipeline/pipeline/stage2_processing.py
python scripts/data_pipeline/pipeline/stage3_aggregation_alignment.py
python scripts/data_pipeline/pipeline/stage4_label_construction.py
