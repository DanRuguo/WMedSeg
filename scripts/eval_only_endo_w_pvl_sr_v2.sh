#!/bin/bash
set -e

# Usage:
#   bash scripts/eval_only_endo_w_pvl_sr_v2.sh <output_dir> <seed> [source_idx] [prompt_design] [extra opts...]
#
# Example:
#   bash scripts/eval_only_endo_w_pvl_sr_v2.sh outputs_endo_w_pvl_sr_v2 666
#   bash scripts/eval_only_endo_w_pvl_sr_v2.sh outputs_endo_w_pvl_sr_v2 666 0 original
#   bash scripts/eval_only_endo_w_pvl_sr_v2.sh outputs_endo_w_pvl_sr_v2 666 2 original TEST.THRESHOLD 0.45

OUTPUT=$1
SEED=$2
SOURCE_IDX=${3:-0}
PROMPT_DESIGN=${4:-original}
EXTRA_OPTS=("${@:5}")

ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")

if [ -z "$OUTPUT" ] || [ -z "$SEED" ]; then
  echo "Usage: bash scripts/eval_only_endo_w_pvl_sr_v2.sh <output_dir> <seed> [source_idx] [prompt_design] [extra opts...]"
  exit 1
fi

if [ "$SOURCE_IDX" -lt 0 ] || [ "$SOURCE_IDX" -ge "${#ENDO[@]}" ]; then
  echo "SOURCE_IDX out of range: $SOURCE_IDX"
  exit 1
fi

SOURCE=${ENDO[$SOURCE_IDX]}

echo "===================================="
echo "EVAL ONLY MODE"
echo "SOURCE DATASET: ${SOURCE}"
echo "TARGET DATASETS: ${ENDO[@]}"
echo "PROMPT DESIGN: ${PROMPT_DESIGN}"
echo "EXTRA OPTS: ${EXTRA_OPTS[@]}"
echo "===================================="

for TARGET in "${ENDO[@]}"; do
  echo "Evaluating SOURCE=${SOURCE} -> TARGET=${TARGET}"

  python test.py \
    --config-file configs/${TARGET}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${SOURCE} \
    --seed ${SEED} \
    --prompt_design ${PROMPT_DESIGN} \
    "${EXTRA_OPTS[@]}"

  python utils/eval.py \
    --config-file configs/${TARGET}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}
done