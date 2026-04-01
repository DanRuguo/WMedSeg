#!/bin/bash
set -e

OUTPUT=$1
SEED=${2:-1}
PROMPT_DESIGN=${3:-original}
SOURCE=Kvasir
TARGETS=(Kvasir ColonDB ClinicDB CVC300 BKAI)

if [ -z "$OUTPUT" ]; then
  echo "Usage: bash scripts/domain_generalization_endo_ood_fix.sh <output_dir> [seed] [prompt_design]"
  exit 1
fi

echo "===================================="
echo "ENDO OOD FIX RUN"
echo "SOURCE DATASET: ${SOURCE}"
echo "TARGET DATASETS: ${TARGETS[@]}"
echo "PROMPT DESIGN: ${PROMPT_DESIGN}"
echo "===================================="

python train.py   --config-file configs/${SOURCE}.yaml   --output-dir ${OUTPUT}   --seed ${SEED}

for TARGET in "${TARGETS[@]}"; do
  echo "Evaluating SOURCE=${SOURCE} -> TARGET=${TARGET}"
  python test.py     --config-file configs/${TARGET}.yaml     --output-dir ${OUTPUT}     --source_dataset ${SOURCE}     --seed ${SEED}     --prompt_design ${PROMPT_DESIGN}

  python utils/eval.py     --config-file configs/${TARGET}.yaml     --output-dir ${OUTPUT}     --seed ${SEED}     --prompt_design ${PROMPT_DESIGN}
done
