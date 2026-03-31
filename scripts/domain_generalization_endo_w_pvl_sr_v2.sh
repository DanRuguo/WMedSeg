#!/bin/bash
set -e

OUTPUT=$1
SEED=$2
SOURCE_IDX=${3:-0}
PROMPT_DESIGN=${4:-original}

ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")

if [ -z "$OUTPUT" ] || [ -z "$SEED" ]; then
  echo "Usage: bash scripts/domain_generalization_endo_w_pvl_sr_v2.sh <output_dir> <seed> [source_idx] [prompt_design]"
  exit 1
fi

if [ "$SOURCE_IDX" -lt 0 ] || [ "$SOURCE_IDX" -ge "${#ENDO[@]}" ]; then
  echo "SOURCE_IDX out of range: $SOURCE_IDX"
  echo "0=Kvasir 1=ColonDB 2=ClinicDB 3=CVC300 4=BKAI"
  exit 1
fi

SOURCE=${ENDO[$SOURCE_IDX]}
TARGETS=()
for i in "${!ENDO[@]}"; do
  if [ "$i" -ne "$SOURCE_IDX" ]; then
    TARGETS+=("${ENDO[$i]}")
  fi
done

echo "===================================="
echo "GROUP: ENDO"
echo "SOURCE DATASET: ${SOURCE}"
echo "TARGET DATASETS: ${TARGETS[@]}"
echo "PROMPT DESIGN: ${PROMPT_DESIGN}"
echo "===================================="

python train.py \
  --config-file configs/${SOURCE}.yaml \
  --output-dir ${OUTPUT} \
  --seed ${SEED}

echo "Evaluating SOURCE=${SOURCE} -> TARGET=${SOURCE} (in-domain)"
python test.py \
  --config-file configs/${SOURCE}.yaml \
  --output-dir ${OUTPUT} \
  --source_dataset ${SOURCE} \
  --seed ${SEED} \
  --prompt_design ${PROMPT_DESIGN}

python utils/eval.py \
  --config-file configs/${SOURCE}.yaml \
  --output-dir ${OUTPUT} \
  --seed ${SEED}

for TARGET in "${TARGETS[@]}"; do
  echo "Evaluating SOURCE=${SOURCE} -> TARGET=${TARGET}"
  python test.py \
    --config-file configs/${TARGET}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${SOURCE} \
    --seed ${SEED} \
    --prompt_design ${PROMPT_DESIGN}

  python utils/eval.py \
    --config-file configs/${TARGET}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}
done
