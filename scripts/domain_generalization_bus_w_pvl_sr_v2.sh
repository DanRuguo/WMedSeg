#!/bin/bash
set -e

OUTPUT=${1:-outputs_bus_w_pvl_sr_v2}
SEED=${2:-666}
shift 2 || true
EXTRA_OPTS=("$@")

SOURCE=BUSI
TARGETS=(BUSBRA BUSUC BUID UDIAT)

CONFIGS=(BUSI BUSBRA BUSUC BUID UDIAT)
for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "configs/${cfg}.yaml" ]; then
        echo "Missing config: configs/${cfg}.yaml"
        exit 1
    fi
done

echo "===================================="
echo "SOURCE DATASET (fully supervised): ${SOURCE}"
echo "TARGET DATASETS: ${TARGETS[*]}"
echo "DECODER: w_pvl_sr + hybrid fuse + deep supervision"
echo "OUTPUT: ${OUTPUT}"
echo "SEED: ${SEED}"
echo "EXTRA OPTS: ${EXTRA_OPTS[*]}"
echo "===================================="

python train.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED} \
    "${EXTRA_OPTS[@]}"

echo "Evaluating SOURCE=${SOURCE} -> TARGET=${SOURCE} (in-domain)"
python test.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${SOURCE} \
    --seed ${SEED} \
    "${EXTRA_OPTS[@]}"

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
        "${EXTRA_OPTS[@]}"

    python utils/eval.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --seed ${SEED}
done
