#!/bin/bash
set -e

OUTPUT=${1:-outputs_wmedseg}
SEED=${2:-666}

# 1) train on source BTMRI
python train.py \
  --config-file configs/BTMRI.yaml \
  --output-dir ${OUTPUT} \
  --seed ${SEED}

# 2) evaluate in-domain on BTMRI
python test.py \
  --config-file configs/BTMRI.yaml \
  --output-dir ${OUTPUT} \
  --source_dataset BTMRI \
  --seed ${SEED}

python utils/eval.py \
  --config-file configs/BTMRI.yaml \
  --output-dir ${OUTPUT} \
  --seed ${SEED}

# 3) evaluate out-of-domain on BRISC using the BTMRI checkpoint
python test.py \
  --config-file configs/BRISC.yaml \
  --output-dir ${OUTPUT} \
  --source_dataset BTMRI \
  --seed ${SEED}

python utils/eval.py \
  --config-file configs/BRISC.yaml \
  --output-dir ${OUTPUT} \
  --seed ${SEED}
