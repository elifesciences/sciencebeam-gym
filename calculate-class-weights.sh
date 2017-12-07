#!/bin/bash

set -e

source prepare-shell.sh

echo "output will be written to: ${CLASS_WEIGHTS_FILENAME}"

python -m sciencebeam_gym.tools.calculate_class_weights \
  --tfrecord-paths "${TRAIN_PREPROC_PATH}/*tfrecord*" \
  --image-key "annotation_image" \
  --color-map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}" \
  --channels="$CHANNEL_NAMES" \
  --out "${CLASS_WEIGHTS_FILENAME}"

echo "output written to: ${CLASS_WEIGHTS_FILENAME}"
