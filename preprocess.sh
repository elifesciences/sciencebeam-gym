#!/bin/bash

source prepare-shell.sh

export IMAGE_WIDTH=256
export IMAGE_HEIGHT=256

if [ $USE_CLOUD == true ]; then
  CLOUD_ARGS="--cloud"
else
  CLOUD_ARGS=""
fi

python -m sciencebeam_gym.trainer.preprocess \
  --data_paths "${DATA_PATH}/train/*/*/*-page1*.png" \
  --output_path "${PREPROC_PATH}/train/" \
  --image_width ${IMAGE_WIDTH} \
  --image_height ${IMAGE_HEIGHT} \
  --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}" \
  --num_workers 1 \
  "${CLOUD_ARGS}"

python -m sciencebeam_gym.trainer.preprocess \
  --data_paths "${DATA_PATH}/test/*/*/*-page1*.png" \
  --output_path "${PREPROC_PATH}/test/" \
  --image_width ${IMAGE_WIDTH} \
  --image_height ${IMAGE_HEIGHT} \
  --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}" \
  --num_workers 1 \
  "${CLOUD_ARGS}"

if [ ! -z "$QUALITATIVE_FOLDER_NAME" ]; then
  python -m sciencebeam_gym.trainer.preprocess \
    --data_paths "${DATA_PATH}/${QUALITATIVE_FOLDER_NAME}/*/*/*-page1*.png" \
    --output_path "${PREPROC_PATH}/${QUALITATIVE_FOLDER_NAME}/" \
    --image_width ${IMAGE_WIDTH} \
    --image_height ${IMAGE_HEIGHT} \
    --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}" \
    --num_workers 1 \
    "${CLOUD_ARGS}"
fi
